// 3x3 "SAME" convolution of 128 channels on Hex boards, as an implicit GEMM on
// the tensor cores (mma.sync m16n8k16, bf16 inputs, f32 accumulation).
//
//   out[r, n] = sum over taps t and input channels c of
//               in[neighbour(r, t), c] * W[n, t * 128 + c]
//
// Rows r = board * cells + cell are positions (NHWC), the neighbour of a
// position outside its board reads zeros (the convolution's zero padding).
// Each CTA computes 128 rows x all 128 output channels; 8 warps of 64 x 32;
// K = 9 taps x 128 channels in steps of 32 through a 3-stage cp.async
// pipeline. The epilogue rounds the f32 sums to bf16 and applies the layer's
// bias, BatchNorm, ReLU and residual exactly as the flax network does in bf16
// (see Epilogue).

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace scacchi_net {

constexpr int CH = 128;               // channels in and out
constexpr int TAPS = 9;
constexpr int KDIM = TAPS * CH;       // 1152
constexpr int BN = 128;
constexpr int BK = 32;
constexpr int KITERS = KDIM / BK;     // 36
constexpr int STAGES = 3;
constexpr int THREADS = 256;
template <int BM> __host__ __device__ constexpr int stage_bytes() { return (BM + BN) * BK * 2; }  // A and B tiles, bf16
constexpr int CONV_SMEM = STAGES * ((128 + BN) * BK * 2);  // 48 KiB, enough for either tile height

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool valid) {
  const int bytes = valid ? 16 : 0;  // zero fill when invalid
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(dst), "l"(src), "r"(bytes));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { asm volatile("cp.async.wait_group %0;\n" ::"n"(N)); }

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void mma_bf16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// A tile row of BK bf16 = 64 bytes = 4 chunks of 16 bytes; XOR swizzle so that
// ldmatrix's 8 rows hit 8 different bank groups.
__device__ __forceinline__ int swz(int row, int chunk) { return row * 64 + ((chunk ^ ((row >> 1) & 3)) << 4); }

// Per-channel constants of the epilogues, float32 (bias already in bf16 values).
struct Epilogue {
  const float* bias;        // [128] conv bias (bf16-representable values)
  const float* bn_mean;     // [128] BatchNorm applied after: running mean
  const float* bn_mul;      // [128] rsqrt(var + eps) * scale
  const float* bn_beta;     // [128] BatchNorm bias
  const __nv_bfloat16* residual;  // [rows, 128] or null: x = conv + bias + residual
  __nv_bfloat16* out_x;     // [rows, 128] or null: the residual stream after this conv
  __nv_bfloat16* out_z;     // [rows, 128]: relu(bn(.)), the next convolution's input
};

struct ConvShape {
  int rows;   // boards * cells
  int side;   // board side
  int cells;  // side * side
};

// One CTA: rows [m0, m0 + BM) x all 128 output channels; BM = 32 MT, warp tile 16 MT x 32.
template <int MT>
__device__ __forceinline__ void conv3x3_tile(const __nv_bfloat16* __restrict__ in, const __nv_bfloat16* __restrict__ w,
                                             const ConvShape& s, const Epilogue& ep, int m0, char* smem) {
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  constexpr int BM = 32 * MT;
  constexpr int STAGE_BYTES = stage_bytes<BM>();
  constexpr int LOADS = BM / 64;  // A chunks per thread per stage: BM rows x 4 chunks / 256
  const int wm = warp >> 2;  // 0..1: row half
  const int wn = warp & 3;   // 0..3: 32-column quarter

  // Loader: each thread moves 2 A chunks and 2 B chunks per stage.
  const int lrow = tid >> 2;   // 0..63
  const int lchunk = tid & 3;  // 16-byte chunk within a 64-byte row
  int a_base[LOADS], a_y[LOADS], a_x[LOADS];
  bool a_ok[LOADS];
#pragma unroll
  for (int i = 0; i < LOADS; ++i) {
    const int r = m0 + lrow + 64 * i;
    a_ok[i] = r < s.rows;
    const int rr = a_ok[i] ? r : 0;
    const int board = rr / s.cells;
    const int cell = rr - board * s.cells;
    a_y[i] = cell / s.side;
    a_x[i] = cell - a_y[i] * s.side;
    a_base[i] = board * s.cells;
  }
  const uint32_t smem0 = smem_u32(smem);

  auto load_stage = [&](int stage, int kk) {
    const int tap = kk >> 2;
    const int dy = tap / 3 - 1, dx = tap % 3 - 1;
    const int c0 = (kk & 3) * BK + lchunk * 8;
    const uint32_t a_dst = smem0 + stage * STAGE_BYTES;
    const uint32_t b_dst = a_dst + BM * BK * 2;
#pragma unroll
    for (int i = 0; i < LOADS; ++i) {
      const int y = a_y[i] + dy, x = a_x[i] + dx;
      const bool ok = a_ok[i] && y >= 0 && y < s.side && x >= 0 && x < s.side;
      const __nv_bfloat16* src = in + (size_t)(ok ? a_base[i] + y * s.side + x : 0) * CH + c0;
      cp_async16(a_dst + swz(lrow + 64 * i, lchunk), src, ok);
    }
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int n = lrow + 64 * i;
      const __nv_bfloat16* src = w + (size_t)n * KDIM + kk * BK + lchunk * 8;
      cp_async16(b_dst + swz(n, lchunk), src, true);
    }
  };

  float acc[MT][4][4];
#pragma unroll
  for (int i = 0; i < MT; ++i)
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int k = 0; k < 4; ++k) acc[i][j][k] = 0.f;

#pragma unroll
  for (int st = 0; st < STAGES - 1; ++st) {
    load_stage(st, st);
    cp_async_commit();
  }

  for (int kk = 0; kk < KITERS; ++kk) {
    cp_async_wait<STAGES - 2>();
    __syncthreads();
    const int next = kk + STAGES - 1;
    if (next < KITERS) load_stage(next % STAGES, next);
    cp_async_commit();

    const uint32_t a_tile = smem0 + (kk % STAGES) * STAGE_BYTES;
    const uint32_t b_tile = a_tile + BM * BK * 2;
#pragma unroll
    for (int ks = 0; ks < BK / 16; ++ks) {
      uint32_t a[MT][4];
#pragma unroll
      for (int mt = 0; mt < MT; ++mt) {
        const int row = wm * (16 * MT) + mt * 16 + (lane & 15);
        const int chunk = ks * 2 + (lane >> 4);
        ldmatrix_x4(a[mt], a_tile + swz(row, chunk));
      }
      uint32_t b[2][4];
#pragma unroll
      for (int np = 0; np < 2; ++np) {
        const int n = wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3);
        const int chunk = ks * 2 + ((lane >> 3) & 1);
        ldmatrix_x4(b[np], b_tile + swz(n, chunk));
      }
#pragma unroll
      for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) mma_bf16(acc[mt][nt], a[mt], b[nt >> 1][(nt & 1) * 2], b[nt >> 1][(nt & 1) * 2 + 1]);
    }
  }
  cp_async_wait<0>();
  __syncthreads();  // the pipeline buffers become the output tile

  // Epilogue: conv output rounded to bf16 (as the bf16 convolution returns
  // it), + bias in bf16, [+ residual in bf16], then relu(bn(.)) via float32.
  // The 128 x 128 bf16 tile goes through shared memory so that every global
  // load and store is a full 16-byte vector of one row (coalesced).
  __nv_bfloat16* tile = reinterpret_cast<__nv_bfloat16*>(smem);  // [128][TS]
  constexpr int TS = CH + 8;                                      // padded row: no bank conflicts
  const int rows_here = min(BM, s.rows - m0);
  auto copy_out = [&](__nv_bfloat16* dst) {
    for (int i = tid; i < BM * (CH / 8); i += THREADS) {
      const int r = i >> 4, c = (i & 15) * 8;
      if (r < rows_here)
        *reinterpret_cast<uint4*>(dst + (size_t)(m0 + r) * CH + c) = *reinterpret_cast<const uint4*>(tile + r * TS + c);
    }
  };
  if (ep.residual) {
    for (int i = tid; i < BM * (CH / 8); i += THREADS) {
      const int r = i >> 4, c = (i & 15) * 8;
      uint4 v = make_uint4(0u, 0u, 0u, 0u);
      if (r < rows_here) v = *reinterpret_cast<const uint4*>(ep.residual + (size_t)(m0 + r) * CH + c);
      *reinterpret_cast<uint4*>(tile + r * TS + c) = v;
    }
    __syncthreads();
  }
  // Pass 1: v = bf16(bf16(acc) + bias) [+ residual], kept in acc; x written to the tile.
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    const int col = wn * 32 + nt * 8 + (lane & 3) * 2;
    const float b0 = ep.bias[col], b1 = ep.bias[col + 1];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int r = wm * (16 * MT) + mt * 16 + (lane >> 2) + half * 8;
        float v0 = __bfloat162float(__float2bfloat16_rn(acc[mt][nt][half * 2]));
        float v1 = __bfloat162float(__float2bfloat16_rn(acc[mt][nt][half * 2 + 1]));
        v0 = __bfloat162float(__float2bfloat16_rn(v0 + b0));
        v1 = __bfloat162float(__float2bfloat16_rn(v1 + b1));
        __nv_bfloat162* t = reinterpret_cast<__nv_bfloat162*>(tile + r * TS + col);
        if (ep.residual) {
          const __nv_bfloat162 res = *t;
          v0 = __bfloat162float(__float2bfloat16_rn(v0 + __bfloat162float(res.x)));
          v1 = __bfloat162float(__float2bfloat16_rn(v1 + __bfloat162float(res.y)));
        }
        acc[mt][nt][half * 2] = v0;
        acc[mt][nt][half * 2 + 1] = v1;
        if (ep.out_x) *t = __floats2bfloat162_rn(v0, v1);
      }
    }
  }
  if (ep.out_x) {
    __syncthreads();
    copy_out(ep.out_x);
    __syncthreads();
  }
  // Pass 2: z = relu(bf16(bn(v))) to the tile, then out.
#pragma unroll
  for (int nt = 0; nt < 4; ++nt) {
    const int col = wn * 32 + nt * 8 + (lane & 3) * 2;
    const float m0c = ep.bn_mean[col], m1c = ep.bn_mean[col + 1];
    const float u0 = ep.bn_mul[col], u1 = ep.bn_mul[col + 1];
    const float e0 = ep.bn_beta[col], e1 = ep.bn_beta[col + 1];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
      for (int half = 0; half < 2; ++half) {
        const int r = wm * (16 * MT) + mt * 16 + (lane >> 2) + half * 8;
        const float z0 = fmaxf(__bfloat162float(__float2bfloat16_rn(fmaf(acc[mt][nt][half * 2] - m0c, u0, e0))), 0.f);
        const float z1 = fmaxf(__bfloat162float(__float2bfloat16_rn(fmaf(acc[mt][nt][half * 2 + 1] - m1c, u1, e1))), 0.f);
        *reinterpret_cast<__nv_bfloat162*>(tile + r * TS + col) = __floats2bfloat162_rn(z0, z1);
      }
    }
  }
  __syncthreads();
  copy_out(ep.out_z);
}

template <int MT>
__global__ void __launch_bounds__(THREADS, 2) conv3x3_kernel(const __nv_bfloat16* __restrict__ in,
                                                             const __nv_bfloat16* __restrict__ w, ConvShape s,
                                                             Epilogue ep) {
  extern __shared__ __align__(128) char smem[];
  conv3x3_tile<MT>(in, w, s, ep, blockIdx.x * 32 * MT, smem);
}

}  // namespace scacchi_net
