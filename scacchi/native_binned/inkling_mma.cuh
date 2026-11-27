// Blackwell / Ampere PTX Tensor Core dense GEMMs for AZInklingBinnedNet.
//
// Computes Y[M, N] = X[M, K] @ W[N, K]^T driven by a device-side active board
// count `*d_boards` (M = (*d_boards) * cells), CUDA-Graph / command-buffer safe.
//
// Modes:
//   - INT8 = false, SWIGLU = false, BN_OUT = 64|128: exact bf16 MMA -> bf16 [M, N].
//   - INT8 = false, SWIGLU = true,  BN_OUT = 64:     exact bf16 MMA + Capped-SwiGLU -> bf16 [M, N_out].
//   - INT8 = true,  SWIGLU = false, BN_OUT = 64|128, IN_GRP64 = false:
//       per-token s8 x per-channel s8 MMA (m16n8k32, 2x rate) -> bf16 [M, N].
//   - INT8 = true,  SWIGLU = true,  BN_OUT = 64:
//       per-token s8 x per-channel s8 MMA + Capped-SwiGLU + FUSED per-64-group INT8 quantization
//       directly from smem -> int8 [M, N_out] + f32 y_scale_grp[M, N_out/64] (zero bf16 round-trip!).
//   - INT8 = true,  SWIGLU = false, BN_OUT = 64, IN_GRP64 = true:
//       per-64-group s8 x per-channel s8 MMA (consumes fused SwiGLU int8 + group scales directly!).

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <type_traits>

#include "conv_mma.cuh"

namespace scacchi_inkling {

using scacchi_net::cp_async16;
using scacchi_net::cp_async_commit;
using scacchi_net::cp_async_wait;
using scacchi_net::ldmatrix_x4;
using scacchi_net::mma_bf16;
using scacchi_net::smem_u32;
using scacchi_net::swz;

__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}

__device__ __forceinline__ void mma_s8s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

constexpr int GEMM_THREADS = 256;
constexpr int GEMM_BM = 128;      // 4 row warps x 32 rows (MT = 2 m16 tiles/warp)
constexpr int GEMM_STAGES = 3;
constexpr int GEMM_BK_BYTES = 64; // 32 bf16 or 64 int8 per k-stage

template <bool SWIGLU, int BN_OUT = 64>
__host__ __device__ constexpr int gemm_bn_w() {
  return SWIGLU ? 128 : BN_OUT;
}

template <bool SWIGLU, int BN_OUT = 64>
__host__ __device__ constexpr int gemm_smem_bytes() {
  constexpr int stage = (GEMM_BM + gemm_bn_w<SWIGLU, BN_OUT>()) * GEMM_BK_BYTES;
  constexpr int pipe = GEMM_STAGES * stage;
  constexpr int epi = GEMM_BM * (BN_OUT + 8) * 2;
  return pipe > epi ? pipe : epi;
}

__device__ __forceinline__ float capped_swiglu_f32(float g, float u) {
  const float sig = 1.0f / (1.0f + __expf(-g));
  return (4.0f * tanhf(g * 0.25f)) * sig * (25.0f * tanhf(u * 0.04f));
}

__device__ __forceinline__ uint8_t gemm_quant_u8_s8(float v, float inv_scale) {
  int q = __float2int_rn(v * inv_scale);
  q = max(-127, min(127, q));
  return static_cast<uint8_t>(static_cast<int8_t>(q));
}

template <bool INT8, bool SWIGLU, int BN_OUT = 64, bool IN_GRP64 = false>
__global__ void __launch_bounds__(GEMM_THREADS, 2) inkling_gemm_kernel(
    const void* __restrict__ X_,
    const float* __restrict__ x_scale,     // [M] f32 (or [M, K/64] when IN_GRP64)
    const void* __restrict__ W_,
    const float* __restrict__ w_scale,     // [N_w] f32 when INT8, else nullptr
    void* __restrict__ Y_,                 // bf16 [M, y_stride] (or int8 [M, N_out] when INT8 && SWIGLU)
    float* __restrict__ y_scale_grp,       // [M, N_out/64] f32 when INT8 && SWIGLU, else nullptr
    int max_boards,
    int cells,
    int K,
    int N_out,
    int y_stride,
    int y_col_offset,
    const int32_t* __restrict__ d_boards) {
  static_assert(!SWIGLU || BN_OUT == 64, "SWIGLU mode requires BN_OUT == 64");
  static_assert(BN_OUT == 64 || BN_OUT == 128, "BN_OUT must be 64 or 128");
  static_assert(!IN_GRP64 || (INT8 && !SWIGLU && BN_OUT == 64), "IN_GRP64 requires INT8 plain BN=64");

  const int active_boards = d_boards ? *d_boards : max_boards;
  const int M = active_boards * cells;
  if (M <= 0) return;

  constexpr int E = INT8 ? 1 : 2;
  constexpr int BKE = GEMM_BK_BYTES / E;  // 32 (bf16) or 64 (int8)
  constexpr int BN_W = gemm_bn_w<SWIGLU, BN_OUT>();
  constexpr int STAGE_BYTES = (GEMM_BM + BN_W) * GEMM_BK_BYTES;
  constexpr int B_OFFSET = GEMM_BM * GEMM_BK_BYTES;
  constexpr bool DUAL_ACC = SWIGLU || (BN_OUT == 128);

  const int m_tiles = (M + GEMM_BM - 1) / GEMM_BM;
  const int n_tiles = N_out / BN_OUT;
  const int total_tiles = m_tiles * n_tiles;

  const char* X = reinterpret_cast<const char*>(X_);
  const char* W = reinterpret_cast<const char*>(W_);
  extern __shared__ __align__(16) char smem[];
  const uint32_t smem0 = smem_u32(smem);

  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int wm = warp >> 1;  // 0..3 (32 rows per warp -> 2 m16 tiles)
  const int wn = warp & 1;   // 0..1 (32 cols per warp per half -> 4 n8 tiles)

  const int lrow = tid >> 2;   // 0..63
  const int lchunk = tid & 3;  // 0..3 (16-byte chunk in 64-byte row)
  const int kiters = K / BKE;

  for (int tile_id = blockIdx.x; tile_id < total_tiles; tile_id += gridDim.x) {
    const int nt = tile_id % n_tiles;
    const int mt = tile_id / n_tiles;
    const int m0 = mt * GEMM_BM;
    const int n0_out = nt * BN_OUT;
    const int w_row0 = nt * BN_W;

    auto load_stage = [&](int stage, int kk) {
      const int k0 = kk * BKE + lchunk * (16 / E);
      const uint32_t a_dst = smem0 + stage * STAGE_BYTES;
      const uint32_t b_dst = a_dst + B_OFFSET;
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const int r = m0 + lrow + 64 * i;
        const bool ok = r < M;
        const char* src = X + ((size_t)(ok ? r : 0) * K + k0) * E;
        cp_async16(a_dst + swz(lrow + 64 * i, lchunk), src, ok);
      }
      constexpr int B_LOADS = BN_W / 64;
#pragma unroll
      for (int i = 0; i < B_LOADS; ++i) {
        const int wr = w_row0 + lrow + 64 * i;
        const char* src = W + ((size_t)wr * K + k0) * E;
        cp_async16(b_dst + swz(lrow + 64 * i, lchunk), src, true);
      }
    };

    using AccT = typename std::conditional<(INT8 && !IN_GRP64), int, float>::type;
    AccT acc0[2][4][4];
    AccT acc1[DUAL_ACC ? 2 : 1][4][4];
#pragma unroll
    for (int i = 0; i < 2; ++i)
#pragma unroll
      for (int j = 0; j < 4; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e) acc0[i][j][e] = AccT(0);
    if constexpr (DUAL_ACC) {
#pragma unroll
      for (int i = 0; i < 2; ++i)
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int e = 0; e < 4; ++e) acc1[i][j][e] = AccT(0);
    }

    const int prefill = kiters < (GEMM_STAGES - 1) ? kiters : (GEMM_STAGES - 1);
    for (int st = 0; st < prefill; ++st) {
      load_stage(st, st);
      cp_async_commit();
    }

    for (int kk = 0; kk < kiters; ++kk) {
      const int inflight = (kk + prefill < kiters) ? prefill : (kiters - 1 - kk);
      if (inflight == 2) cp_async_wait<1>();
      else cp_async_wait<0>();
      __syncthreads();
      const int next = kk + prefill;
      if (next < kiters) {
        load_stage(next % GEMM_STAGES, next);
        cp_async_commit();
      }

      const uint32_t a_tile = smem0 + (kk % GEMM_STAGES) * STAGE_BYTES;
      const uint32_t b_tile = a_tile + B_OFFSET;

      if constexpr (IN_GRP64) {
        int kacc[2][4][4];
#pragma unroll
        for (int mi = 0; mi < 2; ++mi)
#pragma unroll
          for (int nj = 0; nj < 4; ++nj)
#pragma unroll
            for (int e = 0; e < 4; ++e) kacc[mi][nj][e] = 0;
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          uint32_t a[2][4];
#pragma unroll
          for (int mi = 0; mi < 2; ++mi) {
            const int row = wm * 32 + mi * 16 + (lane & 15);
            const int chunk = ks * 2 + (lane >> 4);
            ldmatrix_x4(a[mi], a_tile + swz(row, chunk));
          }
          uint32_t b0[2][4];
#pragma unroll
          for (int np = 0; np < 2; ++np) {
            const int brow = wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3);
            const int chunk = ks * 2 + ((lane >> 3) & 1);
            ldmatrix_x4(b0[np], b_tile + swz(brow, chunk));
          }
#pragma unroll
          for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int nj = 0; nj < 4; ++nj) {
              mma_s8s8(kacc[mi][nj], a[mi], b0[nj >> 1][(nj & 1) * 2], b0[nj >> 1][(nj & 1) * 2 + 1]);
            }
        }
        // Apply per-64-group x_scale for this kk chunk!
#pragma unroll
        for (int mi = 0; mi < 2; ++mi) {
          const int r0 = m0 + wm * 32 + mi * 16 + (lane >> 2);
          const int r1 = r0 + 8;
          const float xs0 = (r0 < M) ? x_scale[(size_t)r0 * kiters + kk] : 0.f;
          const float xs1 = (r1 < M) ? x_scale[(size_t)r1 * kiters + kk] : 0.f;
#pragma unroll
          for (int nj = 0; nj < 4; ++nj) {
            acc0[mi][nj][0] = fmaf(static_cast<float>(kacc[mi][nj][0]), xs0, acc0[mi][nj][0]);
            acc0[mi][nj][1] = fmaf(static_cast<float>(kacc[mi][nj][1]), xs0, acc0[mi][nj][1]);
            acc0[mi][nj][2] = fmaf(static_cast<float>(kacc[mi][nj][2]), xs1, acc0[mi][nj][2]);
            acc0[mi][nj][3] = fmaf(static_cast<float>(kacc[mi][nj][3]), xs1, acc0[mi][nj][3]);
          }
        }
      } else {
#pragma unroll
        for (int ks = 0; ks < 2; ++ks) {
          uint32_t a[2][4];
#pragma unroll
          for (int mi = 0; mi < 2; ++mi) {
            const int row = wm * 32 + mi * 16 + (lane & 15);
            const int chunk = ks * 2 + (lane >> 4);
            ldmatrix_x4(a[mi], a_tile + swz(row, chunk));
          }
          uint32_t b0[2][4];
#pragma unroll
          for (int np = 0; np < 2; ++np) {
            const int brow = wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3);
            const int chunk = ks * 2 + ((lane >> 3) & 1);
            ldmatrix_x4(b0[np], b_tile + swz(brow, chunk));
          }
#pragma unroll
          for (int mi = 0; mi < 2; ++mi)
#pragma unroll
            for (int nj = 0; nj < 4; ++nj) {
              if constexpr (INT8) mma_s8s8(acc0[mi][nj], a[mi], b0[nj >> 1][(nj & 1) * 2], b0[nj >> 1][(nj & 1) * 2 + 1]);
              else mma_bf16(acc0[mi][nj], a[mi], b0[nj >> 1][(nj & 1) * 2], b0[nj >> 1][(nj & 1) * 2 + 1]);
            }

          if constexpr (DUAL_ACC) {
            uint32_t b1[2][4];
#pragma unroll
            for (int np = 0; np < 2; ++np) {
              const int brow = 64 + wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3);
              const int chunk = ks * 2 + ((lane >> 3) & 1);
              ldmatrix_x4(b1[np], b_tile + swz(brow, chunk));
            }
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
#pragma unroll
              for (int nj = 0; nj < 4; ++nj) {
                if constexpr (INT8) mma_s8s8(acc1[mi][nj], a[mi], b1[nj >> 1][(nj & 1) * 2], b1[nj >> 1][(nj & 1) * 2 + 1]);
                else mma_bf16(acc1[mi][nj], a[mi], b1[nj >> 1][(nj & 1) * 2], b1[nj >> 1][(nj & 1) * 2 + 1]);
              }
          }
        }
      }
    }
    cp_async_wait<0>();
    __syncthreads();

    // Coalesced epilogue through smem [128][BN_OUT + 8] bf16.
    __nv_bfloat16* tile = reinterpret_cast<__nv_bfloat16*>(smem);
    constexpr int TS = BN_OUT + 8;
    const int rows_here = min(GEMM_BM, M - m0);

#pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
      const int c_local = wn * 32 + nj * 8 + (lane & 3) * 2;
      const float wg0 = INT8 ? w_scale[w_row0 + c_local] : 1.0f;
      const float wg1 = INT8 ? w_scale[w_row0 + c_local + 1] : 1.0f;
      const float wu0 = (INT8 && DUAL_ACC) ? w_scale[w_row0 + 64 + c_local] : 1.0f;
      const float wu1 = (INT8 && DUAL_ACC) ? w_scale[w_row0 + 64 + c_local + 1] : 1.0f;
#pragma unroll
      for (int mi = 0; mi < 2; ++mi) {
#pragma unroll
        for (int half = 0; half < 2; ++half) {
          const int r_local = wm * 32 + mi * 16 + (lane >> 2) + half * 8;
          const int r_glob = m0 + r_local;
          const float xs = (INT8 && !IN_GRP64 && r_glob < M) ? x_scale[r_glob] : 1.0f;
          float v0 = static_cast<float>(acc0[mi][nj][half * 2]) * xs * wg0;
          float v1 = static_cast<float>(acc0[mi][nj][half * 2 + 1]) * xs * wg1;
          if constexpr (SWIGLU) {
            const float u0 = static_cast<float>(acc1[mi][nj][half * 2]) * xs * wu0;
            const float u1 = static_cast<float>(acc1[mi][nj][half * 2 + 1]) * xs * wu1;
            v0 = capped_swiglu_f32(v0, u0);
            v1 = capped_swiglu_f32(v1, u1);
            *reinterpret_cast<__nv_bfloat162*>(tile + r_local * TS + c_local) = __floats2bfloat162_rn(v0, v1);
          } else {
            *reinterpret_cast<__nv_bfloat162*>(tile + r_local * TS + c_local) = __floats2bfloat162_rn(v0, v1);
            if constexpr (BN_OUT == 128) {
              const float u0 = static_cast<float>(acc1[mi][nj][half * 2]) * xs * wu0;
              const float u1 = static_cast<float>(acc1[mi][nj][half * 2 + 1]) * xs * wu1;
              *reinterpret_cast<__nv_bfloat162*>(tile + r_local * TS + 64 + c_local) = __floats2bfloat162_rn(u0, u1);
            }
          }
        }
      }
    }
    __syncthreads();

    if constexpr (INT8 && SWIGLU) {
      // Fused per-64-group INT8 quantization straight from smem!
      // 256 threads / 128 rows = 2 threads/row (32 cols per thread, 1 xor-1 shuffle for amax, 2x uint4 stores!).
      const int r = tid >> 1;
      const int half = tid & 1;
      const __nv_bfloat162* src2 = reinterpret_cast<const __nv_bfloat162*>(tile + r * TS + half * 32);
      float vals[32];
      float amax = 0.f;
#pragma unroll
      for (int j = 0; j < 16; ++j) {
        const float2 f = __bfloat1622float2(src2[j]);
        vals[2 * j + 0] = f.x;
        vals[2 * j + 1] = f.y;
        amax = fmaxf(amax, fmaxf(fabsf(f.x), fabsf(f.y)));
      }
      const float gmax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
      const float sc = fmaxf(gmax, 1e-6f) * (1.0f / 127.0f);
      const float inv_sc = 127.0f / fmaxf(gmax, 1e-6f);
      if (r < rows_here) {
        int8_t* dst = reinterpret_cast<int8_t*>(Y_) + (size_t)(m0 + r) * y_stride + y_col_offset + n0_out + half * 32;
        uint32_t pk[8];
#pragma unroll
        for (int g = 0; g < 8; ++g) {
          pk[g] = (uint32_t)gemm_quant_u8_s8(vals[4 * g + 0], inv_sc)
                | ((uint32_t)gemm_quant_u8_s8(vals[4 * g + 1], inv_sc) << 8)
                | ((uint32_t)gemm_quant_u8_s8(vals[4 * g + 2], inv_sc) << 16)
                | ((uint32_t)gemm_quant_u8_s8(vals[4 * g + 3], inv_sc) << 24);
        }
        *reinterpret_cast<uint4*>(dst) = make_uint4(pk[0], pk[1], pk[2], pk[3]);
        *reinterpret_cast<uint4*>(dst + 16) = make_uint4(pk[4], pk[5], pk[6], pk[7]);
        if (half == 0) y_scale_grp[(size_t)(m0 + r) * n_tiles + nt] = sc;
      }
    } else {
      __nv_bfloat16* Y = reinterpret_cast<__nv_bfloat16*>(Y_);
      constexpr int CHUNKS_PER_ROW = BN_OUT / 8;  // 8 or 16
      for (int i = tid; i < GEMM_BM * CHUNKS_PER_ROW; i += GEMM_THREADS) {
        const int r = i / CHUNKS_PER_ROW;
        const int c = (i - r * CHUNKS_PER_ROW) * 8;
        if (r < rows_here) {
          *reinterpret_cast<uint4*>(Y + (size_t)(m0 + r) * y_stride + y_col_offset + n0_out + c) =
              *reinterpret_cast<const uint4*>(tile + r * TS + c);
        }
      }
    }
    __syncthreads();
  }
}

}  // namespace scacchi_inkling
