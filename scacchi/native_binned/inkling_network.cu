// Optimized Blackwell / Ampere CUDA inference kernel for AZInklingBinnedNet
// (unlooped Inkling-2 Mini trunk, C=192, H=6, KV=2, D=32, KataGo D2 spatial heads).
//
// Highlights:
//   1. Fused D2-canonicalizing 3x3 stem + embedding RMSNorm + Layer-0 bank seal + attn_norm (+ INT8).
//   2. Fused per-board 2D ShortConv + residual/seal + fla-style online DepthRead + RMSNorm
//      (+ SiLU or INT8 quant), eliminating ALL separate DepthRead passes and 550+ MB of `partial` HBM I/O!
//   3. Dynamic-*d_boards PTX Tensor Core GEMMs (bf16 m16n8k16 or s8xs8 m16n8k32 at 2x rate),
//      with BN=128 tiles for QKV+OutGate, fused [W_gate|W_up]+Capped-SwiGLU+group-64 INT8 quant
//      (zero bf16 round-trip / zero separate quant kernel!), and Tensor Core conv1pgv head projection.
//   4. Low-register (<= 64 regs, 2 CTAs/SM!) super-fused Tensor Core FlashAttention (mma.m16n8k16
//      + ldmatrix.x4.trans) with coalesced OutGate + INT8 quant in Step 4.
//   5. Fused KataGo Policy/Q & Global-Pool Value head with D2 un-transpose and lock-free CacheWrite.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <type_traits>

#include "xla/ffi/api/ffi.h"

#include "cache_key.cuh"
#include "inkling_mma.cuh"

namespace ffi = xla::ffi;

namespace {

using scacchi_net::CacheKey;
using scacchi_net::CacheWrite;
using scacchi_net::cache_try_acquire;
using scacchi_net::hash_cache_key;
using namespace scacchi_inkling;

constexpr int IC = 192;
constexpr int IC2 = IC / 2;        // 96 bf162 pairs
constexpr int IH = 6;
constexpr int IKV = 2;
constexpr int ID = 32;
constexpr int IGROUPS = IH / IKV;  // 3
constexpr int IKVDIM = IKV * ID;   // 64
constexpr int IHEAD_C = 48;
constexpr int IVAL_H = 128;
constexpr float IQK_CLIP = 11.313708498984761f;  // 2 * sqrt(32)
constexpr float IRMS_EPS = 1e-5f;
constexpr float IINV_SQRT2 = 0.7071067811865476f;
constexpr float IINV_SQRT_D = 0.17677669529663687f;  // 1 / sqrt(32)
constexpr float IINV_SQRT_C = 0.07216878364870322f;  // 1 / sqrt(192)

template <typename T>
const T* cptr(const ffi::AnyBuffer& b) {
  return reinterpret_cast<const T*>(b.untyped_data());
}
template <typename T>
T* mptr(ffi::AnyBuffer& b) {
  return reinterpret_cast<T*>(b.untyped_data());
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}

__device__ __forceinline__ int8_t quant_s8(float v, float inv_scale) {
  int q = __float2int_rn(v * inv_scale);
  q = max(-127, min(127, q));
  return static_cast<int8_t>(q);
}

__device__ __forceinline__ uint16_t pack_s8x2(float v0, float v1, float inv_scale) {
  const uint8_t q0 = static_cast<uint8_t>(quant_s8(v0, inv_scale));
  const uint8_t q1 = static_cast<uint8_t>(quant_s8(v1, inv_scale));
  return static_cast<uint16_t>(q0) | (static_cast<uint16_t>(q1) << 8);
}

__device__ __forceinline__ float silu_f32(float x) { return x / (1.0f + __expf(-x)); }

__device__ __forceinline__ int swz_bf16(int row, int d) {
  return row * 32 + (((d >> 3) ^ ((row >> 1) & 3)) << 3) + (d & 7);
}

// ---- Kernel 1: Fused D2-canonicalizing 3x3 stem + emb_norm + Layer-0 bank[0] seal + attn_norm (+ INT8) ----
// When L >= 2 (layer 0 is non-full and always seals bank[0]): writes bank_norm[0] and act0 directly
// with ZERO intermediate `partial` HBM write/read! (Also supports QCOUNT=3 if L == 1.)
template <bool INT8, int QCOUNT>
__global__ void __launch_bounds__(256) inkling_stem_fused_kernel(
    const uint8_t* __restrict__ obs,     // [B, side, side, 4]
    const __nv_bfloat16* __restrict__ w, // [192, 27]
    const float* __restrict__ bias,      // [192]
    const float* __restrict__ emb_s,     // [192]
    const float* __restrict__ emb_b,     // [192]
    const float* __restrict__ l0_norm_sb,// [2 * QCOUNT, 192]
    __nv_bfloat16* __restrict__ bank0,   // [max_M, 192] (bank_norm[0])
    void* __restrict__ out0_,
    float* __restrict__ scale0,
    void* __restrict__ out1_,
    float* __restrict__ scale1,
    void* __restrict__ out2_,
    float* __restrict__ scale2,
    int max_boards,
    int side,
    int cells,
    const int32_t* __restrict__ d_boards) {
  const int active_boards = d_boards ? *d_boards : max_boards;
  const int M = active_boards * cells;
  if (M <= 0) return;

  __shared__ __align__(16) float ws[27 * IC];
  for (int i = threadIdx.x; i < 27 * IC; i += 256) {
    const int n = i / 27, k = i - n * 27;
    ws[k * IC + n] = __bfloat162float(w[i]);
  }
  __syncthreads();

  const int warp_in_cta = threadIdx.x >> 5;
  const int lane = threadIdx.x & 31;
  const float2* ws2 = reinterpret_cast<const float2*>(ws);
  const float2* bias2 = reinterpret_cast<const float2*>(bias);
  const float2* embs2 = reinterpret_cast<const float2*>(emb_s);
  const float2* embb2 = reinterpret_cast<const float2*>(emb_b);
  const float2* nsb2 = reinterpret_cast<const float2*>(l0_norm_sb);

  float2 b_reg[3], es_reg[3], eb_reg[3];
  float2 ns[QCOUNT][3], nbi[QCOUNT][3];
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    const int p = j * 32 + lane;
    b_reg[j] = bias2[p];
    es_reg[j] = embs2[p];
    eb_reg[j] = embb2[p];
#pragma unroll
    for (int q = 0; q < QCOUNT; ++q) {
      ns[q][j] = nsb2[(2 * q) * IC2 + p];
      nbi[q][j] = nsb2[(2 * q + 1) * IC2 + p];
    }
  }

  for (int r = blockIdx.x * 8 + warp_in_cta; r < M; r += gridDim.x * 8) {
    const int board = r / cells;
    const int cell = r - board * cells;
    const int y = cell / side;
    const int x = cell - y * side;
    const bool is_white = obs[(size_t)board * cells * 4 + 2] != 0;

    float2 acc[3] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
    for (int t = 0; t < 9; ++t) {
      const int yy = y + t / 3 - 1;
      const int xx = x + t % 3 - 1;
      if (yy < 0 || yy >= side || xx < 0 || xx >= side) continue;
      const int src_y = is_white ? xx : yy;
      const int src_x = is_white ? yy : xx;
      const uint32_t o = *reinterpret_cast<const uint32_t*>(obs + ((size_t)board * cells + src_y * side + src_x) * 4);
      const bool ch0 = (o & 0xffu) != 0u;
      const bool ch1 = ((o >> 8) & 0xffu) != 0u;
      const float2* w2 = ws2 + (t * 3 + 2) * IC2;
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        const float2 wv = w2[j * 32 + lane];
        acc[j].x += wv.x;
        acc[j].y += wv.y;
      }
      if (ch0) {
        const float2* w0 = ws2 + (t * 3 + 0) * IC2;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          const float2 wv = w0[j * 32 + lane];
          acc[j].x += wv.x;
          acc[j].y += wv.y;
        }
      }
      if (ch1) {
        const float2* w1 = ws2 + (t * 3 + 1) * IC2;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          const float2 wv = w1[j * 32 + lane];
          acc[j].x += wv.x;
          acc[j].y += wv.y;
        }
      }
    }
    float sq = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      acc[j].x = __bfloat162float(__float2bfloat16_rn(acc[j].x + b_reg[j].x));
      acc[j].y = __bfloat162float(__float2bfloat16_rn(acc[j].y + b_reg[j].y));
      sq = fmaf(acc[j].x, acc[j].x, fmaf(acc[j].y, acc[j].y, sq));
    }
    const float rstd = rsqrtf(warp_sum(sq) * (1.0f / IC) + IRMS_EPS);

    // emb_norm output (rounded to bf16 to match stem -> partial exactly), then unit_rms -> bank0 + Layer-0 norm -> act!
    float psq = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const __nv_bfloat162 eb = __floats2bfloat162_rn(
          acc[j].x * rstd * es_reg[j].x + eb_reg[j].x,
          acc[j].y * rstd * es_reg[j].y + eb_reg[j].y);
      acc[j] = __bfloat1622float2(eb);
      psq = fmaf(acc[j].x, acc[j].x, fmaf(acc[j].y, acc[j].y, psq));
    }
    const float prstd = rsqrtf(warp_sum(psq) * (1.0f / IC) + IRMS_EPS);
    __nv_bfloat162* b0_dst = reinterpret_cast<__nv_bfloat162*>(bank0 + (size_t)r * IC);
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      b0_dst[j * 32 + lane] = __floats2bfloat162_rn(acc[j].x * prstd, acc[j].y * prstd);
    }

    // Layer-0 attn RMSNorm (on raw `acc` with prstd already computed!) + optional INT8 quant.
#pragma unroll
    for (int q = 0; q < QCOUNT; ++q) {
      float2 v[3];
      float amax = 0.f;
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        v[j].x = acc[j].x * prstd * ns[q][j].x + nbi[q][j].x;
        v[j].y = acc[j].y * prstd * ns[q][j].y + nbi[q][j].y;
        if constexpr (INT8) amax = fmaxf(amax, fmaxf(fabsf(v[j].x), fabsf(v[j].y)));
      }
      void* dst_p = (q == 0) ? out0_ : (q == 1 ? out1_ : out2_);
      if constexpr (INT8) {
        const float gmax = warp_max(amax);
        const float sc = fmaxf(gmax, 1e-6f) * (1.0f / 127.0f);
        const float inv_sc = 127.0f / fmaxf(gmax, 1e-6f);
        uint16_t* dst = reinterpret_cast<uint16_t*>(reinterpret_cast<int8_t*>(dst_p) + (size_t)r * IC);
#pragma unroll
        for (int j = 0; j < 3; ++j) dst[j * 32 + lane] = pack_s8x2(v[j].x, v[j].y, inv_sc);
        if (lane == 0) {
          float* sp = (q == 0) ? scale0 : (q == 1 ? scale1 : scale2);
          sp[r] = sc;
        }
      } else {
        __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(dst_p) + (size_t)r * IC);
#pragma unroll
        for (int j = 0; j < 3; ++j) dst[j * 32 + lane] = __floats2bfloat162_rn(v[j].x, v[j].y);
      }
    }
  }
}

// ---- Kernel 2: SUPER-FUSED 2D ShortConv + Residual/Seal + fla Online DepthRead + RMSNorm (+ SiLU / INT8) ----
// 1 CTA = 1 board (256 threads = 8 warps, 1 warp/cell, 3 coalesced bf162 pairs/lane = 192 ch).
// Eliminates ALL intermediate `partial` HBM reads (and skips `partial` writes whenever the next op seals/ends!).
template <bool INT8, int QCOUNT, bool APPLY_SILU = false>
__global__ void __launch_bounds__(256) inkling_shortconv_dr_norm_kernel(
    const __nv_bfloat16* __restrict__ tmp,       // [M, 192]
    const float* __restrict__ dw_w,              // [9, 192]
    __nv_bfloat16* __restrict__ partial,         // [M, 192]
    __nv_bfloat16* __restrict__ bank_norm,       // [max_bank, max_M, 192]
    int nb,                                      // number of already-sealed bank entries
    bool replace_partial,                        // true on seals_bank after_attn (skips reading old partial!)
    bool write_partial,                          // false when next sublayer seals or this is final layer!
    bool save_partial_to_bank,                   // true when next layer seals bank
    const float* __restrict__ read_q,            // [QCOUNT, 192]
    const float* __restrict__ norm_sb,           // [2 * QCOUNT, 192]
    void* __restrict__ out0_,
    float* __restrict__ scale0,
    void* __restrict__ out1_,
    float* __restrict__ scale1,
    void* __restrict__ out2_,
    float* __restrict__ scale2,
    int max_boards,
    int side,
    int cells,
    int max_M,
    const int32_t* __restrict__ d_boards) {
  const int active_boards = d_boards ? *d_boards : max_boards;
  const int board = blockIdx.x;
  if (board >= active_boards) return;

  extern __shared__ __align__(16) char sc_smem_raw[];
  __nv_bfloat162* smem_tmp2 = reinterpret_cast<__nv_bfloat162*>(sc_smem_raw);       // [cells, 96]
  float2* smem_dw2 = reinterpret_cast<float2*>(smem_tmp2 + cells * IC2);            // [9, 96]

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const size_t base_row = (size_t)board * cells;

  // Coalesced 16-byte load of tmp[cells, 192] and 8-byte load of dw_w[9, 192] into smem.
  const uint4* src4 = reinterpret_cast<const uint4*>(tmp + base_row * IC);
  uint4* dst4 = reinterpret_cast<uint4*>(smem_tmp2);
  const int nchunks = cells * (IC / 8);
  for (int i = tid; i < nchunks; i += 256) dst4[i] = src4[i];
  const float2* dw2_g = reinterpret_cast<const float2*>(dw_w);
  for (int i = tid; i < 9 * IC2; i += 256) smem_dw2[i] = dw2_g[i];
  __syncthreads();

  const float2* rq2 = reinterpret_cast<const float2*>(read_q);
  const float2* nsb2 = reinterpret_cast<const float2*>(norm_sb);
  float2 rq[QCOUNT][3], ns[QCOUNT][3], nbi[QCOUNT][3];
#pragma unroll
  for (int q = 0; q < QCOUNT; ++q) {
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const int p = j * 32 + lane;
      rq[q][j] = rq2[q * IC2 + p];
      ns[q][j] = nsb2[(2 * q) * IC2 + p];
      nbi[q][j] = nsb2[(2 * q + 1) * IC2 + p];
    }
  }

  auto write_stream = [&](size_t r, int q, float2 (&vec)[3]) {
    float sq = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) sq = fmaf(vec[j].x, vec[j].x, fmaf(vec[j].y, vec[j].y, sq));
    const float rstd = rsqrtf(warp_sum(sq) * (1.0f / IC) + IRMS_EPS);
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      float vx = vec[j].x * rstd * ns[q][j].x + nbi[q][j].x;
      float vy = vec[j].y * rstd * ns[q][j].y + nbi[q][j].y;
      if constexpr (APPLY_SILU) {
        vx = silu_f32(vx);
        vy = silu_f32(vy);
      }
      vec[j] = make_float2(vx, vy);
      if constexpr (INT8) amax = fmaxf(amax, fmaxf(fabsf(vx), fabsf(vy)));
    }
    void* dst_p = (q == 0) ? out0_ : (q == 1 ? out1_ : out2_);
    if constexpr (INT8) {
      const float gmax = warp_max(amax);
      const float sc = fmaxf(gmax, 1e-6f) * (1.0f / 127.0f);
      const float inv_sc = 127.0f / fmaxf(gmax, 1e-6f);
      uint16_t* dst = reinterpret_cast<uint16_t*>(reinterpret_cast<int8_t*>(dst_p) + r * IC);
#pragma unroll
      for (int j = 0; j < 3; ++j) dst[j * 32 + lane] = pack_s8x2(vec[j].x, vec[j].y, inv_sc);
      if (lane == 0) {
        float* sp = (q == 0) ? scale0 : (q == 1 ? scale1 : scale2);
        sp[r] = sc;
      }
    } else {
      __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(dst_p) + r * IC);
#pragma unroll
      for (int j = 0; j < 3; ++j) dst[j * 32 + lane] = __floats2bfloat162_rn(vec[j].x, vec[j].y);
    }
  };

  for (int cell = warp; cell < cells; cell += 8) {
    const int y = cell / side;
    const int x = cell - y * side;
    const size_t r = base_row + cell;

    // 1. 2D 3x3 depthwise ShortConv in smem (3 bf162 pairs per lane = 192 ch).
    float2 br[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) br[j] = __bfloat1622float2(smem_tmp2[cell * IC2 + j * 32 + lane]);
    for (int t = 0; t < 9; ++t) {
      const int yy = y + (t / 3) - 1;
      const int xx = x + (t % 3) - 1;
      if (yy < 0 || yy >= side || xx < 0 || xx >= side) continue;
      const __nv_bfloat162* nptr = smem_tmp2 + (yy * side + xx) * IC2;
      const float2* wptr = smem_dw2 + t * IC2;
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        const float2 nv = __bfloat1622float2(nptr[j * 32 + lane]);
        const float2 wv = wptr[j * 32 + lane];
        br[j].x = fmaf(nv.x, wv.x, br[j].x);
        br[j].y = fmaf(nv.y, wv.y, br[j].y);
      }
    }

    // 2. Residual / block-seal update of `partial` in registers (with bf16 rounding parity).
    __nv_bfloat162* part2 = reinterpret_cast<__nv_bfloat162*>(partial + r * IC);
    float2 p_raw[3];
    float psq = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      float vx = br[j].x * IINV_SQRT2;
      float vy = br[j].y * IINV_SQRT2;
      if (!replace_partial) {
        const float2 old = __bfloat1622float2(part2[j * 32 + lane]);
        vx += old.x;
        vy += old.y;
      }
      const __nv_bfloat162 pb = __floats2bfloat162_rn(vx, vy);
      if (write_partial) part2[j * 32 + lane] = pb;
      p_raw[j] = __bfloat1622float2(pb);
      psq = fmaf(p_raw[j].x, p_raw[j].x, fmaf(p_raw[j].y, p_raw[j].y, psq));
    }

    // 3. unit_rms(partial) + optional seal into bank_norm[nb].
    const float prstd = rsqrtf(warp_sum(psq) * (1.0f / IC) + IRMS_EPS);
    float2 p_norm[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) p_norm[j] = make_float2(p_raw[j].x * prstd, p_raw[j].y * prstd);
    if (save_partial_to_bank) {
      __nv_bfloat162* bdst = reinterpret_cast<__nv_bfloat162*>(bank_norm + ((size_t)nb * max_M + r) * IC);
#pragma unroll
      for (int j = 0; j < 3; ++j) bdst[j * 32 + lane] = __floats2bfloat162_rn(p_norm[j].x, p_norm[j].y);
    }

    // 4. Online fla-style DepthRead across bank_norm[0..nb-1] + p_norm -> RMSNorm (+ SiLU / INT8)!
    if (nb == 0) {
#pragma unroll
      for (int q = 0; q < QCOUNT; ++q) {
        float2 v[3] = {p_norm[0], p_norm[1], p_norm[2]};
        write_stream(r, q, v);
      }
      continue;
    }

    float m_max[QCOUNT], denom[QCOUNT];
    float2 acc[QCOUNT][3];
#pragma unroll
    for (int q = 0; q < QCOUNT; ++q) {
      m_max[q] = -1e30f;
      denom[q] = 0.f;
#pragma unroll
      for (int j = 0; j < 3; ++j) acc[q][j] = make_float2(0.f, 0.f);
    }

    auto absorb_source = [&](const float2 (&src)[3]) {
#pragma unroll
      for (int q = 0; q < QCOUNT; ++q) {
        float dot = 0.f;
#pragma unroll
        for (int j = 0; j < 3; ++j) dot = fmaf(src[j].x, rq[q][j].x, fmaf(src[j].y, rq[q][j].y, dot));
        const float logit = warp_sum(dot) * IINV_SQRT_C;
        const float m_new = fmaxf(m_max[q], logit);
        const float alpha = __expf(m_max[q] - m_new);
        const float beta = __expf(logit - m_new);
        denom[q] = denom[q] * alpha + beta;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          acc[q][j].x = fmaf(beta, src[j].x, acc[q][j].x * alpha);
          acc[q][j].y = fmaf(beta, src[j].y, acc[q][j].y * alpha);
        }
        m_max[q] = m_new;
      }
    };

    for (int bi = 0; bi < nb; ++bi) {
      const __nv_bfloat162* bptr = reinterpret_cast<const __nv_bfloat162*>(bank_norm + ((size_t)bi * max_M + r) * IC);
      float2 bv[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) bv[j] = __bfloat1622float2(bptr[j * 32 + lane]);
      absorb_source(bv);
    }
    absorb_source(p_norm);

#pragma unroll
    for (int q = 0; q < QCOUNT; ++q) {
      const float inv_d = 1.0f / denom[q];
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        acc[q][j].x *= inv_d;
        acc[q][j].y *= inv_d;
      }
      write_stream(r, q, acc[q]);
    }
  }
}

// ---- Kernel 3: Low-Register (2 CTAs/SM!) Super-Fused Tensor Core FlashAttention Kernel ----
// Loops Q heads g=0..2 sequentially so live accumulators drop 3x (<= 64 regs/thread -> 2 CTAs/SM!),
// and moves sigmoid(OutGate) multiplication into Step 4 for 100% coalesced bf162 global loads!
__host__ __device__ inline int inkling_attn_smem_bytes(int cells) {
  const int s_kv_pad = ((cells + 2) + 15) & ~15;
  const int kv_elems = 4 * s_kv_pad * ID;
  const int raw_or_q = max(2 * cells * IKVDIM, 8 * 16 * ID);
  if (cells <= 64) {
    const int step13 = kv_elems + raw_or_q;
    const int step4 = cells * IC;
    return max(step13, step4) * (int)sizeof(__nv_bfloat16);
  }
  const int step12 = kv_elems + 2 * cells * IKVDIM;
  const int step34 = kv_elems + 8 * 16 * ID + cells * IC;
  return max(step12, step34) * (int)sizeof(__nv_bfloat16);
}

template <bool INT8>
__global__ void __launch_bounds__(256, 2) inkling_attn_kernel(
    const __nv_bfloat16* __restrict__ qgkv,  // [M, 512]: [Q(192) | OutGate(192) | K(64) | V(64)]
    const float* __restrict__ q_scale,       // [6, 32]
    const float* __restrict__ sink_k,        // [2, 2, 32] (pre-clipped)
    const float* __restrict__ rope_freqs,    // [2, 16, 2]
    const float* __restrict__ dw_kv,         // [2, 9, 64]
    void* __restrict__ out_,                 // [M, 192] bf16 or int8
    float* __restrict__ out_scale,           // [M] f32 if INT8 else nullptr
    int max_boards,
    int side,
    int cells,
    const int32_t* __restrict__ d_boards) {
  const int active_boards = d_boards ? *d_boards : max_boards;
  const int board = blockIdx.x;
  if (board >= active_boards) return;

  const int S_kv = cells + 2;
  const int S_kv_pad = (S_kv + 15) & ~15;

  extern __shared__ __align__(128) __nv_bfloat16 smem_bf[];
  __nv_bfloat16* k_smem = smem_bf;                          // [2, S_kv_pad, 32] swizzled
  __nv_bfloat16* v_smem = k_smem + 2 * S_kv_pad * ID;       // [2, S_kv_pad, 32] swizzled
  __nv_bfloat16* scratch = v_smem + 2 * S_kv_pad * ID;
  __nv_bfloat16* k_raw = scratch;                           // [cells, 64]
  __nv_bfloat16* v_raw = k_raw + cells * IKVDIM;            // [cells, 64]
  __nv_bfloat16* q_tmp = scratch;                           // [8, 16, 32] swizzled
  __nv_bfloat16* o_smem = (cells <= 64) ? smem_bf : (q_tmp + 8 * 16 * ID);  // [cells, 192]

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const size_t base_row = (size_t)board * cells;

  // Step 1: Load raw K and V for all `cells` into smem via 16-byte uint4.
  for (int i = tid; i < cells * (IKVDIM / 8); i += 256) {
    const int c = i >> 3;
    const int ch = (i & 7) * 8;
    const __nv_bfloat16* row = qgkv + (base_row + c) * 512;
    *reinterpret_cast<uint4*>(k_raw + c * IKVDIM + ch) = *reinterpret_cast<const uint4*>(row + 384 + ch);
    *reinterpret_cast<uint4*>(v_raw + c * IKVDIM + ch) = *reinterpret_cast<const uint4*>(row + 448 + ch);
  }
  __syncthreads();

  // Step 2: 2D 3x3 ShortConv + clip_heads + 2D RoPE on K, and ShortConv + unit_rms on V -> swizzled smem.
  const float half_side = 0.5f * (float)(side - 1);
  const int pair_idx = lane & 15;
  const bool is_hi = (lane >= 16);

  for (int ck = warp; ck < cells * IKV; ck += 8) {
    const int cell = ck >> 1;
    const int kvh = ck & 1;
    const int y = cell / side;
    const int x = cell - y * side;
    const int ch = kvh * ID + lane;

    float k_acc = __bfloat162float(k_raw[cell * IKVDIM + ch]);
    float v_acc = __bfloat162float(v_raw[cell * IKVDIM + ch]);
    for (int t = 0; t < 9; ++t) {
      const int yy = y + t / 3 - 1;
      const int xx = x + t % 3 - 1;
      if (yy < 0 || yy >= side || xx < 0 || xx >= side) continue;
      const int nc = yy * side + xx;
      k_acc = fmaf(__bfloat162float(k_raw[nc * IKVDIM + ch]), dw_kv[t * IKVDIM + ch], k_acc);
      v_acc = fmaf(__bfloat162float(v_raw[nc * IKVDIM + ch]), dw_kv[(9 + t) * IKVDIM + ch], v_acc);
    }
    k_acc *= IINV_SQRT2;
    v_acc *= IINV_SQRT2;

    const float k_norm = sqrtf(fmaxf(warp_sum(k_acc * k_acc), 1e-24f));
    k_acc *= fminf(1.0f, IQK_CLIP / (k_norm + 1e-6f));

    const float sy = (float)y - half_side;
    const float sx = (float)x - half_side;
    const float fx = rope_freqs[(kvh * 16 + pair_idx) * 2 + 0];
    const float fy = rope_freqs[(kvh * 16 + pair_idx) * 2 + 1];
    float s_ang, c_ang;
    __sincosf(sx * fx + sy * fy, &s_ang, &c_ang);
    const float k_partner = __shfl_xor_sync(0xffffffffu, k_acc, 16);
    const float k_rot = is_hi ? (k_partner * s_ang + k_acc * c_ang) : (k_acc * c_ang - k_partner * s_ang);

    const float v_normed = v_acc * rsqrtf(warp_sum(v_acc * v_acc) * (1.0f / ID) + IRMS_EPS);

    const int sw = swz_bf16(cell, lane);
    k_smem[kvh * S_kv_pad * ID + sw] = __float2bfloat16_rn(k_rot);
    v_smem[kvh * S_kv_pad * ID + sw] = __float2bfloat16_rn(v_normed);
  }
  for (int idx = tid; idx < IKV * (S_kv_pad - cells) * ID; idx += 256) {
    const int d = idx & 31;
    const int rem = idx >> 5;
    const int n_extra = S_kv_pad - cells;
    const int kvh = rem / n_extra;
    const int st = rem - kvh * n_extra;
    const int t = cells + st;
    const int sw = swz_bf16(t, d);
    const float kv = (st < 2) ? sink_k[(st * IKV + kvh) * ID + d] : 0.f;
    k_smem[kvh * S_kv_pad * ID + sw] = __float2bfloat16_rn(kv);
    v_smem[kvh * S_kv_pad * ID + sw] = __float2bfloat16_rn(0.f);
  }
  __syncthreads();

  // Step 3: Low-register Tensor Core m16n8k16 FlashAttention (1 head at a time -> <= 64 regs -> 2 CTAs/SM!).
  const int m_tiles = (cells + 15) >> 4;
  const int total_tasks = IKV * m_tiles;
  __nv_bfloat16* q_warp = q_tmp + warp * 16 * ID;
  const uint32_t q_warp_u32 = smem_u32(q_warp);
  const int r0_in_tile = lane >> 2;
  const int r1_in_tile = r0_in_tile + 8;
  const int c_in_n8 = (lane & 3) * 2;

  auto run_head_attn = [&](int kvh, int q_row0, int g, __nv_bfloat162 (&out_h)[8]) {
    const int h = kvh * IGROUPS + g;
    const float fx = rope_freqs[(kvh * 16 + pair_idx) * 2 + 0];
    const float fy = rope_freqs[(kvh * 16 + pair_idx) * 2 + 1];
    const float qs = q_scale[h * ID + lane];

    for (int r = 0; r < 16; ++r) {
      const int cell = q_row0 + r;
      float q_rot = 0.f;
      if (cell < cells) {
        const int y = cell / side;
        const int x = cell - y * side;
        const float sy = (float)y - half_side;
        const float sx = (float)x - half_side;
        float s_ang, c_ang;
        __sincosf(sx * fx + sy * fy, &s_ang, &c_ang);
        float qv = __bfloat162float(qgkv[(base_row + cell) * 512 + h * ID + lane]) * qs;
        const float q_norm = sqrtf(fmaxf(warp_sum(qv * qv), 1e-24f));
        qv *= fminf(1.0f, IQK_CLIP / (q_norm + 1e-6f));
        const float qp = __shfl_xor_sync(0xffffffffu, qv, 16);
        q_rot = (is_hi ? (qp * s_ang + qv * c_ang) : (qv * c_ang - qp * s_ang)) * IINV_SQRT_D;
      }
      q_warp[swz_bf16(r, lane)] = __float2bfloat16_rn(q_rot);
    }
    __syncwarp();
    uint32_t qa0[4], qa1[4];
    const int lrow = lane & 15;
    ldmatrix_x4(qa0, q_warp_u32 + swz(lrow, lane >> 4));
    ldmatrix_x4(qa1, q_warp_u32 + swz(lrow, 2 + (lane >> 4)));

    float m0 = -1e30f, m1 = -1e30f, d0 = 0.f, d1 = 0.f;
    float acc[4][4] = {{0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f}};
    const uint32_t k_base_u32 = smem_u32(k_smem + kvh * S_kv_pad * ID);
    const uint32_t v_base_u32 = smem_u32(v_smem + kvh * S_kv_pad * ID);

    for (int t0 = 0; t0 < S_kv_pad; t0 += 16) {
      uint32_t kb0[4], kb1[4], vb0[4], vb1[4];
      const int k_brow = t0 + (lane & 7) + ((lane >> 4) << 3);
      ldmatrix_x4(kb0, k_base_u32 + swz(k_brow, (lane >> 3) & 1));
      ldmatrix_x4(kb1, k_base_u32 + swz(k_brow, 2 + ((lane >> 3) & 1)));
      const int v_row = t0 + (lane & 7) + (((lane >> 3) & 1) << 3);
      ldmatrix_x4_trans(vb0, v_base_u32 + swz(v_row, lane >> 4));
      ldmatrix_x4_trans(vb1, v_base_u32 + swz(v_row, 2 + (lane >> 4)));

      float s0[4] = {0.f, 0.f, 0.f, 0.f};
      float s1[4] = {0.f, 0.f, 0.f, 0.f};
      mma_bf16(s0, qa0, kb0[0], kb0[1]);
      mma_bf16(s0, qa1, kb1[0], kb1[1]);
      mma_bf16(s1, qa0, kb0[2], kb0[3]);
      mma_bf16(s1, qa1, kb1[2], kb1[3]);

      if (t0 + 16 > S_kv) {
        const int k0 = t0 + c_in_n8;
        if (k0 >= S_kv) { s0[0] = -1e30f; s0[2] = -1e30f; }
        if (k0 + 1 >= S_kv) { s0[1] = -1e30f; s0[3] = -1e30f; }
        if (k0 + 8 >= S_kv) { s1[0] = -1e30f; s1[2] = -1e30f; }
        if (k0 + 9 >= S_kv) { s1[1] = -1e30f; s1[3] = -1e30f; }
      }

      float rm0 = fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s1[0], s1[1]));
      rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffffu, rm0, 1));
      rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffffu, rm0, 2));
      float rm1 = fmaxf(fmaxf(s0[2], s0[3]), fmaxf(s1[2], s1[3]));
      rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffffu, rm1, 1));
      rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffffu, rm1, 2));

      const float mn0 = fmaxf(m0, rm0), mn1 = fmaxf(m1, rm1);
      const float a0 = __expf(m0 - mn0), a1 = __expf(m1 - mn1);
      const float p00 = __expf(s0[0] - mn0), p01 = __expf(s0[1] - mn0);
      const float p02 = __expf(s0[2] - mn1), p03 = __expf(s0[3] - mn1);
      const float p10 = __expf(s1[0] - mn0), p11 = __expf(s1[1] - mn0);
      const float p12 = __expf(s1[2] - mn1), p13 = __expf(s1[3] - mn1);

      float rs0 = (p00 + p01) + (p10 + p11);
      rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1);
      rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2);
      float rs1 = (p02 + p03) + (p12 + p13);
      rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1);
      rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);

      d0 = fmaf(d0, a0, rs0); d1 = fmaf(d1, a1, rs1);
      m0 = mn0; m1 = mn1;
#pragma unroll
      for (int nj = 0; nj < 4; ++nj) {
        acc[nj][0] *= a0; acc[nj][1] *= a0;
        acc[nj][2] *= a1; acc[nj][3] *= a1;
      }

      uint32_t p_reg[4];
      const __nv_bfloat162 pr0 = __floats2bfloat162_rn(p00, p01);
      const __nv_bfloat162 pr1 = __floats2bfloat162_rn(p02, p03);
      const __nv_bfloat162 pr2 = __floats2bfloat162_rn(p10, p11);
      const __nv_bfloat162 pr3 = __floats2bfloat162_rn(p12, p13);
      p_reg[0] = *reinterpret_cast<const uint32_t*>(&pr0);
      p_reg[1] = *reinterpret_cast<const uint32_t*>(&pr1);
      p_reg[2] = *reinterpret_cast<const uint32_t*>(&pr2);
      p_reg[3] = *reinterpret_cast<const uint32_t*>(&pr3);

      mma_bf16(acc[0], p_reg, vb0[0], vb0[1]);
      mma_bf16(acc[1], p_reg, vb0[2], vb0[3]);
      mma_bf16(acc[2], p_reg, vb1[0], vb1[1]);
      mma_bf16(acc[3], p_reg, vb1[2], vb1[3]);
    }
    const float inv_d0 = 1.0f / d0, inv_d1 = 1.0f / d1;
#pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
      out_h[nj] = __floats2bfloat162_rn(acc[nj][0] * inv_d0, acc[nj][1] * inv_d0);
      out_h[4 + nj] = __floats2bfloat162_rn(acc[nj][2] * inv_d1, acc[nj][3] * inv_d1);
    }
  };

  if (cells <= 64) {
    __nv_bfloat162 out_reg[IGROUPS][8];
    const bool active_w = (warp < total_tasks);
    const int kvh = active_w ? (warp / m_tiles) : 0;
    const int q_row0 = active_w ? ((warp - kvh * m_tiles) * 16) : 0;
    if (active_w) {
#pragma unroll
      for (int g = 0; g < IGROUPS; ++g) run_head_attn(kvh, q_row0, g, out_reg[g]);
    }
    __syncthreads();
    if (active_w) {
      const int c0 = q_row0 + r0_in_tile;
      const int c1 = q_row0 + r1_in_tile;
#pragma unroll
      for (int g = 0; g < IGROUPS; ++g) {
        const int h = kvh * IGROUPS + g;
#pragma unroll
        for (int nj = 0; nj < 4; ++nj) {
          const int col = h * ID + nj * 8 + c_in_n8;
          if (c0 < cells) *reinterpret_cast<__nv_bfloat162*>(o_smem + c0 * IC + col) = out_reg[g][nj];
          if (c1 < cells) *reinterpret_cast<__nv_bfloat162*>(o_smem + c1 * IC + col) = out_reg[g][4 + nj];
        }
      }
    }
    __syncthreads();
  } else {
    for (int task = warp; task < total_tasks; task += 8) {
      const int kvh = task / m_tiles;
      const int q_row0 = (task - kvh * m_tiles) * 16;
      const int c0 = q_row0 + r0_in_tile;
      const int c1 = q_row0 + r1_in_tile;
#pragma unroll
      for (int g = 0; g < IGROUPS; ++g) {
        __nv_bfloat162 out_h[8];
        run_head_attn(kvh, q_row0, g, out_h);
        const int h = kvh * IGROUPS + g;
#pragma unroll
        for (int nj = 0; nj < 4; ++nj) {
          const int col = h * ID + nj * 8 + c_in_n8;
          if (c0 < cells) *reinterpret_cast<__nv_bfloat162*>(o_smem + c0 * IC + col) = out_h[nj];
          if (c1 < cells) *reinterpret_cast<__nv_bfloat162*>(o_smem + c1 * IC + col) = out_h[4 + nj];
        }
      }
    }
    __syncthreads();
  }

  // Step 4: 100% coalesced sigmoid(OutGate) multiply + bf16 or per-token INT8 writeout!
  for (int cell = warp; cell < cells; cell += 8) {
    const __nv_bfloat162* o_row = reinterpret_cast<const __nv_bfloat162*>(o_smem + cell * IC);
    const __nv_bfloat162* og_row = reinterpret_cast<const __nv_bfloat162*>(qgkv + (base_row + cell) * 512 + 192);
    float2 v[3];
    float amax = 0.f;
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const int p = j * 32 + lane;
      const float2 of = __bfloat1622float2(o_row[p]);
      const float2 og = __bfloat1622float2(og_row[p]);
      v[j].x = of.x / (1.0f + __expf(-og.x));
      v[j].y = of.y / (1.0f + __expf(-og.y));
      if constexpr (INT8) amax = fmaxf(amax, fmaxf(fabsf(v[j].x), fabsf(v[j].y)));
    }
    if constexpr (INT8) {
      const float gmax = warp_max(amax);
      const float sc = fmaxf(gmax, 1e-6f) * (1.0f / 127.0f);
      const float inv_sc = 127.0f / fmaxf(gmax, 1e-6f);
      uint16_t* dst = reinterpret_cast<uint16_t*>(reinterpret_cast<int8_t*>(out_) + (base_row + cell) * IC);
#pragma unroll
      for (int j = 0; j < 3; ++j) dst[j * 32 + lane] = pack_s8x2(v[j].x, v[j].y, inv_sc);
      if (lane == 0) out_scale[base_row + cell] = sc;
    } else {
      __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<__nv_bfloat16*>(out_) + (base_row + cell) * IC);
#pragma unroll
      for (int j = 0; j < 3; ++j) dst[j * 32 + lane] = __floats2bfloat162_rn(v[j].x, v[j].y);
    }
  }
}

// ---- Kernel 4: Fast KataGo Policy/Q & Value Heads (after Tensor Core conv1pgv GEMM!) + D2 + CacheWrite ----
struct KataHeadsPtrs {
  const float* final_read_q;    // [192]
  const float* final_norm_sb;   // [2, 192]
  const __nv_bfloat16* conv1pgv;// [192, 192]: [conv1p(48); conv1g(48); conv1v(48); zeros(48)]
  const float* head_norm_sb;    // [6, 48] (norm_g s/b, norm_p s/b, val_norm1 s/b)
  const float* linear_g_w;      // [48, 96]
  const float* pol_w;           // [48]
  const float* pol_b;           // [1]
  const float* q_w;             // [K, 48]
  const float* q_b;             // [K]
  const float* pass_pol_w;      // [96]
  const float* pass_pol_b;      // [1]
  const float* pass_q_w;        // [K, 96]
  const float* pass_q_b;        // [K]
  const float* val_l1_w;        // [128, 96]
  const float* val_l1_b;        // [128]
  const float* val_out_w;       // [K, 128]
  const float* val_out_b;       // [K]
};

__global__ void __launch_bounds__(256) inkling_kata_heads_kernel(
    const uint8_t* __restrict__ obs,           // [B, side, side, 4]
    const __nv_bfloat16* __restrict__ pgv_in,  // [M, 192]: [outp(48) | outg(48) | outv(48) | pad(48)]
    KataHeadsPtrs h,
    int max_boards,
    int side,
    int cells,
    int A,
    int K,
    float* __restrict__ logits,  // [B, A] or nullptr
    float* __restrict__ vprobs,  // [B, K]
    float* __restrict__ qprobs,  // [B, A, K]
    const int32_t* __restrict__ d_boards,
    CacheWrite cw) {
  const int active_boards = d_boards ? *d_boards : max_boards;
  const int board = blockIdx.x;
  if (board >= active_boards) return;

  extern __shared__ __align__(16) char smem_raw[];
  __nv_bfloat16* outp_smem = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // [cells, 48]
  float* s_floats = reinterpret_cast<float*>(outp_smem + cells * IHEAD_C);
  float* g_mean = s_floats;           // [48]
  float* g_max = g_mean + 48;         // [48]
  float* v_mean = g_max + 48;         // [48]
  float* v_max = v_mean + 48;         // [48]
  float* g_bias = v_max + 48;         // [48]
  float* val_h = g_bias + 48;         // [128]
  float* v_out_s = val_h + 128;       // [16]
  float* s_pol_w = v_out_s + 16;      // [48]
  float* s_q_w = s_pol_w + 48;        // [16 * 48]
  float* s_q_b = s_q_w + 16 * 48;     // [16]
  float* w_gsum = s_q_b + 16;         // [8 * 48]
  float* w_gmax = w_gsum + 8 * 48;    // [8 * 48]
  float* w_vsum = w_gmax + 8 * 48;    // [8 * 48]
  float* w_vmax = w_vsum + 8 * 48;    // [8 * 48]

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const size_t base_row = (size_t)board * cells;
  const bool is_white = obs[base_row * 4 + 2] != 0;

  if (tid < 48) s_pol_w[tid] = h.pol_w[tid];
  if (tid < K) s_q_b[tid] = h.q_b[tid];
  for (int i = tid; i < K * 48; i += 256) s_q_w[i] = h.q_w[i];
  for (int i = tid; i < 8 * 48; i += 256) {
    w_gsum[i] = 0.f;
    w_gmax[i] = -1e30f;
    w_vsum[i] = 0.f;
    w_vmax[i] = -1e30f;
  }
  __syncthreads();

  const float inv_cells = 1.0f / (float)cells;
  for (int cell = warp; cell < cells; cell += 8) {
    const __nv_bfloat16* row = pgv_in + (base_row + cell) * IC;
    float gg[2] = {0.f, 0.f}, gv[2] = {0.f, 0.f};
    float gsq = 0.f, vsq = 0.f;
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      const int o = pass * 32 + lane;
      if (o < IHEAD_C) {
        outp_smem[cell * IHEAD_C + o] = row[o];
        const float ag = __bfloat162float(row[48 + o]);
        const float av = __bfloat162float(row[96 + o]);
        gg[pass] = ag;
        gv[pass] = av;
        gsq += ag * ag;
        vsq += av * av;
      }
    }
    const float grstd = rsqrtf(warp_sum(gsq) * (1.0f / IHEAD_C) + IRMS_EPS);
    const float vrstd = rsqrtf(warp_sum(vsq) * (1.0f / IHEAD_C) + IRMS_EPS);
#pragma unroll
    for (int pass = 0; pass < 2; ++pass) {
      const int o = pass * 32 + lane;
      if (o < IHEAD_C) {
        const float og = silu_f32(gg[pass] * grstd * h.head_norm_sb[0 * 48 + o] + h.head_norm_sb[1 * 48 + o]);
        const float ov = silu_f32(gv[pass] * vrstd * h.head_norm_sb[4 * 48 + o] + h.head_norm_sb[5 * 48 + o]);
        const int wi = warp * 48 + o;
        w_gsum[wi] += og;
        w_gmax[wi] = fmaxf(w_gmax[wi], og);
        w_vsum[wi] += ov;
        w_vmax[wi] = fmaxf(w_vmax[wi], ov);
      }
    }
  }
  __syncthreads();

  if (tid < 48) {
    float gs = 0.f, gm = -1e30f, vs = 0.f, vm = -1e30f;
#pragma unroll
    for (int w = 0; w < 8; ++w) {
      gs += w_gsum[w * 48 + tid];
      gm = fmaxf(gm, w_gmax[w * 48 + tid]);
      vs += w_vsum[w * 48 + tid];
      vm = fmaxf(vm, w_vmax[w * 48 + tid]);
    }
    g_mean[tid] = gs * inv_cells;
    g_max[tid] = gm;
    v_mean[tid] = vs * inv_cells;
    v_max[tid] = vm;
  }
  __syncthreads();

  if (tid < 48) {
    float acc = 0.f;
    const float* wg = h.linear_g_w + tid * 96;
    for (int j = 0; j < 48; ++j) acc = fmaf(g_mean[j], wg[j], acc);
    for (int j = 0; j < 48; ++j) acc = fmaf(g_max[j], wg[48 + j], acc);
    g_bias[tid] = acc;
  }
  if (tid >= 48 && tid < 48 + IVAL_H) {
    const int o = tid - 48;
    float acc = h.val_l1_b[o];
    const float* wv = h.val_l1_w + o * 96;
    for (int j = 0; j < 48; ++j) acc = fmaf(v_mean[j], wv[j], acc);
    for (int j = 0; j < 48; ++j) acc = fmaf(v_max[j], wv[48 + j], acc);
    val_h[o] = silu_f32(acc);
  }
  __syncthreads();

  if (tid < K) {
    float acc = h.val_out_b[tid];
    const float* wo = h.val_out_w + tid * IVAL_H;
    for (int j = 0; j < IVAL_H; ++j) acc = fmaf(val_h[j], wo[j], acc);
    v_out_s[tid] = acc;
  }
  __syncthreads();
  if (tid == 0) {
    float mx = -1e30f;
    for (int k = 0; k < K; ++k) mx = fmaxf(mx, v_out_s[k]);
    float tot = 0.f;
    for (int k = 0; k < K; ++k) {
      v_out_s[k] = __expf(v_out_s[k] - mx);
      tot += v_out_s[k];
    }
    const float inv_t = 1.0f / tot;
    for (int k = 0; k < K; ++k) vprobs[(size_t)board * K + k] = v_out_s[k] * inv_t;

    if (A == cells + 1) {
      if (logits) {
        float pa = h.pass_pol_b[0];
        for (int j = 0; j < 48; ++j) pa = fmaf(g_mean[j], h.pass_pol_w[j], pa);
        for (int j = 0; j < 48; ++j) pa = fmaf(g_max[j], h.pass_pol_w[48 + j], pa);
        logits[(size_t)board * A + cells] = pa;
      }
      float pq[16], pmx = -1e30f;
      for (int k = 0; k < K; ++k) {
        float acc = h.pass_q_b[k];
        const float* wq = h.pass_q_w + k * 96;
        for (int j = 0; j < 48; ++j) acc = fmaf(g_mean[j], wq[j], acc);
        for (int j = 0; j < 48; ++j) acc = fmaf(g_max[j], wq[48 + j], acc);
        pq[k] = acc;
        pmx = fmaxf(pmx, acc);
      }
      float ptot = 0.f;
      for (int k = 0; k < K; ++k) {
        pq[k] = __expf(pq[k] - pmx);
        ptot += pq[k];
      }
      const float pinv = 1.0f / ptot;
      for (int k = 0; k < K; ++k) qprobs[((size_t)board * A + cells) * K + k] = pq[k] * pinv;
    }
  }

  const float p_ns0 = (lane < 32) ? h.head_norm_sb[2 * 48 + lane] : 0.f;
  const float p_nb0 = (lane < 32) ? h.head_norm_sb[3 * 48 + lane] : 0.f;
  const float p_ns1 = (lane < 16) ? h.head_norm_sb[2 * 48 + 32 + lane] : 0.f;
  const float p_nb1 = (lane < 16) ? h.head_norm_sb[3 * 48 + 32 + lane] : 0.f;

  for (int cell = warp; cell < cells; cell += 8) {
    float hp0 = __bfloat162float(outp_smem[cell * IHEAD_C + lane]) + g_bias[lane];
    float hp1 = (lane < 16) ? (__bfloat162float(outp_smem[cell * IHEAD_C + 32 + lane]) + g_bias[32 + lane]) : 0.f;
    const float psq = hp0 * hp0 + hp1 * hp1;
    const float prstd = rsqrtf(warp_sum(psq) * (1.0f / IHEAD_C) + IRMS_EPS);
    hp0 = silu_f32(hp0 * prstd * p_ns0 + p_nb0);
    hp1 = (lane < 16) ? silu_f32(hp1 * prstd * p_ns1 + p_nb1) : 0.f;

    const int y = cell / side;
    const int x = cell - y * side;
    const int dst_action = is_white ? (x * side + y) : cell;

    if (logits) {
      float pdot = hp0 * s_pol_w[lane];
      if (lane < 16) pdot = fmaf(hp1, s_pol_w[32 + lane], pdot);
      const float pl = warp_sum(pdot) + h.pol_b[0];
      if (lane == 0) logits[(size_t)board * A + dst_action] = pl;
    }

    float ql[16], qmx = -1e30f;
    for (int k = 0; k < K; ++k) {
      const float* wqk = s_q_w + k * IHEAD_C;
      float qdot = hp0 * wqk[lane];
      if (lane < 16) qdot = fmaf(hp1, wqk[32 + lane], qdot);
      ql[k] = warp_sum(qdot) + s_q_b[k];
      qmx = fmaxf(qmx, ql[k]);
    }
    if (lane == 0) {
      float qtot = 0.f;
      for (int k = 0; k < K; ++k) {
        ql[k] = __expf(ql[k] - qmx);
        qtot += ql[k];
      }
      const float qinv = 1.0f / qtot;
      float* qdst = qprobs + ((size_t)board * A + dst_action) * K;
      for (int k = 0; k < K; ++k) qdst[k] = ql[k] * qinv;
    }
  }
  __syncthreads();

  if (cw.table_locks) {
    const CacheKey mk = cw.miss_keys[board];
    if (mk.meta & 0x80000000u) {
      const uint32_t tag = 2u + *cw.cache_wave;
      const uint32_t slot = hash_cache_key(mk);
      __shared__ bool won_slot;
      if (tid == 0) won_slot = cache_try_acquire(cw.table_locks, slot, tag);
      __syncthreads();
      if (won_slot) {
        if (tid == 0) cw.table_keys[slot] = mk;
        if (tid < K) cw.table_v[(size_t)slot * K + tid] = __float2bfloat16_rn(vprobs[(size_t)board * K + tid]);
        const int AK = A * K;
        for (int i = tid; i < AK; i += 256) {
          cw.table_q[(size_t)slot * AK + i] = __float2bfloat16_rn(qprobs[(size_t)board * AK + i]);
        }
        __syncthreads();
        if (tid == 0) {
          __threadfence();
          atomicExch(&cw.table_locks[slot], tag);
        }
      }
    }
  }
}

// ---- Shared launcher for both Full and Compact Inkling FFI handlers ----
ffi::Error configure_inkling_smem(int device) {
  static int configured = -1;
  if (configured == device) return ffi::Error::Success();
  auto set = [](const void* fn, int bytes) {
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  };
  constexpr int smem_64 = gemm_smem_bytes<false, 64>();
  constexpr int smem_128 = gemm_smem_bytes<false, 128>();
  constexpr int smem_swi = gemm_smem_bytes<true, 64>();
  set(reinterpret_cast<const void*>(inkling_gemm_kernel<false, false, 64, false>), smem_64);
  set(reinterpret_cast<const void*>(inkling_gemm_kernel<false, false, 128, false>), smem_128);
  set(reinterpret_cast<const void*>(inkling_gemm_kernel<false, true, 64, false>), smem_swi);
  set(reinterpret_cast<const void*>(inkling_gemm_kernel<true, false, 64, false>), smem_64);
  set(reinterpret_cast<const void*>(inkling_gemm_kernel<true, false, 128, false>), smem_128);
  set(reinterpret_cast<const void*>(inkling_gemm_kernel<true, false, 64, true>), smem_64);
  set(reinterpret_cast<const void*>(inkling_gemm_kernel<true, true, 64, false>), smem_swi);
  set(reinterpret_cast<const void*>(inkling_attn_kernel<false>), 96 * 1024);
  set(reinterpret_cast<const void*>(inkling_attn_kernel<true>), 96 * 1024);
  set(reinterpret_cast<const void*>(inkling_shortconv_dr_norm_kernel<false, 1, false>), 64 * 1024);
  set(reinterpret_cast<const void*>(inkling_shortconv_dr_norm_kernel<false, 3, false>), 64 * 1024);
  set(reinterpret_cast<const void*>(inkling_shortconv_dr_norm_kernel<false, 1, true>), 64 * 1024);
  set(reinterpret_cast<const void*>(inkling_shortconv_dr_norm_kernel<true, 1, false>), 64 * 1024);
  set(reinterpret_cast<const void*>(inkling_shortconv_dr_norm_kernel<true, 3, false>), 64 * 1024);
  set(reinterpret_cast<const void*>(inkling_kata_heads_kernel), 48 * 1024);
  configured = device;
  return ffi::Error::Success();
}

template <bool INT8>
ffi::Error run_inkling_forward(
    cudaStream_t stream,
    int sms,
    int max_boards,
    int side,
    int cells,
    int A,
    int K,
    int L,
    int layers_per_block,
    int FFN,
    const uint8_t* obs,
    const int32_t* d_boards,
    // Stem
    const __nv_bfloat16* stem_w,
    const float* stem_b,
    const float* emb_s,
    const float* emb_b,
    // Per-layer weights
    const float* attn_read_q,   // [L, 3, 192]
    const float* mlp_read_q,    // [L, 192]
    const float* attn_norm_sb,  // [L, 6, 192]
    const float* mlp_norm_sb,   // [L, 2, 192]
    const void* w_qkvg,         // [L, 512, 192] bf16 or int8
    const float* s_qkvg,        // [L, 512] f32
    const void* w_out,          // [L, 192, 192] bf16 or int8
    const float* s_out,         // [L, 192] f32
    const void* w_gu,           // [L, 2*FFN, 192] paired 128-row tiles
    const float* s_gu,          // [L, 2*FFN] f32
    const void* w_down,         // [L, 192, FFN] bf16 or int8
    const float* s_down,        // [L, 192] f32
    const float* q_scale,       // [L, 6, 32]
    const float* sink_k,        // [L, 2, 2, 32]
    const float* rope_freqs,    // [L, 2, 16, 2]
    const float* dw_kv,         // [L, 2, 9, 64]
    const float* dw_post,       // [L, 2, 9, 192]
    // Heads
    KataHeadsPtrs heads,
    // Outputs
    float* logits,
    float* vprobs,
    float* qprobs,
    CacheWrite cw,
    // Scratch buffers
    __nv_bfloat16* partial,     // [max_M, 192]
    __nv_bfloat16* bank_norm,   // [max_bank, max_M, 192]
    __nv_bfloat16* buf_qkvg,    // [max_M, max(512, FFN)] bf16 (also holds SwiGLU int8 + grp64 scales!)
    __nv_bfloat16* buf_act,     // [4, max_M, 192] (act0..act2 streams + disjoint act3 for w_down)
    float* buf_scales) {        // [3, max_M]
  if (max_boards == 0) return ffi::Error::Success();
  const int max_M = max_boards * cells;
  constexpr int E = INT8 ? 1 : 2;
  const int warp_grid = std::max(1, std::min((max_M + 7) / 8, sms * 8));
  const int m_tiles = (max_M + GEMM_BM - 1) / GEMM_BM;
  const int gemm_grid_64 = std::max(1, std::min(m_tiles * 8, sms * 2));
  const int gemm_grid_128 = std::max(1, std::min(m_tiles * 4, sms * 2));
  const int gemm_grid_swi = std::max(1, std::min(m_tiles * (FFN / 64), sms * 2));
  constexpr int smem_64 = gemm_smem_bytes<false, 64>();
  constexpr int smem_128 = gemm_smem_bytes<false, 128>();
  constexpr int smem_swi = gemm_smem_bytes<true, 64>();

  const int attn_smem = inkling_attn_smem_bytes(cells);
  const int sc_dr_smem = cells * IC * (int)sizeof(__nv_bfloat16) + 9 * IC * (int)sizeof(float);
  const int heads_smem =
      cells * IHEAD_C * (int)sizeof(__nv_bfloat16) +
      (48 * 5 + IVAL_H + 16 + 48 + 16 * 48 + 16 + 4 * 8 * 48) * (int)sizeof(float);

  const bool prof = (std::getenv("INKLING_PROFILE") != nullptr) && (d_boards == nullptr);
  cudaEvent_t ev_a = nullptr, ev_b = nullptr;
  float t_stem = 0, t_qkv = 0, t_attn = 0, t_out = 0, t_scdr = 0, t_swi = 0, t_down = 0, t_head = 0;
  auto tic = [&]() { if (prof) cudaEventRecord(ev_a, stream); };
  auto toc = [&](float& acc) {
    if (prof) {
      cudaEventRecord(ev_b, stream);
      cudaEventSynchronize(ev_b);
      float ms = 0.f;
      cudaEventElapsedTime(&ms, ev_a, ev_b);
      acc += ms;
    }
  };
  if (prof) {
    cudaEventCreate(&ev_a);
    cudaEventCreate(&ev_b);
  }

  char* act0 = reinterpret_cast<char*>(buf_act);
  char* act1 = act0 + (size_t)max_M * IC * 2;
  char* act2 = act1 + (size_t)max_M * IC * 2;
  char* act3 = act2 + (size_t)max_M * IC * 2;
  float* sc0 = buf_scales;
  float* sc1 = buf_scales + max_M;
  float* sc2 = buf_scales + (size_t)2 * max_M;

  // For INT8 SwiGLU: pack int8 [M, FFN] and f32 grp64 scales [M, FFN/64] inside buf_qkvg (2 * max(512, FFN) B/row).
  void* swi_out = reinterpret_cast<void*>(buf_qkvg);
  float* swi_grp_scales = reinterpret_cast<float*>(
      reinterpret_cast<char*>(buf_qkvg) + (((size_t)max_M * FFN + 255ULL) & ~255ULL));

  // 1. Fused Stem + emb_norm + bank[0] seal + Layer-0 attn_norm (+ INT8) -> act0 (or act0..2 if L == 1).
  tic();
  if (L == 1) {
    inkling_stem_fused_kernel<INT8, 3><<<warp_grid, 256, 0, stream>>>(
        obs, stem_w, stem_b, emb_s, emb_b, attn_norm_sb, bank_norm,
        act0, sc0, act1, sc1, act2, sc2, max_boards, side, cells, d_boards);
  } else {
    inkling_stem_fused_kernel<INT8, 1><<<warp_grid, 256, 0, stream>>>(
        obs, stem_w, stem_b, emb_s, emb_b, attn_norm_sb, bank_norm,
        act0, sc0, nullptr, nullptr, nullptr, nullptr, max_boards, side, cells, d_boards);
  }
  toc(t_stem);
  int nb = 1;  // Layer 0 always seals bank[0] (0 % layers_per_block == 0).

  for (int i = 0; i < L; ++i) {
    const bool seals_bank = (i % layers_per_block) == 0;
    const bool is_full = (i == L - 1) || (L >= 6 && ((i + 1) % 3) == 0);

    // 2a. QKV + OutGate GEMM(s) -> buf_qkvg [M, 512]
    const char* wq_i = reinterpret_cast<const char*>(w_qkvg) + (size_t)i * 512 * IC * E;
    const float* sq_i = s_qkvg + (size_t)i * 512;
    tic();
    if (is_full) {
      inkling_gemm_kernel<INT8, false, 128, false><<<gemm_grid_128, GEMM_THREADS, smem_128, stream>>>(
          act0, sc0, wq_i, sq_i, buf_qkvg, nullptr, max_boards, cells, IC, 384, 512, 0, d_boards);
      inkling_gemm_kernel<INT8, false, 64, false><<<gemm_grid_64, GEMM_THREADS, smem_64, stream>>>(
          act1, sc1, wq_i + (size_t)384 * IC * E, sq_i + 384, buf_qkvg, nullptr, max_boards, cells, IC, 64, 512, 384, d_boards);
      inkling_gemm_kernel<INT8, false, 64, false><<<gemm_grid_64, GEMM_THREADS, smem_64, stream>>>(
          act2, sc2, wq_i + (size_t)448 * IC * E, sq_i + 448, buf_qkvg, nullptr, max_boards, cells, IC, 64, 512, 448, d_boards);
    } else {
      inkling_gemm_kernel<INT8, false, 128, false><<<gemm_grid_128, GEMM_THREADS, smem_128, stream>>>(
          act0, sc0, wq_i, sq_i, buf_qkvg, nullptr, max_boards, cells, IC, 512, 512, 0, d_boards);
    }
    toc(t_qkv);

    // 2b. Super-fused Low-Register Tensor Core FlashAttention -> act0 [M, 192]
    tic();
    inkling_attn_kernel<INT8><<<max_boards, 256, attn_smem, stream>>>(
        buf_qkvg,
        q_scale + (size_t)i * IH * ID,
        sink_k + (size_t)i * 2 * IKVDIM,
        rope_freqs + (size_t)i * IKV * 16 * 2,
        dw_kv + (size_t)i * 2 * 9 * IKVDIM,
        act0, sc0, max_boards, side, cells, d_boards);
    toc(t_attn);

    // 2c. out_proj GEMM (192->192) -> act1, then FUSED ShortConv(after_attn) + residual/seal + MLP DepthRead/norm!
    const char* wo_i = reinterpret_cast<const char*>(w_out) + (size_t)i * IC * IC * E;
    const float* so_i = s_out + (size_t)i * IC;
    tic();
    inkling_gemm_kernel<INT8, false, 64, false><<<gemm_grid_64, GEMM_THREADS, smem_64, stream>>>(
        act0, sc0, wo_i, so_i, act1, nullptr, max_boards, cells, IC, IC, IC, 0, d_boards);
    toc(t_out);
    tic();
    inkling_shortconv_dr_norm_kernel<INT8, 1, false><<<max_boards, 256, sc_dr_smem, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(act1),
        dw_post + (size_t)(2 * i) * 9 * IC,
        partial, bank_norm, nb,
        seals_bank, /*write_partial=*/true, /*save_partial_to_bank=*/false,
        mlp_read_q + (size_t)i * IC,
        mlp_norm_sb + (size_t)i * 2 * IC,
        act0, sc0, nullptr, nullptr, nullptr, nullptr,
        max_boards, side, cells, max_M, d_boards);
    toc(t_scdr);

    // 2d. Fused [W_gate | W_up] + Capped-SwiGLU (+ fused group-64 INT8 quant when INT8!) -> swi_out
    const char* wgu_i = reinterpret_cast<const char*>(w_gu) + (size_t)i * (2 * FFN) * IC * E;
    const float* sgu_i = s_gu + (size_t)i * (2 * FFN);
    tic();
    inkling_gemm_kernel<INT8, true, 64, false><<<gemm_grid_swi, GEMM_THREADS, smem_swi, stream>>>(
        act0, sc0, wgu_i, sgu_i, swi_out, INT8 ? swi_grp_scales : nullptr,
        max_boards, cells, IC, FFN, FFN, 0, d_boards);
    toc(t_swi);

    // 2e. w_down GEMM (FFN->192, consuming fused group-64 INT8 directly when INT8!) -> act3 (disjoint from act0..act2!)
    const char* wd_i = reinterpret_cast<const char*>(w_down) + (size_t)i * IC * FFN * E;
    const float* sd_i = s_down + (size_t)i * IC;
    tic();
    if constexpr (INT8) {
      inkling_gemm_kernel<true, false, 64, true><<<gemm_grid_64, GEMM_THREADS, smem_64, stream>>>(
          swi_out, swi_grp_scales, wd_i, sd_i, act3, nullptr, max_boards, cells, FFN, IC, IC, 0, d_boards);
    } else {
      inkling_gemm_kernel<false, false, 64, false><<<gemm_grid_64, GEMM_THREADS, smem_64, stream>>>(
          swi_out, nullptr, wd_i, sd_i, act3, nullptr, max_boards, cells, FFN, IC, IC, 0, d_boards);
    }
    toc(t_down);

    // 2f. FUSED ShortConv(after_mlp) + residual + (Next Layer's Attn DepthRead/norm OR Final DepthRead/norm+SiLU)!
    tic();
    if (i + 1 < L) {
      const int nxt = i + 1;
      const bool next_seals = (nxt % layers_per_block) == 0;
      const bool next_full = (nxt == L - 1) || (L >= 6 && ((nxt + 1) % 3) == 0);
      // If next layer seals bank, its after_attn will REPLACE partial -> skip writing partial!
      const bool write_part = !next_seals;
      if (next_full) {
        inkling_shortconv_dr_norm_kernel<INT8, 3, false><<<max_boards, 256, sc_dr_smem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(act3),
            dw_post + (size_t)(2 * i + 1) * 9 * IC,
            partial, bank_norm, nb,
            /*replace_partial=*/false, write_part, next_seals,
            attn_read_q + (size_t)nxt * 3 * IC,
            attn_norm_sb + (size_t)nxt * 6 * IC,
            act0, sc0, act1, sc1, act2, sc2,
            max_boards, side, cells, max_M, d_boards);
      } else {
        inkling_shortconv_dr_norm_kernel<INT8, 1, false><<<max_boards, 256, sc_dr_smem, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(act3),
            dw_post + (size_t)(2 * i + 1) * 9 * IC,
            partial, bank_norm, nb,
            /*replace_partial=*/false, write_part, next_seals,
            attn_read_q + (size_t)nxt * 3 * IC,
            attn_norm_sb + (size_t)nxt * 6 * IC,
            act0, sc0, nullptr, nullptr, nullptr, nullptr,
            max_boards, side, cells, max_M, d_boards);
      }
      if (next_seals) ++nb;
    } else {
      // Final layer: fuse ShortConv(after_mlp) + residual + Final DepthRead + final_norm + SiLU -> act0 (bf16)!
      inkling_shortconv_dr_norm_kernel<false, 1, true><<<max_boards, 256, sc_dr_smem, stream>>>(
          reinterpret_cast<const __nv_bfloat16*>(act3),
          dw_post + (size_t)(2 * i + 1) * 9 * IC,
          partial, bank_norm, nb,
          /*replace_partial=*/false, /*write_partial=*/false, /*save_partial_to_bank=*/false,
          heads.final_read_q, heads.final_norm_sb,
          act0, nullptr, nullptr, nullptr, nullptr, nullptr,
          max_boards, side, cells, max_M, d_boards);
    }
    toc(t_scdr);
  }

  // 3. Tensor Core GEMM [conv1p|conv1g|conv1v] -> buf_qkvg [M, 192], then KataGo heads + D2 + CacheWrite!
  tic();
  inkling_gemm_kernel<false, false, 64, false><<<gemm_grid_64, GEMM_THREADS, smem_64, stream>>>(
      act0, nullptr, heads.conv1pgv, nullptr, buf_qkvg, nullptr, max_boards, cells, IC, IC, IC, 0, d_boards);
  inkling_kata_heads_kernel<<<max_boards, 256, heads_smem, stream>>>(
      obs, buf_qkvg, heads, max_boards, side, cells, A, K,
      logits, vprobs, qprobs, d_boards, cw);
  toc(t_head);

  if (prof) {
    float tot = t_stem + t_qkv + t_attn + t_out + t_scdr + t_swi + t_down + t_head;
    std::fprintf(stderr,
        "[INKLING_PROFILE %s B=%d] tot=%.2fms | stem=%.2f qkv=%.2f attn=%.2f out=%.2f scdr=%.2f swi=%.2f down=%.2f head=%.2f\n",
        INT8 ? "int8" : "bf16", max_boards, tot, t_stem, t_qkv, t_attn, t_out, t_scdr, t_swi, t_down, t_head);
    cudaEventDestroy(ev_a);
    cudaEventDestroy(ev_b);
  }

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(err));
  return ffi::Error::Success();
}

// ---- XLA FFI Entrypoint 1: Full Eval (returns logits, vprobs, qprobs) ----
ffi::Error InklingNetwork(
    cudaStream_t stream,
    int64_t use_int8,
    int64_t layers_per_block,
    ffi::AnyBuffer obs,
    ffi::AnyBuffer stem_w, ffi::AnyBuffer stem_b, ffi::AnyBuffer emb_s, ffi::AnyBuffer emb_b,
    ffi::AnyBuffer attn_read_q, ffi::AnyBuffer mlp_read_q,
    ffi::AnyBuffer attn_norm_sb, ffi::AnyBuffer mlp_norm_sb,
    ffi::AnyBuffer w_qkvg, ffi::AnyBuffer s_qkvg,
    ffi::AnyBuffer w_out, ffi::AnyBuffer s_out,
    ffi::AnyBuffer w_gu, ffi::AnyBuffer s_gu,
    ffi::AnyBuffer w_down, ffi::AnyBuffer s_down,
    ffi::AnyBuffer q_scale, ffi::AnyBuffer sink_k, ffi::AnyBuffer rope_freqs,
    ffi::AnyBuffer dw_kv, ffi::AnyBuffer dw_post,
    ffi::AnyBuffer final_read_q, ffi::AnyBuffer final_norm_sb,
    ffi::AnyBuffer conv1pgv,
    ffi::AnyBuffer head_norm_sb, ffi::AnyBuffer linear_g_w,
    ffi::AnyBuffer pol_w, ffi::AnyBuffer pol_b, ffi::AnyBuffer q_w, ffi::AnyBuffer q_b,
    ffi::AnyBuffer pass_pol_w, ffi::AnyBuffer pass_pol_b, ffi::AnyBuffer pass_q_w, ffi::AnyBuffer pass_q_b,
    ffi::AnyBuffer val_l1_w, ffi::AnyBuffer val_l1_b, ffi::AnyBuffer val_out_w, ffi::AnyBuffer val_out_b,
    ffi::Result<ffi::AnyBuffer> logits,
    ffi::Result<ffi::AnyBuffer> vprobs,
    ffi::Result<ffi::AnyBuffer> qprobs,
    ffi::Result<ffi::AnyBuffer> sc_partial,
    ffi::Result<ffi::AnyBuffer> sc_bank,
    ffi::Result<ffi::AnyBuffer> sc_qkvg,
    ffi::Result<ffi::AnyBuffer> sc_act,
    ffi::Result<ffi::AnyBuffer> sc_scales) {
  auto od = obs.dimensions();
  const int boards = (int)od[0], side = (int)od[1], cells = side * side;
  auto qd = qprobs->dimensions();
  const int A = (int)qd[1], K = (int)qd[2];
  const int L = (int)w_out.dimensions()[0];
  const int FFN = (int)w_down.dimensions()[2];
  if (side < 3 || side > 11 || K > 16 || (FFN % 64) != 0 || FFN > 768)
    return ffi::Error::InvalidArgument("inkling cuda kernel requires Hex 3..11, K<=16, FFN multiple of 64 <= 768");

  int device = 0, sms = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  configure_inkling_smem(device);

  KataHeadsPtrs hp{
      cptr<float>(final_read_q), cptr<float>(final_norm_sb),
      cptr<__nv_bfloat16>(conv1pgv),
      cptr<float>(head_norm_sb), cptr<float>(linear_g_w),
      cptr<float>(pol_w), cptr<float>(pol_b), cptr<float>(q_w), cptr<float>(q_b),
      cptr<float>(pass_pol_w), cptr<float>(pass_pol_b), cptr<float>(pass_q_w), cptr<float>(pass_q_b),
      cptr<float>(val_l1_w), cptr<float>(val_l1_b), cptr<float>(val_out_w), cptr<float>(val_out_b)};
  CacheWrite cw{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};

  if (use_int8) {
    return run_inkling_forward<true>(
        stream, sms, boards, side, cells, A, K, L, (int)layers_per_block, FFN,
        cptr<uint8_t>(obs), nullptr,
        cptr<__nv_bfloat16>(stem_w), cptr<float>(stem_b), cptr<float>(emb_s), cptr<float>(emb_b),
        cptr<float>(attn_read_q), cptr<float>(mlp_read_q), cptr<float>(attn_norm_sb), cptr<float>(mlp_norm_sb),
        w_qkvg.untyped_data(), cptr<float>(s_qkvg), w_out.untyped_data(), cptr<float>(s_out),
        w_gu.untyped_data(), cptr<float>(s_gu), w_down.untyped_data(), cptr<float>(s_down),
        cptr<float>(q_scale), cptr<float>(sink_k), cptr<float>(rope_freqs), cptr<float>(dw_kv), cptr<float>(dw_post),
        hp, mptr<float>(*logits), mptr<float>(*vprobs), mptr<float>(*qprobs), cw,
        mptr<__nv_bfloat16>(*sc_partial), mptr<__nv_bfloat16>(*sc_bank), mptr<__nv_bfloat16>(*sc_qkvg),
        mptr<__nv_bfloat16>(*sc_act), mptr<float>(*sc_scales));
  }
  return run_inkling_forward<false>(
      stream, sms, boards, side, cells, A, K, L, (int)layers_per_block, FFN,
      cptr<uint8_t>(obs), nullptr,
      cptr<__nv_bfloat16>(stem_w), cptr<float>(stem_b), cptr<float>(emb_s), cptr<float>(emb_b),
      cptr<float>(attn_read_q), cptr<float>(mlp_read_q), cptr<float>(attn_norm_sb), cptr<float>(mlp_norm_sb),
      w_qkvg.untyped_data(), cptr<float>(s_qkvg), w_out.untyped_data(), cptr<float>(s_out),
      w_gu.untyped_data(), cptr<float>(s_gu), w_down.untyped_data(), cptr<float>(s_down),
      cptr<float>(q_scale), cptr<float>(sink_k), cptr<float>(rope_freqs), cptr<float>(dw_kv), cptr<float>(dw_post),
      hp, mptr<float>(*logits), mptr<float>(*vprobs), mptr<float>(*qprobs), cw,
      mptr<__nv_bfloat16>(*sc_partial), mptr<__nv_bfloat16>(*sc_bank), mptr<__nv_bfloat16>(*sc_qkvg),
      mptr<__nv_bfloat16>(*sc_act), mptr<float>(*sc_scales));
}

// ---- XLA FFI Entrypoint 2: Compact Search (dynamic *d_boards + PositionCache writeback) ----
ffi::Error InklingNetworkCompact(
    cudaStream_t stream,
    int64_t use_int8,
    int64_t layers_per_block,
    int64_t use_cache,
    ffi::AnyBuffer obs,
    ffi::AnyBuffer d_boards,
    ffi::AnyBuffer miss_keys,
    ffi::AnyBuffer stem_w, ffi::AnyBuffer stem_b, ffi::AnyBuffer emb_s, ffi::AnyBuffer emb_b,
    ffi::AnyBuffer attn_read_q, ffi::AnyBuffer mlp_read_q,
    ffi::AnyBuffer attn_norm_sb, ffi::AnyBuffer mlp_norm_sb,
    ffi::AnyBuffer w_qkvg, ffi::AnyBuffer s_qkvg,
    ffi::AnyBuffer w_out, ffi::AnyBuffer s_out,
    ffi::AnyBuffer w_gu, ffi::AnyBuffer s_gu,
    ffi::AnyBuffer w_down, ffi::AnyBuffer s_down,
    ffi::AnyBuffer q_scale, ffi::AnyBuffer sink_k, ffi::AnyBuffer rope_freqs,
    ffi::AnyBuffer dw_kv, ffi::AnyBuffer dw_post,
    ffi::AnyBuffer final_read_q, ffi::AnyBuffer final_norm_sb,
    ffi::AnyBuffer conv1pgv,
    ffi::AnyBuffer head_norm_sb, ffi::AnyBuffer linear_g_w,
    ffi::AnyBuffer pol_w, ffi::AnyBuffer pol_b, ffi::AnyBuffer q_w, ffi::AnyBuffer q_b,
    ffi::AnyBuffer pass_pol_w, ffi::AnyBuffer pass_pol_b, ffi::AnyBuffer pass_q_w, ffi::AnyBuffer pass_q_b,
    ffi::AnyBuffer val_l1_w, ffi::AnyBuffer val_l1_b, ffi::AnyBuffer val_out_w, ffi::AnyBuffer val_out_b,
    ffi::AnyBuffer cache_keys, ffi::AnyBuffer cache_locks, ffi::AnyBuffer cache_v, ffi::AnyBuffer cache_q,
    ffi::AnyBuffer cache_wave,
    ffi::Result<ffi::AnyBuffer> vprobs,
    ffi::Result<ffi::AnyBuffer> qprobs,
    ffi::Result<ffi::AnyBuffer> r_cache_keys,
    ffi::Result<ffi::AnyBuffer> r_cache_locks,
    ffi::Result<ffi::AnyBuffer> r_cache_v,
    ffi::Result<ffi::AnyBuffer> r_cache_q,
    ffi::Result<ffi::AnyBuffer> sc_partial,
    ffi::Result<ffi::AnyBuffer> sc_bank,
    ffi::Result<ffi::AnyBuffer> sc_qkvg,
    ffi::Result<ffi::AnyBuffer> sc_act,
    ffi::Result<ffi::AnyBuffer> sc_scales) {
  auto od = obs.dimensions();
  const int max_boards = (int)od[0], side = (int)od[1], cells = side * side;
  auto qd = qprobs->dimensions();
  const int A = (int)qd[1], K = (int)qd[2];
  const int L = (int)w_out.dimensions()[0];
  const int FFN = (int)w_down.dimensions()[2];
  if (side < 3 || side > 11 || K > 16 || (FFN % 64) != 0 || FFN > 768)
    return ffi::Error::InvalidArgument("inkling compact kernel requires Hex 3..11, K<=16, FFN multiple of 64 <= 768");

  int device = 0, sms = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  configure_inkling_smem(device);

  KataHeadsPtrs hp{
      cptr<float>(final_read_q), cptr<float>(final_norm_sb),
      cptr<__nv_bfloat16>(conv1pgv),
      cptr<float>(head_norm_sb), cptr<float>(linear_g_w),
      cptr<float>(pol_w), cptr<float>(pol_b), cptr<float>(q_w), cptr<float>(q_b),
      cptr<float>(pass_pol_w), cptr<float>(pass_pol_b), cptr<float>(pass_q_w), cptr<float>(pass_q_b),
      cptr<float>(val_l1_w), cptr<float>(val_l1_b), cptr<float>(val_out_w), cptr<float>(val_out_b)};
  CacheWrite cw{nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
  if (use_cache) {
    cw.miss_keys = reinterpret_cast<const CacheKey*>(miss_keys.untyped_data());
    cw.table_keys = reinterpret_cast<CacheKey*>(r_cache_keys->untyped_data());
    cw.table_locks = reinterpret_cast<uint32_t*>(r_cache_locks->untyped_data());
    cw.table_v = mptr<__nv_bfloat16>(*r_cache_v);
    cw.table_q = mptr<__nv_bfloat16>(*r_cache_q);
    cw.cache_wave = cptr<uint32_t>(cache_wave);
  }
  const int32_t* d_cnt = cptr<int32_t>(d_boards);

  if (use_int8) {
    return run_inkling_forward<true>(
        stream, sms, max_boards, side, cells, A, K, L, (int)layers_per_block, FFN,
        cptr<uint8_t>(obs), d_cnt,
        cptr<__nv_bfloat16>(stem_w), cptr<float>(stem_b), cptr<float>(emb_s), cptr<float>(emb_b),
        cptr<float>(attn_read_q), cptr<float>(mlp_read_q), cptr<float>(attn_norm_sb), cptr<float>(mlp_norm_sb),
        w_qkvg.untyped_data(), cptr<float>(s_qkvg), w_out.untyped_data(), cptr<float>(s_out),
        w_gu.untyped_data(), cptr<float>(s_gu), w_down.untyped_data(), cptr<float>(s_down),
        cptr<float>(q_scale), cptr<float>(sink_k), cptr<float>(rope_freqs), cptr<float>(dw_kv), cptr<float>(dw_post),
        hp, nullptr, mptr<float>(*vprobs), mptr<float>(*qprobs), cw,
        mptr<__nv_bfloat16>(*sc_partial), mptr<__nv_bfloat16>(*sc_bank), mptr<__nv_bfloat16>(*sc_qkvg),
        mptr<__nv_bfloat16>(*sc_act), mptr<float>(*sc_scales));
  }
  return run_inkling_forward<false>(
      stream, sms, max_boards, side, cells, A, K, L, (int)layers_per_block, FFN,
      cptr<uint8_t>(obs), d_cnt,
      cptr<__nv_bfloat16>(stem_w), cptr<float>(stem_b), cptr<float>(emb_s), cptr<float>(emb_b),
      cptr<float>(attn_read_q), cptr<float>(mlp_read_q), cptr<float>(attn_norm_sb), cptr<float>(mlp_norm_sb),
      w_qkvg.untyped_data(), cptr<float>(s_qkvg), w_out.untyped_data(), cptr<float>(s_out),
      w_gu.untyped_data(), cptr<float>(s_gu), w_down.untyped_data(), cptr<float>(s_down),
      cptr<float>(q_scale), cptr<float>(sink_k), cptr<float>(rope_freqs), cptr<float>(dw_kv), cptr<float>(dw_post),
      hp, nullptr, mptr<float>(*vprobs), mptr<float>(*qprobs), cw,
      mptr<__nv_bfloat16>(*sc_partial), mptr<__nv_bfloat16>(*sc_bank), mptr<__nv_bfloat16>(*sc_qkvg),
      mptr<__nv_bfloat16>(*sc_act), mptr<float>(*sc_scales));
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ScacchiInklingNetwork, InklingNetwork,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("use_int8")
        .Attr<int64_t>("layers_per_block")
        .Arg<ffi::AnyBuffer>()  // obs
        .Arg<ffi::AnyBuffer>()  // stem_w
        .Arg<ffi::AnyBuffer>()  // stem_b
        .Arg<ffi::AnyBuffer>()  // emb_s
        .Arg<ffi::AnyBuffer>()  // emb_b
        .Arg<ffi::AnyBuffer>()  // attn_read_q
        .Arg<ffi::AnyBuffer>()  // mlp_read_q
        .Arg<ffi::AnyBuffer>()  // attn_norm_sb
        .Arg<ffi::AnyBuffer>()  // mlp_norm_sb
        .Arg<ffi::AnyBuffer>()  // w_qkvg
        .Arg<ffi::AnyBuffer>()  // s_qkvg
        .Arg<ffi::AnyBuffer>()  // w_out
        .Arg<ffi::AnyBuffer>()  // s_out
        .Arg<ffi::AnyBuffer>()  // w_gu
        .Arg<ffi::AnyBuffer>()  // s_gu
        .Arg<ffi::AnyBuffer>()  // w_down
        .Arg<ffi::AnyBuffer>()  // s_down
        .Arg<ffi::AnyBuffer>()  // q_scale
        .Arg<ffi::AnyBuffer>()  // sink_k
        .Arg<ffi::AnyBuffer>()  // rope_freqs
        .Arg<ffi::AnyBuffer>()  // dw_kv
        .Arg<ffi::AnyBuffer>()  // dw_post
        .Arg<ffi::AnyBuffer>()  // final_read_q
        .Arg<ffi::AnyBuffer>()  // final_norm_sb
        .Arg<ffi::AnyBuffer>()  // conv1pgv
        .Arg<ffi::AnyBuffer>()  // head_norm_sb
        .Arg<ffi::AnyBuffer>()  // linear_g_w
        .Arg<ffi::AnyBuffer>()  // pol_w
        .Arg<ffi::AnyBuffer>()  // pol_b
        .Arg<ffi::AnyBuffer>()  // q_w
        .Arg<ffi::AnyBuffer>()  // q_b
        .Arg<ffi::AnyBuffer>()  // pass_pol_w
        .Arg<ffi::AnyBuffer>()  // pass_pol_b
        .Arg<ffi::AnyBuffer>()  // pass_q_w
        .Arg<ffi::AnyBuffer>()  // pass_q_b
        .Arg<ffi::AnyBuffer>()  // val_l1_w
        .Arg<ffi::AnyBuffer>()  // val_l1_b
        .Arg<ffi::AnyBuffer>()  // val_out_w
        .Arg<ffi::AnyBuffer>()  // val_out_b
        .Ret<ffi::AnyBuffer>()  // logits
        .Ret<ffi::AnyBuffer>()  // vprobs
        .Ret<ffi::AnyBuffer>()  // qprobs
        .Ret<ffi::AnyBuffer>()  // sc_partial
        .Ret<ffi::AnyBuffer>()  // sc_bank
        .Ret<ffi::AnyBuffer>()  // sc_qkvg
        .Ret<ffi::AnyBuffer>()  // sc_act
        .Ret<ffi::AnyBuffer>(), // sc_scales
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ScacchiInklingNetworkCompact, InklingNetworkCompact,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("use_int8")
        .Attr<int64_t>("layers_per_block")
        .Attr<int64_t>("use_cache")
        .Arg<ffi::AnyBuffer>()  // 0: obs
        .Arg<ffi::AnyBuffer>()  // 1: d_boards
        .Arg<ffi::AnyBuffer>()  // 2: miss_keys
        .Arg<ffi::AnyBuffer>()  // 3: stem_w
        .Arg<ffi::AnyBuffer>()  // 4: stem_b
        .Arg<ffi::AnyBuffer>()  // 5: emb_s
        .Arg<ffi::AnyBuffer>()  // 6: emb_b
        .Arg<ffi::AnyBuffer>()  // 7: attn_read_q
        .Arg<ffi::AnyBuffer>()  // 8: mlp_read_q
        .Arg<ffi::AnyBuffer>()  // 9: attn_norm_sb
        .Arg<ffi::AnyBuffer>()  // 10: mlp_norm_sb
        .Arg<ffi::AnyBuffer>()  // 11: w_qkvg
        .Arg<ffi::AnyBuffer>()  // 12: s_qkvg
        .Arg<ffi::AnyBuffer>()  // 13: w_out
        .Arg<ffi::AnyBuffer>()  // 14: s_out
        .Arg<ffi::AnyBuffer>()  // 15: w_gu
        .Arg<ffi::AnyBuffer>()  // 16: s_gu
        .Arg<ffi::AnyBuffer>()  // 17: w_down
        .Arg<ffi::AnyBuffer>()  // 18: s_down
        .Arg<ffi::AnyBuffer>()  // 19: q_scale
        .Arg<ffi::AnyBuffer>()  // 20: sink_k
        .Arg<ffi::AnyBuffer>()  // 21: rope_freqs
        .Arg<ffi::AnyBuffer>()  // 22: dw_kv
        .Arg<ffi::AnyBuffer>()  // 23: dw_post
        .Arg<ffi::AnyBuffer>()  // 24: final_read_q
        .Arg<ffi::AnyBuffer>()  // 25: final_norm_sb
        .Arg<ffi::AnyBuffer>()  // 26: conv1pgv
        .Arg<ffi::AnyBuffer>()  // 27: head_norm_sb
        .Arg<ffi::AnyBuffer>()  // 28: linear_g_w
        .Arg<ffi::AnyBuffer>()  // 29: pol_w
        .Arg<ffi::AnyBuffer>()  // 30: pol_b
        .Arg<ffi::AnyBuffer>()  // 31: q_w
        .Arg<ffi::AnyBuffer>()  // 32: q_b
        .Arg<ffi::AnyBuffer>()  // 33: pass_pol_w
        .Arg<ffi::AnyBuffer>()  // 34: pass_pol_b
        .Arg<ffi::AnyBuffer>()  // 35: pass_q_w
        .Arg<ffi::AnyBuffer>()  // 36: pass_q_b
        .Arg<ffi::AnyBuffer>()  // 37: val_l1_w
        .Arg<ffi::AnyBuffer>()  // 38: val_l1_b
        .Arg<ffi::AnyBuffer>()  // 39: val_out_w
        .Arg<ffi::AnyBuffer>()  // 40: val_out_b
        .Arg<ffi::AnyBuffer>()  // 41: cache_keys
        .Arg<ffi::AnyBuffer>()  // 42: cache_locks
        .Arg<ffi::AnyBuffer>()  // 43: cache_v (bf16)
        .Arg<ffi::AnyBuffer>()  // 44: cache_q (bf16)
        .Arg<ffi::AnyBuffer>()  // 45: cache_wave (u32[1])
        .Ret<ffi::AnyBuffer>()  // 0: vprobs
        .Ret<ffi::AnyBuffer>()  // 1: qprobs
        .Ret<ffi::AnyBuffer>()  // 2: r_cache_keys (alias 41)
        .Ret<ffi::AnyBuffer>()  // 3: r_cache_locks (alias 42)
        .Ret<ffi::AnyBuffer>()  // 4: r_cache_v (alias 43)
        .Ret<ffi::AnyBuffer>()  // 5: r_cache_q (alias 44)
        .Ret<ffi::AnyBuffer>()  // 6: sc_partial
        .Ret<ffi::AnyBuffer>()  // 7: sc_bank
        .Ret<ffi::AnyBuffer>()  // 8: sc_qkvg
        .Ret<ffi::AnyBuffer>()  // 9: sc_act
        .Ret<ffi::AnyBuffer>(), // 10: sc_scales
    {ffi::Traits::kCmdBufferCompatible});
