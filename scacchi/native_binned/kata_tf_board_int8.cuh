// Single-kernel SMEM-resident INT8 + FP8-FlashAttention megakernel for flat AZKataTransformerBinnedNet (Hex 3..11).
//
// Architecture: 3x3 D2-canonical stem -> L flat Pre-RMSNorm blocks (MHA QK-Norm + per-head 2D RoPE + SwiGLU)
//               -> final RMSNorm + SiLU -> KataGo spatial Policy/Q + Global-Pool Value heads (+ 512K PositionCache).
//
// Design highlights (incorporating every Nsight Compute lesson from Inkling):
//   - Hex 3..8  (cells <= 64):  1-half `kata_tf_board_int8_kernel_1h` (48.5 KB total SMEM, <= 128 regs -> 2 CTAs/SM!):
//       * Residual stream X[64, 192] bf16 is 100% SMEM-resident (XOR-swizzled: 0 bank conflicts, 0 extra bytes,
//         ZERO L2 intermediate reads/writes across the entire trunk + heads!).
//       * 2-head streaming MHA (`hp = 0..2`): Q/K RMSNorm in 2 warp shuffles, 2D RoPE in-register (0 shuffles!),
//         Q packed directly to FP8 `mma_e4m3` A-fragments via 4 warp shuffles (0 SMEM for Q, 0 `scratch_bf`!).
//       * Exact S=64 (2 FP8 tiles of 32, zero sink padding!) FlashAttention-3 zero-shuffle V-permutation.
//   - Hex 9..11 (64 < cells <= 121): 2-half `kata_tf_board_int8_kernel_2h` (47.6 KB total SMEM, <= 128 regs -> 2 CTAs/SM!).
//   - Coalesced transposed Head weights (`linear_g_w[96,48]`, `val_l1_w[96,128]`, `val_out_w[128,K]`, `pass_q_w[96,K]`),
//     `mma_bf16` Tensor Core `conv1pgv`, parallel `val_out`/`pass_q`/`pass_pol`, and 0 local memory spills!

#pragma once

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <cstdio>

#include "cache_key.cuh"
#include "inkling_mma.cuh"

namespace scacchi_kata_tf {

using scacchi_inkling::mma_s8s8;
using scacchi_inkling::tanh_approx_f32;
using scacchi_net::CacheKey;
using scacchi_net::CacheWrite;
using scacchi_net::cache_try_acquire;
using scacchi_net::cp_async16;
using scacchi_net::cp_async_commit;
using scacchi_net::cp_async_wait;
using scacchi_net::hash_cache_key;
using scacchi_net::ldmatrix_x4;
using scacchi_net::mma_bf16;
using scacchi_net::mma_e4m3;
using scacchi_net::smem_u32;
using scacchi_net::swz;

constexpr int K_C = 192;
constexpr int K_C2 = 96;
constexpr int K_H = 6;
constexpr int K_D = 32;
constexpr int K_HEAD_C = 48;
constexpr int K_VAL_H = 128;
constexpr float K_RMS_EPS = 1e-5f;
constexpr float K_INV_SQRT_D_LOG2E = 0.2550348688296215f;  // (1 / sqrt(32)) * log2(e)

struct KataTfHeadsPtrs {
  const float* final_norm_sb;   // [2, 192]
  const __nv_bfloat16* conv1pgv;// [192, 192] ([conv1p(48); conv1g(48); conv1v(48); 0(48)])
  const float* head_norm_sb;    // [6, 48] ([g_s, g_b, p_s, p_b, v_s, v_b])
  const float* linear_g_w;      // [96, 48] TRANSPOSED for coalesced tid<48 reads!
  const float* pol_w;           // [48]
  const float* pol_b;           // [1]
  const float* q_w;             // [K, 48]
  const float* q_b;             // [K]
  const float* pass_pol_w;      // [96]
  const float* pass_pol_b;      // [1]
  const float* pass_q_w;        // [96, K] TRANSPOSED for coalesced k<K reads!
  const float* pass_q_b;        // [K]
  const float* val_l1_w;        // [96, 128] TRANSPOSED for coalesced o<128 reads!
  const float* val_l1_b;        // [128]
  const float* val_out_w;       // [128, K] TRANSPOSED for coalesced k<K reads!
  const float* val_out_b;       // [K]
};

struct KataTfParams {
  const uint8_t* obs;
  const int32_t* d_boards;
  int max_boards;
  int side;
  int cells;
  int A;
  int K;
  int L;
  int FFN;
  const __nv_bfloat16* stem_w;  // [192, 27]
  const float* stem_b;          // [192]
  const float* attn_norm_sb;    // [L, 2, 192]
  const float* ffn_norm_sb;     // [L, 2, 192]
  const float* qk_norm_sb;      // [L, 4, 32] ([q_s, q_b, k_s, k_b])
  const float* rope_freqs;      // [L, 6, 16, 2]
  const int8_t* w_qkv;          // [L, 3, 192, 192] ([Q; K; V])
  const float* s_qkv;           // [L, 3, 192]
  const int8_t* w_out;          // [L, 192, 192]
  const float* s_out;           // [L, 192]
  const int8_t* w_gu;           // [L, 2*FFN, 192] paired 128-row [gate64 | up64] tiles
  const float* s_gu;            // [L, 2*FFN]
  const int8_t* w_down;         // [L, 192, FFN]
  const float* s_down;          // [L, 192]
  const float* qkv_fp8_scales;  // [L, 14]: [q_sc, k_sc, v_sc[6], inv_v_sc[6]]
  KataTfHeadsPtrs heads;
  float* logits;                // [B, A] or nullptr in compact mode
  float* vprobs;                // [B, K]
  float* qprobs;                // [B, A, K]
  CacheWrite cw;
  // L2 scratch (only used by 2-half Hex 9..11; unused on Hex 3..8!)
  __nv_bfloat16* cta_x;         // [gridDim.x, cells, 192]
  __nv_bfloat16* cta_tmp;       // [gridDim.x, cells, 192]
  int8_t* cta_act_s8;           // [gridDim.x, cells, 192]
  float* cta_act_sc;            // [gridDim.x, cells]
};

__device__ __forceinline__ float kt_warp_sum(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
  return v;
}

__device__ __forceinline__ float kt_warp_max(float v) {
#pragma unroll
  for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
  return v;
}

__device__ __forceinline__ int8_t kt_quant_s8(float v, float inv_scale) {
  int q = __float2int_rn(v * inv_scale);
  q = max(-127, min(127, q));
  return static_cast<int8_t>(q);
}

__device__ __forceinline__ uint16_t kt_pack_s8x2(float v0, float v1, float inv_scale) {
  return static_cast<uint16_t>(static_cast<uint8_t>(kt_quant_s8(v0, inv_scale))) |
         (static_cast<uint16_t>(static_cast<uint8_t>(kt_quant_s8(v1, inv_scale))) << 8);
}

__device__ __forceinline__ uint16_t kt_pack_e4m3x2(float lo, float hi) {
  uint16_t out;
  asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(out) : "f"(hi), "f"(lo));
  return out;
}

__device__ __forceinline__ uint8_t kt_cvt_e4m3(float v) {
  return static_cast<uint8_t>(kt_pack_e4m3x2(v, 0.f) & 0xffu);
}

__device__ __forceinline__ uint32_t kt_pack_e4m3x4(float a, float b, float c, float d) {
  return (uint32_t)kt_pack_e4m3x2(a, b) | ((uint32_t)kt_pack_e4m3x2(c, d) << 16);
}

__device__ __forceinline__ float kt_silu(float x) {
  return x * fmaf(0.5f, tanh_approx_f32(0.5f * x), 0.5f);
}

__device__ __forceinline__ int kt_swz32(int row, int chunk) {
  return row * 32 + ((chunk ^ ((row >> 2) & 1)) << 4);
}

__device__ __forceinline__ int kt_swz_e4m3_32(int row, int d) {
  return row * 32 + (((d >> 4) ^ ((row >> 2) & 1)) << 4) + (d & 15);
}

__device__ __forceinline__ int kt_v_key_perm(int k) {
  return (k & ~15) | ((k & 8) >> 2) | ((k & 6) << 1) | (k & 1);
}

// XOR-swizzled index into X_smem[64, 3, 32] of __nv_bfloat162 (0 bank conflicts for BOTH row-wise & MMA access!).
__device__ __forceinline__ int kt_x_idx(int row, int panel, int p32) {
  return (row * 3 + panel) * 32 + (p32 ^ ((row & 7) << 2));
}

__device__ __forceinline__ void kt_store_act_s8_row(
    char* act_s8, int ROWS, int r, int lane, const float2 (&v)[3], float inv_sc) {
  const int chunk = lane >> 3;
  const int in_chunk = (lane & 7) * 2;
  const int sw_chunk = chunk ^ ((r >> 1) & 3);
#pragma unroll
  for (int j = 0; j < 3; ++j) {
    *reinterpret_cast<uint16_t*>(act_s8 + (j * ROWS + r) * 64 + (sw_chunk << 4) + in_chunk) =
        kt_pack_s8x2(v[j].x, v[j].y, inv_sc);
  }
}

// Pack 4 MMA output columns (2 uint16_t pairs across 4 row-lanes) into mma_e4m3 A-fragment uint32_t!
// ZERO SMEM for Q!
__device__ __forceinline__ uint32_t kt_pack_qa_half(int lane, uint16_t h0, uint16_t h1) {
  const int base = lane & ~3;
  const int g = lane & 3;
  const int s0 = base | ((g & 1) << 1);
  const int s1 = s0 | 1;
  const uint32_t a0 = __shfl_sync(0xffffffffu, (uint32_t)h0, s0);
  const uint32_t a1 = __shfl_sync(0xffffffffu, (uint32_t)h0, s1);
  const uint32_t b0 = __shfl_sync(0xffffffffu, (uint32_t)h1, s0);
  const uint32_t b1 = __shfl_sync(0xffffffffu, (uint32_t)h1, s1);
  return (g < 2) ? (a0 | (a1 << 16)) : (b0 | (b1 << 16));
}

// ============================================================================
// Kernel A: Hex 3..8 (cells <= 64, 100% SMEM-resident X, 48.5 KB SMEM -> 2 CTAs/SM!)
// ZERO L2 intermediate reads/writes across the entire network!
// ============================================================================
__host__ __device__ constexpr int kata_tf_smem_bytes_1h() {
  // X_smem(24,576) + act_s8(12,288) + act_sc(256) + overlay(12,288) = 49,408 B (48.25 KB)!
  // Plus ~256 B static = 49,664 B <= 50,176 B (49 KB) -> 2 CTAs/SM on SM 12.0!
  return 64 * K_C * 2 + 64 * K_C + 64 * 4 + 12288;
}

__global__ void __launch_bounds__(256, 2) kata_tf_board_int8_kernel_1h(
    const __grid_constant__ KataTfParams P) {
  const int active_boards = P.d_boards ? *P.d_boards : P.max_boards;
  if (active_boards <= 0) return;

  constexpr int ROWS = 64;
  constexpr int SKV_FP8 = 64;
  constexpr int V_TILES64 = 1;

  extern __shared__ __align__(128) char smem_raw[];
  __shared__ uint8_t cell_y[64];
  __shared__ uint8_t cell_x[64];
  __shared__ bool won_slot;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wm = warp >> 1;
  const int wn = warp & 1;

  const int side = P.side;
  const int cells = P.cells;
  const int A = P.A;
  const int K = P.K;
  const int L = P.L;
  const int FFN = P.FFN;

  if (tid < cells) {
    const int y = tid / side;
    cell_y[tid] = (uint8_t)y;
    cell_x[tid] = (uint8_t)(tid - y * side);
  }
  __syncthreads();

  // ---- SMEM Partition (49,408 B total) ----
  // [0 .. 24,575]:      X_smem[64, 3, 32] bf162 (XOR-swizzled residual stream — 100% SMEM-resident!)
  // [24,576 .. 36,863]: act_s8[3, 64, 64] int8 (swizzled)
  // [36,864 .. 37,119]: act_sc[64] f32
  // [37,120 .. 49,407]: 12,288 B overlay:
  //   - K GEMM:         w_stage_a (37,120..45,311, 8 KB 2-buf), k_smem (45,312..49,407, 4 KB)
  //   - V/Q GEMM+Attn:  w_stage_1 (37,120..41,215, 4 KB 1-buf), v_smem (41,216..45,311, 4 KB), k_smem (45,312..49,407, 4 KB)
  //   - O/MLP quant:    sub_s8 (37,120..41,215, 4 KB), sub_sc (41,216..41,471), half_max (41,472..41,983)
  //   - O/MLP accum:    sub_s8 (37,120..41,215, 4 KB), w_stage_b (41,216..49,407, 8 KB 2-buf!)
  //   - Stem ws:        at 24,576..45,743 (27 * 196 * 4 = 21,168 B)
  //   - Stage 3 Heads:  h_final_swz at 24,576..49,151 (24 KB), then [0..24,575] holds h_outp(6 KB)+h_stats(3.8 KB)+h_wstage(8 KB)!
  __nv_bfloat162* X_smem2 = reinterpret_cast<__nv_bfloat162*>(smem_raw);
  char* act_s8 = smem_raw + ROWS * K_C * 2;
  float* act_sc = reinterpret_cast<float*>(act_s8 + ROWS * K_C);
  char* ov_base = smem_raw + ROWS * K_C * 2 + ROWS * K_C + ROWS * 4;  // offset 37,120

  char* w_stage_a = ov_base;                   // 8 KB at 37,120 (K GEMM 2-buf, V/Q GEMM 1-buf)
  uint8_t* v_smem = reinterpret_cast<uint8_t*>(ov_base + 4096);  // 4 KB at 41,216
  uint8_t* k_smem = reinterpret_cast<uint8_t*>(ov_base + 8192);  // 4 KB at 45,312

  char* sub_s8 = ov_base;                      // 4 KB at 37,120
  float* sub_sc = reinterpret_cast<float*>(ov_base + 4096);      // 256 B at 41,216
  float* half_max = sub_sc + ROWS;             // 512 B at 41,472
  char* w_stage_b = ov_base + 4096;            // 8 KB at 41,216 (O-proj & MLP 2-buf!)

  // Stage 3 Heads overlay:
  __nv_bfloat16* h_outp = reinterpret_cast<__nv_bfloat16*>(smem_raw);                 // 6,144 B
  float* h_stats = reinterpret_cast<float*>(h_outp + ROWS * K_HEAD_C);                // 3,840 B
  char* h_wstage = smem_raw + 10240;                                                  // 8,192 B (128B aligned)
  __nv_bfloat16* h_gv_tmp = reinterpret_cast<__nv_bfloat16*>(h_wstage);               // 6,144 B
  char* h_final_swz = smem_raw + ROWS * K_C * 2;                                      // 24,576 B

  const uint32_t act_u32 = smem_u32(act_s8);
  const uint32_t wsta_u32 = smem_u32(w_stage_a);
  const uint32_t wstb_u32 = smem_u32(w_stage_b);
  const uint32_t sub_u32 = smem_u32(sub_s8);
  const uint32_t ksm_u32 = smem_u32(k_smem);
  const uint32_t vsm_u32 = smem_u32(v_smem);
  const uint32_t hfin_u32 = smem_u32(h_final_swz);
  const uint32_t hwst_u32 = smem_u32(h_wstage);

  const int S_pad32 = (cells + 31) & ~31;
  const float half_side = 0.5f * (float)(side - 1);
  const int r0_in_tile = lane >> 2;
  const int c_in_n8 = (lane & 3) * 2;
  const int my_r0 = wm * 16 + r0_in_tile;
  const int my_r1 = my_r0 + 8;

  // ---- GEMM helpers ----
  // Double-buffered 192->64 GEMM using 8 KB w_stage at `w_u32` (used for K).
  auto gemm_192_to_64_2buf = [&](const int8_t* W_mat, uint32_t w_u32, int (&acc)[4][4]) {
    const int lrow = tid >> 2, lchunk = tid & 3;
    auto load_w = [&](int st, int kk) {
      cp_async16(w_u32 + (st & 1) * 4096 + swz(lrow, lchunk),
                 W_mat + (size_t)lrow * K_C + kk * 64 + lchunk * 16, true);
    };
#pragma unroll
    for (int nj = 0; nj < 4; ++nj)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[nj][e] = 0;
    load_w(0, 0);
    cp_async_commit();
#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
      cp_async_wait<0>();
      __syncthreads();
      if (kk + 1 < 3) { load_w(kk + 1, kk + 1); cp_async_commit(); }
      const uint32_t b_tile = w_u32 + (kk & 1) * 4096;
      const uint32_t a_panel = act_u32 + (uint32_t)(kk * ROWS * 64);
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        uint32_t a[4];
        ldmatrix_x4(a, a_panel + swz(wm * 16 + (lane & 15), ks * 2 + (lane >> 4)));
        uint32_t b[2][4];
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          const int brow = wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3);
          ldmatrix_x4(b[np], b_tile + swz(brow, ks * 2 + ((lane >> 3) & 1)));
        }
#pragma unroll
        for (int nj = 0; nj < 4; ++nj) mma_s8s8(acc[nj], a, b[nj >> 1][(nj & 1) * 2], b[nj >> 1][(nj & 1) * 2 + 1]);
      }
    }
    __syncthreads();
  };

  // Single-buffered 192->64 GEMM using 4 KB w_stage at `w_u32` (used for V and Q while k_smem/v_smem live).
  auto gemm_192_to_64_1buf = [&](const int8_t* W_mat, uint32_t w_u32, int (&acc)[4][4]) {
    const int lrow = tid >> 2, lchunk = tid & 3;
#pragma unroll
    for (int nj = 0; nj < 4; ++nj)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[nj][e] = 0;
#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
      cp_async16(w_u32 + swz(lrow, lchunk),
                 W_mat + (size_t)lrow * K_C + kk * 64 + lchunk * 16, true);
      cp_async_commit();
      cp_async_wait<0>();
      __syncthreads();
      const uint32_t a_panel = act_u32 + (uint32_t)(kk * ROWS * 64);
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        uint32_t a[4];
        ldmatrix_x4(a, a_panel + swz(wm * 16 + (lane & 15), ks * 2 + (lane >> 4)));
        uint32_t b[2][4];
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          const int brow = wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3);
          ldmatrix_x4(b[np], w_u32 + swz(brow, ks * 2 + ((lane >> 3) & 1)));
        }
#pragma unroll
        for (int nj = 0; nj < 4; ++nj) mma_s8s8(acc[nj], a, b[nj >> 1][(nj & 1) * 2], b[nj >> 1][(nj & 1) * 2 + 1]);
      }
      __syncthreads();
    }
  };

  // Single-act-load fused [W_gate(64) | W_up(64)] GEMM using 8 KB w_stage_b.
  auto gemm_192_to_64x2 = [&](const int8_t* W_pair, int (&acc0)[4][4], int (&acc1)[4][4]) {
    const int lrow = tid >> 2, lchunk = tid & 3;
#pragma unroll
    for (int nj = 0; nj < 4; ++nj)
#pragma unroll
      for (int e = 0; e < 4; ++e) { acc0[nj][e] = 0; acc1[nj][e] = 0; }
#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
      cp_async16(wstb_u32 + swz(lrow, lchunk), W_pair + (size_t)lrow * K_C + kk * 64 + lchunk * 16, true);
      cp_async16(wstb_u32 + 4096 + swz(lrow, lchunk), W_pair + (size_t)(64 + lrow) * K_C + kk * 64 + lchunk * 16, true);
      cp_async_commit();
      cp_async_wait<0>();
      __syncthreads();
      const uint32_t a_panel = act_u32 + (uint32_t)(kk * ROWS * 64);
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        uint32_t a[4];
        ldmatrix_x4(a, a_panel + swz(wm * 16 + (lane & 15), ks * 2 + (lane >> 4)));
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t b[4];
          const int brow = wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3);
          ldmatrix_x4(b, wstb_u32 + swz(brow, ks * 2 + ((lane >> 3) & 1)));
          mma_s8s8(acc0[np * 2 + 0], a, b[0], b[1]);
          mma_s8s8(acc0[np * 2 + 1], a, b[2], b[3]);
        }
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t b[4];
          const int brow = wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3);
          ldmatrix_x4(b, wstb_u32 + 4096 + swz(brow, ks * 2 + ((lane >> 3) & 1)));
          mma_s8s8(acc1[np * 2 + 0], a, b[0], b[1]);
          mma_s8s8(acc1[np * 2 + 1], a, b[2], b[3]);
        }
      }
      __syncthreads();
    }
  };

  // Quantize 64-col register outputs -> sub_s8 (37,120), returning row scales (xs0, xs1) so w_stage_b (41,216) is free!
  auto quant_regs_to_sub_s8 = [&](const __nv_bfloat162 (&val_bf)[8], float& out_xs0, float& out_xs1) {
    float am0 = 0.f, am1 = 0.f;
#pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
      const float2 f0 = __bfloat1622float2(val_bf[nj]);
      const float2 f1 = __bfloat1622float2(val_bf[4 + nj]);
      am0 = fmaxf(am0, fmaxf(fabsf(f0.x), fabsf(f0.y)));
      am1 = fmaxf(am1, fmaxf(fabsf(f1.x), fabsf(f1.y)));
    }
    am0 = fmaxf(am0, __shfl_xor_sync(0xffffffffu, am0, 1));
    am0 = fmaxf(am0, __shfl_xor_sync(0xffffffffu, am0, 2));
    am1 = fmaxf(am1, __shfl_xor_sync(0xffffffffu, am1, 1));
    am1 = fmaxf(am1, __shfl_xor_sync(0xffffffffu, am1, 2));
    if ((lane & 3) == 0) {
      if (my_r0 < cells) half_max[my_r0 * 2 + wn] = am0;
      if (my_r1 < cells) half_max[my_r1 * 2 + wn] = am1;
    }
    __syncthreads();

    const float gm0 = (my_r0 < cells) ? fmaxf(fmaxf(half_max[my_r0 * 2], half_max[my_r0 * 2 + 1]), 1e-6f) : 1.0f;
    const float gm1 = (my_r1 < cells) ? fmaxf(fmaxf(half_max[my_r1 * 2], half_max[my_r1 * 2 + 1]), 1e-6f) : 1.0f;
    const float is0 = (my_r0 < cells) ? (127.0f / gm0) : 0.f;
    const float is1 = (my_r1 < cells) ? (127.0f / gm1) : 0.f;
    out_xs0 = (my_r0 < cells) ? (gm0 * (1.0f / 127.0f)) : 0.f;
    out_xs1 = (my_r1 < cells) ? (gm1 * (1.0f / 127.0f)) : 0.f;

    const int sw0_mask = (my_r0 >> 1) & 3;
    const int sw1_mask = (my_r1 >> 1) & 3;
#pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
      const int col = wn * 32 + nj * 8 + c_in_n8;
      const int ch = col >> 4, in_ch = col & 15;
      const float2 f0 = __bfloat1622float2(val_bf[nj]);
      const float2 f1 = __bfloat1622float2(val_bf[4 + nj]);
      const uint32_t q0_lo = kt_pack_s8x2(f0.x, f0.y, is0);
      const uint32_t q0_hi = __shfl_xor_sync(0xffffffffu, q0_lo, 1);
      const uint32_t q1_lo = kt_pack_s8x2(f1.x, f1.y, is1);
      const uint32_t q1_hi = __shfl_xor_sync(0xffffffffu, q1_lo, 1);
      if ((lane & 1) == 0) {
        *reinterpret_cast<uint32_t*>(sub_s8 + my_r0 * 64 + ((ch ^ sw0_mask) << 4) + in_ch) = q0_lo | (q0_hi << 16);
        *reinterpret_cast<uint32_t*>(sub_s8 + my_r1 * 64 + ((ch ^ sw1_mask) << 4) + in_ch) = q1_lo | (q1_hi << 16);
      }
    }
    __syncthreads();  // half_max is now dead -> w_stage_b (41,216) is 100% free!
  };

  auto accum_64_to_192 = [&](const int8_t* W_full, int K_full, int k_off, float xs0, float xs1, float (&out_acc)[12][4]) {
    const int lrow = tid >> 2, lchunk = tid & 3;
    auto load_w = [&](int st, int rc) {
      cp_async16(wstb_u32 + (st & 1) * 4096 + swz(lrow, lchunk),
                 W_full + (size_t)(rc * 64 + lrow) * K_full + k_off + lchunk * 16, true);
    };
    load_w(0, 0);
    cp_async_commit();
#pragma unroll
    for (int rc = 0; rc < 3; ++rc) {
      cp_async_wait<0>();
      __syncthreads();
      if (rc + 1 < 3) { load_w(rc + 1, rc + 1); cp_async_commit(); }
      const uint32_t b_tile = wstb_u32 + (rc & 1) * 4096;
      int kacc[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        uint32_t a[4];
        ldmatrix_x4(a, sub_u32 + swz(wm * 16 + (lane & 15), ks * 2 + (lane >> 4)));
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t b[4];
          const int brow = wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3);
          ldmatrix_x4(b, b_tile + swz(brow, ks * 2 + ((lane >> 3) & 1)));
          mma_s8s8(kacc[np * 2 + 0], a, b[0], b[1]);
          mma_s8s8(kacc[np * 2 + 1], a, b[2], b[3]);
        }
      }
#pragma unroll
      for (int nj = 0; nj < 4; ++nj) {
        const int d = rc * 4 + nj;
        out_acc[d][0] = fmaf((float)kacc[nj][0], xs0, out_acc[d][0]);
        out_acc[d][1] = fmaf((float)kacc[nj][1], xs0, out_acc[d][1]);
        out_acc[d][2] = fmaf((float)kacc[nj][2], xs1, out_acc[d][2]);
        out_acc[d][3] = fmaf((float)kacc[nj][3], xs1, out_acc[d][3]);
      }
    }
    __syncthreads();
  };

  // Add MMA-layout out_acc * scale into SMEM-resident X_smem2 IN PLACE (0 bank conflicts!).
  auto add_to_x_smem = [&](const float (&acc)[12][4], const float* s_out) {
#pragma unroll
    for (int rc = 0; rc < 3; ++rc) {
#pragma unroll
      for (int nj = 0; nj < 4; ++nj) {
        const int idx = rc * 4 + nj;
        const int col = rc * 64 + wn * 32 + nj * 8 + c_in_n8;
        const int panel = col >> 6;
        const int p32 = (col >> 1) & 31;
        const float ws0 = s_out[col], ws1 = s_out[col + 1];
        if (my_r0 < cells) {
          const int xi = kt_x_idx(my_r0, panel, p32);
          const float2 cur = __bfloat1622float2(X_smem2[xi]);
          X_smem2[xi] = __floats2bfloat162_rn(fmaf(acc[idx][0], ws0, cur.x), fmaf(acc[idx][1], ws1, cur.y));
        }
        if (my_r1 < cells) {
          const int xi = kt_x_idx(my_r1, panel, p32);
          const float2 cur = __bfloat1622float2(X_smem2[xi]);
          X_smem2[xi] = __floats2bfloat162_rn(fmaf(acc[idx][2], ws0, cur.x), fmaf(acc[idx][3], ws1, cur.y));
        }
      }
    }
    __syncthreads();
  };

  // Fused RMSNorm(X_smem) -> INT8 act_s8 + act_sc (ZERO L2 reads, ZERO bank conflicts!).
  auto norm_x_to_act_s8 = [&](const float* nsb_ptr) {
    const float2* nsb2 = reinterpret_cast<const float2*>(nsb_ptr);
    float2 ns[3], nbi[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const int p = j * 32 + lane;
      ns[j] = nsb2[p];
      nbi[j] = nsb2[K_C2 + p];
    }
    for (int cell = warp; cell < cells; cell += 8) {
      float2 v[3];
      float sq = 0.f;
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        v[j] = __bfloat1622float2(X_smem2[kt_x_idx(cell, j, lane)]);
        sq = fmaf(v[j].x, v[j].x, fmaf(v[j].y, v[j].y, sq));
      }
      const float rstd = rsqrtf(kt_warp_sum(sq) * (1.0f / K_C) + K_RMS_EPS);
      float amax = 0.f;
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        v[j].x = fmaf(v[j].x * rstd, ns[j].x, nbi[j].x);
        v[j].y = fmaf(v[j].y * rstd, ns[j].y, nbi[j].y);
        amax = fmaxf(amax, fmaxf(fabsf(v[j].x), fabsf(v[j].y)));
      }
      const float gmax = fmaxf(kt_warp_max(amax), 1e-6f);
      kt_store_act_s8_row(act_s8, ROWS, cell, lane, v, 127.0f / gmax);
      if (lane == 0) act_sc[cell] = gmax * (1.0f / 127.0f);
    }
    for (int cell = cells + warp; cell < ROWS; cell += 8) {
      float2 zv[3] = {make_float2(0.f, 0.f), make_float2(0.f, 0.f), make_float2(0.f, 0.f)};
      kt_store_act_s8_row(act_s8, ROWS, cell, lane, zv, 0.f);
      if (lane == 0) act_sc[cell] = 0.f;
    }
    __syncthreads();
  };

  auto store_v_fp8 = [&](int h_local, int d, int k_idx, uint8_t val) {
    const int k_perm = kt_v_key_perm(k_idx);
    const int sw_ch = (k_perm >> 4) ^ ((d >> 1) & 3);
    v_smem[(h_local * K_D + d) * 64 + (sw_ch << 4) + (k_perm & 15)] = val;
  };

  // Helper: in-register Q/K RMSNorm (2 shuffles!) + 2D RoPE (0 shuffles!) for a head pair `hp`.
  // Takes MMA int32 `acc[4][4]`, multiplies by `act_sc * w_sc`, applies RMSNorm(32) and per-head 2D RoPE!
  auto apply_qknorm_rope = [&](const int (&acc)[4][4], float xs0, float xs1, const float* ws_ptr,
                               const float* qkn_s, const float* qkn_b, const float* rf_h,
                               float2 (&v0)[4], float2 (&v1)[4]) {
    float sq0 = 0.f, sq1 = 0.f;
#pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
      const int col = wn * 32 + nj * 8 + c_in_n8;
      const float w0 = ws_ptr[col], w1 = ws_ptr[col + 1];
      v0[nj] = make_float2((float)acc[nj][0] * xs0 * w0, (float)acc[nj][1] * xs0 * w1);
      v1[nj] = make_float2((float)acc[nj][2] * xs1 * w0, (float)acc[nj][3] * xs1 * w1);
      sq0 = fmaf(v0[nj].x, v0[nj].x, fmaf(v0[nj].y, v0[nj].y, sq0));
      sq1 = fmaf(v1[nj].x, v1[nj].x, fmaf(v1[nj].y, v1[nj].y, sq1));
    }
    sq0 += __shfl_xor_sync(0xffffffffu, sq0, 1);
    sq0 += __shfl_xor_sync(0xffffffffu, sq0, 2);
    sq1 += __shfl_xor_sync(0xffffffffu, sq1, 1);
    sq1 += __shfl_xor_sync(0xffffffffu, sq1, 2);
    const float rstd0 = rsqrtf(sq0 * (1.0f / K_D) + K_RMS_EPS);
    const float rstd1 = rsqrtf(sq1 * (1.0f / K_D) + K_RMS_EPS);
#pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
      const int d = nj * 8 + c_in_n8;
      const float ns0 = qkn_s[d], ns1 = qkn_s[d + 1];
      const float nb0 = qkn_b[d], nb1 = qkn_b[d + 1];
      v0[nj].x = fmaf(v0[nj].x * rstd0, ns0, nb0);
      v0[nj].y = fmaf(v0[nj].y * rstd0, ns1, nb1);
      v1[nj].x = fmaf(v1[nj].x * rstd1, ns0, nb0);
      v1[nj].y = fmaf(v1[nj].y * rstd1, ns1, nb1);
    }
    // 2D RoPE in-register: nj=0 pairs with nj=2 (d0, d1 in 0..7); nj=1 pairs with nj=3 (d2, d3 in 8..15)!
    const float sx0 = (my_r0 < cells) ? ((float)cell_x[my_r0] - half_side) : 0.f;
    const float sy0 = (my_r0 < cells) ? ((float)cell_y[my_r0] - half_side) : 0.f;
    const float sx1 = (my_r1 < cells) ? ((float)cell_x[my_r1] - half_side) : 0.f;
    const float sy1 = (my_r1 < cells) ? ((float)cell_y[my_r1] - half_side) : 0.f;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int d_lo = p * 8 + c_in_n8;
      const float fx0 = rf_h[d_lo * 2 + 0], fy0 = rf_h[d_lo * 2 + 1];
      const float fx1 = rf_h[(d_lo + 1) * 2 + 0], fy1 = rf_h[(d_lo + 1) * 2 + 1];
      float s, c, a, b;
      // Row r0, even elem
      __sincosf(fmaf(sx0, fx0, sy0 * fy0), &s, &c);
      a = v0[p].x; b = v0[p + 2].x;
      v0[p].x = fmaf(a, c, -b * s); v0[p + 2].x = fmaf(a, s, b * c);
      // Row r0, odd elem
      __sincosf(fmaf(sx0, fx1, sy0 * fy1), &s, &c);
      a = v0[p].y; b = v0[p + 2].y;
      v0[p].y = fmaf(a, c, -b * s); v0[p + 2].y = fmaf(a, s, b * c);
      // Row r1, even elem
      __sincosf(fmaf(sx1, fx0, sy1 * fy0), &s, &c);
      a = v1[p].x; b = v1[p + 2].x;
      v1[p].x = fmaf(a, c, -b * s); v1[p + 2].x = fmaf(a, s, b * c);
      // Row r1, odd elem
      __sincosf(fmaf(sx1, fx1, sy1 * fy1), &s, &c);
      a = v1[p].y; b = v1[p + 2].y;
      v1[p].y = fmaf(a, c, -b * s); v1[p + 2].y = fmaf(a, s, b * c);
    }
  };

  for (int board = blockIdx.x; board < active_boards; board += gridDim.x) {
    const uint8_t* obs_b = P.obs + (size_t)board * cells * 4;
    const bool is_white = obs_b[2] != 0;

    // ---- 1. Stem 3x3 -> X_smem (100% SMEM-resident) + Layer-0 attn_norm -> act_s8 ----
    {
      constexpr int WS_STRIDE = K_C + 4;
      float* ws = reinterpret_cast<float*>(smem_raw + ROWS * K_C * 2);  // 21,168 B at 24,576
      for (int idx = tid; idx < 27 * K_C; idx += 256) {
        const int n = idx / 27, k = idx - n * 27;
        ws[k * WS_STRIDE + n] = __bfloat162float(P.stem_w[idx]);
      }
      __syncthreads();

      const float2* bias2 = reinterpret_cast<const float2*>(P.stem_b);
      float2 b_reg[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) b_reg[j] = bias2[j * 32 + lane];

      for (int cell = warp; cell < cells; cell += 8) {
        const int y = (int)cell_y[cell], x = (int)cell_x[cell];
        float2 acc[3] = {b_reg[0], b_reg[1], b_reg[2]};
#pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
          const int yy = y + dy;
          if (yy < 0 || yy >= side) continue;
#pragma unroll
          for (int dx = -1; dx <= 1; ++dx) {
            const int xx = x + dx;
            if (xx < 0 || xx >= side) continue;
            const int t = (dy + 1) * 3 + (dx + 1);
            const int sy = is_white ? xx : yy;
            const int sx = is_white ? yy : xx;
            const uint32_t o = *reinterpret_cast<const uint32_t*>(obs_b + (sy * side + sx) * 4);
            const bool ch0 = (o & 0xffu) != 0u;
            const bool ch1 = ((o >> 8) & 0xffu) != 0u;
            const float2* w2 = reinterpret_cast<const float2*>(ws + (t * 3 + 2) * WS_STRIDE);
#pragma unroll
            for (int j = 0; j < 3; ++j) {
              const float2 wv = w2[j * 32 + lane];
              acc[j].x += wv.x; acc[j].y += wv.y;
            }
            if (ch0) {
              const float2* w0 = reinterpret_cast<const float2*>(ws + (t * 3 + 0) * WS_STRIDE);
#pragma unroll
              for (int j = 0; j < 3; ++j) {
                const float2 wv = w0[j * 32 + lane];
                acc[j].x += wv.x; acc[j].y += wv.y;
              }
            }
            if (ch1) {
              const float2* w1 = reinterpret_cast<const float2*>(ws + (t * 3 + 1) * WS_STRIDE);
#pragma unroll
              for (int j = 0; j < 3; ++j) {
                const float2 wv = w1[j * 32 + lane];
                acc[j].x += wv.x; acc[j].y += wv.y;
              }
            }
          }
        }
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          X_smem2[kt_x_idx(cell, j, lane)] = __floats2bfloat162_rn(acc[j].x, acc[j].y);
        }
      }
      __syncthreads();
    }
    norm_x_to_act_s8(P.attn_norm_sb);

    // ---- 2. L Flat Transformer Blocks (100% SMEM-resident!) ----
    for (int i = 0; i < L; ++i) {
      const int8_t* wq_i = P.w_qkv + (size_t)i * 3 * K_C * K_C;
      const int8_t* wk_i = wq_i + (size_t)K_C * K_C;
      const int8_t* wv_i = wk_i + (size_t)K_C * K_C;
      const float* sq_i = P.s_qkv + (size_t)i * 3 * K_C;
      const float* sk_i = sq_i + K_C;
      const float* sv_i = sk_i + K_C;
      const int8_t* wo_i = P.w_out + (size_t)i * K_C * K_C;
      const float* so_i = P.s_out + (size_t)i * K_C;
      const float* qkn_i = P.qk_norm_sb + (size_t)i * 4 * K_D;
      const float* rf_i = P.rope_freqs + (size_t)i * K_H * 16 * 2;
      const float* fp8_sc = P.qkv_fp8_scales + (size_t)i * 14;
      const float q_fp8_sc = fp8_sc[0];
      const float k_fp8_sc = fp8_sc[1];
      const float s_unscale = K_INV_SQRT_D_LOG2E / (q_fp8_sc * k_fp8_sc);

      float out_acc[12][4];
#pragma unroll
      for (int nj = 0; nj < 12; ++nj)
#pragma unroll
        for (int e = 0; e < 4; ++e) out_acc[nj][e] = 0.f;

      const float xs_r0 = (my_r0 < cells) ? act_sc[my_r0] : 0.f;
      const float xs_r1 = (my_r1 < cells) ? act_sc[my_r1] : 0.f;

      // 2-head streaming MHA (3 head pairs x 2 heads = 6 heads)
      for (int hp = 0; hp < 3; ++hp) {
        const int h = hp * 2 + wn;
        const float* rf_h = rf_i + h * 16 * 2;
        const float v_fp8_sc = fp8_sc[2 + h];
        const float inv_v_fp8_sc = fp8_sc[8 + h];

        // 2a. K (2 heads) -> k_norm + RoPE in registers -> k_smem (4 KB, 0 bank conflicts!)
        {
          int kacc[4][4];
          gemm_192_to_64_2buf(wk_i + (size_t)(hp * 64) * K_C, wsta_u32, kacc);
          float2 k0[4], k1[4];
          apply_qknorm_rope(kacc, xs_r0, xs_r1, sk_i + hp * 64, qkn_i + 2 * K_D, qkn_i + 3 * K_D, rf_h, k0, k1);
          uint8_t* kh_sm = k_smem + wn * SKV_FP8 * K_D;
#pragma unroll
          for (int nj = 0; nj < 4; ++nj) {
            const int d = nj * 8 + c_in_n8;
            const uint32_t p0_lo = (my_r0 < cells) ? kt_pack_e4m3x2(k0[nj].x * k_fp8_sc, k0[nj].y * k_fp8_sc) : 0u;
            const uint32_t p0_hi = __shfl_xor_sync(0xffffffffu, p0_lo, 1);
            const uint32_t p1_lo = (my_r1 < cells) ? kt_pack_e4m3x2(k1[nj].x * k_fp8_sc, k1[nj].y * k_fp8_sc) : 0u;
            const uint32_t p1_hi = __shfl_xor_sync(0xffffffffu, p1_lo, 1);
            if ((lane & 1) == 0) {
              *reinterpret_cast<uint32_t*>(kh_sm + kt_swz_e4m3_32(my_r0, d)) = p0_lo | (p0_hi << 16);
              *reinterpret_cast<uint32_t*>(kh_sm + kt_swz_e4m3_32(my_r1, d)) = p1_lo | (p1_hi << 16);
            }
          }
        }

        // 2b. V (2 heads) -> v_smem (4 KB, FlashAttention-3 zero-shuffle permutation!)
        {
          int vacc[4][4];
          gemm_192_to_64_1buf(wv_i + (size_t)(hp * 64) * K_C, wsta_u32, vacc);
          const float sv_r0 = xs_r0 * v_fp8_sc;
          const float sv_r1 = xs_r1 * v_fp8_sc;
#pragma unroll
          for (int nj = 0; nj < 4; ++nj) {
            const int col = wn * 32 + nj * 8 + c_in_n8;
            const int d = nj * 8 + c_in_n8;
            const float w0 = sv_i[hp * 64 + col], w1 = sv_i[hp * 64 + col + 1];
            store_v_fp8(wn, d,     my_r0, (my_r0 < cells) ? kt_cvt_e4m3((float)vacc[nj][0] * sv_r0 * w0) : 0u);
            store_v_fp8(wn, d + 1, my_r0, (my_r0 < cells) ? kt_cvt_e4m3((float)vacc[nj][1] * sv_r0 * w1) : 0u);
            store_v_fp8(wn, d,     my_r1, (my_r1 < cells) ? kt_cvt_e4m3((float)vacc[nj][2] * sv_r1 * w0) : 0u);
            store_v_fp8(wn, d + 1, my_r1, (my_r1 < cells) ? kt_cvt_e4m3((float)vacc[nj][3] * sv_r1 * w1) : 0u);
          }
        }

        // 2c. Q (2 heads) -> q_norm + RoPE -> direct in-register qa[4] (0 SMEM!) -> FP8 FlashAttention!
        __nv_bfloat162 attn_reg[8];
        {
          int qacc[4][4];
          gemm_192_to_64_1buf(wq_i + (size_t)(hp * 64) * K_C, wsta_u32, qacc);
          // Note: gemm_192_to_64_1buf ends with __syncthreads(), so k_smem & v_smem are 100% visible!
          float2 q0[4], q1[4];
          apply_qknorm_rope(qacc, xs_r0, xs_r1, sq_i + hp * 64, qkn_i, qkn_i + K_D, rf_h, q0, q1);
          uint16_t p0[4], p1[4];
#pragma unroll
          for (int nj = 0; nj < 4; ++nj) {
            p0[nj] = kt_pack_e4m3x2(q0[nj].x * q_fp8_sc, q0[nj].y * q_fp8_sc);
            p1[nj] = kt_pack_e4m3x2(q1[nj].x * q_fp8_sc, q1[nj].y * q_fp8_sc);
          }
          uint32_t qa[4];
          qa[0] = kt_pack_qa_half(lane, p0[0], p0[1]);
          qa[1] = kt_pack_qa_half(lane, p1[0], p1[1]);
          qa[2] = kt_pack_qa_half(lane, p0[2], p0[3]);
          qa[3] = kt_pack_qa_half(lane, p1[2], p1[3]);

          const uint32_t k_base_u32 = ksm_u32 + (uint32_t)(wn * SKV_FP8 * K_D);
          const uint32_t v_base_u32 = vsm_u32 + (uint32_t)(wn * V_TILES64 * K_D * 64);

          float m0 = -1e30f, m1 = -1e30f, d0 = 0.f, d1 = 0.f;
          float aacc[4][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};

          for (int t0 = 0; t0 < S_pad32; t0 += 32) {
            uint32_t kb0[4], kb1[4];
            const int kb_row0 = t0 + (lane & 7) + ((lane >> 4) << 3);
            const int kb_ch = (lane >> 3) & 1;
            ldmatrix_x4(kb0, k_base_u32 + kt_swz32(kb_row0, kb_ch));
            ldmatrix_x4(kb1, k_base_u32 + kt_swz32(kb_row0 + 16, kb_ch));

            float s0[4] = {0.f,0.f,0.f,0.f}, s1[4] = {0.f,0.f,0.f,0.f};
            float s2[4] = {0.f,0.f,0.f,0.f}, s3[4] = {0.f,0.f,0.f,0.f};
            mma_e4m3(s0, qa, kb0[0], kb0[1]);
            mma_e4m3(s1, qa, kb0[2], kb0[3]);
            mma_e4m3(s2, qa, kb1[0], kb1[1]);
            mma_e4m3(s3, qa, kb1[2], kb1[3]);
#pragma unroll
            for (int e = 0; e < 4; ++e) {
              s0[e] *= s_unscale; s1[e] *= s_unscale;
              s2[e] *= s_unscale; s3[e] *= s_unscale;
            }
            if (t0 + 32 > cells) {
              const int k0 = t0 + c_in_n8;
              if (k0 >= cells)      { s0[0] = -1e30f; s0[2] = -1e30f; }
              if (k0 + 1 >= cells)  { s0[1] = -1e30f; s0[3] = -1e30f; }
              if (k0 + 8 >= cells)  { s1[0] = -1e30f; s1[2] = -1e30f; }
              if (k0 + 9 >= cells)  { s1[1] = -1e30f; s1[3] = -1e30f; }
              if (k0 + 16 >= cells) { s2[0] = -1e30f; s2[2] = -1e30f; }
              if (k0 + 17 >= cells) { s2[1] = -1e30f; s2[3] = -1e30f; }
              if (k0 + 24 >= cells) { s3[0] = -1e30f; s3[2] = -1e30f; }
              if (k0 + 25 >= cells) { s3[1] = -1e30f; s3[3] = -1e30f; }
            }

            float rm0 = fmaxf(fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s1[0], s1[1])),
                              fmaxf(fmaxf(s2[0], s2[1]), fmaxf(s3[0], s3[1])));
            rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffffu, rm0, 1));
            rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffffu, rm0, 2));
            float rm1 = fmaxf(fmaxf(fmaxf(s0[2], s0[3]), fmaxf(s1[2], s1[3])),
                              fmaxf(fmaxf(s2[2], s2[3]), fmaxf(s3[2], s3[3])));
            rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffffu, rm1, 1));
            rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffffu, rm1, 2));

            const float mn0 = fmaxf(m0, rm0), mn1 = fmaxf(m1, rm1);
            const float a0 = exp2f(m0 - mn0), a1 = exp2f(m1 - mn1);
            const float b0_s = 8.0f - mn0, b1_s = 8.0f - mn1;
            const float p00 = exp2f(s0[0] + b0_s), p01 = exp2f(s0[1] + b0_s);
            const float p10 = exp2f(s1[0] + b0_s), p11 = exp2f(s1[1] + b0_s);
            const float p20 = exp2f(s2[0] + b0_s), p21 = exp2f(s2[1] + b0_s);
            const float p30 = exp2f(s3[0] + b0_s), p31 = exp2f(s3[1] + b0_s);
            const float p02 = exp2f(s0[2] + b1_s), p03 = exp2f(s0[3] + b1_s);
            const float p12 = exp2f(s1[2] + b1_s), p13 = exp2f(s1[3] + b1_s);
            const float p22 = exp2f(s2[2] + b1_s), p23 = exp2f(s2[3] + b1_s);
            const float p32 = exp2f(s3[2] + b1_s), p33 = exp2f(s3[3] + b1_s);

            float rs0 = ((p00 + p01) + (p10 + p11)) + ((p20 + p21) + (p30 + p31));
            rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1);
            rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2);
            float rs1 = ((p02 + p03) + (p12 + p13)) + ((p22 + p23) + (p32 + p33));
            rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1);
            rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);

            d0 = fmaf(d0, a0, rs0); d1 = fmaf(d1, a1, rs1);
            m0 = mn0; m1 = mn1;
#pragma unroll
            for (int nj = 0; nj < 4; ++nj) {
              aacc[nj][0] *= a0; aacc[nj][1] *= a0;
              aacc[nj][2] *= a1; aacc[nj][3] *= a1;
            }
            uint32_t p_reg[4];
            p_reg[0] = kt_pack_e4m3x4(p00, p01, p10, p11);
            p_reg[1] = kt_pack_e4m3x4(p02, p03, p12, p13);
            p_reg[2] = kt_pack_e4m3x4(p20, p21, p30, p31);
            p_reg[3] = kt_pack_e4m3x4(p22, p23, p32, p33);

            const int v_ch = ((t0 & 32) >> 4) + ((lane >> 3) & 1);
            uint32_t vb0[4], vb1[4];
            const int vd0 = (lane & 7) + ((lane >> 4) << 3);
            ldmatrix_x4(vb0, v_base_u32 + swz(vd0, v_ch));
            ldmatrix_x4(vb1, v_base_u32 + swz(16 + vd0, v_ch));

            mma_e4m3(aacc[0], p_reg, vb0[0], vb0[1]);
            mma_e4m3(aacc[1], p_reg, vb0[2], vb0[3]);
            mma_e4m3(aacc[2], p_reg, vb1[0], vb1[1]);
            mma_e4m3(aacc[3], p_reg, vb1[2], vb1[3]);
          }
          const float inv_d0 = inv_v_fp8_sc / d0;
          const float inv_d1 = inv_v_fp8_sc / d1;
#pragma unroll
          for (int nj = 0; nj < 4; ++nj) {
            attn_reg[nj] = __floats2bfloat162_rn(aacc[nj][0] * inv_d0, aacc[nj][1] * inv_d0);
            attn_reg[4 + nj] = __floats2bfloat162_rn(aacc[nj][2] * inv_d1, aacc[nj][3] * inv_d1);
          }
        }
        __syncthreads();  // k_smem & v_smem now dead -> overlay sub_s8 + w_stage_b!

        // 2d. Quantize attn_reg -> sub_s8 -> O_proj accumulation into out_acc!
        float oxs0, oxs1;
        quant_regs_to_sub_s8(attn_reg, oxs0, oxs1);
        accum_64_to_192(wo_i, K_C, hp * 64, oxs0, oxs1, out_acc);
      }

      // Residual add X += attn_out (in SMEM!) + fused FFN RMSNorm -> act_s8!
      add_to_x_smem(out_acc, so_i);
      norm_x_to_act_s8(P.ffn_norm_sb + (size_t)i * 2 * K_C);

      // 2e. SwiGLU FFN: fused [W_gate | W_up] -> silu(g)*u -> W_down!
      float down_acc[12][4];
#pragma unroll
      for (int nj = 0; nj < 12; ++nj)
#pragma unroll
        for (int e = 0; e < 4; ++e) down_acc[nj][e] = 0.f;

      const int8_t* wgu_i = P.w_gu + (size_t)i * (2 * FFN) * K_C;
      const float* sgu_i = P.s_gu + (size_t)i * (2 * FFN);
      const int8_t* wd_i = P.w_down + (size_t)i * K_C * FFN;
      const float* sd_i = P.s_down + (size_t)i * K_C;
      const int n_chunks = FFN >> 6;
      const float mxs0 = (my_r0 < cells) ? act_sc[my_r0] : 0.f;
      const float mxs1 = (my_r1 < cells) ? act_sc[my_r1] : 0.f;

      for (int cc = 0; cc < n_chunks; ++cc) {
        __nv_bfloat162 swi_bf[8];
        {
          int gacc[4][4], uacc[4][4];
          gemm_192_to_64x2(wgu_i + (size_t)(cc * 128) * K_C, gacc, uacc);
          const float* sg_c = sgu_i + cc * 128;
          const float* su_c = sg_c + 64;
#pragma unroll
          for (int nj = 0; nj < 4; ++nj) {
            const int col = wn * 32 + nj * 8 + c_in_n8;
            const float wg0 = sg_c[col], wg1 = sg_c[col + 1];
            const float wu0 = su_c[col], wu1 = su_c[col + 1];
            if (my_r0 < cells) {
              swi_bf[nj] = __floats2bfloat162_rn(
                  kt_silu((float)gacc[nj][0] * mxs0 * wg0) * ((float)uacc[nj][0] * mxs0 * wu0),
                  kt_silu((float)gacc[nj][1] * mxs0 * wg1) * ((float)uacc[nj][1] * mxs0 * wu1));
            }
            if (my_r1 < cells) {
              swi_bf[4 + nj] = __floats2bfloat162_rn(
                  kt_silu((float)gacc[nj][2] * mxs1 * wg0) * ((float)uacc[nj][2] * mxs1 * wu0),
                  kt_silu((float)gacc[nj][3] * mxs1 * wg1) * ((float)uacc[nj][3] * mxs1 * wu1));
            }
          }
        }
        float dxs0, dxs1;
        quant_regs_to_sub_s8(swi_bf, dxs0, dxs1);
        accum_64_to_192(wd_i, FFN, cc * 64, dxs0, dxs1, down_acc);
      }

      add_to_x_smem(down_acc, sd_i);
      if (i + 1 < L) {
        norm_x_to_act_s8(P.attn_norm_sb + (size_t)(i + 1) * 2 * K_C);
      }
    }

    // ---- 3. Fused Final RMSNorm + SiLU (X_smem -> h_final_swz) + Parallel KataGo Heads ----
    {
      const float2* fnsb2 = reinterpret_cast<const float2*>(P.heads.final_norm_sb);
      float2 fns[3], fnb[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        const int p = j * 32 + lane;
        fns[j] = fnsb2[p];
        fnb[j] = fnsb2[K_C2 + p];
      }
      // Write silu(final_norm(X_smem)) directly into swizzled h_final_swz[6 panels, 64 rows, 32 bf16]!
      // Each warp owns 1 row `cell`, lane owns 3 pairs (j=0..2 -> ch = j*64 + lane*2 .. +1).
      // Panel kp = ch >> 5 (0..5), in_panel_ch = ch & 31, chunk = in_panel_ch >> 3 (0..3), in_chunk_b = (in_panel_ch & 7)*2!
      for (int cell = warp; cell < cells; cell += 8) {
        float2 v[3];
        float sq = 0.f;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          v[j] = __bfloat1622float2(X_smem2[kt_x_idx(cell, j, lane)]);
          sq = fmaf(v[j].x, v[j].x, fmaf(v[j].y, v[j].y, sq));
        }
        const float rstd = rsqrtf(kt_warp_sum(sq) * (1.0f / K_C) + K_RMS_EPS);
        const int sw_mask = (cell >> 1) & 3;
        const int kp_off = lane >> 4;  // 0 or 1 within pair of 32-ch panels
        const int ch_in_32 = (lane & 15) * 2;
        const int chunk = (ch_in_32 >> 3) ^ sw_mask;
        const int in_ch_b = (ch_in_32 & 7) * 2;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          const __nv_bfloat162 o = __floats2bfloat162_rn(
              kt_silu(fmaf(v[j].x * rstd, fns[j].x, fnb[j].x)),
              kt_silu(fmaf(v[j].y * rstd, fns[j].y, fnb[j].y)));
          const int kp = j * 2 + kp_off;
          *reinterpret_cast<__nv_bfloat162*>(h_final_swz + (kp * ROWS + cell) * 64 + (chunk << 4) + in_ch_b) = o;
        }
      }
      for (int idx = tid; idx < (ROWS - cells) * 24; idx += 256) {
        const int cell = cells + idx / 24, rem = idx - (cell - cells) * 24;
        *reinterpret_cast<uint4*>(h_final_swz + (rem >> 2) * ROWS * 64 + swz(cell, rem & 3)) = make_uint4(0u, 0u, 0u, 0u);
      }
      __syncthreads();  // X_smem [0..24,575] is now FREE for h_outp, h_stats, h_wstage!

      float* g_mean = h_stats;
      float* g_max = g_mean + 48;
      float* v_mean = g_max + 48;
      float* v_max = v_mean + 48;
      float* w_sum = v_max + 48;
      float* w_max = w_sum + 8 * 48;
      const float inv_cells = 1.0f / (float)cells;

      auto run_conv1_tc = [&](int row0, __nv_bfloat16* dst) {
        const int lrow = tid >> 2, lchunk = tid & 3;
        auto load_w = [&](int st, int kp) {
          const bool ok = lrow < K_HEAD_C;
          cp_async16(hwst_u32 + (st & 1) * 4096 + swz(lrow, lchunk),
                     P.heads.conv1pgv + (size_t)(row0 + (ok ? lrow : 0)) * K_C + kp * 32 + lchunk * 8, ok);
        };
        float facc[4][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
        load_w(0, 0);
        cp_async_commit();
#pragma unroll
        for (int kp = 0; kp < 6; ++kp) {
          cp_async_wait<0>();
          __syncthreads();
          if (kp + 1 < 6) { load_w(kp + 1, kp + 1); cp_async_commit(); }
          const uint32_t b_tile = hwst_u32 + (kp & 1) * 4096;
          const uint32_t a_panel = hfin_u32 + (uint32_t)(kp * ROWS * 64);
#pragma unroll
          for (int ks = 0; ks < 2; ++ks) {
            uint32_t a[4];
            ldmatrix_x4(a, a_panel + swz(wm * 16 + (lane & 15), ks * 2 + (lane >> 4)));
            uint32_t b[2][4];
#pragma unroll
            for (int np = 0; np < 2; ++np) {
              const int brow = wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3);
              ldmatrix_x4(b[np], b_tile + swz(brow, ks * 2 + ((lane >> 3) & 1)));
            }
#pragma unroll
            for (int nj = 0; nj < 4; ++nj) mma_bf16(facc[nj], a, b[nj >> 1][(nj & 1) * 2], b[nj >> 1][(nj & 1) * 2 + 1]);
          }
        }
        __syncthreads();
#pragma unroll
        for (int nj = 0; nj < 4; ++nj) {
          const int col = wn * 32 + nj * 8 + c_in_n8;
          if (col < K_HEAD_C) {
            if (my_r0 < cells) *reinterpret_cast<__nv_bfloat162*>(dst + my_r0 * K_HEAD_C + col) =
                __floats2bfloat162_rn(facc[nj][0], facc[nj][1]);
            if (my_r1 < cells) *reinterpret_cast<__nv_bfloat162*>(dst + my_r1 * K_HEAD_C + col) =
                __floats2bfloat162_rn(facc[nj][2], facc[nj][3]);
          }
        }
        __syncthreads();
      };

      auto reduce_gv_pool = [&](int norm_row_s, int norm_row_b, float* out_mean, float* out_max) {
        for (int idx = tid; idx < 8 * 48; idx += 256) { w_sum[idx] = 0.f; w_max[idx] = -1e30f; }
        __syncthreads();
        const float ns0 = P.heads.head_norm_sb[norm_row_s * 48 + lane];
        const float nb0 = P.heads.head_norm_sb[norm_row_b * 48 + lane];
        const float ns1 = (lane < 16) ? P.heads.head_norm_sb[norm_row_s * 48 + 32 + lane] : 0.f;
        const float nb1 = (lane < 16) ? P.heads.head_norm_sb[norm_row_b * 48 + 32 + lane] : 0.f;
        for (int cell = warp; cell < cells; cell += 8) {
          const float v0 = __bfloat162float(h_gv_tmp[cell * K_HEAD_C + lane]);
          const float v1 = (lane < 16) ? __bfloat162float(h_gv_tmp[cell * K_HEAD_C + 32 + lane]) : 0.f;
          const float rstd = rsqrtf(kt_warp_sum(v0 * v0 + v1 * v1) * (1.0f / K_HEAD_C) + K_RMS_EPS);
          const float o0 = kt_silu(fmaf(v0 * rstd, ns0, nb0));
          w_sum[warp * 48 + lane] += o0;
          w_max[warp * 48 + lane] = fmaxf(w_max[warp * 48 + lane], o0);
          if (lane < 16) {
            const float o1 = kt_silu(fmaf(v1 * rstd, ns1, nb1));
            w_sum[warp * 48 + 32 + lane] += o1;
            w_max[warp * 48 + 32 + lane] = fmaxf(w_max[warp * 48 + 32 + lane], o1);
          }
        }
        __syncthreads();
        if (tid < 48) {
          float s = 0.f, m = -1e30f;
#pragma unroll
          for (int w = 0; w < 8; ++w) { s += w_sum[w * 48 + tid]; m = fmaxf(m, w_max[w * 48 + tid]); }
          out_mean[tid] = s * inv_cells;
          out_max[tid] = m;
        }
        __syncthreads();
      };

      run_conv1_tc(0, h_outp);
      run_conv1_tc(48, h_gv_tmp);
      reduce_gv_pool(0, 1, g_mean, g_max);
      run_conv1_tc(96, h_gv_tmp);
      reduce_gv_pool(4, 5, v_mean, v_max);

      // Parallel dense head matmuls with 100% COALESCED transposed weight reads!
      float* g_bias = reinterpret_cast<float*>(h_final_swz);
      float* val_h = g_bias + 48;
      float* v_out_s = val_h + K_VAL_H;
      float* s_pol_w = v_out_s + 16;
      float* s_q_w = s_pol_w + 48;
      float* s_q_b = s_q_w + 16 * 48;
      float* s_pq = s_q_b + 16;
      float* s_ql = s_pq + 16;

      if (tid < 48) s_pol_w[tid] = P.heads.pol_w[tid];
      if (tid < K) s_q_b[tid] = P.heads.q_b[tid];
      for (int idx = tid; idx < K * 48; idx += 256) s_q_w[idx] = P.heads.q_w[idx];

      if (tid < 48) {
        float acc = 0.f;
        for (int j = 0; j < 48; ++j) acc = fmaf(g_mean[j], P.heads.linear_g_w[j * 48 + tid], acc);
        for (int j = 0; j < 48; ++j) acc = fmaf(g_max[j], P.heads.linear_g_w[(48 + j) * 48 + tid], acc);
        g_bias[tid] = acc;
      } else if (tid < 48 + K_VAL_H) {
        const int o = tid - 48;
        float acc = P.heads.val_l1_b[o];
        for (int j = 0; j < 48; ++j) acc = fmaf(v_mean[j], P.heads.val_l1_w[j * K_VAL_H + o], acc);
        for (int j = 0; j < 48; ++j) acc = fmaf(v_max[j], P.heads.val_l1_w[(48 + j) * K_VAL_H + o], acc);
        val_h[o] = kt_silu(acc);
      } else if (A == cells + 1 && tid >= 176 && tid < 176 + K) {
        const int k = tid - 176;
        float acc = P.heads.pass_q_b[k];
        for (int j = 0; j < 48; ++j) acc = fmaf(g_mean[j], P.heads.pass_q_w[j * K + k], acc);
        for (int j = 0; j < 48; ++j) acc = fmaf(g_max[j], P.heads.pass_q_w[(48 + j) * K + k], acc);
        s_pq[k] = acc;
      } else if (A == cells + 1 && P.logits && tid == 192) {
        float pa = P.heads.pass_pol_b[0];
        for (int j = 0; j < 48; ++j) pa = fmaf(g_mean[j], P.heads.pass_pol_w[j], pa);
        for (int j = 0; j < 48; ++j) pa = fmaf(g_max[j], P.heads.pass_pol_w[48 + j], pa);
        P.logits[(size_t)board * A + cells] = pa;
      }
      __syncthreads();

      if (tid < K) {
        float acc = P.heads.val_out_b[tid];
        for (int j = 0; j < K_VAL_H; ++j) acc = fmaf(val_h[j], P.heads.val_out_w[j * K + tid], acc);
        v_out_s[tid] = acc;
      }
      if (A == cells + 1 && warp == 1) {
        const float v = (lane < K) ? s_pq[lane] : -1e30f;
        const float mx = kt_warp_max(v);
        const float e = (lane < K) ? __expf(v - mx) : 0.f;
        const float tot = kt_warp_sum(e);
        if (lane < K) P.qprobs[((size_t)board * A + cells) * K + lane] = e / tot;
      }
      __syncwarp();
      if (warp == 0) {
        const float v = (lane < K) ? v_out_s[lane] : -1e30f;
        const float mx = kt_warp_max(v);
        const float e = (lane < K) ? __expf(v - mx) : 0.f;
        const float tot = kt_warp_sum(e);
        if (lane < K) P.vprobs[(size_t)board * K + lane] = e / tot;
      }

      const float p_ns0 = P.heads.head_norm_sb[2 * 48 + lane];
      const float p_nb0 = P.heads.head_norm_sb[3 * 48 + lane];
      const float p_ns1 = (lane < 16) ? P.heads.head_norm_sb[2 * 48 + 32 + lane] : 0.f;
      const float p_nb1 = (lane < 16) ? P.heads.head_norm_sb[3 * 48 + 32 + lane] : 0.f;
      float* w_ql = s_ql + warp * 16;

      for (int cell = warp; cell < cells; cell += 8) {
        float hp0 = __bfloat162float(h_outp[cell * K_HEAD_C + lane]) + g_bias[lane];
        float hp1 = (lane < 16) ? (__bfloat162float(h_outp[cell * K_HEAD_C + 32 + lane]) + g_bias[32 + lane]) : 0.f;
        const float prstd = rsqrtf(kt_warp_sum(hp0 * hp0 + hp1 * hp1) * (1.0f / K_HEAD_C) + K_RMS_EPS);
        hp0 = kt_silu(fmaf(hp0 * prstd, p_ns0, p_nb0));
        hp1 = (lane < 16) ? kt_silu(fmaf(hp1 * prstd, p_ns1, p_nb1)) : 0.f;

        const int y = (int)cell_y[cell], x = (int)cell_x[cell];
        const int dst_action = is_white ? (x * side + y) : cell;

        if (P.logits) {
          float pdot = hp0 * s_pol_w[lane];
          if (lane < 16) pdot = fmaf(hp1, s_pol_w[32 + lane], pdot);
          const float pl = kt_warp_sum(pdot) + P.heads.pol_b[0];
          if (lane == 0) P.logits[(size_t)board * A + dst_action] = pl;
        }

        float qmx = -1e30f;
        for (int k = 0; k < K; ++k) {
          const float* wqk = s_q_w + k * K_HEAD_C;
          float qdot = hp0 * wqk[lane];
          if (lane < 16) qdot = fmaf(hp1, wqk[32 + lane], qdot);
          const float v = kt_warp_sum(qdot) + s_q_b[k];
          if (lane == 0) w_ql[k] = v;
          qmx = fmaxf(qmx, v);
        }
        if (lane == 0) {
          float qtot = 0.f;
          for (int k = 0; k < K; ++k) { w_ql[k] = __expf(w_ql[k] - qmx); qtot += w_ql[k]; }
          const float qinv = 1.0f / qtot;
          float* qdst = P.qprobs + ((size_t)board * A + dst_action) * K;
          for (int k = 0; k < K; ++k) qdst[k] = w_ql[k] * qinv;
        }
      }
      __syncthreads();

      if (P.cw.table_locks) {
        const CacheKey mk = P.cw.miss_keys[board];
        if (mk.meta & 0x80000000u) {
          const uint32_t tag = 2u + *P.cw.cache_wave;
          const uint32_t slot = hash_cache_key(mk);
          if (tid == 0) won_slot = cache_try_acquire(P.cw.table_locks, slot, tag);
          __syncthreads();
          if (won_slot) {
            if (tid == 0) P.cw.table_keys[slot] = mk;
            if (tid < K) P.cw.table_v[(size_t)slot * K + tid] = __float2bfloat16_rn(P.vprobs[(size_t)board * K + tid]);
            const int AK = A * K;
            for (int idx = tid; idx < AK; idx += 256) {
              P.cw.table_q[(size_t)slot * AK + idx] = __float2bfloat16_rn(P.qprobs[(size_t)board * AK + idx]);
            }
            __syncthreads();
            if (tid == 0) { __threadfence(); atomicExch(&P.cw.table_locks[slot], tag); }
          }
        }
      }
      __syncthreads();
    }
  }
}

// ============================================================================
// Kernel B: Hex 9..11 (64 < cells <= 121, 2-half MW=1, 47.6 KB total SMEM -> 2 CTAs/SM!!)
// ============================================================================
__host__ __device__ constexpr int kata_tf_smem_bytes_2h() {
  constexpr int ROWS = 64;
  constexpr int MAX_CELLS = 121;
  constexpr int ACT_END = ((ROWS * K_C + ROWS * 4) + 127) & ~127;  // 12,544 B
  // Trunk: act_s8(12.25K) + k_smem(8K) + v_smem(8K) + w_stage_2buf(8K) = 36.25 KB!
  constexpr int s_trunk = ACT_END + 2 * 128 * K_D + 2 * 2 * K_D * 64 + 8192;
  // Heads: h_outp[121,48](11.6K) + h_stats(3.84K) -> 15,616 + h_final_swz(24.576K) + h_wstage(8.192K) = 48,384 B (47.25 KB)!
  constexpr int h_off1 = ((MAX_CELLS * K_HEAD_C * 2 + 960 * 4) + 127) & ~127;
  constexpr int s_heads = h_off1 + ROWS * K_C * 2 + 8192;
  return s_trunk > s_heads ? s_trunk : s_heads;  // 48,384 B + 384 B static = 48,768 B <= 50,176 B (49 KB -> 2 CTAs/SM!)
}

__global__ void __launch_bounds__(256, 2) kata_tf_board_int8_kernel_2h(
    const __grid_constant__ KataTfParams P) {
  const int active_boards = P.d_boards ? *P.d_boards : P.max_boards;
  if (active_boards <= 0) return;

  constexpr int ROWS = 64;
  constexpr int MAX_CELLS = 121;
  constexpr int SKV_FP8 = 128;
  constexpr int V_TILES64 = 2;

  extern __shared__ __align__(128) char smem_raw[];
  __shared__ uint8_t cell_y[128];
  __shared__ uint8_t cell_x[128];
  __shared__ bool won_slot;

  const int tid = threadIdx.x;
  const int warp = tid >> 5;
  const int lane = tid & 31;
  const int wm = warp >> 1;
  const int wn = warp & 1;

  const int side = P.side;
  const int cells = P.cells;
  const int A = P.A;
  const int K = P.K;
  const int L = P.L;
  const int FFN = P.FFN;

  if (tid < cells) {
    const int y = tid / side;
    cell_y[tid] = (uint8_t)y;
    cell_x[tid] = (uint8_t)(tid - y * side);
  }
  __syncthreads();

  // Trunk SMEM (36.25 KB): act_s8(12.25K) + k_smem(8K) + v_smem(8K) + w_stage(8K)
  char* act_s8 = smem_raw;
  float* act_sc = reinterpret_cast<float*>(act_s8 + ROWS * K_C);
  constexpr int ACT_END = ((ROWS * K_C + ROWS * 4) + 127) & ~127;
  uint8_t* k_smem = reinterpret_cast<uint8_t*>(smem_raw + ACT_END);
  uint8_t* v_smem = k_smem + 2 * SKV_FP8 * K_D;
  char* w_stage = reinterpret_cast<char*>(v_smem + 2 * V_TILES64 * K_D * 64);

  // sub_s8 + sub_sc + half_max overlay on k_smem (when k_smem/v_smem not needed) OR during O-proj right after FlashAttn!
  // Wait: in 2h Attention, we do Half 1 FlashAttn while k_smem/v_smem MUST STAY LIVE for Half 0 FlashAttn!
  // So put sub_s8(4K) + half_max(512B) at w_stage + 8192 = 37,120..41,728 (< 48,384 B!!)!
  // OH! 37,120 + 4,608 = 41,728 B << 48,384 B -> ZERO OVERLAY CONFLICT WITH k_smem/v_smem!!
  char* sub_s8 = w_stage + 8192;
  float* half_max = reinterpret_cast<float*>(sub_s8 + ROWS * 64);

  // Heads SMEM (47.25 KB)
  __nv_bfloat16* h_outp = reinterpret_cast<__nv_bfloat16*>(smem_raw);
  float* h_stats = reinterpret_cast<float*>(h_outp + MAX_CELLS * K_HEAD_C);
  constexpr int H_OFF1 = ((MAX_CELLS * K_HEAD_C * 2 + 960 * 4) + 127) & ~127;
  char* h_final_swz = smem_raw + H_OFF1;
  char* h_wstage = h_final_swz + ROWS * K_C * 2;
  __nv_bfloat16* h_gv_tmp = reinterpret_cast<__nv_bfloat16*>(h_wstage);

  const uint32_t act_u32 = smem_u32(act_s8);
  const uint32_t wst_u32 = smem_u32(w_stage);
  const uint32_t sub_u32 = smem_u32(sub_s8);
  const uint32_t ksm_u32 = smem_u32(k_smem);
  const uint32_t vsm_u32 = smem_u32(v_smem);
  const uint32_t hfin_u32 = smem_u32(h_final_swz);
  const uint32_t hwst_u32 = smem_u32(h_wstage);

  const int S_pad32 = (cells + 31) & ~31;
  const float half_side = 0.5f * (float)(side - 1);
  const int r0_in_tile = lane >> 2;
  const int c_in_n8 = (lane & 3) * 2;
  const int my_r0 = wm * 16 + r0_in_tile;
  const int my_r1 = my_r0 + 8;

  const size_t cta_row_base = (size_t)blockIdx.x * cells;
  __nv_bfloat16* my_X = P.cta_x + cta_row_base * K_C;
  __nv_bfloat16* my_tmp = P.cta_tmp + cta_row_base * K_C;
  int8_t* my_act_s8 = P.cta_act_s8 + cta_row_base * K_C;
  float* my_act_sc = P.cta_act_sc + cta_row_base;

  auto load_act_s8_half = [&](int row0) {
    const int lrow = tid >> 2, lchunk = tid & 3;
    const bool ok = (row0 + lrow) < cells;
#pragma unroll
    for (int kp = 0; kp < 3; ++kp) {
      cp_async16(act_u32 + kp * 4096 + swz(lrow, lchunk),
                 my_act_s8 + (size_t)(ok ? (row0 + lrow) : 0) * K_C + kp * 64 + lchunk * 16, ok);
    }
    if (tid < 64) act_sc[tid] = (row0 + tid < cells) ? my_act_sc[row0 + tid] : 0.f;
    cp_async_commit();
    cp_async_wait<0>();
    __syncthreads();
  };

  auto gemm_192_to_64 = [&](const int8_t* W_mat, int (&acc)[4][4]) {
    const int lrow = tid >> 2, lchunk = tid & 3;
    auto load_w = [&](int st, int kk) {
      cp_async16(wst_u32 + (st & 1) * 4096 + swz(lrow, lchunk),
                 W_mat + (size_t)lrow * K_C + kk * 64 + lchunk * 16, true);
    };
#pragma unroll
    for (int nj = 0; nj < 4; ++nj)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[nj][e] = 0;
    load_w(0, 0);
    cp_async_commit();
#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
      cp_async_wait<0>();
      __syncthreads();
      if (kk + 1 < 3) { load_w(kk + 1, kk + 1); cp_async_commit(); }
      const uint32_t b_tile = wst_u32 + (kk & 1) * 4096;
      const uint32_t a_panel = act_u32 + (uint32_t)(kk * ROWS * 64);
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        uint32_t a[4];
        ldmatrix_x4(a, a_panel + swz(wm * 16 + (lane & 15), ks * 2 + (lane >> 4)));
        uint32_t b[2][4];
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          const int brow = wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3);
          ldmatrix_x4(b[np], b_tile + swz(brow, ks * 2 + ((lane >> 3) & 1)));
        }
#pragma unroll
        for (int nj = 0; nj < 4; ++nj) mma_s8s8(acc[nj], a, b[nj >> 1][(nj & 1) * 2], b[nj >> 1][(nj & 1) * 2 + 1]);
      }
    }
    __syncthreads();
  };

  auto gemm_192_to_64x2 = [&](const int8_t* W_pair, int (&acc0)[4][4], int (&acc1)[4][4]) {
    const int lrow = tid >> 2, lchunk = tid & 3;
#pragma unroll
    for (int nj = 0; nj < 4; ++nj)
#pragma unroll
      for (int e = 0; e < 4; ++e) { acc0[nj][e] = 0; acc1[nj][e] = 0; }
#pragma unroll
    for (int kk = 0; kk < 3; ++kk) {
      cp_async16(wst_u32 + swz(lrow, lchunk), W_pair + (size_t)lrow * K_C + kk * 64 + lchunk * 16, true);
      cp_async16(wst_u32 + 4096 + swz(lrow, lchunk), W_pair + (size_t)(64 + lrow) * K_C + kk * 64 + lchunk * 16, true);
      cp_async_commit();
      cp_async_wait<0>();
      __syncthreads();
      const uint32_t a_panel = act_u32 + (uint32_t)(kk * ROWS * 64);
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        uint32_t a[4];
        ldmatrix_x4(a, a_panel + swz(wm * 16 + (lane & 15), ks * 2 + (lane >> 4)));
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t b[4];
          const int brow = wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3);
          ldmatrix_x4(b, wst_u32 + swz(brow, ks * 2 + ((lane >> 3) & 1)));
          mma_s8s8(acc0[np * 2 + 0], a, b[0], b[1]);
          mma_s8s8(acc0[np * 2 + 1], a, b[2], b[3]);
        }
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t b[4];
          const int brow = wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3);
          ldmatrix_x4(b, wst_u32 + 4096 + swz(brow, ks * 2 + ((lane >> 3) & 1)));
          mma_s8s8(acc1[np * 2 + 0], a, b[0], b[1]);
          mma_s8s8(acc1[np * 2 + 1], a, b[2], b[3]);
        }
      }
      __syncthreads();
    }
  };

  auto quant_regs_to_sub_s8 = [&](const __nv_bfloat162 (&val_bf)[8], int n_rows, float& out_xs0, float& out_xs1) {
    float am0 = 0.f, am1 = 0.f;
#pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
      const float2 f0 = __bfloat1622float2(val_bf[nj]);
      const float2 f1 = __bfloat1622float2(val_bf[4 + nj]);
      am0 = fmaxf(am0, fmaxf(fabsf(f0.x), fabsf(f0.y)));
      am1 = fmaxf(am1, fmaxf(fabsf(f1.x), fabsf(f1.y)));
    }
    am0 = fmaxf(am0, __shfl_xor_sync(0xffffffffu, am0, 1));
    am0 = fmaxf(am0, __shfl_xor_sync(0xffffffffu, am0, 2));
    am1 = fmaxf(am1, __shfl_xor_sync(0xffffffffu, am1, 1));
    am1 = fmaxf(am1, __shfl_xor_sync(0xffffffffu, am1, 2));
    if ((lane & 3) == 0) {
      if (my_r0 < n_rows) half_max[my_r0 * 2 + wn] = am0;
      if (my_r1 < n_rows) half_max[my_r1 * 2 + wn] = am1;
    }
    __syncthreads();

    const float gm0 = (my_r0 < n_rows) ? fmaxf(fmaxf(half_max[my_r0 * 2], half_max[my_r0 * 2 + 1]), 1e-6f) : 1.0f;
    const float gm1 = (my_r1 < n_rows) ? fmaxf(fmaxf(half_max[my_r1 * 2], half_max[my_r1 * 2 + 1]), 1e-6f) : 1.0f;
    const float is0 = (my_r0 < n_rows) ? (127.0f / gm0) : 0.f;
    const float is1 = (my_r1 < n_rows) ? (127.0f / gm1) : 0.f;
    out_xs0 = (my_r0 < n_rows) ? (gm0 * (1.0f / 127.0f)) : 0.f;
    out_xs1 = (my_r1 < n_rows) ? (gm1 * (1.0f / 127.0f)) : 0.f;

    const int sw0_mask = (my_r0 >> 1) & 3;
    const int sw1_mask = (my_r1 >> 1) & 3;
#pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
      const int col = wn * 32 + nj * 8 + c_in_n8;
      const int ch = col >> 4, in_ch = col & 15;
      const float2 f0 = __bfloat1622float2(val_bf[nj]);
      const float2 f1 = __bfloat1622float2(val_bf[4 + nj]);
      const uint32_t q0_lo = kt_pack_s8x2(f0.x, f0.y, is0);
      const uint32_t q0_hi = __shfl_xor_sync(0xffffffffu, q0_lo, 1);
      const uint32_t q1_lo = kt_pack_s8x2(f1.x, f1.y, is1);
      const uint32_t q1_hi = __shfl_xor_sync(0xffffffffu, q1_lo, 1);
      if ((lane & 1) == 0) {
        *reinterpret_cast<uint32_t*>(sub_s8 + my_r0 * 64 + ((ch ^ sw0_mask) << 4) + in_ch) = q0_lo | (q0_hi << 16);
        *reinterpret_cast<uint32_t*>(sub_s8 + my_r1 * 64 + ((ch ^ sw1_mask) << 4) + in_ch) = q1_lo | (q1_hi << 16);
      }
    }
    __syncthreads();
  };

  auto accum_64_to_192 = [&](const int8_t* W_full, int K_full, int k_off, float xs0, float xs1, float (&out_acc)[12][4]) {
    const int lrow = tid >> 2, lchunk = tid & 3;
    auto load_w = [&](int st, int rc) {
      cp_async16(wst_u32 + (st & 1) * 4096 + swz(lrow, lchunk),
                 W_full + (size_t)(rc * 64 + lrow) * K_full + k_off + lchunk * 16, true);
    };
    load_w(0, 0);
    cp_async_commit();
#pragma unroll
    for (int rc = 0; rc < 3; ++rc) {
      cp_async_wait<0>();
      __syncthreads();
      if (rc + 1 < 3) { load_w(rc + 1, rc + 1); cp_async_commit(); }
      const uint32_t b_tile = wst_u32 + (rc & 1) * 4096;
      int kacc[4][4] = {{0,0,0,0},{0,0,0,0},{0,0,0,0},{0,0,0,0}};
#pragma unroll
      for (int ks = 0; ks < 2; ++ks) {
        uint32_t a[4];
        ldmatrix_x4(a, sub_u32 + swz(wm * 16 + (lane & 15), ks * 2 + (lane >> 4)));
#pragma unroll
        for (int np = 0; np < 2; ++np) {
          uint32_t b[4];
          const int brow = wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3);
          ldmatrix_x4(b, b_tile + swz(brow, ks * 2 + ((lane >> 3) & 1)));
          mma_s8s8(kacc[np * 2 + 0], a, b[0], b[1]);
          mma_s8s8(kacc[np * 2 + 1], a, b[2], b[3]);
        }
      }
#pragma unroll
      for (int nj = 0; nj < 4; ++nj) {
        const int d = rc * 4 + nj;
        out_acc[d][0] = fmaf((float)kacc[nj][0], xs0, out_acc[d][0]);
        out_acc[d][1] = fmaf((float)kacc[nj][1], xs0, out_acc[d][1]);
        out_acc[d][2] = fmaf((float)kacc[nj][2], xs1, out_acc[d][2]);
        out_acc[d][3] = fmaf((float)kacc[nj][3], xs1, out_acc[d][3]);
      }
    }
    __syncthreads();
  };

  auto store_v_fp8 = [&](int h_local, int d, int k_idx, uint8_t val) {
    const int k_perm = kt_v_key_perm(k_idx);
    const int t64 = k_perm >> 6, k_in = k_perm & 63;
    const int sw_ch = (k_in >> 4) ^ ((d >> 1) & 3);
    v_smem[((h_local * V_TILES64 + t64) * K_D + d) * 64 + (sw_ch << 4) + (k_in & 15)] = val;
  };

  auto norm_l2_x_to_l2_act = [&](const float* nsb_ptr) {
    const float2* nsb2 = reinterpret_cast<const float2*>(nsb_ptr);
    float2 ns[3], nbi[3];
#pragma unroll
    for (int j = 0; j < 3; ++j) {
      const int p = j * 32 + lane;
      ns[j] = nsb2[p]; nbi[j] = nsb2[K_C2 + p];
    }
    for (int cell = warp; cell < cells; cell += 8) {
      const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(my_X + cell * K_C);
      float2 v[3];
      float sq = 0.f;
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        v[j] = __bfloat1622float2(xp[j * 32 + lane]);
        sq = fmaf(v[j].x, v[j].x, fmaf(v[j].y, v[j].y, sq));
      }
      const float rstd = rsqrtf(kt_warp_sum(sq) * (1.0f / K_C) + K_RMS_EPS);
      float amax = 0.f;
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        v[j].x = fmaf(v[j].x * rstd, ns[j].x, nbi[j].x);
        v[j].y = fmaf(v[j].y * rstd, ns[j].y, nbi[j].y);
        amax = fmaxf(amax, fmaxf(fabsf(v[j].x), fabsf(v[j].y)));
      }
      const float gmax = fmaxf(kt_warp_max(amax), 1e-6f);
      const float inv_sc = 127.0f / gmax;
      uint16_t* dst = reinterpret_cast<uint16_t*>(my_act_s8 + cell * K_C);
#pragma unroll
      for (int j = 0; j < 3; ++j) dst[j * 32 + lane] = kt_pack_s8x2(v[j].x, v[j].y, inv_sc);
      if (lane == 0) my_act_sc[cell] = gmax * (1.0f / 127.0f);
    }
    __syncthreads();
  };

  auto apply_qknorm_rope_h = [&](int row0, const int (&acc)[4][4], float xs0, float xs1, const float* ws_ptr,
                                 const float* qkn_s, const float* qkn_b, const float* rf_h,
                                 float2 (&v0)[4], float2 (&v1)[4]) {
    const int gr0 = row0 + my_r0, gr1 = row0 + my_r1;
    float sq0 = 0.f, sq1 = 0.f;
#pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
      const int col = wn * 32 + nj * 8 + c_in_n8;
      const float w0 = ws_ptr[col], w1 = ws_ptr[col + 1];
      v0[nj] = make_float2((float)acc[nj][0] * xs0 * w0, (float)acc[nj][1] * xs0 * w1);
      v1[nj] = make_float2((float)acc[nj][2] * xs1 * w0, (float)acc[nj][3] * xs1 * w1);
      sq0 = fmaf(v0[nj].x, v0[nj].x, fmaf(v0[nj].y, v0[nj].y, sq0));
      sq1 = fmaf(v1[nj].x, v1[nj].x, fmaf(v1[nj].y, v1[nj].y, sq1));
    }
    sq0 += __shfl_xor_sync(0xffffffffu, sq0, 1);
    sq0 += __shfl_xor_sync(0xffffffffu, sq0, 2);
    sq1 += __shfl_xor_sync(0xffffffffu, sq1, 1);
    sq1 += __shfl_xor_sync(0xffffffffu, sq1, 2);
    const float rstd0 = rsqrtf(sq0 * (1.0f / K_D) + K_RMS_EPS);
    const float rstd1 = rsqrtf(sq1 * (1.0f / K_D) + K_RMS_EPS);
#pragma unroll
    for (int nj = 0; nj < 4; ++nj) {
      const int d = nj * 8 + c_in_n8;
      const float ns0 = qkn_s[d], ns1 = qkn_s[d + 1];
      const float nb0 = qkn_b[d], nb1 = qkn_b[d + 1];
      v0[nj].x = fmaf(v0[nj].x * rstd0, ns0, nb0);
      v0[nj].y = fmaf(v0[nj].y * rstd0, ns1, nb1);
      v1[nj].x = fmaf(v1[nj].x * rstd1, ns0, nb0);
      v1[nj].y = fmaf(v1[nj].y * rstd1, ns1, nb1);
    }
    const float sx0 = (gr0 < cells) ? ((float)cell_x[gr0] - half_side) : 0.f;
    const float sy0 = (gr0 < cells) ? ((float)cell_y[gr0] - half_side) : 0.f;
    const float sx1 = (gr1 < cells) ? ((float)cell_x[gr1] - half_side) : 0.f;
    const float sy1 = (gr1 < cells) ? ((float)cell_y[gr1] - half_side) : 0.f;
#pragma unroll
    for (int p = 0; p < 2; ++p) {
      const int d_lo = p * 8 + c_in_n8;
      const float fx0 = rf_h[d_lo * 2 + 0], fy0 = rf_h[d_lo * 2 + 1];
      const float fx1 = rf_h[(d_lo + 1) * 2 + 0], fy1 = rf_h[(d_lo + 1) * 2 + 1];
      float s, c, a, b;
      __sincosf(fmaf(sx0, fx0, sy0 * fy0), &s, &c);
      a = v0[p].x; b = v0[p + 2].x;
      v0[p].x = fmaf(a, c, -b * s); v0[p + 2].x = fmaf(a, s, b * c);
      __sincosf(fmaf(sx0, fx1, sy0 * fy1), &s, &c);
      a = v0[p].y; b = v0[p + 2].y;
      v0[p].y = fmaf(a, c, -b * s); v0[p + 2].y = fmaf(a, s, b * c);
      __sincosf(fmaf(sx1, fx0, sy1 * fy0), &s, &c);
      a = v1[p].x; b = v1[p + 2].x;
      v1[p].x = fmaf(a, c, -b * s); v1[p + 2].x = fmaf(a, s, b * c);
      __sincosf(fmaf(sx1, fx1, sy1 * fy1), &s, &c);
      a = v1[p].y; b = v1[p + 2].y;
      v1[p].y = fmaf(a, c, -b * s); v1[p + 2].y = fmaf(a, s, b * c);
    }
  };

  for (int board = blockIdx.x; board < active_boards; board += gridDim.x) {
    const uint8_t* obs_b = P.obs + (size_t)board * cells * 4;
    const bool is_white = obs_b[2] != 0;

    // ---- 1. Stem 3x3 -> L2 my_X + Layer-0 attn_norm -> L2 my_act_s8 ----
    {
      constexpr int WS_STRIDE = K_C + 4;
      float* ws = reinterpret_cast<float*>(smem_raw);  // 21,168 B
      for (int idx = tid; idx < 27 * K_C; idx += 256) {
        const int n = idx / 27, k = idx - n * 27;
        ws[k * WS_STRIDE + n] = __bfloat162float(P.stem_w[idx]);
      }
      __syncthreads();

      const float2* bias2 = reinterpret_cast<const float2*>(P.stem_b);
      const float2* nsb2 = reinterpret_cast<const float2*>(P.attn_norm_sb);
      float2 b_reg[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) b_reg[j] = bias2[j * 32 + lane];

      for (int cell = warp; cell < cells; cell += 8) {
        const int y = (int)cell_y[cell], x = (int)cell_x[cell];
        float2 acc[3] = {b_reg[0], b_reg[1], b_reg[2]};
#pragma unroll
        for (int dy = -1; dy <= 1; ++dy) {
          const int yy = y + dy;
          if (yy < 0 || yy >= side) continue;
#pragma unroll
          for (int dx = -1; dx <= 1; ++dx) {
            const int xx = x + dx;
            if (xx < 0 || xx >= side) continue;
            const int t = (dy + 1) * 3 + (dx + 1);
            const int sy = is_white ? xx : yy;
            const int sx = is_white ? yy : xx;
            const uint32_t o = *reinterpret_cast<const uint32_t*>(obs_b + (sy * side + sx) * 4);
            const bool ch0 = (o & 0xffu) != 0u;
            const bool ch1 = ((o >> 8) & 0xffu) != 0u;
            const float2* w2 = reinterpret_cast<const float2*>(ws + (t * 3 + 2) * WS_STRIDE);
#pragma unroll
            for (int j = 0; j < 3; ++j) {
              const float2 wv = w2[j * 32 + lane];
              acc[j].x += wv.x; acc[j].y += wv.y;
            }
            if (ch0) {
              const float2* w0 = reinterpret_cast<const float2*>(ws + (t * 3 + 0) * WS_STRIDE);
#pragma unroll
              for (int j = 0; j < 3; ++j) {
                const float2 wv = w0[j * 32 + lane];
                acc[j].x += wv.x; acc[j].y += wv.y;
              }
            }
            if (ch1) {
              const float2* w1 = reinterpret_cast<const float2*>(ws + (t * 3 + 1) * WS_STRIDE);
#pragma unroll
              for (int j = 0; j < 3; ++j) {
                const float2 wv = w1[j * 32 + lane];
                acc[j].x += wv.x; acc[j].y += wv.y;
              }
            }
          }
        }
        __nv_bfloat162* xdst = reinterpret_cast<__nv_bfloat162*>(my_X + cell * K_C);
        float sq = 0.f;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          const __nv_bfloat162 xb = __floats2bfloat162_rn(acc[j].x, acc[j].y);
          xdst[j * 32 + lane] = xb;
          acc[j] = __bfloat1622float2(xb);
          sq = fmaf(acc[j].x, acc[j].x, fmaf(acc[j].y, acc[j].y, sq));
        }
        const float rstd = rsqrtf(kt_warp_sum(sq) * (1.0f / K_C) + K_RMS_EPS);
        float amax = 0.f;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          const int p = j * 32 + lane;
          const float2 ns0 = nsb2[p], nb0 = nsb2[K_C2 + p];
          acc[j].x = fmaf(acc[j].x * rstd, ns0.x, nb0.x);
          acc[j].y = fmaf(acc[j].y * rstd, ns0.y, nb0.y);
          amax = fmaxf(amax, fmaxf(fabsf(acc[j].x), fabsf(acc[j].y)));
        }
        const float gmax = fmaxf(kt_warp_max(amax), 1e-6f);
        const float inv_sc = 127.0f / gmax;
        uint16_t* adst = reinterpret_cast<uint16_t*>(my_act_s8 + cell * K_C);
#pragma unroll
        for (int j = 0; j < 3; ++j) adst[j * 32 + lane] = kt_pack_s8x2(acc[j].x, acc[j].y, inv_sc);
        if (lane == 0) my_act_sc[cell] = gmax * (1.0f / 127.0f);
      }
      __syncthreads();
    }

    // ---- 2. L Flat Transformer Blocks (2-half MW=1, 2 CTAs/SM!) ----
    for (int i = 0; i < L; ++i) {
      const int8_t* wq_i = P.w_qkv + (size_t)i * 3 * K_C * K_C;
      const int8_t* wk_i = wq_i + (size_t)K_C * K_C;
      const int8_t* wv_i = wk_i + (size_t)K_C * K_C;
      const float* sq_i = P.s_qkv + (size_t)i * 3 * K_C;
      const float* sk_i = sq_i + K_C;
      const float* sv_i = sk_i + K_C;
      const int8_t* wo_i = P.w_out + (size_t)i * K_C * K_C;
      const float* so_i = P.s_out + (size_t)i * K_C;
      const float* qkn_i = P.qk_norm_sb + (size_t)i * 4 * K_D;
      const float* rf_i = P.rope_freqs + (size_t)i * K_H * 16 * 2;
      const float* fp8_sc = P.qkv_fp8_scales + (size_t)i * 14;
      const float q_fp8_sc = fp8_sc[0];
      const float k_fp8_sc = fp8_sc[1];
      const float s_unscale = K_INV_SQRT_D_LOG2E / (q_fp8_sc * k_fp8_sc);

      // 2-head streaming MHA across Half 0 (0..63) and Half 1 (64..cells-1)
      for (int hp = 0; hp < 3; ++hp) {
        const int h = hp * 2 + wn;
        const float* rf_h = rf_i + h * 16 * 2;
        const float v_fp8_sc = fp8_sc[2 + h];
        const float inv_v_fp8_sc = fp8_sc[8 + h];

        // Fill k_smem & v_smem for both halves (Half 0, then Half 1!)
        for (int half = 0; half < 2; ++half) {
          const int row0 = half * 64;
          const int gr0 = row0 + my_r0, gr1 = row0 + my_r1;
          load_act_s8_half(row0);
          const float xs0 = (gr0 < cells) ? act_sc[my_r0] : 0.f;
          const float xs1 = (gr1 < cells) ? act_sc[my_r1] : 0.f;

          int kacc[4][4];
          gemm_192_to_64(wk_i + (size_t)(hp * 64) * K_C, kacc);
          float2 k0[4], k1[4];
          apply_qknorm_rope_h(row0, kacc, xs0, xs1, sk_i + hp * 64, qkn_i + 2 * K_D, qkn_i + 3 * K_D, rf_h, k0, k1);
          uint8_t* kh_sm = k_smem + wn * SKV_FP8 * K_D;
#pragma unroll
          for (int nj = 0; nj < 4; ++nj) {
            const int d = nj * 8 + c_in_n8;
            const uint32_t p0_lo = (gr0 < cells) ? kt_pack_e4m3x2(k0[nj].x * k_fp8_sc, k0[nj].y * k_fp8_sc) : 0u;
            const uint32_t p0_hi = __shfl_xor_sync(0xffffffffu, p0_lo, 1);
            const uint32_t p1_lo = (gr1 < cells) ? kt_pack_e4m3x2(k1[nj].x * k_fp8_sc, k1[nj].y * k_fp8_sc) : 0u;
            const uint32_t p1_hi = __shfl_xor_sync(0xffffffffu, p1_lo, 1);
            if ((lane & 1) == 0) {
              *reinterpret_cast<uint32_t*>(kh_sm + kt_swz_e4m3_32(gr0, d)) = p0_lo | (p0_hi << 16);
              *reinterpret_cast<uint32_t*>(kh_sm + kt_swz_e4m3_32(gr1, d)) = p1_lo | (p1_hi << 16);
            }
          }

          int vacc[4][4];
          gemm_192_to_64(wv_i + (size_t)(hp * 64) * K_C, vacc);
          const float sv_r0 = xs0 * v_fp8_sc;
          const float sv_r1 = xs1 * v_fp8_sc;
#pragma unroll
          for (int nj = 0; nj < 4; ++nj) {
            const int col = wn * 32 + nj * 8 + c_in_n8;
            const int d = nj * 8 + c_in_n8;
            const float w0 = sv_i[hp * 64 + col], w1 = sv_i[hp * 64 + col + 1];
            store_v_fp8(wn, d,     gr0, (gr0 < cells) ? kt_cvt_e4m3((float)vacc[nj][0] * sv_r0 * w0) : 0u);
            store_v_fp8(wn, d + 1, gr0, (gr0 < cells) ? kt_cvt_e4m3((float)vacc[nj][1] * sv_r0 * w1) : 0u);
            store_v_fp8(wn, d,     gr1, (gr1 < cells) ? kt_cvt_e4m3((float)vacc[nj][2] * sv_r1 * w0) : 0u);
            store_v_fp8(wn, d + 1, gr1, (gr1 < cells) ? kt_cvt_e4m3((float)vacc[nj][3] * sv_r1 * w1) : 0u);
          }
        }
        __syncthreads();

        // Run Q + FlashAttn + O-proj for Half 1 FIRST (act_s8 ALREADY holds Half 1!), then Half 0!
        for (int pass = 0; pass < 2; ++pass) {
          const int half = 1 - pass;
          const int row0 = half * 64;
          const int n_rows = min(64, cells - row0);
          if (pass == 1) load_act_s8_half(0);
          const float xs0 = (my_r0 < n_rows) ? act_sc[my_r0] : 0.f;
          const float xs1 = (my_r1 < n_rows) ? act_sc[my_r1] : 0.f;

          __nv_bfloat162 attn_reg[8];
          {
            int qacc[4][4];
            gemm_192_to_64(wq_i + (size_t)(hp * 64) * K_C, qacc);
            float2 q0[4], q1[4];
            apply_qknorm_rope_h(row0, qacc, xs0, xs1, sq_i + hp * 64, qkn_i, qkn_i + K_D, rf_h, q0, q1);
            uint16_t p0[4], p1[4];
#pragma unroll
            for (int nj = 0; nj < 4; ++nj) {
              p0[nj] = kt_pack_e4m3x2(q0[nj].x * q_fp8_sc, q0[nj].y * q_fp8_sc);
              p1[nj] = kt_pack_e4m3x2(q1[nj].x * q_fp8_sc, q1[nj].y * q_fp8_sc);
            }
            uint32_t qa[4];
            qa[0] = kt_pack_qa_half(lane, p0[0], p0[1]);
            qa[1] = kt_pack_qa_half(lane, p1[0], p1[1]);
            qa[2] = kt_pack_qa_half(lane, p0[2], p0[3]);
            qa[3] = kt_pack_qa_half(lane, p1[2], p1[3]);

            const uint32_t k_base_u32 = ksm_u32 + (uint32_t)(wn * SKV_FP8 * K_D);
            const uint32_t v_base_u32 = vsm_u32 + (uint32_t)(wn * V_TILES64 * K_D * 64);

            float m0 = -1e30f, m1 = -1e30f, d0 = 0.f, d1 = 0.f;
            float aacc[4][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};

            for (int t0 = 0; t0 < S_pad32; t0 += 32) {
              uint32_t kb0[4], kb1[4];
              const int kb_row0 = t0 + (lane & 7) + ((lane >> 4) << 3);
              const int kb_ch = (lane >> 3) & 1;
              ldmatrix_x4(kb0, k_base_u32 + kt_swz32(kb_row0, kb_ch));
              ldmatrix_x4(kb1, k_base_u32 + kt_swz32(kb_row0 + 16, kb_ch));

              float s0[4] = {0.f,0.f,0.f,0.f}, s1[4] = {0.f,0.f,0.f,0.f};
              float s2[4] = {0.f,0.f,0.f,0.f}, s3[4] = {0.f,0.f,0.f,0.f};
              mma_e4m3(s0, qa, kb0[0], kb0[1]);
              mma_e4m3(s1, qa, kb0[2], kb0[3]);
              mma_e4m3(s2, qa, kb1[0], kb1[1]);
              mma_e4m3(s3, qa, kb1[2], kb1[3]);
#pragma unroll
              for (int e = 0; e < 4; ++e) {
                s0[e] *= s_unscale; s1[e] *= s_unscale;
                s2[e] *= s_unscale; s3[e] *= s_unscale;
              }
              if (t0 + 32 > cells) {
                const int k0 = t0 + c_in_n8;
                if (k0 >= cells)      { s0[0] = -1e30f; s0[2] = -1e30f; }
                if (k0 + 1 >= cells)  { s0[1] = -1e30f; s0[3] = -1e30f; }
                if (k0 + 8 >= cells)  { s1[0] = -1e30f; s1[2] = -1e30f; }
                if (k0 + 9 >= cells)  { s1[1] = -1e30f; s1[3] = -1e30f; }
                if (k0 + 16 >= cells) { s2[0] = -1e30f; s2[2] = -1e30f; }
                if (k0 + 17 >= cells) { s2[1] = -1e30f; s2[3] = -1e30f; }
                if (k0 + 24 >= cells) { s3[0] = -1e30f; s3[2] = -1e30f; }
                if (k0 + 25 >= cells) { s3[1] = -1e30f; s3[3] = -1e30f; }
              }

              float rm0 = fmaxf(fmaxf(fmaxf(s0[0], s0[1]), fmaxf(s1[0], s1[1])),
                                fmaxf(fmaxf(s2[0], s2[1]), fmaxf(s3[0], s3[1])));
              rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffffu, rm0, 1));
              rm0 = fmaxf(rm0, __shfl_xor_sync(0xffffffffu, rm0, 2));
              float rm1 = fmaxf(fmaxf(fmaxf(s0[2], s0[3]), fmaxf(s1[2], s1[3])),
                                fmaxf(fmaxf(s2[2], s2[3]), fmaxf(s3[2], s3[3])));
              rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffffu, rm1, 1));
              rm1 = fmaxf(rm1, __shfl_xor_sync(0xffffffffu, rm1, 2));

              const float mn0 = fmaxf(m0, rm0), mn1 = fmaxf(m1, rm1);
              const float a0 = exp2f(m0 - mn0), a1 = exp2f(m1 - mn1);
              const float b0_s = 8.0f - mn0, b1_s = 8.0f - mn1;
              const float p00 = exp2f(s0[0] + b0_s), p01 = exp2f(s0[1] + b0_s);
              const float p10 = exp2f(s1[0] + b0_s), p11 = exp2f(s1[1] + b0_s);
              const float p20 = exp2f(s2[0] + b0_s), p21 = exp2f(s2[1] + b0_s);
              const float p30 = exp2f(s3[0] + b0_s), p31 = exp2f(s3[1] + b0_s);
              const float p02 = exp2f(s0[2] + b1_s), p03 = exp2f(s0[3] + b1_s);
              const float p12 = exp2f(s1[2] + b1_s), p13 = exp2f(s1[3] + b1_s);
              const float p22 = exp2f(s2[2] + b1_s), p23 = exp2f(s2[3] + b1_s);
              const float p32 = exp2f(s3[2] + b1_s), p33 = exp2f(s3[3] + b1_s);

              float rs0 = ((p00 + p01) + (p10 + p11)) + ((p20 + p21) + (p30 + p31));
              rs0 += __shfl_xor_sync(0xffffffffu, rs0, 1);
              rs0 += __shfl_xor_sync(0xffffffffu, rs0, 2);
              float rs1 = ((p02 + p03) + (p12 + p13)) + ((p22 + p23) + (p32 + p33));
              rs1 += __shfl_xor_sync(0xffffffffu, rs1, 1);
              rs1 += __shfl_xor_sync(0xffffffffu, rs1, 2);

              d0 = fmaf(d0, a0, rs0); d1 = fmaf(d1, a1, rs1);
              m0 = mn0; m1 = mn1;
#pragma unroll
              for (int nj = 0; nj < 4; ++nj) {
                aacc[nj][0] *= a0; aacc[nj][1] *= a0;
                aacc[nj][2] *= a1; aacc[nj][3] *= a1;
              }
              uint32_t p_reg[4];
              p_reg[0] = kt_pack_e4m3x4(p00, p01, p10, p11);
              p_reg[1] = kt_pack_e4m3x4(p02, p03, p12, p13);
              p_reg[2] = kt_pack_e4m3x4(p20, p21, p30, p31);
              p_reg[3] = kt_pack_e4m3x4(p22, p23, p32, p33);

              const uint32_t vt_tile = v_base_u32 + (uint32_t)((t0 >> 6) * K_D * 64);
              const int v_ch = ((t0 & 32) >> 4) + ((lane >> 3) & 1);
              uint32_t vb0[4], vb1[4];
              const int vd0 = (lane & 7) + ((lane >> 4) << 3);
              ldmatrix_x4(vb0, vt_tile + swz(vd0, v_ch));
              ldmatrix_x4(vb1, vt_tile + swz(16 + vd0, v_ch));

              mma_e4m3(aacc[0], p_reg, vb0[0], vb0[1]);
              mma_e4m3(aacc[1], p_reg, vb0[2], vb0[3]);
              mma_e4m3(aacc[2], p_reg, vb1[0], vb1[1]);
              mma_e4m3(aacc[3], p_reg, vb1[2], vb1[3]);
            }
            const float inv_d0 = inv_v_fp8_sc / d0;
            const float inv_d1 = inv_v_fp8_sc / d1;
#pragma unroll
            for (int nj = 0; nj < 4; ++nj) {
              attn_reg[nj] = __floats2bfloat162_rn(aacc[nj][0] * inv_d0, aacc[nj][1] * inv_d0);
              attn_reg[4 + nj] = __floats2bfloat162_rn(aacc[nj][2] * inv_d1, aacc[nj][3] * inv_d1);
            }
          }

          float oxs0, oxs1;
          quant_regs_to_sub_s8(attn_reg, n_rows, oxs0, oxs1);
          float oacc[12][4];
#pragma unroll
          for (int nj = 0; nj < 12; ++nj)
#pragma unroll
            for (int e = 0; e < 4; ++e) oacc[nj][e] = 0.f;
          accum_64_to_192(wo_i, K_C, hp * 64, oxs0, oxs1, oacc);

          // Coalesced O-proj residual add into L2 my_X via 8 KB w_stage (0 uncoalesced sectors, 0 my_tmp traffic!)
          __nv_bfloat162* st2 = reinterpret_cast<__nv_bfloat162*>(w_stage);
#pragma unroll
          for (int rc = 0; rc < 3; ++rc) {
#pragma unroll
            for (int nj = 0; nj < 4; ++nj) {
              const int idx = rc * 4 + nj;
              const int col = rc * 64 + wn * 32 + nj * 8 + c_in_n8;
              const int p32 = (wn * 32 + nj * 8 + c_in_n8) >> 1;
              const float ws0 = so_i[col], ws1 = so_i[col + 1];
              if (my_r0 < n_rows) {
                st2[my_r0 * 32 + (p32 ^ ((my_r0 & 7) << 2))] = __floats2bfloat162_rn(oacc[idx][0] * ws0, oacc[idx][1] * ws1);
              }
              if (my_r1 < n_rows) {
                st2[my_r1 * 32 + (p32 ^ ((my_r1 & 7) << 2))] = __floats2bfloat162_rn(oacc[idx][2] * ws0, oacc[idx][3] * ws1);
              }
            }
            __syncthreads();
            for (int r = warp; r < n_rows; r += 8) {
              __nv_bfloat162* xp = reinterpret_cast<__nv_bfloat162*>(my_X + (row0 + r) * K_C + rc * 64) + lane;
              const float2 cur = __bfloat1622float2(*xp);
              const float2 add = __bfloat1622float2(st2[r * 32 + (lane ^ ((r & 7) << 2))]);
              *xp = __floats2bfloat162_rn(cur.x + add.x, cur.y + add.y);
            }
            __syncthreads();
          }
        }
      }

      // FFN RMSNorm(my_X) -> my_act_s8 (in L2)
      norm_l2_x_to_l2_act(P.ffn_norm_sb + (size_t)i * 2 * K_C);

      // SwiGLU FFN across Half 0 & Half 1 (ONLY 1 act_s8 load per half!)
      const int8_t* wgu_i = P.w_gu + (size_t)i * (2 * FFN) * K_C;
      const float* sgu_i = P.s_gu + (size_t)i * (2 * FFN);
      const int8_t* wd_i = P.w_down + (size_t)i * K_C * FFN;
      const float* sd_i = P.s_down + (size_t)i * K_C;
      const int n_chunks = FFN >> 6;

      for (int half = 0; half < 2; ++half) {
        const int row0 = half * 64;
        const int n_rows = min(64, cells - row0);
        load_act_s8_half(row0);
        const float mxs0 = (my_r0 < n_rows) ? act_sc[my_r0] : 0.f;
        const float mxs1 = (my_r1 < n_rows) ? act_sc[my_r1] : 0.f;

        float down_acc[12][4];
#pragma unroll
        for (int nj = 0; nj < 12; ++nj)
#pragma unroll
          for (int e = 0; e < 4; ++e) down_acc[nj][e] = 0.f;

        for (int cc = 0; cc < n_chunks; ++cc) {
          __nv_bfloat162 swi_bf[8];
          {
            int gacc[4][4], uacc[4][4];
            gemm_192_to_64x2(wgu_i + (size_t)(cc * 128) * K_C, gacc, uacc);
            const float* sg_c = sgu_i + cc * 128;
            const float* su_c = sg_c + 64;
#pragma unroll
            for (int nj = 0; nj < 4; ++nj) {
              const int col = wn * 32 + nj * 8 + c_in_n8;
              const float wg0 = sg_c[col], wg1 = sg_c[col + 1];
              const float wu0 = su_c[col], wu1 = su_c[col + 1];
              if (my_r0 < n_rows) {
                swi_bf[nj] = __floats2bfloat162_rn(
                    kt_silu((float)gacc[nj][0] * mxs0 * wg0) * ((float)uacc[nj][0] * mxs0 * wu0),
                    kt_silu((float)gacc[nj][1] * mxs0 * wg1) * ((float)uacc[nj][1] * mxs0 * wu1));
              }
              if (my_r1 < n_rows) {
                swi_bf[4 + nj] = __floats2bfloat162_rn(
                    kt_silu((float)gacc[nj][2] * mxs1 * wg0) * ((float)uacc[nj][2] * mxs1 * wu0),
                    kt_silu((float)gacc[nj][3] * mxs1 * wg1) * ((float)uacc[nj][3] * mxs1 * wu1));
              }
            }
          }
          float dxs0, dxs1;
          quant_regs_to_sub_s8(swi_bf, n_rows, dxs0, dxs1);
          accum_64_to_192(wd_i, FFN, cc * 64, dxs0, dxs1, down_acc);
        }

        // Coalesced FFN residual add into L2 my_X via 8 KB w_stage (0 uncoalesced sectors!)
        {
          __nv_bfloat162* st2 = reinterpret_cast<__nv_bfloat162*>(w_stage);
#pragma unroll
          for (int rc = 0; rc < 3; ++rc) {
#pragma unroll
            for (int nj = 0; nj < 4; ++nj) {
              const int idx = rc * 4 + nj;
              const int col = rc * 64 + wn * 32 + nj * 8 + c_in_n8;
              const int p32 = (wn * 32 + nj * 8 + c_in_n8) >> 1;
              const float ws0 = sd_i[col], ws1 = sd_i[col + 1];
              if (my_r0 < n_rows) {
                st2[my_r0 * 32 + (p32 ^ ((my_r0 & 7) << 2))] = __floats2bfloat162_rn(down_acc[idx][0] * ws0, down_acc[idx][1] * ws1);
              }
              if (my_r1 < n_rows) {
                st2[my_r1 * 32 + (p32 ^ ((my_r1 & 7) << 2))] = __floats2bfloat162_rn(down_acc[idx][2] * ws0, down_acc[idx][3] * ws1);
              }
            }
            __syncthreads();
            for (int r = warp; r < n_rows; r += 8) {
              __nv_bfloat162* xp = reinterpret_cast<__nv_bfloat162*>(my_X + (row0 + r) * K_C + rc * 64) + lane;
              const float2 cur = __bfloat1622float2(*xp);
              const float2 add = __bfloat1622float2(st2[r * 32 + (lane ^ ((r & 7) << 2))]);
              *xp = __floats2bfloat162_rn(cur.x + add.x, cur.y + add.y);
            }
            __syncthreads();
          }
        }
      }

      if (i + 1 < L) {
        norm_l2_x_to_l2_act(P.attn_norm_sb + (size_t)(i + 1) * 2 * K_C);
      }
    }

    // ---- 3. Final RMSNorm + SiLU (in-place on my_X -> my_tmp) + 2-half Parallel KataGo Heads ----
    {
      const float2* fnsb2 = reinterpret_cast<const float2*>(P.heads.final_norm_sb);
      float2 fns[3], fnb[3];
#pragma unroll
      for (int j = 0; j < 3; ++j) {
        const int p = j * 32 + lane;
        fns[j] = fnsb2[p]; fnb[j] = fnsb2[K_C2 + p];
      }
      for (int cell = warp; cell < cells; cell += 8) {
        const __nv_bfloat162* xp = reinterpret_cast<const __nv_bfloat162*>(my_X + cell * K_C);
        float2 v[3];
        float sq = 0.f;
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          v[j] = __bfloat1622float2(xp[j * 32 + lane]);
          sq = fmaf(v[j].x, v[j].x, fmaf(v[j].y, v[j].y, sq));
        }
        const float rstd = rsqrtf(kt_warp_sum(sq) * (1.0f / K_C) + K_RMS_EPS);
        __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(my_tmp + cell * K_C);
#pragma unroll
        for (int j = 0; j < 3; ++j) {
          dst[j * 32 + lane] = __floats2bfloat162_rn(
              kt_silu(fmaf(v[j].x * rstd, fns[j].x, fnb[j].x)),
              kt_silu(fmaf(v[j].y * rstd, fns[j].y, fnb[j].y)));
        }
      }
      __syncthreads();

      float* g_mean = h_stats;
      float* g_max = g_mean + 48;
      float* v_mean = g_max + 48;
      float* v_max = v_mean + 48;
      float* w_sum = v_max + 48;
      float* w_max = w_sum + 8 * 48;
      const float inv_cells = 1.0f / (float)cells;

      auto run_conv1_tc_h = [&](int row0_w, int n_rows, __nv_bfloat16* dst) {
        const int lrow = tid >> 2, lchunk = tid & 3;
        auto load_w = [&](int st, int kp) {
          const bool ok = lrow < K_HEAD_C;
          cp_async16(hwst_u32 + (st & 1) * 4096 + swz(lrow, lchunk),
                     P.heads.conv1pgv + (size_t)(row0_w + (ok ? lrow : 0)) * K_C + kp * 32 + lchunk * 8, ok);
        };
        float facc[4][4] = {{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f},{0.f,0.f,0.f,0.f}};
        load_w(0, 0);
        cp_async_commit();
#pragma unroll
        for (int kp = 0; kp < 6; ++kp) {
          cp_async_wait<0>();
          __syncthreads();
          if (kp + 1 < 6) { load_w(kp + 1, kp + 1); cp_async_commit(); }
          const uint32_t b_tile = hwst_u32 + (kp & 1) * 4096;
          const uint32_t a_panel = hfin_u32 + (uint32_t)(kp * ROWS * 64);
#pragma unroll
          for (int ks = 0; ks < 2; ++ks) {
            uint32_t a[4];
            ldmatrix_x4(a, a_panel + swz(wm * 16 + (lane & 15), ks * 2 + (lane >> 4)));
            uint32_t b[2][4];
#pragma unroll
            for (int np = 0; np < 2; ++np) {
              const int brow = wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3);
              ldmatrix_x4(b[np], b_tile + swz(brow, ks * 2 + ((lane >> 3) & 1)));
            }
#pragma unroll
            for (int nj = 0; nj < 4; ++nj) mma_bf16(facc[nj], a, b[nj >> 1][(nj & 1) * 2], b[nj >> 1][(nj & 1) * 2 + 1]);
          }
        }
        __syncthreads();
#pragma unroll
        for (int nj = 0; nj < 4; ++nj) {
          const int col = wn * 32 + nj * 8 + c_in_n8;
          if (col < K_HEAD_C) {
            if (my_r0 < n_rows) *reinterpret_cast<__nv_bfloat162*>(dst + my_r0 * K_HEAD_C + col) =
                __floats2bfloat162_rn(facc[nj][0], facc[nj][1]);
            if (my_r1 < n_rows) *reinterpret_cast<__nv_bfloat162*>(dst + my_r1 * K_HEAD_C + col) =
                __floats2bfloat162_rn(facc[nj][2], facc[nj][3]);
          }
        }
        __syncthreads();
      };

      auto accum_gv_half = [&](int norm_row_s, int norm_row_b, int n_rows) {
        const float ns0 = P.heads.head_norm_sb[norm_row_s * 48 + lane];
        const float nb0 = P.heads.head_norm_sb[norm_row_b * 48 + lane];
        const float ns1 = (lane < 16) ? P.heads.head_norm_sb[norm_row_s * 48 + 32 + lane] : 0.f;
        const float nb1 = (lane < 16) ? P.heads.head_norm_sb[norm_row_b * 48 + 32 + lane] : 0.f;
        for (int cell = warp; cell < n_rows; cell += 8) {
          const float v0 = __bfloat162float(h_gv_tmp[cell * K_HEAD_C + lane]);
          const float v1 = (lane < 16) ? __bfloat162float(h_gv_tmp[cell * K_HEAD_C + 32 + lane]) : 0.f;
          const float rstd = rsqrtf(kt_warp_sum(v0 * v0 + v1 * v1) * (1.0f / K_HEAD_C) + K_RMS_EPS);
          const float o0 = kt_silu(fmaf(v0 * rstd, ns0, nb0));
          w_sum[warp * 48 + lane] += o0;
          w_max[warp * 48 + lane] = fmaxf(w_max[warp * 48 + lane], o0);
          if (lane < 16) {
            const float o1 = kt_silu(fmaf(v1 * rstd, ns1, nb1));
            w_sum[warp * 48 + 32 + lane] += o1;
            w_max[warp * 48 + 32 + lane] = fmaxf(w_max[warp * 48 + 32 + lane], o1);
          }
        }
        __syncthreads();
      };

      auto load_hfinal_half = [&](int row0, int n_rows) {
        const uint4* src4 = reinterpret_cast<const uint4*>(my_tmp + row0 * K_C);
        for (int idx = tid; idx < n_rows * 24; idx += 256) {
          const int r = idx / 24, rem = idx - r * 24;
          *reinterpret_cast<uint4*>(h_final_swz + (rem >> 2) * ROWS * 64 + swz(r, rem & 3)) = src4[idx];
        }
        for (int idx = tid; idx < (ROWS - n_rows) * 24; idx += 256) {
          const int r = n_rows + idx / 24, rem = idx - (r - n_rows) * 24;
          *reinterpret_cast<uint4*>(h_final_swz + (rem >> 2) * ROWS * 64 + swz(r, rem & 3)) = make_uint4(0u, 0u, 0u, 0u);
        }
        __syncthreads();
      };

      // Pass G (conv1p + conv1g across both halves)
      for (int idx = tid; idx < 8 * 48; idx += 256) { w_sum[idx] = 0.f; w_max[idx] = -1e30f; }
      __syncthreads();
      for (int half = 0; half < 2; ++half) {
        const int row0 = half * 64, n_rows = min(64, cells - row0);
        load_hfinal_half(row0, n_rows);
        run_conv1_tc_h(0, n_rows, h_outp + row0 * K_HEAD_C);
        run_conv1_tc_h(48, n_rows, h_gv_tmp);
        accum_gv_half(0, 1, n_rows);
      }
      if (tid < 48) {
        float s = 0.f, m = -1e30f;
#pragma unroll
        for (int w = 0; w < 8; ++w) { s += w_sum[w * 48 + tid]; m = fmaxf(m, w_max[w * 48 + tid]); }
        g_mean[tid] = s * inv_cells; g_max[tid] = m;
      }
      __syncthreads();

      // Pass V (conv1v across both halves; Half 1 is ALREADY in h_final_swz!)
      for (int idx = tid; idx < 8 * 48; idx += 256) { w_sum[idx] = 0.f; w_max[idx] = -1e30f; }
      __syncthreads();
      for (int pass = 0; pass < 2; ++pass) {
        const int half = 1 - pass;
        const int row0 = half * 64, n_rows = min(64, cells - row0);
        if (pass == 1) load_hfinal_half(0, n_rows);
        run_conv1_tc_h(96, n_rows, h_gv_tmp);
        accum_gv_half(4, 5, n_rows);
      }
      if (tid < 48) {
        float s = 0.f, m = -1e30f;
#pragma unroll
        for (int w = 0; w < 8; ++w) { s += w_sum[w * 48 + tid]; m = fmaxf(m, w_max[w * 48 + tid]); }
        v_mean[tid] = s * inv_cells; v_max[tid] = m;
      }
      __syncthreads();

      // Parallel dense head compute with 100% coalesced transposed weight reads!
      float* g_bias = reinterpret_cast<float*>(h_final_swz);
      float* val_h = g_bias + 48;
      float* v_out_s = val_h + K_VAL_H;
      float* s_pol_w = v_out_s + 16;
      float* s_q_w = s_pol_w + 48;
      float* s_q_b = s_q_w + 16 * 48;
      float* s_pq = s_q_b + 16;
      float* s_ql = s_pq + 16;

      if (tid < 48) s_pol_w[tid] = P.heads.pol_w[tid];
      if (tid < K) s_q_b[tid] = P.heads.q_b[tid];
      for (int idx = tid; idx < K * 48; idx += 256) s_q_w[idx] = P.heads.q_w[idx];

      if (tid < 48) {
        float acc = 0.f;
        for (int j = 0; j < 48; ++j) acc = fmaf(g_mean[j], P.heads.linear_g_w[j * 48 + tid], acc);
        for (int j = 0; j < 48; ++j) acc = fmaf(g_max[j], P.heads.linear_g_w[(48 + j) * 48 + tid], acc);
        g_bias[tid] = acc;
      } else if (tid < 48 + K_VAL_H) {
        const int o = tid - 48;
        float acc = P.heads.val_l1_b[o];
        for (int j = 0; j < 48; ++j) acc = fmaf(v_mean[j], P.heads.val_l1_w[j * K_VAL_H + o], acc);
        for (int j = 0; j < 48; ++j) acc = fmaf(v_max[j], P.heads.val_l1_w[(48 + j) * K_VAL_H + o], acc);
        val_h[o] = kt_silu(acc);
      } else if (A == cells + 1 && tid >= 176 && tid < 176 + K) {
        const int k = tid - 176;
        float acc = P.heads.pass_q_b[k];
        for (int j = 0; j < 48; ++j) acc = fmaf(g_mean[j], P.heads.pass_q_w[j * K + k], acc);
        for (int j = 0; j < 48; ++j) acc = fmaf(g_max[j], P.heads.pass_q_w[(48 + j) * K + k], acc);
        s_pq[k] = acc;
      } else if (A == cells + 1 && P.logits && tid == 192) {
        float pa = P.heads.pass_pol_b[0];
        for (int j = 0; j < 48; ++j) pa = fmaf(g_mean[j], P.heads.pass_pol_w[j], pa);
        for (int j = 0; j < 48; ++j) pa = fmaf(g_max[j], P.heads.pass_pol_w[48 + j], pa);
        P.logits[(size_t)board * A + cells] = pa;
      }
      __syncthreads();

      if (tid < K) {
        float acc = P.heads.val_out_b[tid];
        for (int j = 0; j < K_VAL_H; ++j) acc = fmaf(val_h[j], P.heads.val_out_w[j * K + tid], acc);
        v_out_s[tid] = acc;
      }
      if (A == cells + 1 && warp == 1) {
        const float v = (lane < K) ? s_pq[lane] : -1e30f;
        const float mx = kt_warp_max(v);
        const float e = (lane < K) ? __expf(v - mx) : 0.f;
        const float tot = kt_warp_sum(e);
        if (lane < K) P.qprobs[((size_t)board * A + cells) * K + lane] = e / tot;
      }
      __syncwarp();
      if (warp == 0) {
        const float v = (lane < K) ? v_out_s[lane] : -1e30f;
        const float mx = kt_warp_max(v);
        const float e = (lane < K) ? __expf(v - mx) : 0.f;
        const float tot = kt_warp_sum(e);
        if (lane < K) P.vprobs[(size_t)board * K + lane] = e / tot;
      }

      const float p_ns0 = P.heads.head_norm_sb[2 * 48 + lane];
      const float p_nb0 = P.heads.head_norm_sb[3 * 48 + lane];
      const float p_ns1 = (lane < 16) ? P.heads.head_norm_sb[2 * 48 + 32 + lane] : 0.f;
      const float p_nb1 = (lane < 16) ? P.heads.head_norm_sb[3 * 48 + 32 + lane] : 0.f;
      float* w_ql = s_ql + warp * 16;

      for (int cell = warp; cell < cells; cell += 8) {
        float hp0 = __bfloat162float(h_outp[cell * K_HEAD_C + lane]) + g_bias[lane];
        float hp1 = (lane < 16) ? (__bfloat162float(h_outp[cell * K_HEAD_C + 32 + lane]) + g_bias[32 + lane]) : 0.f;
        const float prstd = rsqrtf(kt_warp_sum(hp0 * hp0 + hp1 * hp1) * (1.0f / K_HEAD_C) + K_RMS_EPS);
        hp0 = kt_silu(fmaf(hp0 * prstd, p_ns0, p_nb0));
        hp1 = (lane < 16) ? kt_silu(fmaf(hp1 * prstd, p_ns1, p_nb1)) : 0.f;

        const int y = (int)cell_y[cell], x = (int)cell_x[cell];
        const int dst_action = is_white ? (x * side + y) : cell;

        if (P.logits) {
          float pdot = hp0 * s_pol_w[lane];
          if (lane < 16) pdot = fmaf(hp1, s_pol_w[32 + lane], pdot);
          const float pl = kt_warp_sum(pdot) + P.heads.pol_b[0];
          if (lane == 0) P.logits[(size_t)board * A + dst_action] = pl;
        }

        float qmx = -1e30f;
        for (int k = 0; k < K; ++k) {
          const float* wqk = s_q_w + k * K_HEAD_C;
          float qdot = hp0 * wqk[lane];
          if (lane < 16) qdot = fmaf(hp1, wqk[32 + lane], qdot);
          const float v = kt_warp_sum(qdot) + s_q_b[k];
          if (lane == 0) w_ql[k] = v;
          qmx = fmaxf(qmx, v);
        }
        if (lane == 0) {
          float qtot = 0.f;
          for (int k = 0; k < K; ++k) { w_ql[k] = __expf(w_ql[k] - qmx); qtot += w_ql[k]; }
          const float qinv = 1.0f / qtot;
          float* qdst = P.qprobs + ((size_t)board * A + dst_action) * K;
          for (int k = 0; k < K; ++k) qdst[k] = w_ql[k] * qinv;
        }
      }
      __syncthreads();

      if (P.cw.table_locks) {
        const CacheKey mk = P.cw.miss_keys[board];
        if (mk.meta & 0x80000000u) {
          const uint32_t tag = 2u + *P.cw.cache_wave;
          const uint32_t slot = hash_cache_key(mk);
          if (tid == 0) won_slot = cache_try_acquire(P.cw.table_locks, slot, tag);
          __syncthreads();
          if (won_slot) {
            if (tid == 0) P.cw.table_keys[slot] = mk;
            if (tid < K) P.cw.table_v[(size_t)slot * K + tid] = __float2bfloat16_rn(P.vprobs[(size_t)board * K + tid]);
            const int AK = A * K;
            for (int idx = tid; idx < AK; idx += 256) {
              P.cw.table_q[(size_t)slot * AK + idx] = __float2bfloat16_rn(P.qprobs[(size_t)board * AK + idx]);
            }
            __syncthreads();
            if (tid == 0) { __threadfence(); atomicExch(&P.cw.table_locks[slot], tag); }
          }
        }
      }
      __syncthreads();
    }
  }
}

inline cudaError_t launch_kata_tf_board_int8(const KataTfParams& params, cudaStream_t stream) {
  static int cached_sms = 0;
  if (cached_sms == 0) {
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    cached_sms = (sms > 0) ? sms : 1;
  }
  const int grid = std::min(params.max_boards, cached_sms * 2);
  if (grid <= 0) return cudaSuccess;

  if (params.cells <= 64) {
    constexpr int smem = kata_tf_smem_bytes_1h();
    static bool attr_set_1h = false;
    if (!attr_set_1h) {
      cudaFuncSetAttribute(kata_tf_board_int8_kernel_1h, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      attr_set_1h = true;
    }
    kata_tf_board_int8_kernel_1h<<<grid, 256, smem, stream>>>(params);
  } else {
    constexpr int smem = kata_tf_smem_bytes_2h();
    static bool attr_set_2h = false;
    if (!attr_set_2h) {
      cudaFuncSetAttribute(kata_tf_board_int8_kernel_2h, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
      attr_set_2h = true;
    }
    kata_tf_board_int8_kernel_2h<<<grid, 256, smem, stream>>>(params);
  }
  return cudaGetLastError();
}

}  // namespace scacchi_kata_tf
