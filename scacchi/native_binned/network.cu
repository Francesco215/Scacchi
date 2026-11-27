// The binned AlphaZero network (scacchi.network.AZBinnedNet, ResNet v2, 128
// channels) for Hex positions, inference only.
//
// Modes:
//   - kernel == 0 (bf16) and kernel == 1 (fp8), or kernel == 2 (int8) on sides
//     12..15: 14 CUDA kernels (stem + 12 convs + heads).
//   - kernel == 2 (int8) on sides 3..11: single-kernel board-resident INT8
//     megakernel (`conv_board_int8.cuh`), keeping X and Zq in shared memory
//     across stem + all 12 residual convolutions + heads with zero intermediate
//     global memory traffic.
//   - ScacchiBinnedNetworkCompact: same single-kernel INT8 megakernel driven by
//     a device-side compacted board count `*d_boards` (with optional lock-free
//     GPU position-cache writeback), CUDA Graph / XLA command-buffer safe.

#include <algorithm>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdint>

#include "xla/ffi/api/ffi.h"
#include <cuda_fp8.h>

#include "conv_mma.cuh"
#include "conv_int8.cuh"
#include "conv_board_int8.cuh"

namespace ffi = xla::ffi;

namespace {

using namespace scacchi_net;

constexpr int BLOCKS = 6;
constexpr int CONVS = 2 * BLOCKS;
constexpr int STEM_BIAS = 0;
__host__ __device__ constexpr int bn1_row(int i) { return 1 + 8 * i; }
__host__ __device__ constexpr int b1_row(int i) { return 4 + 8 * i; }
__host__ __device__ constexpr int bn2_row(int i) { return 5 + 8 * i; }
__host__ __device__ constexpr int b2_row(int i) { return 8 + 8 * i; }
constexpr int FINAL_BN = 1 + 8 * BLOCKS;
constexpr int VEC_ROWS = FINAL_BN + 3;  // 52
constexpr int HEAD_CH = 5;

constexpr int HEADS_SMEM = 96 * 1024;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float bn_relu(float v, float mean, float mul, float beta) {
  return fmaxf(bf(fmaf(v - mean, mul, beta)), 0.f);
}

// ---- stem: 3x3 convolution 4 -> 128 of the observation -----------------------
__global__ void __launch_bounds__(256) stem_kernel(const uint8_t* __restrict__ obs, const __nv_bfloat16* __restrict__ w,
                                                   const float* __restrict__ vec, int rows, int side, int cells,
                                                   __nv_bfloat16* __restrict__ x_out, __nv_bfloat16* __restrict__ z_out,
                                                   uint8_t* __restrict__ zq_out, const float* __restrict__ qinv,
                                                   const int32_t* __restrict__ d_boards = nullptr) {
  const int actual_rows = d_boards ? (*d_boards * cells) : rows;
  if (actual_rows <= 0) return;
  __shared__ __align__(16) float ws[36 * CH];
  for (int i = threadIdx.x; i < CH * 36; i += blockDim.x) {
    const int n = i / 36, k = i - n * 36;
    ws[k * CH + n] = __bfloat162float(w[i]);
  }
  __syncthreads();
  const int n0 = (threadIdx.x & 15) * 8;
  for (int r = blockIdx.x * 16 + (threadIdx.x >> 4); r < actual_rows; r += gridDim.x * 16) {
    const int board = r / cells, cell = r - board * cells, y = cell / side, x = cell - y * side;
    float acc[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) acc[j] = 0.f;
    for (int t = 0; t < 9; ++t) {
      const int yy = y + t / 3 - 1, xx = x + t % 3 - 1;
      if (yy < 0 || yy >= side || xx < 0 || xx >= side) continue;
      const uint32_t o = *reinterpret_cast<const uint32_t*>(obs + (size_t)(board * cells + yy * side + xx) * 4);
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        if (!((o >> (8 * c)) & 0xffu)) continue;
        const float4* wr = reinterpret_cast<const float4*>(ws + (t * 4 + c) * CH + n0);
        const float4 w0 = wr[0], w1 = wr[1];
        acc[0] += w0.x; acc[1] += w0.y; acc[2] += w0.z; acc[3] += w0.w;
        acc[4] += w1.x; acc[5] += w1.y; acc[6] += w1.z; acc[7] += w1.w;
      }
    }
    __align__(16) __nv_bfloat16 xo[8], zo[8];
#pragma unroll
    for (int j = 0; j < 8; ++j) {
      const int n = n0 + j;
      const float v = bf(bf(acc[j]) + vec[STEM_BIAS * CH + n]);
      xo[j] = __float2bfloat16_rn(v);
      zo[j] = __float2bfloat16_rn(bn_relu(v, vec[bn1_row(0) * CH + n], vec[(bn1_row(0) + 1) * CH + n], vec[(bn1_row(0) + 2) * CH + n]));
    }
    *reinterpret_cast<uint4*>(x_out + (size_t)r * CH + n0) = *reinterpret_cast<const uint4*>(xo);
    if (zq_out) {
      uint32_t q[2] = {0u, 0u};
#pragma unroll
      for (int j = 0; j < 8; ++j) {
        const float z = __bfloat162float(zo[j]);
        const uint32_t byte = qinv ? i8_quant(z, qinv[n0 + j]) : (uint32_t)__nv_cvt_float_to_fp8(z, __NV_SATFINITE, __NV_E4M3);
        q[j >> 2] |= byte << (8 * (j & 3));
      }
      *reinterpret_cast<uint2*>(zq_out + (size_t)r * CH + n0) = make_uint2(q[0], q[1]);
    } else {
      *reinterpret_cast<uint4*>(z_out + (size_t)r * CH + n0) = *reinterpret_cast<const uint4*>(zo);
    }
  }
}

constexpr int STEM_CTAS_PER_SM = 4;
constexpr int HEADS_MIN_CTAS_PER_SM = 4;

// ---- heads ----------------------------------------------------------------------
struct Heads {
  const __nv_bfloat16* conv_w;
  const float* conv_v;
  const __nv_bfloat16* pol_w;
  const float* pol_b;
  const __nv_bfloat16* val_w;
  const float* val_b;
  const float* vout_w;
  const float* vout_b;
  const __nv_bfloat16* q_w;
  const float* q_b;
  const float* qout_w;
  const float* qout_b;
};

template <int HB>
__global__ void __launch_bounds__(256) heads_kernel(const __nv_bfloat16* __restrict__ z, Heads h, int boards, int cells,
                                                    int A, int K, float* __restrict__ logits, float* __restrict__ vprobs,
                                                    float* __restrict__ qprobs,
                                                    const int32_t* __restrict__ d_boards = nullptr,
                                                    CacheWrite cw = {nullptr, nullptr, nullptr, nullptr, nullptr}) {
  const int actual_boards = d_boards ? *d_boards : boards;
  const int b0 = blockIdx.x * HB;
  if (b0 >= actual_boards) return;
  extern __shared__ __align__(16) float sh[];
  const int nb = min(HB, actual_boards - b0);
  const int tid = threadIdx.x;
  float* feat = sh;
  float* vh = feat + HB * cells * HEAD_CH;
  float* qh = vh + HB * CH;
  float* ql = qh + HB * CH;
  const int AK = A * K;

  float* cws = ql;
  for (int i = tid; i < HEAD_CH * CH; i += 256) {
    const int o = i / CH, c = i - o * CH;
    cws[c * HEAD_CH + o] = __bfloat162float(h.conv_w[i]);
  }
  __syncthreads();
  for (int rr = tid; rr < nb * cells; rr += 256) {
    const uint4* zr = reinterpret_cast<const uint4*>(z + ((size_t)b0 * cells + rr) * CH);
    float part[HEAD_CH] = {0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll 4
    for (int q = 0; q < CH / 8; ++q) {
      const uint4 raw = zr[q];
      const __nv_bfloat162* p2 = reinterpret_cast<const __nv_bfloat162*>(&raw);
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const float lo = __low2float(p2[e]), hi = __high2float(p2[e]);
        const float* w0 = cws + (q * 8 + 2 * e) * HEAD_CH;
#pragma unroll
        for (int o = 0; o < HEAD_CH; ++o) part[o] = fmaf(hi, w0[HEAD_CH + o], fmaf(lo, w0[o], part[o]));
      }
    }
#pragma unroll
    for (int o = 0; o < HEAD_CH; ++o) {
      const float v = bf(bf(part[o]) + h.conv_v[o]);
      feat[rr * HEAD_CH + o] = bn_relu(v, h.conv_v[HEAD_CH + o], h.conv_v[2 * HEAD_CH + o], h.conv_v[3 * HEAD_CH + o]);
    }
  }
  __syncthreads();

  if (logits) {
    for (int a = tid; a < A; a += 256) {
      float acc[HB];
#pragma unroll
      for (int bb = 0; bb < HB; ++bb) acc[bb] = 0.f;
      for (int i = 0; i < 2 * cells; ++i) {
        const float wv = __bfloat162float(h.pol_w[(size_t)i * A + a]);
        const int c = i >> 1, ch = i & 1;
#pragma unroll
        for (int bb = 0; bb < HB; ++bb) acc[bb] = fmaf(feat[(bb * cells + c) * HEAD_CH + ch], wv, acc[bb]);
      }
      const float bias = h.pol_b[a];
      for (int bb = 0; bb < nb; ++bb) logits[(size_t)(b0 + bb) * A + a] = bf(bf(acc[bb]) + bias);
    }
  }
  {
    const int j = tid & (CH - 1);
    const bool is_q = tid >= CH;
    float acc[HB];
#pragma unroll
    for (int bb = 0; bb < HB; ++bb) acc[bb] = 0.f;
    if (!is_q) {
      for (int c = 0; c < cells; ++c) {
        const float wv = __bfloat162float(h.val_w[(size_t)c * CH + j]);
#pragma unroll
        for (int bb = 0; bb < HB; ++bb) acc[bb] = fmaf(feat[(bb * cells + c) * HEAD_CH + 2], wv, acc[bb]);
      }
    } else {
      for (int i = 0; i < 2 * cells; ++i) {
        const float wv = __bfloat162float(h.q_w[(size_t)i * CH + j]);
        const int c = i >> 1, ch = 3 + (i & 1);
#pragma unroll
        for (int bb = 0; bb < HB; ++bb) acc[bb] = fmaf(feat[(bb * cells + c) * HEAD_CH + ch], wv, acc[bb]);
      }
    }
    const float bias = is_q ? h.q_b[j] : h.val_b[j];
    float* dst = is_q ? qh : vh;
#pragma unroll
    for (int bb = 0; bb < HB; ++bb) dst[bb * CH + j] = fmaxf(bf(bf(acc[bb]) + bias), 0.f);
  }
  __syncthreads();

  float* vl = ql;
  for (int it = tid; it < nb * K; it += 256) {
    const int bb = it / K, k = it - bb * K;
    float sacc = 0.f;
    for (int j = 0; j < CH; ++j) sacc = fmaf(vh[bb * CH + j], h.vout_w[j * K + k], sacc);
    vl[bb * K + k] = sacc + h.vout_b[k];
  }
  __syncthreads();
  if (tid < nb) {
    const int bb = tid;
    float mx = -INFINITY;
    for (int k = 0; k < K; ++k) mx = fmaxf(mx, vl[bb * K + k]);
    float tot = 0.f;
    for (int k = 0; k < K; ++k) tot += expf(vl[bb * K + k] - mx);
    for (int k = 0; k < K; ++k) vprobs[(size_t)(b0 + bb) * K + k] = expf(vl[bb * K + k] - mx) / tot;
  }
  __syncthreads();
  for (int j = tid; j < AK; j += 256) {
    float acc[HB];
#pragma unroll
    for (int bb = 0; bb < HB; ++bb) acc[bb] = 0.f;
    for (int i = 0; i < CH; ++i) {
      const float wv = h.qout_w[(size_t)i * AK + j];
#pragma unroll
      for (int bb = 0; bb < HB; ++bb) acc[bb] = fmaf(qh[bb * CH + i], wv, acc[bb]);
    }
    const float bias = h.qout_b[j];
#pragma unroll
    for (int bb = 0; bb < HB; ++bb) ql[bb * AK + j] = acc[bb] + bias;
  }
  __syncthreads();
  for (int it = tid; it < nb * A; it += 256) {
    const int bb = it / A, a = it - bb * A;
    const float* l = ql + bb * AK + a * K;
    float mx = -INFINITY;
    for (int k = 0; k < K; ++k) mx = fmaxf(mx, l[k]);
    float tot = 0.f;
    for (int k = 0; k < K; ++k) tot += expf(l[k] - mx);
    float* out = qprobs + ((size_t)(b0 + bb) * A + a) * K;
    for (int k = 0; k < K; ++k) out[k] = expf(l[k] - mx) / tot;
  }
  if (cw.miss_keys) {
    __syncthreads();
    for (int bb = 0; bb < nb; ++bb) {
      const CacheKey mk = cw.miss_keys[b0 + bb];
      if (!(mk.meta & 0x80000000u)) continue;
      const uint32_t slot = hash_cache_key(mk);
      __shared__ int won_slot;
      if (tid == 0) {
        won_slot = (atomicCAS(cw.table_locks + slot, 0u, 1u) == 0u) ? (int)slot : -1;
        if (won_slot >= 0) cw.table_keys[slot] = mk;
      }
      __syncthreads();
      if (won_slot >= 0) {
        const float* vp = vprobs + (size_t)(b0 + bb) * K;
        const float* qp = qprobs + (size_t)(b0 + bb) * AK;
        for (int k = tid; k < K; k += 256) cw.table_v[(size_t)slot * K + k] = vp[k];
        for (int i = tid; i < AK; i += 256) cw.table_q[(size_t)slot * AK + i] = qp[i];
        __syncthreads();
        if (tid == 0) {
          __threadfence();
          cw.table_locks[slot] = 2u;
        }
      }
      __syncthreads();
    }
  }
}

template <typename T>
T* ptr(ffi::AnyBuffer& b) { return reinterpret_cast<T*>(b.untyped_data()); }

ffi::Error configure_kernels_for_device(int device) {
  static std::atomic<uint64_t> configured_mask{0};
  const uint64_t bit = 1ull << (device & 63);
  if (configured_mask.load() & bit) return ffi::Error::Success();
  cudaError_t e = cudaFuncSetAttribute(conv3x3_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, CONV_SMEM);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(conv3x3_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, CONV_SMEM);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(conv3x3_kernel<4, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, CONV_SMEM);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(conv3x3_kernel<2, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, CONV_SMEM);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(conv3x3_int8_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, I8_SMEM_PAIR);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(heads_kernel<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, HEADS_SMEM);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(heads_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, HEADS_SMEM);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(heads_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, HEADS_SMEM);
  if (e == cudaSuccess) e = cudaFuncSetAttribute(board_net_int8_kernel<2, 1, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, board_net_smem<2>());
  if (e == cudaSuccess) e = cudaFuncSetAttribute(board_net_int8_kernel<2, 2, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, board_net_smem<2>());
  if (e == cudaSuccess) e = cudaFuncSetAttribute(board_net_int8_kernel<2, 4, 2>, cudaFuncAttributeMaxDynamicSharedMemorySize, board_net_smem<2>());
  if (e == cudaSuccess) e = cudaFuncSetAttribute(board_net_int8_kernel<4, 1, 1>, cudaFuncAttributeMaxDynamicSharedMemorySize, board_net_smem<4>());
  if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  configured_mask.fetch_or(bit);
  return ffi::Error::Success();
}

ffi::Error launch_board_resident_int8(
    cudaStream_t stream, int sms, const uint8_t* obs, const int32_t* d_boards,
    int max_boards, int side, int cells, int A, int K,
    const __nv_bfloat16* stem_w, const int8_t* trunk_w, const float* wscale,
    const float* qinv, const float* vec, const NetHeads& nh,
    float* logits, float* vprobs, float* qprobs, const CacheWrite& cw) {
  if (cells <= 64) {
    const int smem = board_net_smem<2>();
    if (cells <= 16) {
      const int grid = d_boards ? (2 * sms) : std::min(2 * sms, (max_boards + 3) / 4);
      board_net_int8_kernel<2, 4, 2><<<std::max(1, grid), B_THREADS, smem, stream>>>(
          obs, d_boards, max_boards, side, cells, A, K, stem_w, trunk_w, wscale, qinv, vec, nh, logits, vprobs, qprobs, cw);
    } else if (cells <= 32) {
      const int grid = d_boards ? (2 * sms) : std::min(2 * sms, (max_boards + 1) / 2);
      board_net_int8_kernel<2, 2, 2><<<std::max(1, grid), B_THREADS, smem, stream>>>(
          obs, d_boards, max_boards, side, cells, A, K, stem_w, trunk_w, wscale, qinv, vec, nh, logits, vprobs, qprobs, cw);
    } else {
      const int grid = d_boards ? (2 * sms) : std::min(2 * sms, max_boards);
      board_net_int8_kernel<2, 1, 2><<<std::max(1, grid), B_THREADS, smem, stream>>>(
          obs, d_boards, max_boards, side, cells, A, K, stem_w, trunk_w, wscale, qinv, vec, nh, logits, vprobs, qprobs, cw);
    }
  } else {
    const int smem = board_net_smem<4>();
    const int grid = d_boards ? sms : std::min(sms, max_boards);
    board_net_int8_kernel<4, 1, 1><<<std::max(1, grid), B_THREADS, smem, stream>>>(
        obs, d_boards, max_boards, side, cells, A, K, stem_w, trunk_w, wscale, qinv, vec, nh, logits, vprobs, qprobs, cw);
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  return ffi::Error::Success();
}

// kernel: 0 bf16, 1 FP8 (e4m3), 2 INT8 (board-resident single-kernel on sides 3..11), 3 INT8 legacy 14-kernel.
ffi::Error Network(cudaStream_t stream, int64_t kernel, ffi::AnyBuffer obs, ffi::AnyBuffer stem_w, ffi::AnyBuffer trunk_w,
                   ffi::AnyBuffer wscale, ffi::AnyBuffer qinv,
                   ffi::AnyBuffer vec, ffi::AnyBuffer head_cw, ffi::AnyBuffer head_cv, ffi::AnyBuffer pol_w,
                   ffi::AnyBuffer pol_b, ffi::AnyBuffer val_w, ffi::AnyBuffer val_b, ffi::AnyBuffer vout_w,
                   ffi::AnyBuffer vout_b, ffi::AnyBuffer q_w, ffi::AnyBuffer q_b, ffi::AnyBuffer qout_w,
                   ffi::AnyBuffer qout_b, ffi::Result<ffi::AnyBuffer> logits, ffi::Result<ffi::AnyBuffer> vprobs,
                   ffi::Result<ffi::AnyBuffer> qprobs, ffi::Result<ffi::AnyBuffer> xs, ffi::Result<ffi::AnyBuffer> zs,
                   ffi::Result<ffi::AnyBuffer> z2s) {
  auto od = obs.dimensions();
  if (od.size() != 4 || od[1] != od[2] || od[3] != 4) return ffi::Error::InvalidArgument("obs must be [B, N, N, 4]");
  const int boards = (int)od[0], side = (int)od[1], cells = side * side;
  auto ld = logits->dimensions();
  auto qd = qprobs->dimensions();
  if (ld.size() != 2 || qd.size() != 3) return ffi::Error::InvalidArgument("bad output ranks");
  const int A = (int)ld[1], K = (int)qd[2];
  if ((int)stem_w.element_count() != CH * 36 || (int64_t)trunk_w.element_count() != (int64_t)CONVS * CH * KDIM ||
      (int)vec.element_count() != VEC_ROWS * CH || (int)head_cw.element_count() != HEAD_CH * CH ||
      (int)head_cv.element_count() != 4 * HEAD_CH || (int)pol_w.element_count() != 2 * cells * A ||
      (int)val_w.element_count() != cells * CH || (int)vout_w.element_count() != CH * K ||
      (int)q_w.element_count() != 2 * cells * CH || (int)qout_w.element_count() != CH * A * K || K > 16)
    return ffi::Error::InvalidArgument("network parameter shapes disagree");
  if (boards == 0) return ffi::Error::Success();

  int sms = 0, device = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  ffi::Error cfg_err = configure_kernels_for_device(device);
  if (cfg_err.failure()) return cfg_err;

  if (kernel < 0 || kernel > 3) return ffi::Error::InvalidArgument("kernel must be 0..3");
  const bool fp8 = kernel == 1;
  const bool int8 = (kernel == 2 || kernel == 3);
  if (int8 && !i8_supports(side)) return ffi::Error::InvalidArgument("the int8 network needs a board side from 3 to 15");
  if (kernel && ((int)wscale.element_count() != CONVS * CH || (int)qinv.element_count() != CONVS * CH))
    return ffi::Error::InvalidArgument("wscale and qinv must be [12, 128]");

  // Use single-kernel board-resident INT8 megakernel when batch is smaller than 1 SM wave
  // (where 1 launch avoids 13 inter-kernel barriers), and G=4 L2-resident conv3x3_int8_kernel
  // on larger batches (both match bit-for-bit to 0 ULP!).
  if (kernel == 2 && side >= 3 && side <= 11 && boards < sms) {
    NetHeads nh{ptr<const __nv_bfloat16>(head_cw), ptr<const float>(head_cv), ptr<const __nv_bfloat16>(pol_w),
                ptr<const float>(pol_b), ptr<const __nv_bfloat16>(val_w), ptr<const float>(val_b),
                ptr<const float>(vout_w), ptr<const float>(vout_b), ptr<const __nv_bfloat16>(q_w),
                ptr<const float>(q_b), ptr<const float>(qout_w), ptr<const float>(qout_b)};
    CacheWrite no_cache{nullptr, nullptr, nullptr, nullptr, nullptr};
    return launch_board_resident_int8(
        stream, sms, ptr<const uint8_t>(obs), nullptr, boards, side, cells, A, K,
        ptr<const __nv_bfloat16>(stem_w), ptr<const int8_t>(trunk_w), ptr<const float>(wscale),
        ptr<const float>(qinv), ptr<const float>(vec), nh,
        ptr<float>(*logits), ptr<float>(*vprobs), ptr<float>(*qprobs), no_cache);
  }

  const int rows = boards * cells;
  const ConvShape s{rows, side, cells};
  __nv_bfloat16* X = ptr<__nv_bfloat16>(*xs);
  __nv_bfloat16* Z = ptr<__nv_bfloat16>(*zs);
  __nv_bfloat16* Z2 = ptr<__nv_bfloat16>(*z2s);
  const float* V = ptr<const float>(vec);
  const char* W = ptr<const char>(trunk_w);
  const size_t conv_bytes = (size_t)CH * KDIM * (kernel ? 1 : 2);
  const float* WS = ptr<const float>(wscale);
  const float* QI = ptr<const float>(qinv);
  uint8_t* Zq = reinterpret_cast<uint8_t*>(Z);
  uint8_t* Z2q = reinterpret_cast<uint8_t*>(Z2);

  const int stem_grid = std::max(1, std::min((rows + 15) / 16, STEM_CTAS_PER_SM * sms));
  stem_kernel<<<stem_grid, 256, 0, stream>>>(ptr<const uint8_t>(obs), ptr<const __nv_bfloat16>(stem_w), V, rows,
                                             side, cells, X, Z, kernel ? Zq : nullptr, int8 ? QI : nullptr);
  const bool tall = (rows + 127) / 128 >= 4 * sms;
  const int bm = tall ? 128 : 64;
  const int grid = (rows + bm - 1) / bm;
  auto conv = [&](const void* in, const void* w, const Epilogue& e) {
    if (fp8) {
      if (tall) conv3x3_kernel<4, true><<<grid, THREADS, CONV_SMEM, stream>>>(in, w, s, e);
      else conv3x3_kernel<2, true><<<grid, THREADS, CONV_SMEM, stream>>>(in, w, s, e);
    } else {
      if (tall) conv3x3_kernel<4><<<grid, THREADS, CONV_SMEM, stream>>>(in, w, s, e);
      else conv3x3_kernel<2><<<grid, THREADS, CONV_SMEM, stream>>>(in, w, s, e);
    }
  };
  const I8Plan plan = int8 ? i8_plan(boards, side, sms) : I8Plan{};
  auto conv8 = [&](const uint8_t* in, int j, const Epilogue& e) {
    conv3x3_int8_kernel<<<plan.grid, I8_THREADS, plan.smem, stream>>>(
        in, reinterpret_cast<const int8_t*>(W + (size_t)j * conv_bytes), s, e, e.out_zq ? QI + (j + 1) * CH : nullptr,
        plan.boards_per_unit);
  };
  for (int i = 0; i < BLOCKS; ++i) {
    Epilogue e1{V + b1_row(i) * CH, V + bn2_row(i) * CH, V + (bn2_row(i) + 1) * CH, V + (bn2_row(i) + 2) * CH,
                nullptr, nullptr, Z2};
    if (kernel) { e1.wscale = WS + (2 * i) * CH; e1.out_zq = Z2q; }
    if (int8) conv8(Zq, 2 * i, e1);
    else conv(Z, W + (size_t)(2 * i) * conv_bytes, e1);
    const int next = i + 1 < BLOCKS ? bn1_row(i + 1) : FINAL_BN;
    Epilogue e2{V + b2_row(i) * CH, V + next * CH, V + (next + 1) * CH, V + (next + 2) * CH, X, X, Z};
    if (kernel) { e2.wscale = WS + (2 * i + 1) * CH; if (i + 1 < BLOCKS) e2.out_zq = Zq; }
    if (int8) conv8(Z2q, 2 * i + 1, e2);
    else conv(Z2, W + (size_t)(2 * i + 1) * conv_bytes, e2);
  }
  Heads h{ptr<const __nv_bfloat16>(head_cw), ptr<const float>(head_cv), ptr<const __nv_bfloat16>(pol_w),
          ptr<const float>(pol_b), ptr<const __nv_bfloat16>(val_w), ptr<const float>(val_b), ptr<const float>(vout_w),
          ptr<const float>(vout_b), ptr<const __nv_bfloat16>(q_w), ptr<const float>(q_b), ptr<const float>(qout_w),
          ptr<const float>(qout_b)};
  auto heads_smem = [&](int hb) { return sizeof(float) * (hb * cells * HEAD_CH + 2 * hb * CH + std::max(hb * A * K, HEAD_CH * CH)); };
  if (heads_smem(8) <= HEADS_SMEM && (boards + 7) / 8 >= HEADS_MIN_CTAS_PER_SM * sms) {
    heads_kernel<8><<<(boards + 7) / 8, 256, heads_smem(8), stream>>>(Z, h, boards, cells, A, K, ptr<float>(*logits),
                                                                      ptr<float>(*vprobs), ptr<float>(*qprobs));
  } else if (heads_smem(4) <= HEADS_SMEM) {
    heads_kernel<4><<<(boards + 3) / 4, 256, heads_smem(4), stream>>>(Z, h, boards, cells, A, K, ptr<float>(*logits),
                                                                      ptr<float>(*vprobs), ptr<float>(*qprobs));
  } else {
    return ffi::Error::InvalidArgument("heads do not fit in shared memory");
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  return ffi::Error::Success();
}

// Compact dynamic-count INT8 network for search: evaluates only `*d_boards`
// compacted active leaves (skipping policy logits when unneeded, and writing
// newly evaluated V/Q beliefs into the GPU position cache).
ffi::Error NetworkCompact(
    cudaStream_t stream, int64_t use_cache, int64_t small_hint,
    ffi::AnyBuffer obs, ffi::AnyBuffer d_boards, ffi::AnyBuffer miss_keys,
    ffi::AnyBuffer stem_w, ffi::AnyBuffer trunk_w, ffi::AnyBuffer wscale, ffi::AnyBuffer qinv,
    ffi::AnyBuffer vec, ffi::AnyBuffer head_cw, ffi::AnyBuffer head_cv, ffi::AnyBuffer pol_w,
    ffi::AnyBuffer pol_b, ffi::AnyBuffer val_w, ffi::AnyBuffer val_b, ffi::AnyBuffer vout_w,
    ffi::AnyBuffer vout_b, ffi::AnyBuffer q_w, ffi::AnyBuffer q_b, ffi::AnyBuffer qout_w,
    ffi::AnyBuffer qout_b,
    ffi::AnyBuffer cache_keys, ffi::AnyBuffer cache_locks, ffi::AnyBuffer cache_v, ffi::AnyBuffer cache_q,
    ffi::Result<ffi::AnyBuffer> vprobs, ffi::Result<ffi::AnyBuffer> qprobs,
    ffi::Result<ffi::AnyBuffer> r_cache_keys, ffi::Result<ffi::AnyBuffer> r_cache_locks,
    ffi::Result<ffi::AnyBuffer> r_cache_v, ffi::Result<ffi::AnyBuffer> r_cache_q,
    ffi::Result<ffi::AnyBuffer> xs, ffi::Result<ffi::AnyBuffer> zs, ffi::Result<ffi::AnyBuffer> z2s) {
  auto od = obs.dimensions();
  if (od.size() != 4 || od[1] != od[2] || od[3] != 4) return ffi::Error::InvalidArgument("obs must be [B, N, N, 4]");
  const int max_boards = (int)od[0], side = (int)od[1], cells = side * side;
  auto qd = qprobs->dimensions();
  const int A = (int)qd[1], K = (int)qd[2];
  if (side < 3 || side > 11 || K > 16) return ffi::Error::InvalidArgument("compact int8 network supports Hex 3..11 and K <= 16");
  if (max_boards == 0) return ffi::Error::Success();

  int sms = 0, device = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
  ffi::Error cfg_err = configure_kernels_for_device(device);
  if (cfg_err.failure()) return cfg_err;

  CacheWrite cw{nullptr, nullptr, nullptr, nullptr, nullptr};
  if (use_cache) {
    cw.miss_keys = reinterpret_cast<const CacheKey*>(miss_keys.untyped_data());
    cw.table_keys = reinterpret_cast<CacheKey*>(r_cache_keys->untyped_data());
    cw.table_locks = reinterpret_cast<uint32_t*>(r_cache_locks->untyped_data());
    cw.table_v = ptr<float>(*r_cache_v);
    cw.table_q = ptr<float>(*r_cache_q);
  }
  const int32_t* d_cnt = ptr<const int32_t>(d_boards);

  if (small_hint || max_boards < sms) {
    NetHeads nh{ptr<const __nv_bfloat16>(head_cw), ptr<const float>(head_cv), ptr<const __nv_bfloat16>(pol_w),
                ptr<const float>(pol_b), ptr<const __nv_bfloat16>(val_w), ptr<const float>(val_b),
                ptr<const float>(vout_w), ptr<const float>(vout_b), ptr<const __nv_bfloat16>(q_w),
                ptr<const float>(q_b), ptr<const float>(qout_w), ptr<const float>(qout_b)};
    return launch_board_resident_int8(
        stream, sms, ptr<const uint8_t>(obs), d_cnt, max_boards, side, cells, A, K,
        ptr<const __nv_bfloat16>(stem_w), ptr<const int8_t>(trunk_w), ptr<const float>(wscale),
        ptr<const float>(qinv), ptr<const float>(vec), nh,
        nullptr, ptr<float>(*vprobs), ptr<float>(*qprobs), cw);
  }

  // Large/medium wave: dynamic-*d_boards G-board L2-cached INT8 pipeline (0.68ms @ 1280, scales linearly down to 0).
  const int rows = max_boards * cells;
  const ConvShape s{rows, side, cells};
  __nv_bfloat16* X = ptr<__nv_bfloat16>(*xs);
  __nv_bfloat16* Z = ptr<__nv_bfloat16>(*zs);
  __nv_bfloat16* Z2 = ptr<__nv_bfloat16>(*z2s);
  const float* V = ptr<const float>(vec);
  const char* W = ptr<const char>(trunk_w);
  const size_t conv_bytes = (size_t)CH * KDIM;
  const float* WS = ptr<const float>(wscale);
  const float* QI = ptr<const float>(qinv);
  uint8_t* Zq = reinterpret_cast<uint8_t*>(Z);
  uint8_t* Z2q = reinterpret_cast<uint8_t*>(Z2);

  const int stem_grid = std::max(1, std::min((rows + 15) / 16, STEM_CTAS_PER_SM * sms));
  stem_kernel<<<stem_grid, 256, 0, stream>>>(
      ptr<const uint8_t>(obs), ptr<const __nv_bfloat16>(stem_w), V, rows, side, cells, X, Z, Zq, QI, d_cnt);
  const I8Plan plan = i8_plan(max_boards, side, sms);
  auto conv8 = [&](const uint8_t* in_zq, int j, const Epilogue& e) {
    conv3x3_int8_kernel<<<plan.grid, I8_THREADS, plan.smem, stream>>>(
        in_zq, reinterpret_cast<const int8_t*>(W + (size_t)j * conv_bytes), s, e,
        e.out_zq ? QI + (j + 1) * CH : nullptr, plan.boards_per_unit, d_cnt);
  };
  for (int i = 0; i < BLOCKS; ++i) {
    Epilogue e1{V + b1_row(i) * CH, V + bn2_row(i) * CH, V + (bn2_row(i) + 1) * CH, V + (bn2_row(i) + 2) * CH,
                nullptr, nullptr, Z2, WS + (2 * i) * CH, Z2q};
    conv8(Zq, 2 * i, e1);
    const int next = i + 1 < BLOCKS ? bn1_row(i + 1) : FINAL_BN;
    Epilogue e2{V + b2_row(i) * CH, V + next * CH, V + (next + 1) * CH, V + (next + 2) * CH,
                X, X, Z, WS + (2 * i + 1) * CH, (i + 1 < BLOCKS) ? Zq : nullptr};
    conv8(Z2q, 2 * i + 1, e2);
  }
  Heads h{ptr<const __nv_bfloat16>(head_cw), ptr<const float>(head_cv), ptr<const __nv_bfloat16>(pol_w),
          ptr<const float>(pol_b), ptr<const __nv_bfloat16>(val_w), ptr<const float>(val_b), ptr<const float>(vout_w),
          ptr<const float>(vout_b), ptr<const __nv_bfloat16>(q_w), ptr<const float>(q_b), ptr<const float>(qout_w),
          ptr<const float>(qout_b)};
  auto heads_smem = [&](int hb) { return sizeof(float) * (hb * cells * HEAD_CH + 2 * hb * CH + std::max(hb * A * K, HEAD_CH * CH)); };
  if (heads_smem(8) <= HEADS_SMEM && (max_boards + 7) / 8 >= HEADS_MIN_CTAS_PER_SM * sms) {
    heads_kernel<8><<<(max_boards + 7) / 8, 256, heads_smem(8), stream>>>(
        Z, h, max_boards, cells, A, K, nullptr, ptr<float>(*vprobs), ptr<float>(*qprobs), d_cnt, cw);
  } else {
    heads_kernel<4><<<(max_boards + 3) / 4, 256, heads_smem(4), stream>>>(
        Z, h, max_boards, cells, A, K, nullptr, ptr<float>(*vprobs), ptr<float>(*qprobs), d_cnt, cw);
  }
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  return ffi::Error::Success();
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ScacchiBinnedNetwork, Network,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("kernel")
        .Arg<ffi::AnyBuffer>()   // obs
        .Arg<ffi::AnyBuffer>()   // stem_w
        .Arg<ffi::AnyBuffer>()   // trunk_w
        .Arg<ffi::AnyBuffer>()   // wscale
        .Arg<ffi::AnyBuffer>()   // qinv
        .Arg<ffi::AnyBuffer>()   // vec
        .Arg<ffi::AnyBuffer>()   // head_cw
        .Arg<ffi::AnyBuffer>()   // head_cv
        .Arg<ffi::AnyBuffer>()   // pol_w
        .Arg<ffi::AnyBuffer>()   // pol_b
        .Arg<ffi::AnyBuffer>()   // val_w
        .Arg<ffi::AnyBuffer>()   // val_b
        .Arg<ffi::AnyBuffer>()   // vout_w
        .Arg<ffi::AnyBuffer>()   // vout_b
        .Arg<ffi::AnyBuffer>()   // q_w
        .Arg<ffi::AnyBuffer>()   // q_b
        .Arg<ffi::AnyBuffer>()   // qout_w
        .Arg<ffi::AnyBuffer>()   // qout_b
        .Ret<ffi::AnyBuffer>()   // logits
        .Ret<ffi::AnyBuffer>()   // vprobs
        .Ret<ffi::AnyBuffer>()   // qprobs
        .Ret<ffi::AnyBuffer>()   // scratch x
        .Ret<ffi::AnyBuffer>()   // scratch z
        .Ret<ffi::AnyBuffer>(),  // scratch z2
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ScacchiBinnedNetworkCompact, NetworkCompact,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("use_cache")
        .Attr<int64_t>("small_hint")
        .Arg<ffi::AnyBuffer>()   // obs         [B, N, N, 4] bool
        .Arg<ffi::AnyBuffer>()   // d_boards    [1] i32
        .Arg<ffi::AnyBuffer>()   // miss_keys   [B, 10] u32 (40 bytes/entry = CacheKey)
        .Arg<ffi::AnyBuffer>()   // stem_w
        .Arg<ffi::AnyBuffer>()   // trunk_w
        .Arg<ffi::AnyBuffer>()   // wscale
        .Arg<ffi::AnyBuffer>()   // qinv
        .Arg<ffi::AnyBuffer>()   // vec
        .Arg<ffi::AnyBuffer>()   // head_cw
        .Arg<ffi::AnyBuffer>()   // head_cv
        .Arg<ffi::AnyBuffer>()   // pol_w
        .Arg<ffi::AnyBuffer>()   // pol_b
        .Arg<ffi::AnyBuffer>()   // val_w
        .Arg<ffi::AnyBuffer>()   // val_b
        .Arg<ffi::AnyBuffer>()   // vout_w
        .Arg<ffi::AnyBuffer>()   // vout_b
        .Arg<ffi::AnyBuffer>()   // q_w
        .Arg<ffi::AnyBuffer>()   // q_b
        .Arg<ffi::AnyBuffer>()   // qout_w
        .Arg<ffi::AnyBuffer>()   // qout_b
        .Arg<ffi::AnyBuffer>()   // cache_keys  [CACHE_SLOTS, 10] u32
        .Arg<ffi::AnyBuffer>()   // cache_locks [CACHE_SLOTS] u32
        .Arg<ffi::AnyBuffer>()   // cache_v     [CACHE_SLOTS, K] f32
        .Arg<ffi::AnyBuffer>()   // cache_q     [CACHE_SLOTS, A, K] f32
        .Ret<ffi::AnyBuffer>()   // vprobs      [B, K] f32
        .Ret<ffi::AnyBuffer>()   // qprobs      [B, A, K] f32
        .Ret<ffi::AnyBuffer>()   // r_cache_keys
        .Ret<ffi::AnyBuffer>()   // r_cache_locks
        .Ret<ffi::AnyBuffer>()   // r_cache_v
        .Ret<ffi::AnyBuffer>()   // r_cache_q
        .Ret<ffi::AnyBuffer>()   // scratch x
        .Ret<ffi::AnyBuffer>()   // scratch z
        .Ret<ffi::AnyBuffer>(),  // scratch z2
    {ffi::Traits::kCmdBufferCompatible});
