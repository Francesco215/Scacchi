// The binned AlphaZero network (scacchi.network.AZBinnedNet, ResNet v2, 128
// channels) for Hex positions, inference only, as 14 CUDA kernels behind one
// XLA FFI call: the stem, 12 tensor-core 3x3 convolutions whose epilogues apply
// bias, BatchNorm, ReLU and the residual (no separate elementwise passes), and
// one kernel for the policy, value and Q heads.
//
// Precision follows the flax module (dtype=bfloat16): convolution and dense
// products take bf16 inputs and accumulate in f32, every layer output is
// rounded to bf16, biases and residuals are added in bf16, BatchNorm runs in
// f32 on the bf16 value and is rounded back, and the two output layers of the
// beliefs are f32 (here in full f32 FMA; XLA may use TF32 for them). The order
// of the f32 accumulations differs from cuDNN's, so outputs agree with the XLA
// network to bf16 rounding, not bit for bit. Each position's result does not
// depend on the batch it is evaluated in.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstdint>

#include "xla/ffi/api/ffi.h"
#include <cuda_fp8.h>

#include "conv_mma.cuh"

namespace ffi = xla::ffi;

namespace {

using namespace scacchi_net;

constexpr int BLOCKS = 6;
constexpr int CONVS = 2 * BLOCKS;
// Rows of the per-channel table vec[ROWS][128] (float32).
constexpr int STEM_BIAS = 0;
__host__ __device__ constexpr int bn1_row(int i) { return 1 + 8 * i; }   // mean, mul, beta
__host__ __device__ constexpr int b1_row(int i) { return 4 + 8 * i; }
__host__ __device__ constexpr int bn2_row(int i) { return 5 + 8 * i; }   // mean, mul, beta
__host__ __device__ constexpr int b2_row(int i) { return 8 + 8 * i; }
constexpr int FINAL_BN = 1 + 8 * BLOCKS;                                 // 49: mean, mul, beta
constexpr int VEC_ROWS = FINAL_BN + 3;                                   // 52
constexpr int HEAD_CH = 5;  // policy 2, value 1, q 2

constexpr int HEADS_SMEM = 96 * 1024;

__device__ __forceinline__ float bf(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }
__device__ __forceinline__ float bn_relu(float v, float mean, float mul, float beta) {
  return fmaxf(bf(fmaf(v - mean, mul, beta)), 0.f);
}

// ---- stem: 3x3 convolution 4 -> 128 of the observation -----------------------
// x = conv(obs) + bias (bf16), z = relu(bn1_0(x)). One thread: one row, 8 channels.
// Weights in shared memory as [tap * 4 + c][128], so a thread's 8 channels are
// two float4 loads and a warp reads consecutive banks.
__global__ void __launch_bounds__(256) stem_kernel(const uint8_t* __restrict__ obs, const __nv_bfloat16* __restrict__ w,
                                                   const float* __restrict__ vec, int rows, int side, int cells,
                                                   __nv_bfloat16* __restrict__ x_out, __nv_bfloat16* __restrict__ z_out,
                                                   uint8_t* __restrict__ zq_out) {
  __shared__ __align__(16) float ws[36 * CH];
  for (int i = threadIdx.x; i < CH * 36; i += blockDim.x) {
    const int n = i / 36, k = i - n * 36;
    ws[k * CH + n] = __bfloat162float(w[i]);
  }
  __syncthreads();
  const int n0 = (threadIdx.x & 15) * 8;
  for (int r = blockIdx.x * 16 + (threadIdx.x >> 4); r < rows; r += gridDim.x * 16) {
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
    for (int j = 0; j < 8; ++j)
      q[j >> 2] |= (uint32_t)__nv_cvt_float_to_fp8(__bfloat162float(zo[j]), __NV_SATFINITE, __NV_E4M3) << (8 * (j & 3));
    *reinterpret_cast<uint2*>(zq_out + (size_t)r * CH + n0) = make_uint2(q[0], q[1]);
  } else {
    *reinterpret_cast<uint4*>(z_out + (size_t)r * CH + n0) = *reinterpret_cast<const uint4*>(zo);
  }
  }
}

// ---- heads ----------------------------------------------------------------------
struct Heads {
  const __nv_bfloat16* conv_w;  // [5][128]: policy 0, 1, value, q 0, 1
  const float* conv_v;          // [4][5]: bias, bn mean, bn mul, bn beta
  const __nv_bfloat16* pol_w;   // [2C][A]
  const float* pol_b;           // [A]
  const __nv_bfloat16* val_w;   // [C][128]
  const float* val_b;           // [128]
  const float* vout_w;          // [128][K]
  const float* vout_b;          // [K]
  const __nv_bfloat16* q_w;     // [2C][128]
  const float* q_b;             // [128]
  const float* qout_w;          // [128][A K]
  const float* qout_b;          // [A K]
};

// HB boards per CTA, 256 threads. Shared: head features, hidden layers, Q logits.
template <int HB>
__global__ void __launch_bounds__(256) heads_kernel(const __nv_bfloat16* __restrict__ z, Heads h, int boards, int cells,
                                                    int A, int K, float* __restrict__ logits, float* __restrict__ vprobs,
                                                    float* __restrict__ qprobs) {
  extern __shared__ __align__(16) float sh[];
  const int b0 = blockIdx.x * HB;
  const int nb = min(HB, boards - b0);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  float* feat = sh;                           // [HB][cells][5]
  float* vh = feat + HB * cells * HEAD_CH;    // [HB][128]
  float* qh = vh + HB * CH;                   // [HB][128]
  float* ql = qh + HB * CH;                   // [HB][A K]
  const int AK = A * K;

  // 1x1 convolutions to 5 channels, + bias (bf16), BatchNorm, ReLU: one
  // thread per position, weights broadcast from shared memory.
  float* cws = ql;  // [128][5], before the Q logits use this space
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

  // Dense layers for all HB boards at once: a thread owns output columns and
  // reads each weight once per CTA (coalesced over columns).
  // Policy: flatten (cell, channel 0..1) -> A logits, bf16 dense layer.
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
  // Value and Q hidden layers (bf16 dense, ReLU): threads 0..127 value, 128..255 Q.
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

  // Value beliefs: f32 dense layer to K logits (one thread per board and atom), softmax.
  float* vl = ql;  // [HB][K], read before the Q logits overwrite it
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
  // Q beliefs: f32 dense layer to A K logits (weights read once per CTA).
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
}

template <typename T>
T* ptr(ffi::AnyBuffer& b) { return reinterpret_cast<T*>(b.untyped_data()); }

ffi::Error Network(cudaStream_t stream, int64_t fp8, ffi::AnyBuffer obs, ffi::AnyBuffer stem_w, ffi::AnyBuffer trunk_w,
                   ffi::AnyBuffer wscale,
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
  const int rows = boards * cells;
  const ConvShape s{rows, side, cells};
  __nv_bfloat16* X = ptr<__nv_bfloat16>(*xs);
  __nv_bfloat16* Z = ptr<__nv_bfloat16>(*zs);
  __nv_bfloat16* Z2 = ptr<__nv_bfloat16>(*z2s);
  const float* V = ptr<const float>(vec);
  const char* W = ptr<const char>(trunk_w);
  const size_t conv_bytes = (size_t)CH * KDIM * (fp8 ? 1 : 2);
  const float* WS = ptr<const float>(wscale);
  if (fp8 && (int)wscale.element_count() != CONVS * CH) return ffi::Error::InvalidArgument("wscale must be [12, 128]");
  // FP8: the convolution inputs Z and Z2 are e4m3 bytes in the same scratch buffers.
  uint8_t* Zq = reinterpret_cast<uint8_t*>(Z);
  uint8_t* Z2q = reinterpret_cast<uint8_t*>(Z2);

  static std::atomic<bool> configured{false};
  if (!configured.load()) {
    cudaError_t e = cudaFuncSetAttribute(conv3x3_kernel<4>, cudaFuncAttributeMaxDynamicSharedMemorySize, CONV_SMEM);
    if (e == cudaSuccess) e = cudaFuncSetAttribute(conv3x3_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, CONV_SMEM);
    if (e == cudaSuccess) e = cudaFuncSetAttribute(conv3x3_kernel<4, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, CONV_SMEM);
    if (e == cudaSuccess) e = cudaFuncSetAttribute(conv3x3_kernel<2, true>, cudaFuncAttributeMaxDynamicSharedMemorySize, CONV_SMEM);
    if (e == cudaSuccess) e = cudaFuncSetAttribute(heads_kernel<16>, cudaFuncAttributeMaxDynamicSharedMemorySize, HEADS_SMEM);
    if (e == cudaSuccess) e = cudaFuncSetAttribute(heads_kernel<8>, cudaFuncAttributeMaxDynamicSharedMemorySize, HEADS_SMEM);
    if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
    configured.store(true);
  }
  stem_kernel<<<(rows + 511) / 512, 256, 0, stream>>>(ptr<const uint8_t>(obs), ptr<const __nv_bfloat16>(stem_w), V, rows,
                                                    side, cells, X, Z, fp8 ? Zq : nullptr);
  // 128-row tiles when they fill at least two waves (2 CTAs per SM), else 64.
  int sms = 0, device = 0;
  cudaGetDevice(&device);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device);
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
  for (int i = 0; i < BLOCKS; ++i) {
    Epilogue e1{V + b1_row(i) * CH, V + bn2_row(i) * CH, V + (bn2_row(i) + 1) * CH, V + (bn2_row(i) + 2) * CH,
                nullptr, nullptr, Z2};
    if (fp8) { e1.wscale = WS + (2 * i) * CH; e1.out_zq = Z2q; }
    conv(Z, W + (size_t)(2 * i) * conv_bytes, e1);
    const int next = i + 1 < BLOCKS ? bn1_row(i + 1) : FINAL_BN;
    Epilogue e2{V + b2_row(i) * CH, V + next * CH, V + (next + 1) * CH, V + (next + 2) * CH, X, X, Z};
    // The last convolution's z feeds the heads, which read bf16.
    if (fp8) { e2.wscale = WS + (2 * i + 1) * CH; if (i + 1 < BLOCKS) e2.out_zq = Zq; }
    conv(Z2, W + (size_t)(2 * i + 1) * conv_bytes, e2);
  }
  Heads h{ptr<const __nv_bfloat16>(head_cw), ptr<const float>(head_cv), ptr<const __nv_bfloat16>(pol_w),
          ptr<const float>(pol_b), ptr<const __nv_bfloat16>(val_w), ptr<const float>(val_b), ptr<const float>(vout_w),
          ptr<const float>(vout_b), ptr<const __nv_bfloat16>(q_w), ptr<const float>(q_b), ptr<const float>(qout_w),
          ptr<const float>(qout_b)};
  auto heads_smem = [&](int hb) { return sizeof(float) * (hb * cells * HEAD_CH + 2 * hb * CH + hb * A * K); };
  if (heads_smem(8) <= HEADS_SMEM) {
    heads_kernel<8><<<(boards + 7) / 8, 256, heads_smem(8), stream>>>(Z, h, boards, cells, A, K, ptr<float>(*logits),
                                                                      ptr<float>(*vprobs), ptr<float>(*qprobs));
  } else {
    return ffi::Error::InvalidArgument("heads do not fit in shared memory");
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
        .Attr<int64_t>("fp8")    // 1: trunk convolutions on e4m3 inputs
        .Arg<ffi::AnyBuffer>()   // obs      [B, N, N, 4] bool
        .Arg<ffi::AnyBuffer>()   // stem_w   [128, 36] bf16
        .Arg<ffi::AnyBuffer>()   // trunk_w  [12, 128, 1152] bf16, or e4m3 bytes when fp8
        .Arg<ffi::AnyBuffer>()   // wscale   [12, 128] f32 weight scale per output channel (fp8)
        .Arg<ffi::AnyBuffer>()   // vec      [52, 128] f32
        .Arg<ffi::AnyBuffer>()   // head_cw  [5, 128] bf16
        .Arg<ffi::AnyBuffer>()   // head_cv  [4, 5] f32
        .Arg<ffi::AnyBuffer>()   // pol_w    [2C, A] bf16
        .Arg<ffi::AnyBuffer>()   // pol_b    [A] f32
        .Arg<ffi::AnyBuffer>()   // val_w    [C, 128] bf16
        .Arg<ffi::AnyBuffer>()   // val_b    [128] f32
        .Arg<ffi::AnyBuffer>()   // vout_w   [128, K] f32
        .Arg<ffi::AnyBuffer>()   // vout_b   [K] f32
        .Arg<ffi::AnyBuffer>()   // q_w      [2C, 128] bf16
        .Arg<ffi::AnyBuffer>()   // q_b      [128] f32
        .Arg<ffi::AnyBuffer>()   // qout_w   [128, A K] f32
        .Arg<ffi::AnyBuffer>()   // qout_b   [A K] f32
        .Ret<ffi::AnyBuffer>()   // logits   [B, A] f32
        .Ret<ffi::AnyBuffer>()   // vprobs   [B, K] f32
        .Ret<ffi::AnyBuffer>()   // qprobs   [B, A, K] f32
        .Ret<ffi::AnyBuffer>()   // scratch x  [B C, 128] bf16
        .Ret<ffi::AnyBuffer>()   // scratch z  [B C, 128] bf16
        .Ret<ffi::AnyBuffer>(),  // scratch z2 [B C, 128] bf16
    {ffi::Traits::kCmdBufferCompatible});
