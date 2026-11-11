// Fused binned backup, one warp per lane (batch element).
//
//   value = weighted_sum(edges, best_policy(edges, invalid), points)
//
// Same function as scacchi.binned.best_policy followed by
// scacchi.binned.weighted_sum (lattice 129), computed in float32 with plain
// FMAs and IEEE division: no TF32, no fast-math, no truncation (no move, atom,
// lattice cell or quadrature node is dropped). Only the order of additions and
// products differs from the XLA program; the one place with more than float32
// is the normalisation of the policy (a double sum and quotient, see stage 2).
// Lanes with repaired[b] == false are skipped and written as zeros. Nothing
// uses atomics, so results are bitwise reproducible and do not depend on which
// other lanes are in the batch.
//
// One warp handles one lane; the stages are in backup_lane.cuh, shared with
// the fused search step (search.cu).
//
// Tuning knob (environment, read once): SCACCHI_BINNED_WARPS (warps per block;
// results do not depend on it). Build flags, off by default: -DBACKUP_FAST_RECIP
// (Newton reciprocal instead of IEEE division in stage 2; not IEEE, so not
// used by the search), -DBACKUP_PROFILE (cycle counts of one warp, printed).

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cstdint>
#include <cstdlib>

#include "xla/ffi/api/ffi.h"
#include "backup_lane.cuh"

namespace ffi = xla::ffi;

namespace {

using namespace scacchi_backup;

template <int KT>
__global__ void __launch_bounds__(256) backup_kernel(
    const float* __restrict__ edges, const uint8_t* __restrict__ invalid, const float* __restrict__ points,
    const uint8_t* __restrict__ repaired, const float* __restrict__ nodes, const float* __restrict__ qweights,
    float* __restrict__ out, int B, Dims d, Offsets off, int warp_floats) {
  extern __shared__ __align__(16) float smem[];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int b = blockIdx.x * (blockDim.x >> 5) + wid;
  if (b >= B) return;
  const int K = KT > 0 ? KT : d.K;
  float* o = out + (size_t)b * K;
  if (!repaired[b]) {
    for (int k = lane; k < K; k += 32) o[k] = 0.f;
    return;
  }
  const Smem S = carve(smem + (size_t)wid * warp_floats, off);
  load_edges(S, edges + (size_t)b * d.A * K, d.A, K, d.KS, lane);
  __syncwarp();
  backup_lane<KT>(S, invalid + (size_t)b * d.A, points, nodes, qweights, o, d, lane);
}

template <int KT>
ffi::Error launch(cudaStream_t stream, const float* edges, const uint8_t* invalid, const float* points, const uint8_t* repaired,
                  const float* nodes, const float* qweights, float* out, int B, const Dims& d, int forced_warps) {
  const Offsets off = make_offsets(d);
  const int warp_floats = round4(off.total);
  const size_t per_warp = (size_t)warp_floats * sizeof(float);
  int warps = forced_warps;
  if (warps <= 0) warps = (int)std::max<size_t>(1, std::min<size_t>(8, (24 * 1024) / per_warp));
  const size_t smem = per_warp * warps;
  static std::atomic<size_t> configured{0};
  if (smem > configured.load()) {
    cudaError_t e = cudaFuncSetAttribute(backup_kernel<KT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)smem);
    if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
    configured.store(smem);
  }
  const int blocks = (B + warps - 1) / warps;
  backup_kernel<KT><<<blocks, warps * 32, smem, stream>>>(edges, invalid, points, repaired, nodes, qweights, out, B, d, off, warp_floats);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  return ffi::Error::Success();
}

ffi::Error Launch(cudaStream_t stream, ffi::AnyBuffer edges, ffi::AnyBuffer invalid, ffi::AnyBuffer points,
                  ffi::AnyBuffer repaired, ffi::AnyBuffer nodes, ffi::AnyBuffer qweights, ffi::Result<ffi::AnyBuffer> out) {
  auto dims = edges.dimensions();
  if (dims.size() != 3) return ffi::Error::InvalidArgument("edges must be [B, A, K]");
  const int B = (int)dims[0], A = (int)dims[1], K = (int)dims[2];
  if (A < 1 || A > MAXA || K < 2 || K > MAXK) return ffi::Error::InvalidArgument("need 1 <= A <= 128 and 2 <= K <= 129");
  if (edges.element_type() != ffi::F32 || points.element_type() != ffi::F32 || nodes.element_type() != ffi::F32 ||
      qweights.element_type() != ffi::F32 || out->element_type() != ffi::F32)
    return ffi::Error::InvalidArgument("float operands must be float32");
  if (invalid.element_type() != ffi::PRED || repaired.element_type() != ffi::PRED)
    return ffi::Error::InvalidArgument("invalid and repaired must be bool");
  const Dims d = make_dims(A, K);
  if ((int)nodes.element_count() != d.Q || (int)qweights.element_count() != d.Q)
    return ffi::Error::InvalidArgument("quadrature must have ceil(A / 2) nodes");
  if ((int)points.element_count() != K || (int)invalid.element_count() != B * A || (int)repaired.element_count() != B)
    return ffi::Error::InvalidArgument("operand shapes disagree");
  if (B == 0) return ffi::Error::Success();

  static const int forced_warps = [] { const char* s = getenv("SCACCHI_BINNED_WARPS"); return s ? atoi(s) : 0; }();
  const float* pe = (const float*)edges.untyped_data();
  const uint8_t* pi = (const uint8_t*)invalid.untyped_data();
  const float* pp = (const float*)points.untyped_data();
  const uint8_t* pr = (const uint8_t*)repaired.untyped_data();
  const float* pn = (const float*)nodes.untyped_data();
  const float* pw = (const float*)qweights.untyped_data();
  float* po = (float*)out->untyped_data();
  // Atom counts the search uses are compiled in (loops unroll); others take the generic path.
  switch (K) {
    case 9:  return launch<9>(stream, pe, pi, pp, pr, pn, pw, po, B, d, forced_warps);
    case 65: return launch<65>(stream, pe, pi, pp, pr, pn, pw, po, B, d, forced_warps);
    default: return launch<0>(stream, pe, pi, pp, pr, pn, pw, po, B, d, forced_warps);
  }
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ScacchiBinnedBackup, Launch,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Arg<ffi::AnyBuffer>()   // edges    [B, A, K] f32
        .Arg<ffi::AnyBuffer>()   // invalid  [B, A]    bool
        .Arg<ffi::AnyBuffer>()   // points   [K]       f32
        .Arg<ffi::AnyBuffer>()   // repaired [B]       bool
        .Arg<ffi::AnyBuffer>()   // nodes    [Q]       f32, Gauss-Legendre on [0, 1]
        .Arg<ffi::AnyBuffer>()   // weights  [Q]       f32
        .Ret<ffi::AnyBuffer>(),  // value    [B, K]    f32
    {ffi::Traits::kCmdBufferCompatible});
