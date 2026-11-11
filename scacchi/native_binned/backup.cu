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
// One warp handles one lane; its data stays in shared memory and registers:
//   0  load the [A, K] edge beliefs; per-row exclusive cumulative sums
//   1  J[q, i]   = prod over legal a of (below[a,i] + t_q h[a,i])
//   2  policy[a] = sum_q w_q sum_i h[a,i] J[q,i] / (below[a,i] + t_q h[a,i])
//   3  rank the moves by decreasing weight (stable); moves of weight exactly
//      zero only scale the law by their total mass, which normalisation removes
//   4  chain: put rank 1 on the lattice, then convolve ranks 2..npos one
//      after another. A rank's law is dense on `cells` lattice cells (binned.
//      _rank_cells); the partial law lives in shared memory and each thread
//      owns 4 outputs. Mass at or past the last cell is the "lump", carried
//      as a scalar and updated from the suffix sums of the rank's law.
//   5  split the lattice law onto the atoms and normalise.
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

namespace ffi = xla::ffi;

namespace {

constexpr int LAT = 129;          // lattice cells, as binned.weighted_sum
constexpr int TOP = LAT - 1;      // last cell, absorbing
constexpr int OFF = 68;           // zero cells in front of the partial law (>= max cells - 1, multiple of 4)
constexpr int PCELLS = OFF + 132; // partial law buffer: zeros, 128 cells, room for cell 128
constexpr int LAWL = 72;          // dense law of one rank >= 2: at most 67 cells, padded to a multiple of 4
constexpr int QPM = 9;            // quadrature nodes per work item in stage 2
constexpr int MAXA = 128;
constexpr int MAXK = 129;
constexpr unsigned FULL = 0xffffffffu;
constexpr float TINY = FLT_MIN;   // jnp.finfo(float32).tiny

#ifdef BACKUP_PROFILE
#define PROF_DECL long long prof_t[16]; long long prof_acc[4] = {0, 0, 0, 0};
#define PROF_MARK(i) do { __syncwarp(); prof_t[i] = clock64(); } while (0)
#define PROF_ACC(slot, a, bb) prof_acc[slot] += prof_t[bb] - prof_t[a]
#else
#define PROF_DECL
#define PROF_MARK(i)
#define PROF_ACC(slot, a, bb)
#endif

__host__ __device__ constexpr int round4(int x) { return (x + 3) & ~3; }

struct Dims {
  int A, K, Q, KS, NP, QJ;
};

inline Dims make_dims(int A, int K) {
  Dims d;
  d.A = A;
  d.K = K;
  d.Q = (A + 1) / 2;
  d.KS = (K & 1) ? K : K + 1;  // odd row stride: lanes reading different rows never share a bank
  d.NP = (d.Q + QPM - 1) / QPM;
  d.QJ = round4(d.NP * QPM);
  return d;
}

struct Offsets {
  // per-lane data that lives through the whole kernel
  int h, w, ordw, vlist, wv, pts, tq, wq, regionA;
  // stages 1-2
  int bp, J, regionB1;
  // chain, final stage
  int pb0, pb1, law, atom, fr, up, regionB2;
  int total;
};

inline Offsets make_offsets(const Dims& d) {
  Offsets o;
  int x = 0;
  o.h = x;      x += round4(d.A * d.KS);
  o.w = x;      x += round4(d.A);
  o.ordw = x;   x += 2 * round4(d.A);
  o.vlist = x;  x += round4(d.A);
  o.wv = x;     x += round4(d.A);
  o.pts = x;    x += round4(d.K);
  o.tq = x;     x += d.QJ;
  o.wq = x;     x += d.QJ;
  o.regionA = x;
  int y = 0;
  o.bp = y;     y += std::max(round4(d.A * d.KS), round4(d.A * d.NP));  // below[] in stage 1, partial policies in stage 2
  o.J = y;      y += round4(d.QJ * d.K);
  o.regionB1 = y;
  int z = 0;
  o.pb0 = z;    z += PCELLS;
  o.pb1 = z;    z += PCELLS;
  const int x0 = z;
  o.law = x0;
  o.atom = x0 + LAWL;  // K atoms of (shift, mass), 2 floats each
  o.fr = x0;       // the lattice-to-atom tables reuse law and atom space after the chain
  o.up = x0 + 132;
  z += round4(std::max(LAWL + 2 * d.K, 264));
  o.regionB2 = z;
  o.total = o.regionA + std::max(o.regionB1, o.regionB2);
  return o;
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(FULL, v, d);
  return v;
}

__device__ __forceinline__ float warp_prod(float v) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) v *= __shfl_xor_sync(FULL, v, d);
  return v;
}

// h / x for normal positive x. IEEE division by default (as the XLA program);
// -DBACKUP_FAST_RECIP uses the hardware reciprocal plus one Newton step, within
// about 1-2 ulp, which is ~10% faster overall and changed no measured error.
__device__ __forceinline__ float divide(float h, float x) {
#ifdef BACKUP_FAST_RECIP
  float r;
  asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
  return h * fmaf(r, fmaf(-x, r, 1.0f), r);
#else
  return h / x;
#endif
}

struct Smem {
  float *h, *w, *wv, *pts, *tq, *wq;
  float2* ordw;
  int* vlist;
  float *bp, *J;
  float *pb0, *pb1, *law, *fr;
  float2* atom;
  int* up;
};

__device__ __forceinline__ Smem carve(float* base, const Offsets& o) {
  Smem s;
  s.h = base + o.h;
  s.w = base + o.w;
  s.ordw = reinterpret_cast<float2*>(base + o.ordw);
  s.vlist = reinterpret_cast<int*>(base + o.vlist);
  s.wv = base + o.wv;
  s.pts = base + o.pts;
  s.tq = base + o.tq;
  s.wq = base + o.wq;
  float* b = base + o.regionA;
  s.bp = b + o.bp;
  s.J = b + o.J;
  s.pb0 = b + o.pb0;
  s.pb1 = b + o.pb1;
  s.law = b + o.law;
  s.atom = reinterpret_cast<float2*>(b + o.atom);
  s.fr = b + o.fr;
  s.up = reinterpret_cast<int*>(b + o.up);
  return s;
}

// Shift of every atom of an edge with weight w, in lattice cells (a real
// number, as binned.weighted_sum.on_lattice), and the atom's mass.
template <int KT>
__device__ __forceinline__ void prep_atoms(const Smem& S, float w, const float* __restrict__ hr, int K, int lane) {
  for (int j = lane; j < K; j += 32) S.atom[j] = make_float2((w * S.pts[j]) * (float)TOP, hr[j]);
  __syncwarp();
}

// Cells c = lane + 32 s (s < nslots) of the law of one edge on the lattice.
// Splitting a mass h at real position x between its two neighbouring cells
// puts h (1 - part) on floor(x) and h part on floor(x) + 1, part = x - floor(x):
// a hat function of c - x. Written as min(1 - (x - c), x - (c - 1)), clipped at
// zero, so that both values are the exact numbers binned.weighted_sum uses
// (x - c and x - (c - 1) are exact where they are selected). The weights sum to
// at most one, so x <= 128 / rank and nothing needs clamping to the last cell.
// Every lane sums over all atoms: no search, no divergence, broadcast loads.
template <int SMAX, int KT>
__device__ __forceinline__ void law_cells(float (&lv)[SMAX], const Smem& S, int K, int lane, int nslots) {
  constexpr int UK = KT > 0 ? (KT > 16 ? 8 : KT) : 4;
#pragma unroll
  for (int s = 0; s < SMAX; ++s) {
    float acc = 0.f;
    if (s < nslots) {
      const float c = (float)(lane + 32 * s);
      const float cm1 = c - 1.f;
#pragma unroll UK
      for (int j = 0; j < K; ++j) {
        const float2 a2 = S.atom[j];
        acc = fmaf(a2.y, fmaxf(fminf(1.f - (a2.x - c), a2.x - cm1), 0.f), acc);
      }
    }
    lv[s] = acc;
  }
}

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
  const int A = d.A, Q = d.Q, KS = d.KS, NP = d.NP, QJ = d.QJ;
  constexpr int UK = KT > 0 ? (KT > 16 ? 8 : KT) : 4;
  float* o = out + (size_t)b * K;
  if (!repaired[b]) {
    for (int k = lane; k < K; k += 32) o[k] = 0.f;
    return;
  }
  const Smem S = carve(smem + (size_t)wid * warp_floats, off);
  PROF_DECL
  PROF_MARK(0);

  // ---- stage 0: load -------------------------------------------------------
  const float* e = edges + (size_t)b * A * K;
  const uint8_t* inv = invalid + (size_t)b * A;
  const float Kinv = 1.0f / (float)K;
  for (int k = lane; k < K; k += 32) S.pts[k] = points[k];
  for (int q = lane; q < QJ; q += 32) {
    S.tq[q] = q < Q ? nodes[q] : 0.f;
    S.wq[q] = q < Q ? qweights[q] : 0.f;
  }
  int nv = 0;  // legal moves, compacted in increasing action order
  for (int a0 = 0; a0 < A; a0 += 32) {
    const int a = a0 + lane;
    const bool ok = a < A && inv[a] == 0;
    const unsigned m = __ballot_sync(FULL, ok);
    if (ok) S.vlist[nv + __popc(m & ((1u << lane) - 1u))] = a;
    nv += __popc(m);
  }
  for (int a = lane; a < A; a += 32) S.w[a] = 0.f;
  for (int idx = lane; idx < A * K; idx += 32) {
    const int a = (int)(((float)idx + 0.5f) * Kinv);
    S.h[a * KS + (idx - a * K)] = __ldg(e + idx);
  }
  __syncwarp();
  for (int a = lane; a < A; a += 32) {
    if (!inv[a]) {
      const float* row = S.h + a * KS;
      float* br = S.bp + a * KS;
      float cs = 0.f;
#pragma unroll UK
      for (int k = 0; k < K; ++k) {
        const float x = row[k];
        cs += x;
        br[k] = fmaxf(cs - x, 0.f);  // mass strictly below atom k
      }
    }
  }
  __syncwarp();
  PROF_MARK(1);

  // ---- stage 1: J[q, i] = prod_a (below[a,i] + t_q h[a,i]) over legal a ------
  {
    const int nJ = (QJ >> 2) * K;
    for (int it = lane; it < nJ; it += 32) {
      const int g = (int)(((float)it + 0.5f) * Kinv);
      const int i = it - g * K;
      const float t0 = S.tq[4 * g], t1 = S.tq[4 * g + 1], t2 = S.tq[4 * g + 2], t3 = S.tq[4 * g + 3];
      float j0 = 1.f, j1 = 1.f, j2 = 1.f, j3 = 1.f;
#pragma unroll 4
      for (int v = 0; v < nv; ++v) {
        const int a = S.vlist[v];
        const float hh = S.h[a * KS + i], lo = S.bp[a * KS + i];
        j0 *= fmaf(t0, hh, lo);
        j1 *= fmaf(t1, hh, lo);
        j2 *= fmaf(t2, hh, lo);
        j3 *= fmaf(t3, hh, lo);
      }
      *reinterpret_cast<float4*>(S.J + i * QJ + 4 * g) = make_float4(j0, j1, j2, j3);
    }
  }
  __syncwarp();
  PROF_MARK(2);

  // ---- stage 2: policy -----------------------------------------------------
  {
    const float nvinv = nv > 0 ? 1.0f / (float)nv : 0.f;
    const int nItems = nv * NP;
    for (int it = lane; it < nItems; it += 32) {
      const int p = (int)(((float)it + 0.5f) * nvinv);
      const int v = it - p * nv;
      const float* hr = S.h + S.vlist[v] * KS;
      const int q0 = p * QPM;
      float tqr[QPM], bn[QPM];
#pragma unroll
      for (int qq = 0; qq < QPM; ++qq) {
        tqr[qq] = S.tq[q0 + qq];
        bn[qq] = 0.f;
      }
      float cs = 0.f;
#pragma unroll UK
      for (int i = 0; i < K; ++i) {
        const float hh = hr[i];
        cs += hh;
        const float lo = fmaxf(cs - hh, 0.f);
        const float* Ji = S.J + i * QJ + q0;
#pragma unroll
        for (int qq = 0; qq < QPM; ++qq) {
          const float own = fmaf(tqr[qq], hh, lo);
          const float share = divide(hh, fmaxf(own, TINY));
          bn[qq] = fmaf(Ji[qq], share, bn[qq]);
        }
      }
      float pol = 0.f;
#pragma unroll
      for (int qq = 0; qq < QPM; ++qq) pol = fmaf(S.wq[q0 + qq], bn[qq], pol);
      S.bp[v * NP + p] = pol;
    }
  }
  __syncwarp();
  // Normalise in double: the weights then sum to one up to the rounding of each
  // weight to float32 (about 1e-9 for 49 moves). A float32 total would leave the
  // sum off by up to 1e-7, which the lattice law amplifies when mass piles up
  // at the last cell.
  double tot = 0.0;
  for (int v = lane; v < nv; v += 32) {
    float s = 0.f;
    for (int p = 0; p < NP; ++p) s += S.bp[v * NP + p];
    S.wv[v] = s;
    tot += (double)s;
  }
#pragma unroll
  for (int dd = 16; dd > 0; dd >>= 1) tot += __shfl_xor_sync(FULL, tot, dd);
  __syncwarp();
  {
    const bool good = tot > 0.0;
    const double denom = fmax(tot, (double)TINY);
    const float uniform = 1.f / fmaxf((float)nv, 1.f);
    for (int v = lane; v < nv; v += 32) {
      const float x = good ? (float)((double)S.wv[v] / denom) : uniform;
      S.wv[v] = x;
      S.w[S.vlist[v]] = x;
    }
  }
  __syncwarp();
  PROF_MARK(3);

  // ---- stage 3: rank by decreasing weight, ties by action index -------------
  int npos = 0;
  for (int v0 = 0; v0 < nv; v0 += 32) {
    const int v = v0 + lane;
    bool pos = false;
    if (v < nv) {
      const float wa = S.wv[v];
      pos = wa > 0.f;
      if (pos) {
        int rank = 0;
        for (int u = 0; u < nv; ++u) {
          const float wu = S.wv[u];
          rank += (wu > wa) | ((wu == wa) & (u < v));
        }
        S.ordw[rank] = make_float2(__int_as_float(S.vlist[v]), wa);
      }
    }
    npos += __popc(__ballot_sync(FULL, pos));
  }
  // Edges with weight exactly zero shift by zero cells: they only scale the
  // law by their total mass, which the final normalisation divides out.
  float scale = 1.f;
  for (int a = lane; a < A; a += 32) {
    if (!(S.w[a] > 0.f)) {
      const float* row = S.h + a * KS;
      float cs = 0.f;
      for (int k = 0; k < K; ++k) cs += row[k];
      scale *= cs;
    }
  }
  scale = warp_prod(scale);
  __syncwarp();
  PROF_MARK(4);

  // ---- stage 4: the chain of convolutions ------------------------------------
  float* cur = S.pb0;
  float* nxt = S.pb1;
  for (int i = lane; i < OFF; i += 32) { S.pb0[i] = 0.f; S.pb1[i] = 0.f; }
  float lump = 0.f;
  if (npos == 0) {
    for (int c = lane; c < TOP; c += 32) cur[OFF + c] = c == 0 ? 1.f : 0.f;
  } else {
    const float2 ow = S.ordw[0];
    PROF_MARK(11);
    prep_atoms<KT>(S, ow.y, S.h + __float_as_int(ow.x) * KS, K, lane);
    PROF_MARK(12);
    float lv[5];
    law_cells<5, KT>(lv, S, K, lane, 5);
    PROF_MARK(13);
#pragma unroll
    for (int s = 0; s < 4; ++s) cur[OFF + lane + 32 * s] = lv[s];
    lump = __shfl_sync(FULL, lv[4], 0);
    PROF_MARK(14);
  }
  __syncwarp();
  PROF_MARK(5);

  int first = 2;
  for (int r = 1; r < npos; ++r) {
    // Group of ranks with the same number of cells, as binned._rank_cells.
    while (r + 1 > 2 * first - 2) first = 2 * first - 1;
    const int C = min(LAT, (LAT - 1) / first + 3);
    const int ng = (C + 3) >> 2;
    const float2 ow = S.ordw[r];
    prep_atoms<KT>(S, ow.y, S.h + __float_as_int(ow.x) * KS, K, lane);
    float lv[3] = {0.f, 0.f, 0.f};
    const int S3 = (4 * ng + 31) >> 5;
    law_cells<3, KT>(lv, S, K, lane, (C + 31) >> 5);
#pragma unroll
    for (int s = 0; s < 3; ++s) {
      const int c = lane + 32 * s;
      if (c < 4 * ng) S.law[c] = lv[s];
    }
    __syncwarp();
    PROF_MARK(6);

    // Mass that reaches the last cell: lump' = lump T[0] + sum_{j=1}^{C-1} p[128 - j] T[j],
    // T[j] = sum_{m >= j} law[m], by suffix scans over the lanes' cells.
    {
      float Tv[3] = {0.f, 0.f, 0.f};
      float above = 0.f;
#pragma unroll
      for (int sl = 2; sl >= 0; --sl) {
        if (sl < S3) {
          float v = lv[sl];
#pragma unroll
          for (int dd = 1; dd < 32; dd <<= 1) {
            const float u = __shfl_down_sync(FULL, v, dd);
            v += (lane + dd < 32) ? u : 0.f;
          }
          Tv[sl] = v + above;
          above += __shfl_sync(FULL, v, 0);
        }
      }
      float cross = 0.f;
#pragma unroll
      for (int sl = 0; sl < 3; ++sl) {
        const int j = lane + 32 * sl;
        if (sl < S3 && j >= 1 && j < C) cross = fmaf(cur[OFF + TOP - j], Tv[sl], cross);
      }
      cross = warp_sum(cross);
      lump = fmaf(lump, __shfl_sync(FULL, Tv[0], 0), cross);
    }
    PROF_MARK(7);

    // new[n] = sum_m law[m] p[n - m], n < 128; thread t owns n = 4t..4t+3 and
    // slides a two-chunk window over p while m advances by 4.
    const float4* pc = reinterpret_cast<const float4*>(cur + OFF);
    const float4* lw4 = reinterpret_cast<const float4*>(S.law);
    float4 c0 = pc[lane];
    float a0v = 0.f, a1v = 0.f, a2v = 0.f, a3v = 0.f;
#pragma unroll 2
    for (int g = 0; g < ng; ++g) {
      const float4 lw = lw4[g];
      const float4 pr = pc[lane - g - 1];
      a0v = fmaf(lw.x, c0.x, a0v); a0v = fmaf(lw.y, pr.w, a0v); a0v = fmaf(lw.z, pr.z, a0v); a0v = fmaf(lw.w, pr.y, a0v);
      a1v = fmaf(lw.x, c0.y, a1v); a1v = fmaf(lw.y, c0.x, a1v); a1v = fmaf(lw.z, pr.w, a1v); a1v = fmaf(lw.w, pr.z, a1v);
      a2v = fmaf(lw.x, c0.z, a2v); a2v = fmaf(lw.y, c0.y, a2v); a2v = fmaf(lw.z, c0.x, a2v); a2v = fmaf(lw.w, pr.w, a2v);
      a3v = fmaf(lw.x, c0.w, a3v); a3v = fmaf(lw.y, c0.z, a3v); a3v = fmaf(lw.z, c0.y, a3v); a3v = fmaf(lw.w, c0.x, a3v);
      c0 = pr;
    }
    reinterpret_cast<float4*>(nxt + OFF)[lane] = make_float4(a0v, a1v, a2v, a3v);
    __syncwarp();
    PROF_MARK(8);
    PROF_ACC(0, 5, 6); PROF_ACC(1, 6, 7); PROF_ACC(2, 7, 8);
#ifdef BACKUP_PROFILE
    prof_t[5] = prof_t[8];
#endif
    float* t = cur; cur = nxt; nxt = t;
  }
  PROF_MARK(9);

  // ---- stage 5: lattice -> atoms, normalise -----------------------------------
  if (lane == 0) cur[OFF + TOP] = lump;
  for (int s = 0; s < 5; ++s) {
    const int c = lane + 32 * s;
    if (c < LAT) {
      const float fine = (float)c * 0.0078125f;  // c / 128, exact
      int cnt = 0;                               // searchsorted(points, fine, "right")
#pragma unroll UK
      for (int k = 0; k < K; ++k) cnt += (S.pts[k] <= fine);
      const int u = min(max(cnt, 1), K - 1);
      const float den = S.pts[u] - S.pts[u - 1];
      S.up[c] = u;
      S.fr[c] = fminf(fmaxf((fine - S.pts[u - 1]) / den, 0.f), 1.f);
    }
  }
  __syncwarp();
  float rk[5];
  float tsum = 0.f;
#pragma unroll
  for (int s = 0; s < 5; ++s) {
    const int k = lane + 32 * s;
    float r = 0.f;
    if (k < K) {
      int lo = 0, hi = LAT;  // cells with up in {k, k + 1} are contiguous: [lo, end)
      while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (S.up[mid] >= k) hi = mid; else lo = mid + 1;
      }
      int end = lo, top = LAT;
      while (end < top) {
        const int mid = (end + top) >> 1;
        if (S.up[mid] > k + 1) top = mid; else end = mid + 1;
      }
#pragma unroll 4
      for (int c = lo; c < end; ++c) {
        const float pcell = cur[OFF + c];
        const float f = S.fr[c];
        r = fmaf(pcell, S.up[c] == k ? f : 1.f - f, r);
      }
      r *= scale;
      tsum += r;
    }
    rk[s] = r;
  }
  tsum = warp_sum(tsum);
  const float denom = fmaxf(tsum, TINY);
#pragma unroll
  for (int s = 0; s < 5; ++s) {
    const int k = lane + 32 * s;
    if (k < K) o[k] = rk[s] / denom;
  }
#ifdef BACKUP_PROFILE
  PROF_MARK(10);
  if (b == (B >> 1) && lane == 0) {
    printf("PROF r1 split: zero %lld prep %lld law %lld rest %lld\n", prof_t[11]-prof_t[4], prof_t[12]-prof_t[11], prof_t[13]-prof_t[12], prof_t[5]-prof_t[13]);
    printf("PROF r1 rest split: stores+shfl %lld final-sync %lld\n", prof_t[14]-prof_t[13], prof_t[5]-prof_t[14]);
    printf("PROF cycles: load %lld J %lld policy %lld rank %lld rank1 %lld | per-step law %lld lump %lld conv %lld (steps %d) | final %lld total %lld\n",
           prof_t[1] - prof_t[0], prof_t[2] - prof_t[1], prof_t[3] - prof_t[2], prof_t[4] - prof_t[3], prof_t[5] - prof_t[4],
           prof_acc[0], prof_acc[1], prof_acc[2], npos - 1, prof_t[10] - prof_t[9], prof_t[10] - prof_t[0]);
  }
#endif
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
