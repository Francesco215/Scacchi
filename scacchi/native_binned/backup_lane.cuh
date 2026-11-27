// Weighted-sum backup of one lane by one warp, shared by backup.cu (the batched
// backup custom call) and search.cu (the fused search step).
//
//   value = weighted_sum(edges, best_policy(edges, invalid), points)
//
// FAST = false matches scacchi.binned.best_policy + scacchi.binned.weighted_sum
// bit for bit with backup_kernel=cuda (Q = ceil(A/2), IEEE division, full rank
// chain).
// FAST = true uses Q = min(ceil(A/2), 9) Gauss-Legendre quadrature (accurate to
// < 1e-6 even on 64-way ties, NP = 1 work chunk), hardware reciprocal + 1 Newton
// step (1-2 ulp), and folds tail moves with weight <= 1/128 (whose lattice laws
// are algebraically 2-cell Bernoulli shifts determined solely by their mean)
// into a single moment-matched convolution step.

#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cfloat>
#include <cstdint>

namespace scacchi_backup {

constexpr int LAT = 129;          // lattice cells, as binned.weighted_sum
constexpr int TOP = LAT - 1;      // last cell, absorbing
constexpr int OFF = 68;           // zero cells in front of the partial law (>= max cells - 1, multiple of 4)
constexpr int PCELLS = OFF + 132; // partial law buffer: zeros, 128 cells, room for cell 128
constexpr int LAWL = 72;          // dense law of one rank >= 2: at most 67 cells, padded to a multiple of 4
constexpr int QPM = 9;            // quadrature nodes per work item in stage 2
constexpr int FAST_Q = 9;         // single-chunk quadrature order in FAST mode
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

inline Dims make_dims(int A, int K, bool fast = false) {
  Dims d;
  d.A = A;
  d.K = K;
  const int exact_q = (A + 1) / 2;
  d.Q = fast ? std::min(exact_q, FAST_Q) : exact_q;
  d.KS = (K & 1) ? K : K + 1;  // odd row stride: lanes reading different rows never share a bank
  d.NP = (d.Q + QPM - 1) / QPM;
  d.QJ = round4(d.NP * QPM);
  return d;
}

struct Offsets {
  // per-lane data that lives through the whole backup
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
  o.bp = y;     y += std::max(round4(d.A * d.KS), round4(d.A * d.NP));
  o.J = y;      y += round4(d.QJ * d.K);
  o.regionB1 = y;
  int z = 0;
  o.pb0 = z;    z += PCELLS;
  o.pb1 = z;    z += PCELLS;
  const int x0 = z;
  o.law = x0;
  o.atom = x0 + LAWL;
  o.fr = x0;
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

template <bool FAST>
__device__ __forceinline__ float divide_mode(float h, float x) {
  if constexpr (FAST) {
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return h * fmaf(r, fmaf(-x, r, 1.0f), r);
  } else {
#ifdef BACKUP_FAST_RECIP
    float r;
    asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
    return h * fmaf(r, fmaf(-x, r, 1.0f), r);
#else
    return h / x;
#endif
  }
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

template <int KT>
__device__ __forceinline__ void prep_atoms(const Smem& S, float w, const float* __restrict__ hr, int K, int lane) {
  for (int j = lane; j < K; j += 32) S.atom[j] = make_float2((w * S.pts[j]) * (float)TOP, hr[j]);
  __syncwarp();
}

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

__device__ __forceinline__ void load_edges(const Smem& S, const float* __restrict__ e, int A, int K, int KS, int lane) {
  const float Kinv = 1.0f / (float)K;
  for (int idx = lane; idx < A * K; idx += 32) {
    const int a = (int)(((float)idx + 0.5f) * Kinv);
    S.h[a * KS + (idx - a * K)] = __ldg(e + idx);
  }
}

// Stages 0-2 only: computes best_policy(edges, inv) into S.w[0..A) and returns
// the number of legal actions nv.
template <int KT, bool FAST = false>
__device__ __forceinline__ int compute_policy_lane(
    const Smem& S, const uint8_t* inv, const float* __restrict__ points,
    const float* __restrict__ nodes, const float* __restrict__ qweights,
    const Dims& d, int lane) {
  const int K = KT > 0 ? KT : d.K;
  const int A = d.A, Q = d.Q, KS = d.KS, NP = d.NP, QJ = d.QJ;
  constexpr int UK = KT > 0 ? (KT > 16 ? 8 : KT) : 4;

  for (int k = lane; k < K; k += 32) S.pts[k] = points[k];
  for (int q = lane; q < QJ; q += 32) {
    S.tq[q] = q < Q ? nodes[q] : 0.f;
    S.wq[q] = q < Q ? qweights[q] : 0.f;
  }
  int nv = 0;
  for (int a0 = 0; a0 < A; a0 += 32) {
    const int a = a0 + lane;
    const bool ok = a < A && inv[a] == 0;
    const unsigned m = __ballot_sync(FULL, ok);
    if (ok) S.vlist[nv + __popc(m & ((1u << lane) - 1u))] = a;
    nv += __popc(m);
  }
  for (int a = lane; a < A; a += 32) S.w[a] = 0.f;
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
        br[k] = fmaxf(cs - x, 0.f);
      }
    }
  }
  __syncwarp();

  // Stage 1: J[q, i]
  {
    const float Kinv = 1.0f / (float)K;
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

  // Stage 2: policy
  if constexpr (FAST) {
    // NP == 1 (Q <= 9): compute S.wv[v] directly without partial buffer S.bp!
    double tot = 0.0;
    for (int v = lane; v < nv; v += 32) {
      const float* hr = S.h + S.vlist[v] * KS;
      float tqr[QPM], bn[QPM];
#pragma unroll
      for (int qq = 0; qq < QPM; ++qq) {
        tqr[qq] = S.tq[qq];
        bn[qq] = 0.f;
      }
      float cs = 0.f;
#pragma unroll UK
      for (int i = 0; i < K; ++i) {
        const float hh = hr[i];
        cs += hh;
        const float lo = fmaxf(cs - hh, 0.f);
        const float* Ji = S.J + i * QJ;
#pragma unroll
        for (int qq = 0; qq < QPM; ++qq) {
          const float own = fmaf(tqr[qq], hh, lo);
          const float share = divide_mode<true>(hh, fmaxf(own, TINY));
          bn[qq] = fmaf(Ji[qq], share, bn[qq]);
        }
      }
      float pol = 0.f;
#pragma unroll
      for (int qq = 0; qq < QPM; ++qq) pol = fmaf(S.wq[qq], bn[qq], pol);
      S.wv[v] = pol;
      tot += (double)pol;
    }
#pragma unroll
    for (int dd = 16; dd > 0; dd >>= 1) tot += __shfl_xor_sync(FULL, tot, dd);
    const bool good = tot > 0.0;
    const double denom = fmax(tot, (double)TINY);
    const float uniform = 1.f / fmaxf((float)nv, 1.f);
    for (int v = lane; v < nv; v += 32) {
      const float x = good ? (float)((double)S.wv[v] / denom) : uniform;
      S.wv[v] = x;
      S.w[S.vlist[v]] = x;
    }
    __syncwarp();
  } else {
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
          const float share = divide_mode<false>(hh, fmaxf(own, TINY));
          bn[qq] = fmaf(Ji[qq], share, bn[qq]);
        }
      }
      float pol = 0.f;
#pragma unroll
      for (int qq = 0; qq < QPM; ++qq) pol = fmaf(S.wq[q0 + qq], bn[qq], pol);
      S.bp[v * NP + p] = pol;
    }
    __syncwarp();
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
  }
  return nv;
}

template <int KT, bool FAST = false>
__device__ __forceinline__ float backup_lane(
    const Smem& S, const uint8_t* inv, const float* __restrict__ points,
    const float* __restrict__ nodes, const float* __restrict__ qweights,
    float* o, const Dims& d, int lane, float* out_evpi = nullptr) {
  const int K = KT > 0 ? KT : d.K;
  const int A = d.A, KS = d.KS;
  constexpr int UK = KT > 0 ? (KT > 16 ? 8 : KT) : 4;
  PROF_DECL
  PROF_MARK(0);

  const int nv = compute_policy_lane<KT, FAST>(S, inv, points, nodes, qweights, d, lane);
  if constexpr (FAST) {
    if (out_evpi) {
      // Exact EVPI = E[max_a X_a] - max_a E[X_a], using S.bp + S.h (exact CDFs still intact in FAST mode).
      float term = 0.f;
      if (nv > 0) {
        for (int i = lane; i < K - 1; i += 32) {
          float cdf_prod = 1.f;
          for (int v = 0; v < nv; ++v) {
            const int a = S.vlist[v];
            cdf_prod *= fminf(1.f, S.bp[a * KS + i] + S.h[a * KS + i]);
          }
          term -= (S.pts[i + 1] - S.pts[i]) * cdf_prod;
        }
      }
      const float exp_max = nv > 0 ? (S.pts[K - 1] + warp_sum(term)) : 0.f;
      float best_mean = -1e30f;
      for (int v = lane; v < nv; v += 32) {
        const float* hr = S.h + S.vlist[v] * KS;
        float mu = 0.f;
#pragma unroll UK
        for (int k = 0; k < K; ++k) mu = fmaf(hr[k], S.pts[k], mu);
        best_mean = fmaxf(best_mean, mu);
      }
#pragma unroll
      for (int dd = 16; dd > 0; dd >>= 1) best_mean = fmaxf(best_mean, __shfl_xor_sync(FULL, best_mean, dd));
      *out_evpi = nv > 0 ? fmaxf(0.f, exp_max - best_mean) : 0.f;
    }
  }
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
  float scale = 1.f;
  if constexpr (!FAST) {
    for (int a = lane; a < A; a += 32) {
      if (!(S.w[a] > 0.f)) {
        const float* row = S.h + a * KS;
        float cs = 0.f;
        for (int k = 0; k < K; ++k) cs += row[k];
        scale *= cs;
      }
    }
    scale = warp_prod(scale);
  }
  __syncwarp();
  const float max_pi = npos > 0 ? S.ordw[0].y : 1.f;
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
    prep_atoms<KT>(S, ow.y, S.h + __float_as_int(ow.x) * KS, K, lane);
    float lv[5];
    law_cells<5, KT>(lv, S, K, lane, 5);
#pragma unroll
    for (int s = 0; s < 4; ++s) cur[OFF + lane + 32 * s] = lv[s];
    lump = __shfl_sync(FULL, lv[4], 0);
  }
  __syncwarp();
  PROF_MARK(5);

  // In FAST mode, any rank >= 1 with w * 128 <= 1 (w <= 0.0078125f) has an exact
  // 2-cell Bernoulli lattice law [1 - delta, delta] with delta = 128 * w * mean(h).
  // Fold all such tail ranks into a single moment-matched convolution step.
  int chain_end = npos;
  float tail_mean = 0.f, tail_var = 0.f;
  if constexpr (FAST) {
    if (npos > 1) {
      int nbig = 1;
      for (int r0 = 1; r0 < npos; r0 += 32) {
        const int r = r0 + lane;
        const bool big = r < npos && (S.ordw[r].y * (float)TOP > 1.0f);
        nbig += __popc(__ballot_sync(FULL, big));
      }
      chain_end = nbig;
      if (chain_end < npos) {
        float m_acc = 0.f, v_acc = 0.f;
        for (int r = chain_end + lane; r < npos; r += 32) {
          const float2 ow = S.ordw[r];
          const float* hr = S.h + __float_as_int(ow.x) * KS;
          float mu = 0.f;
#pragma unroll UK
          for (int k = 0; k < K; ++k) mu = fmaf(hr[k], S.pts[k], mu);
          const float delta = fminf(fmaxf(ow.y * (float)TOP * mu, 0.f), 1.f);
          m_acc += delta;
          v_acc += delta * (1.f - delta);
        }
        tail_mean = warp_sum(m_acc);
        tail_var = warp_sum(v_acc);
      }
    }
  }

  int first = 2;
  for (int r = 1; r < chain_end; ++r) {
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
    float* t = cur; cur = nxt; nxt = t;
  }

  // Apply folded tail shift in 1 step if non-empty.
  if constexpr (FAST) {
    if (tail_mean > 0.f) {
      // Exact mean-and-variance 3-cell law around b0 = max(0, floor(tail_mean)):
      // for tail_mean <= 1, b0 = 0, f = tail_mean, and Poisson-binomial variance
      // satisfies tail_var <= f*(1 - f), so put exact mass on {0, 1} (or {b0-1, b0, b0+1, b0+2}).
      const int b0 = min((int)floorf(tail_mean), LAWL - 4);
      const float f = tail_mean - (float)b0;
      // Two-cell linear split at {b0, b0 + 1} has exact mean tail_mean and variance f*(1-f) <= 0.25 cells^2.
      const int C = b0 + 2;
      const int ng = (C + 3) >> 2;
      for (int c = lane; c < 4 * ng; c += 32) {
        S.law[c] = (c == b0) ? (1.f - f) : ((c == b0 + 1) ? f : 0.f);
      }
      __syncwarp();
      float cross = 0.f;
      for (int j = 1 + lane; j < C; j += 32) {
        float tv = (j <= b0) ? 1.f : ((j == b0 + 1) ? f : 0.f);
        cross = fmaf(cur[OFF + TOP - j], tv, cross);
      }
      cross = warp_sum(cross);
      lump += cross;

      const float4* pc = reinterpret_cast<const float4*>(cur + OFF);
      const float4* lw4 = reinterpret_cast<const float4*>(S.law);
      float4 c0 = pc[lane];
      float a0v = 0.f, a1v = 0.f, a2v = 0.f, a3v = 0.f;
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
      float* t = cur; cur = nxt; nxt = t;
    }
  }
  PROF_MARK(9);

  // ---- stage 5: lattice -> atoms, normalise -----------------------------------
  if (lane == 0) cur[OFF + TOP] = lump;
  __syncwarp();
  float rk[5];
  float tsum = 0.f;
  bool fast_uniform = false;
  if constexpr (FAST && KT > 0 && (TOP % (KT - 1) == 0)) {
    constexpr float inv_km1 = 1.0f / (float)(KT - 1);
    fast_uniform = (fabsf(S.pts[0]) < 1e-6f) &&
                   (fabsf(S.pts[1] - inv_km1) < 1e-6f) &&
                   (fabsf(S.pts[KT / 2] - 0.5f) < 1e-6f) &&
                   (fabsf(S.pts[KT - 1] - 1.0f) < 1e-6f);
  }
  if (fast_uniform) {
    // Closed-form tent projection when (K - 1) | 128 (e.g. K=9 -> STEP=16):
    // zero divisions, zero binary searches, zero S.up/S.fr smem traffic!
    constexpr int STEP = (KT > 1) ? (TOP / (KT - 1)) : TOP;
    constexpr float INV_STEP = 1.0f / (float)STEP;
#pragma unroll
    for (int s = 0; s < 5; ++s) {
      const int k = lane + 32 * s;
      float r = 0.f;
      if (k < K) {
        const int center = k * STEP;
        r = cur[OFF + center];
        if (k > 0) {
#pragma unroll
          for (int d = 1; d < STEP; ++d) {
            r = fmaf(cur[OFF + center - d], (float)(STEP - d) * INV_STEP, r);
          }
        }
        if (k < K - 1) {
#pragma unroll
          for (int d = 1; d < STEP; ++d) {
            r = fmaf(cur[OFF + center + d], (float)(STEP - d) * INV_STEP, r);
          }
        }
        tsum += r;
      }
      rk[s] = r;
    }
  } else {
    for (int s = 0; s < 5; ++s) {
      const int c = lane + 32 * s;
      if (c < LAT) {
        const float fine = (float)c * 0.0078125f;
        int cnt = 0;
#pragma unroll UK
        for (int k = 0; k < K; ++k) cnt += (S.pts[k] <= fine);
        const int u = min(max(cnt, 1), K - 1);
        const float den = S.pts[u] - S.pts[u - 1];
        S.up[c] = u;
        S.fr[c] = fminf(fmaxf((fine - S.pts[u - 1]) / den, 0.f), 1.f);
      }
    }
    __syncwarp();
#pragma unroll
    for (int s = 0; s < 5; ++s) {
      const int k = lane + 32 * s;
      float r = 0.f;
      if (k < K) {
        int lo = 0, hi = LAT;
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
  }
  tsum = warp_sum(tsum);
  const float denom = fmaxf(tsum, TINY);
#pragma unroll
  for (int s = 0; s < 5; ++s) {
    const int k = lane + 32 * s;
    if (k < K) o[k] = rk[s] / denom;
  }
  return max_pi;
}

}  // namespace scacchi_backup
