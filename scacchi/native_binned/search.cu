// Fused step of the binned Thompson tree search on Hex, one warp per lane.
//
// Provides two XLA FFI handlers:
//   1. ScacchiBinnedSearchStep: exact bitwise match to the JAX binned search
//      (used by search_kernel=fused in bitwise reference mode and existing tests).
//   2. ScacchiBinnedSearchFastStep: full fast search pipeline with:
//        - in-place topological subtree reuse across moves (F_SUBTREE_INIT),
//        - dynamic per-lane node capacity and confidence early-stopping,
//        - terminal-leaf & depth-revisit network-eval skipping,
//        - read-only GPU position-cache probing + atomic active-board compaction
//          for board_net_int8_kernel (CUDA-Graph / command-buffer compatible),
//        - fast Q=9 + Newton-reciprocal + tail-folded weighted-sum backup with
//          single-changed-child refresh & pre-backup proof check,
//        - fused root policy & SearchSummary readout (F_READOUT).

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "xla/ffi/api/ffi.h"
#include "backup_lane.cuh"
#include "cache_key.cuh"

namespace ffi = xla::ffi;

namespace {

using namespace scacchi_backup;
using scacchi_net::CACHE_MASK;
using scacchi_net::CACHE_SLOTS;
using scacchi_net::CacheKey;
using scacchi_net::hash_cache_key;

static_assert(sizeof(CacheKey) == 40, "CacheKey must be 5 x uint64_t (40 bytes)");

constexpr int NOUT = -1;       // dirichlet_mctx.outcomes.NO_OUTCOME
constexpr int NDIST = -1;      // dirichlet_mctx.outcomes.NO_DISTANCE
constexpr int PEND = 8;
constexpr int MAX_CELLS = 127; // int8 board ids (Hex 11 has 121 cells)
constexpr int MAX_WARPS = 4;   // lanes per block, at most; chosen for occupancy

struct Geometry {
  int B, N, A, K, C, size, max_depth;
  Dims d;
  Offsets off;
  int warp_floats;
};

struct TreeRefs {
  int32_t* parents;   // [B, N]
  int32_t* children;  // [B, N, A]
  int32_t* to_play;   // [B, N]
  int8_t* node_out;   // [B, N]
  int32_t* node_pay;  // [B, N]
  int8_t* edge_out;   // [B, N, A]
  int32_t* edge_pay;  // [B, N, A]
  float* vprior;      // [B, N, K]
  float* valpha;      // [B, N, K]
  float* ealpha;      // [B, N, A, K]
  uint8_t* invalid;   // [B, N, A]
  int8_t* board;      // [B, N, C]
  int32_t* steps;     // [B, N]
  int32_t* pending;   // [B, PEND]
};

struct Inputs {
  const float* net_value;  // [B, K]
  const float* net_q;      // [B, A, K]
  const uint32_t* keys;    // [B, 2]
  const int32_t* order;    // [B, 2] pgx _player_order
  const float* points;     // [K]
  const float* qnodes;     // [Q]
  const float* qweights;   // [Q]
  const int32_t* sim;      // []
  uint8_t* obs;            // [B, size, size, 4]
};

// ---- jax.random, partitionable threefry2x32 ---------------------------------

__device__ __forceinline__ uint32_t rotl32(uint32_t x, int r) { return (x << r) | (x >> (32 - r)); }

__device__ __forceinline__ uint2 threefry(uint32_t k0, uint32_t k1, uint32_t x0, uint32_t x1) {
  const uint32_t k2 = k0 ^ k1 ^ 0x1BD11BDAu;
  x0 += k0;
  x1 += k1;
#define TF_ROUND(r) x0 += x1; x1 = rotl32(x1, r); x1 ^= x0;
  TF_ROUND(13) TF_ROUND(15) TF_ROUND(26) TF_ROUND(6)
  x0 += k1; x1 += k2 + 1u;
  TF_ROUND(17) TF_ROUND(29) TF_ROUND(16) TF_ROUND(24)
  x0 += k2; x1 += k0 + 2u;
  TF_ROUND(13) TF_ROUND(15) TF_ROUND(26) TF_ROUND(6)
  x0 += k0; x1 += k1 + 3u;
  TF_ROUND(17) TF_ROUND(29) TF_ROUND(16) TF_ROUND(24)
  x0 += k1; x1 += k2 + 4u;
  TF_ROUND(13) TF_ROUND(15) TF_ROUND(26) TF_ROUND(6)
  x0 += k2; x1 += k0 + 5u;
#undef TF_ROUND
  return make_uint2(x0, x1);
}

__device__ __forceinline__ uint2 split_key(uint2 key, uint32_t i) { return threefry(key.x, key.y, 0u, i); }

__device__ __forceinline__ uint32_t random_bits(uint2 key, uint32_t i) {
  const uint2 r = threefry(key.x, key.y, 0u, i);
  return r.x ^ r.y;
}

__device__ __forceinline__ float uniform01(uint2 key, uint32_t i) {
  return __uint_as_float((random_bits(key, i) >> 9) | 0x3F800000u) - 1.0f;
}

__device__ __forceinline__ float gumbel_low(uint2 key, uint32_t i) {
  const float f = uniform01(key, i);
  const float u = fmaxf(FLT_MIN, f + FLT_MIN);
  return -logf(-logf(u));
}

// ---- small helpers ------------------------------------------------------------

__device__ __forceinline__ int warp_sum_int(int v) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) v += __shfl_xor_sync(FULL, v, d);
  return v;
}
__device__ __forceinline__ int warp_max_int(int v) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) v = max(v, __shfl_xor_sync(FULL, v, d));
  return v;
}
__device__ __forceinline__ int warp_min_int(int v) {
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) v = min(v, __shfl_xor_sync(FULL, v, d));
  return v;
}

__device__ __forceinline__ int align_cat(int outcome, int source_player, int target_player) {
  if (outcome == NOUT) return NOUT;
  return source_player == target_player ? outcome : 1 - outcome;
}

__device__ __forceinline__ int aligned_atom(int k, int K, bool flip) { return flip ? K - 1 - k : k; }

// ---- descent -------------------------------------------------------------------

template <int KT>
__device__ int sample_action(const TreeRefs& t, const Geometry& g, int b, int n, uint2 sample_key, uint2 tie_key, int lane) {
  const int K = KT > 0 ? KT : g.K;
  const int A = g.A, N = g.N;
  const size_t bN = (size_t)b * N;
  const size_t e = (bN + n) * A;
  const int npl = t.to_play[bN + n];
  int my_idx[4];
  bool my_blocked[4];
  int best = -1;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int a = lane + 32 * j;
    my_idx[j] = -1;
    my_blocked[j] = true;
    if (a < A) {
      const int eo = t.edge_out[e + a];
      const bool blocked = t.invalid[e + a] != 0 || eo != NOUT;
      const int ch = t.children[e + a];
      const bool visited = ch != -1;
      const bool use_child_prior = visited && eo == NOUT && t.edge_pay[e + a] <= 0;
      const float* row;
      bool flip = false;
      if (use_child_prior) {
        row = t.vprior + (bN + ch) * K;
        flip = t.to_play[bN + ch] != npl;
      } else {
        row = t.ealpha + (e + a) * K;
      }
      const float u = uniform01(sample_key, (uint32_t)a);
      float cdf = 0.f;
      int count = 0;
#pragma unroll 4
      for (int k = 0; k < K; ++k) {
        cdf = (k == 0) ? row[aligned_atom(0, K, flip)] : cdf + row[aligned_atom(k, K, flip)];
        count += (u >= cdf);
      }
      const int idx = min(count, K - 1);
      my_idx[j] = idx;
      my_blocked[j] = blocked;
      if (!blocked) best = max(best, idx);
    }
  }
  best = warp_max_int(best);
  float bv = -INFINITY;
  int ba = 0x7fffffff;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const int a = lane + 32 * j;
    if (a < A && !my_blocked[j] && my_idx[j] == best) {
      const float gval = gumbel_low(tie_key, (uint32_t)a);
      if (gval > bv || (gval == bv && a < ba)) { bv = gval; ba = a; }
    }
  }
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
    const float ov = __shfl_xor_sync(FULL, bv, d);
    const int oa = __shfl_xor_sync(FULL, ba, d);
    if (ov > bv || (ov == bv && oa < ba)) { bv = ov; ba = oa; }
  }
  return ba == 0x7fffffff ? 0 : ba;
}

__device__ __forceinline__ bool any_searchable(const TreeRefs& t, const Geometry& g, size_t e, int lane) {
  bool any = false;
  for (int a = lane; a < g.A; a += 32) any |= (t.invalid[e + a] == 0) && (t.edge_out[e + a] == NOUT);
  return __any_sync(FULL, any);
}

// ---- Hex move -----------------------------------------------------------------

struct Move {
  int terminal_outcome;
  int to_play;
  int new_step;
};

__device__ Move hex_step_board(const TreeRefs& t, const Geometry& g, const int32_t* order_all,
                               int b, int parent, int action, int child_slot, int lane) {
  const int C = g.C, size = g.size, N = g.N;
  const size_t bN = (size_t)b * N;
  const int8_t* pb = t.board + (bN + parent) * C;
  int8_t* cb = t.board + (bN + child_slot) * C;
  const int step = t.steps[bN + parent];
  const int set_id = action + 1;
  const int x = action / size, y = action % size;
  const int xs[6] = {x, x + 1, x - 1, x + 1, x - 1, x};
  const int ys[6] = {y - 1, y - 1, y, y, y + 1, y + 1};
  int ids[6];
#pragma unroll
  for (int i = 0; i < 6; ++i) {
    const bool on = xs[i] >= 0 && xs[i] < size && ys[i] >= 0 && ys[i] < size;
    const int v = on ? (int)pb[xs[i] * size + ys[i]] : 0;
    ids[i] = v > 0 ? v : 0;
  }
  const int new_step = step + 1;
  const int color = new_step & 1;
  const int target = -set_id;
  bool top = false, bottom = false;
  for (int c = lane; c < C; c += 32) {
    const int v = pb[c];
    bool merge = c == action;
    if (v > 0) {
#pragma unroll
      for (int i = 0; i < 6; ++i) merge |= (v == ids[i]);
    }
    const int nv = -(merge ? set_id : v);
    cb[c] = (int8_t)nv;
    if (nv == target) {
      const int cx = c / size, cy = c % size;
      if (color == 0) { top |= cy == 0; bottom |= cy == size - 1; }
      else { top |= cx == 0; bottom |= cx == size - 1; }
    }
  }
  const bool terminal = __any_sync(FULL, top) && __any_sync(FULL, bottom);
  if (lane == 0) t.steps[bN + child_slot] = new_step;
  const int* order = order_all + 2 * b;
  const int child_player = order[color];
  Move m;
  m.to_play = child_player;
  m.terminal_outcome = NOUT;
  m.new_step = new_step;
  if (terminal) {
    const int parent_player = t.to_play[bN + parent];
    const int reward = (order[parent_player] == color) ? -1 : 1;
    const int parent_outcome = (reward + 1) / 2;
    m.terminal_outcome = align_cat(parent_outcome, parent_player, child_player);
  }
  __syncwarp();
  return m;
}

__device__ __forceinline__ void write_board_obs(const int8_t* cb, int C, int step, uint8_t* o, int lane) {
  const int color = step & 1;
  for (int c = lane; c < C; c += 32) {
    const int v = cb[c];
    o[4 * c + 0] = v > 0;
    o[4 * c + 1] = v < 0;
    o[4 * c + 2] = color != 0;
    o[4 * c + 3] = step == 1;
  }
}

__device__ __forceinline__ CacheKey build_cache_key(const int8_t* cb, int C, int step, int lane) {
  uint32_t m0 = __ballot_sync(FULL, lane < C && cb[lane] > 0);
  uint32_t o0 = __ballot_sync(FULL, lane < C && cb[lane] < 0);
  uint32_t m1 = __ballot_sync(FULL, (32 + lane) < C && cb[32 + lane] > 0);
  uint32_t o1 = __ballot_sync(FULL, (32 + lane) < C && cb[32 + lane] < 0);
  uint32_t m2 = __ballot_sync(FULL, (64 + lane) < C && cb[64 + lane] > 0);
  uint32_t o2 = __ballot_sync(FULL, (64 + lane) < C && cb[64 + lane] < 0);
  uint32_t m3 = __ballot_sync(FULL, (96 + lane) < C && cb[96 + lane] > 0);
  uint32_t o3 = __ballot_sync(FULL, (96 + lane) < C && cb[96 + lane] < 0);
  CacheKey k;
  k.my_lo = (uint64_t)m0 | ((uint64_t)m1 << 32);
  k.op_lo = (uint64_t)o0 | ((uint64_t)o1 << 32);
  k.my_hi = (uint64_t)m2 | ((uint64_t)m3 << 32);
  k.op_hi = (uint64_t)o2 | ((uint64_t)o3 << 32);
  k.meta = 0x80000000u | ((step & 1) ? 1u : 0u) | ((step == 1) ? 2u : 0u);
  k.pad = 0u;
  return k;
}

// ============================================================================
// 1. Reference bitwise-identical search step (ScacchiBinnedSearchStep)
// ============================================================================

template <int KT>
__device__ void descend(const TreeRefs& t, const Geometry& g, const Inputs& in, int b, int sim, int lane) {
  const int N = g.N, A = g.A;
  const size_t bN = (size_t)b * N;
  int32_t* pend = t.pending + (size_t)b * PEND;
  const bool root_active = t.node_out[bN] == NOUT && any_searchable(t, g, bN * A, lane);
  if (!root_active) {
    if (lane < PEND) pend[lane] = lane == 3 ? NOUT : 0;
    uint8_t* o = in.obs + (size_t)b * g.C * 4;
    for (int i = lane; i < g.C * 4; i += 32) o[i] = 0;
    return;
  }
  uint2 key = make_uint2(in.keys[2 * b], in.keys[2 * b + 1]);
  int node = 0, depth = 0, action = 0;
  while (true) {
    const uint2 next_key = split_key(key, 0u);
    const uint2 selection_key = split_key(key, 1u);
    const uint2 sample_key = split_key(selection_key, 0u);
    const uint2 tie_key = split_key(selection_key, 1u);
    action = sample_action<KT>(t, g, b, node, sample_key, tie_key, lane);
    const int child = t.children[(bN + node) * A + action];
    const bool visited = child != -1;
    const int safe = visited ? child : 0;
    depth += 1;
    const bool selectable = t.node_out[bN + safe] == NOUT && any_searchable(t, g, (bN + safe) * A, lane);
    if (!(visited && selectable && depth < g.max_depth)) break;
    key = next_key;
    node = child;
  }
  const Move m = hex_step_board(t, g, in.order, b, node, action, sim + 1, lane);
  write_board_obs(t.board + (bN + sim + 1) * g.C, g.C, m.new_step, in.obs + (size_t)b * g.C * 4, lane);
  if (lane == 0) {
    pend[0] = node;
    pend[1] = action;
    pend[2] = 1;
    pend[3] = m.terminal_outcome;
    pend[4] = m.to_play;
  }
}

template <int KT>
__device__ void repair_node(const TreeRefs& t, const Geometry& g, const Inputs& in, const Smem& S, int b, int n,
                            bool leaf_active, int leaf_action, int leaf_player, int lane) {
  const int K = KT > 0 ? KT : g.K;
  const int A = g.A, N = g.N, KS = g.d.KS;
  const size_t bN = (size_t)b * N;
  const size_t e = (bN + n) * A;
  const int npl = t.to_play[bN + n];
  const bool leaf_flip = leaf_player != npl;
  const float* leaf_value = in.net_value + (size_t)b * K;
  int count_delta = 0;
  for (int a = lane; a < A; a += 32) {
    const int eo = t.edge_out[e + a];
    const int ep = t.edge_pay[e + a];
    const bool unresolved = eo == NOUT;
    const int ch = t.children[e + a];
    const bool visited = ch != -1;
    const bool direct = leaf_active && a == leaf_action && unresolved;
    int pay = ep + (direct ? 1 : 0);
    int source = direct ? 1 : 0;
    bool child_flip = false;
    if (visited) {
      child_flip = t.to_play[bN + ch] != npl;
      const int cpay = t.node_pay[bN + ch];
      if (unresolved && t.node_out[bN + ch] == NOUT && cpay > 0) {
        source = 2;
        pay = 1 + cpay;
      }
    }
    const bool use_child_prior = visited && unresolved && pay <= 0;
    if (unresolved && t.invalid[e + a] == 0) count_delta += pay - ep;
    float* h = S.h + a * KS;
    float* stored = t.ealpha + (e + a) * K;
    if (eo != NOUT) {
      const int hot = eo * (K - 1);
      for (int k = 0; k < K; ++k) h[k] = k == hot ? 1.f : 0.f;
    } else if (use_child_prior) {
      const float* row = t.vprior + (bN + ch) * K;
      for (int k = 0; k < K; ++k) h[k] = row[aligned_atom(k, K, child_flip)];
    } else if (source == 0) {
      for (int k = 0; k < K; ++k) h[k] = stored[k];
    } else {
      const float* row = source == 1 ? leaf_value : t.valpha + (bN + ch) * K;
      const bool flip = source == 1 ? leaf_flip : child_flip;
      for (int k = 0; k < K; ++k) {
        const float v = row[aligned_atom(k, K, flip)];
        h[k] = v;
        stored[k] = v;
      }
    }
    if (unresolved && pay != ep) t.edge_pay[e + a] = pay;
  }
  count_delta = warp_sum_int(count_delta);
  const int node_pay_old = t.node_pay[bN + n];
  const int n_down = node_pay_old + count_delta;
  __syncwarp();
  if (n_down > 0) {
    backup_lane<KT, false>(S, t.invalid + e, in.points, in.qnodes, in.qweights, t.valpha + (bN + n) * K, g.d, lane);
  }
  if (lane == 0) t.node_pay[bN + n] = n_down;
  __syncwarp();

  bool has_legal = false, not_known = false, has_win = false;
  int win_min = INT_MAX, loss_max = INT_MIN;
  for (int a = lane; a < A; a += 32) {
    const bool legal = t.invalid[e + a] == 0;
    const int eo = t.edge_out[e + a];
    const bool known = legal && eo != NOUT;
    has_legal |= legal;
    not_known |= legal && !known;
    if (known && eo == 1) { has_win = true; win_min = min(win_min, t.edge_pay[e + a]); }
    if (known && eo == 0) loss_max = max(loss_max, t.edge_pay[e + a]);
  }
  has_legal = __any_sync(FULL, has_legal);
  not_known = __any_sync(FULL, not_known);
  has_win = __any_sync(FULL, has_win);
  win_min = warp_min_int(win_min);
  loss_max = warp_max_int(loss_max);
  const bool all_categorical = has_legal && !not_known;
  const int candidate = has_win ? 1 : (all_categorical ? 0 : NOUT);
  const int distance = has_win ? win_min : loss_max;
  const bool publish_node = t.node_out[bN + n] == NOUT && candidate != NOUT;
  if (!publish_node) return;
  const int parent = t.parents[bN + n];
  const bool has_parent = parent != -1;
  const int sp = has_parent ? parent : 0;
  const size_t pe = (bN + sp) * A;
  int incoming = A;
  for (int a0 = 0; a0 < A; a0 += 32) {
    const int a = a0 + lane;
    const unsigned m = __ballot_sync(FULL, a < A && t.children[pe + a] == n);
    if (m != 0u && incoming == A) incoming = a0 + __ffs(m) - 1;
  }
  const bool has_incoming = incoming < A;
  const int inc = has_incoming ? incoming : 0;
  __syncwarp();
  if (lane == 0) {
    const bool publish_edge = has_parent && has_incoming && t.edge_out[pe + inc] == NOUT;
    if (publish_edge) {
      const int old_count = t.edge_pay[pe + inc];
      t.node_pay[bN + sp] = t.node_pay[bN + sp] + 1 + n_down - old_count;
      t.edge_out[pe + inc] = (int8_t)align_cat(candidate, npl, t.to_play[bN + sp]);
      t.edge_pay[pe + inc] = distance + 1;
    }
    t.node_pay[bN + n] = distance;
    t.node_out[bN + n] = (int8_t)candidate;
  }
  __syncwarp();
}

template <int KT>
__device__ void finish(const TreeRefs& t, const Geometry& g, const Inputs& in, const Smem& S, int b, int sim, int lane) {
  const int K = KT > 0 ? KT : g.K;
  const int A = g.A, N = g.N, C = g.C;
  const size_t bN = (size_t)b * N;
  const int32_t* pend = t.pending + (size_t)b * PEND;
  const int parent = pend[0], action = pend[1], active = pend[2], tout = pend[3], child_player = pend[4];
  if (!active) return;
  const int fresh = sim + 1;
  const size_t pa = (bN + parent) * A + action;
  const bool initialize = t.children[pa] == -1;
  const int parent_player = t.to_play[bN + parent];
  const bool publish_terminal = tout != NOUT;
  const int old_edge_count = t.edge_pay[pa];
  const int parent_support = t.node_pay[bN + parent] + (1 - old_edge_count);
  __syncwarp();
  if (initialize) {
    if (lane == 0) {
      t.parents[bN + fresh] = parent;
      t.children[pa] = fresh;
      t.to_play[bN + fresh] = child_player;
      t.node_out[bN + fresh] = (int8_t)tout;
    }
    const float* v = in.net_value + (size_t)b * K;
    for (int k = lane; k < K; k += 32) {
      t.vprior[(bN + fresh) * K + k] = v[k];
      t.valpha[(bN + fresh) * K + k] = v[k];
    }
    const float* q = in.net_q + (size_t)b * A * K;
    float* dst = t.ealpha + (bN + fresh) * A * K;
    for (int i = lane; i < A * K; i += 32) dst[i] = q[i];
    const int8_t* cb = t.board + (bN + fresh) * C;
    uint8_t* inv = t.invalid + (bN + fresh) * A;
    for (int a = lane; a < A; a += 32) inv[a] = publish_terminal ? 0 : (a < C ? cb[a] != 0 : 1);
  }
  if (publish_terminal && lane == 0) {
    t.node_pay[bN + parent] = parent_support;
    t.edge_out[pa] = (int8_t)align_cat(tout, child_player, parent_player);
    t.edge_pay[pa] = 1;
  }
  __syncwarp();
  int node = parent;
  bool leaf_active = true;
  while (true) {
    repair_node<KT>(t, g, in, S, b, node, leaf_active, action, child_player, lane);
    if (node == 0) break;
    node = t.parents[bN + node];
    leaf_active = false;
  }
}

enum : int {
  FINISH = 1,
  DESCEND = 2,
  F_SUBTREE_INIT = 4,
  F_ROOT_FINISH = 8,
  F_READOUT = 16,
};

template <int KT>
__global__ void __launch_bounds__(32 * MAX_WARPS) step_kernel(TreeRefs t, Inputs in, Geometry g, int flags) {
  extern __shared__ __align__(16) float smem[];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int b = blockIdx.x * (blockDim.x >> 5) + wid;
  if (b >= g.B) return;
  const Smem S = carve(smem + (size_t)wid * g.warp_floats, g.off);
  const int sim = *in.sim;
  if (flags & FINISH) finish<KT>(t, g, in, S, b, sim - 1, lane);
  if (flags & DESCEND) descend<KT>(t, g, in, b, sim, lane);
}

// ============================================================================
// 2. Fast Search Pipeline (ScacchiBinnedSearchFastStep)
// ============================================================================

struct FastConfig {
  int max_depth;
  int min_stop_nodes;
  int use_cache;
  int support_search_evidence;
  float early_stop_prob;
  float policy_temp;
};

struct FastInputs {
  const float* net_value;       // [B, K] compacted
  const float* net_q;           // [B, A, K] compacted
  const uint32_t* keys;         // [B, 2] per-lane simulation or tie keys
  const int32_t* order;         // [B, 2]
  const float* points;          // [K]
  const float* qnodes;          // [Q] (Q <= 9)
  const float* qweights;        // [Q]
  const int32_t* prev_actions;  // [B]
  const uint8_t* reset_mask;    // [B] bool
  const int8_t* root_boards;    // [B, C]
  const int32_t* root_steps;    // [B]
  const int32_t* root_to_play;  // [B]
  const int8_t* root_term_out;  // [B]
  const uint8_t* root_invalid;  // [B, A]
  const uint8_t* root_obs;      // [B, C * 4]
  const int32_t* lane_caps;     // [B] node capacity per lane (<= N)
  const CacheKey* cache_keys;   // [CACHE_SLOTS]
  const uint32_t* cache_locks;  // [CACHE_SLOTS]
  const float* cache_v;         // [CACHE_SLOTS, K]
  const float* cache_q;         // [CACHE_SLOTS, A * K]
};

struct FastOutputs {
  uint8_t* obs_compact;         // [B, C * 4]
  int32_t* d_num_active;        // [1]
  CacheKey* miss_keys;          // [B]
  float* out_policy;            // [B, A]
  float* out_qalpha;            // [B, A, K]
  float* out_valpha;            // [B, K]
  float* out_visits;            // [B, A]
  int8_t* out_q_cat_out;        // [B, A]
  int32_t* out_q_cat_dist;      // [B, A]
  int8_t* out_v_cat_out;        // [B]
  int32_t* out_v_cat_dist;      // [B]
  int32_t* out_search_action;   // [B]
};

__device__ __forceinline__ int probe_or_enqueue(
    const FastConfig& cfg, const FastInputs& in, const FastOutputs& out,
    const int8_t* board_ptr, const uint8_t* obs_src, int C, int step, int lane) {
  const CacheKey mk = build_cache_key(board_ptr, C, step, lane);
  if (cfg.use_cache) {
    const uint32_t h = hash_cache_key(mk);
    int hit = 0;
    if (lane == 0 && in.cache_locks[h] == 2u) {
      const CacheKey tk = in.cache_keys[h];
      hit = (tk.my_lo == mk.my_lo && tk.my_hi == mk.my_hi &&
             tk.op_lo == mk.op_lo && tk.op_hi == mk.op_hi && tk.meta == mk.meta);
    }
    if (__shfl_sync(FULL, hit, 0)) {
      return -2 - (int)h;
    }
  }
  int slot = 0;
  if (lane == 0) {
    slot = atomicAdd(out.d_num_active, 1);
    if (cfg.use_cache) out.miss_keys[slot] = mk;
  }
  slot = __shfl_sync(FULL, slot, 0);
  uint8_t* dst_obs = out.obs_compact + (size_t)slot * C * 4;
  if (obs_src) {
    for (int i = lane; i < C * 4; i += 32) dst_obs[i] = obs_src[i];
  } else {
    write_board_obs(board_ptr, C, step, dst_obs, lane);
  }
  return slot;
}

template <int KT>
__device__ void fast_subtree_init(
    const TreeRefs& t, const Geometry& g, const FastConfig& cfg,
    const FastInputs& in, const FastOutputs& out, const Smem& S, int b, int lane) {
  const int K = KT > 0 ? KT : g.K;
  const int N = g.N, A = g.A, C = g.C;
  const size_t bN = (size_t)b * N;
  int32_t* pend = t.pending + (size_t)b * PEND;
  const int prev_a = in.prev_actions[b];
  const int old_size = pend[6];
  const bool can_try = (in.reset_mask[b] == 0) && (in.root_term_out[b] == NOUT) &&
                       (prev_a >= 0) && (prev_a < A) && (old_size > 1) && (old_size <= N);
  const int c = can_try ? t.children[bN * A + prev_a] : -1;

  if (c >= 1 && c < old_size) {
    // Topological in-place subtree compaction: every kept j >= c maps to dst < j.
    int* remap = S.vlist;  // >= 64 ints available
    for (int i = lane; i < N; i += 32) remap[i] = -1;
    __syncwarp();
    int kept = 0;
    if (lane == 0) {
      remap[c] = 0;
      kept = 1;
      for (int j = c + 1; j < old_size; ++j) {
        const int p = t.parents[bN + j];
        if (p >= 0 && remap[p] >= 0) remap[j] = kept++;
      }
    }
    kept = __shfl_sync(FULL, kept, 0);
    __syncwarp();

    for (int j = c; j < old_size; ++j) {
      const int dst = remap[j];
      if (dst < 0) continue;
      const int new_par = (j == c) ? -1 : remap[t.parents[bN + j]];
      const int tp = t.to_play[bN + j];
      const int8_t no = t.node_out[bN + j];
      const int np = t.node_pay[bN + j];
      const int st = t.steps[bN + j];
      __syncwarp();
      if (lane == 0) {
        t.parents[bN + dst] = new_par;
        t.to_play[bN + dst] = tp;
        t.node_out[bN + dst] = no;
        t.node_pay[bN + dst] = np;
        t.steps[bN + dst] = st;
      }
      for (int k = lane; k < K; k += 32) {
        t.vprior[(bN + dst) * K + k] = t.vprior[(bN + j) * K + k];
        t.valpha[(bN + dst) * K + k] = t.valpha[(bN + j) * K + k];
      }
      for (int cell = lane; cell < C; cell += 32) {
        t.board[(bN + dst) * C + cell] = t.board[(bN + j) * C + cell];
      }
      for (int a = lane; a < A; a += 32) {
        const int old_ch = t.children[(bN + j) * A + a];
        t.children[(bN + dst) * A + a] = (old_ch >= 0) ? remap[old_ch] : -1;
        t.edge_out[(bN + dst) * A + a] = t.edge_out[(bN + j) * A + a];
        t.edge_pay[(bN + dst) * A + a] = t.edge_pay[(bN + j) * A + a];
        t.invalid[(bN + dst) * A + a] = t.invalid[(bN + j) * A + a];
      }
      for (int idx = lane; idx < A * K; idx += 32) {
        t.ealpha[(bN + dst) * A * K + idx] = t.ealpha[(bN + j) * A * K + idx];
      }
      __syncwarp();
    }

    for (int dst = kept; dst < N; ++dst) {
      if (lane == 0) {
        t.parents[bN + dst] = -1;
        t.to_play[bN + dst] = 0;
        t.node_out[bN + dst] = NOUT;
        t.node_pay[bN + dst] = 0;
        t.steps[bN + dst] = 0;
      }
      for (int a = lane; a < A; a += 32) {
        t.children[(bN + dst) * A + a] = -1;
        t.edge_out[(bN + dst) * A + a] = NOUT;
        t.edge_pay[(bN + dst) * A + a] = 0;
        t.invalid[(bN + dst) * A + a] = 1;
      }
    }
    if (lane == 0) {
      pend[0] = 0; pend[1] = 0; pend[2] = 0; pend[3] = NOUT;
      pend[4] = 0; pend[5] = -1; pend[6] = kept; pend[7] = 0;
    }
    __syncwarp();
    return;
  }

  // Fresh / reset root initialization.
  for (int dst = 0; dst < N; ++dst) {
    if (lane == 0) {
      t.parents[bN + dst] = -1;
      t.to_play[bN + dst] = (dst == 0) ? in.root_to_play[b] : 0;
      t.node_out[bN + dst] = (dst == 0) ? in.root_term_out[b] : (int8_t)NOUT;
      t.node_pay[bN + dst] = 0;
      t.steps[bN + dst] = (dst == 0) ? in.root_steps[b] : 0;
    }
    for (int a = lane; a < A; a += 32) {
      t.children[(bN + dst) * A + a] = -1;
      t.edge_out[(bN + dst) * A + a] = NOUT;
      t.edge_pay[(bN + dst) * A + a] = 0;
      t.invalid[(bN + dst) * A + a] = (dst == 0) ? in.root_invalid[(size_t)b * A + a] : 1;
    }
  }
  const int8_t* rb = in.root_boards + (size_t)b * C;
  for (int cell = lane; cell < C; cell += 32) {
    t.board[bN * C + cell] = rb[cell];
  }
  __syncwarp();

  int eval_src = -1;
  if (in.root_term_out[b] == NOUT) {
    eval_src = probe_or_enqueue(
        cfg, in, out, t.board + bN * C, in.root_obs + (size_t)b * C * 4, C, in.root_steps[b], lane);
  } else {
    // Uniform placeholder priors for terminal roots (never read by categorical targets).
    const float uni = 1.f / (float)K;
    for (int k = lane; k < K; k += 32) {
      t.vprior[bN * K + k] = uni;
      t.valpha[bN * K + k] = uni;
    }
    for (int i = lane; i < A * K; i += 32) t.ealpha[bN * A * K + i] = uni;
  }
  if (lane == 0) {
    pend[0] = 0; pend[1] = 0; pend[2] = 0; pend[3] = NOUT;
    pend[4] = 0; pend[5] = eval_src; pend[6] = 1; pend[7] = 0;
  }
  __syncwarp();
}

template <int KT>
__device__ void fast_root_finish(
    const TreeRefs& t, const Geometry& g, const FastInputs& in, int b, int lane) {
  const int K = KT > 0 ? KT : g.K;
  const int A = g.A, N = g.N;
  const size_t bN = (size_t)b * N;
  int32_t* pend = t.pending + (size_t)b * PEND;
  const int src = pend[5];
  if (src == -1) return;
  const float* v = (src >= 0) ? (in.net_value + (size_t)src * K) : (in.cache_v + (size_t)(-2 - src) * K);
  const float* q = (src >= 0) ? (in.net_q + (size_t)src * A * K) : (in.cache_q + (size_t)(-2 - src) * A * K);
  for (int k = lane; k < K; k += 32) {
    t.vprior[bN * K + k] = v[k];
    t.valpha[bN * K + k] = v[k];
  }
  for (int i = lane; i < A * K; i += 32) {
    t.ealpha[bN * A * K + i] = q[i];
  }
  if (lane == 0) pend[5] = -1;
  __syncwarp();
}

template <int KT>
__device__ void fast_repair_node(
    const TreeRefs& t, const Geometry& g, const FastConfig& cfg, const Smem& S,
    int b, int n, int changed_action, bool leaf_active, const float* leaf_value, int leaf_player,
    int* out_incoming_on_parent, int lane) {
  const int K = KT > 0 ? KT : g.K;
  const int A = g.A, N = g.N, KS = g.d.KS;
  const size_t bN = (size_t)b * N;
  const size_t e = (bN + n) * A;
  const int npl = t.to_play[bN + n];
  const bool leaf_flip = leaf_player != npl;
  int count_delta = 0;

  // Only `changed_action` can have a new leaf/child message in this simulation;
  // all other edges read `t.ealpha` directly without random global child gathers.
  for (int a = lane; a < A; a += 32) {
    const int eo = t.edge_out[e + a];
    const int ep = t.edge_pay[e + a];
    const bool unresolved = eo == NOUT;
    int pay = ep;
    int source = 0;  // 0 stored, 1 leaf, 2 changed child
    bool child_flip = false;
    int ch = -1;
    if (a == changed_action && unresolved) {
      if (leaf_active) {
        pay = ep + 1;
        source = 1;
      } else {
        ch = t.children[e + a];
        if (ch != -1) {
          child_flip = t.to_play[bN + ch] != npl;
          const int cpay = t.node_pay[bN + ch];
          if (t.node_out[bN + ch] == NOUT && cpay > 0) {
            source = 2;
            pay = 1 + cpay;
          }
        }
      }
      if (t.invalid[e + a] == 0) count_delta += pay - ep;
      if (pay != ep) t.edge_pay[e + a] = pay;
    }

    float* h = S.h + a * KS;
    float* stored = t.ealpha + (e + a) * K;
    if (eo != NOUT) {
      const int hot = eo * (K - 1);
      for (int k = 0; k < K; ++k) h[k] = k == hot ? 1.f : 0.f;
    } else if (source == 0) {
      for (int k = 0; k < K; ++k) h[k] = stored[k];
    } else {
      const float* row = (source == 1) ? leaf_value : (t.valpha + (bN + ch) * K);
      const bool flip = (source == 1) ? leaf_flip : child_flip;
      for (int k = 0; k < K; ++k) {
        const float v = row[aligned_atom(k, K, flip)];
        h[k] = v;
        stored[k] = v;
      }
    }
  }
  count_delta = warp_sum_int(count_delta);
  const int n_down = t.node_pay[bN + n] + count_delta;
  if (lane == 0) t.node_pay[bN + n] = n_down;
  __syncwarp();

  // Check categorical win/loss BEFORE running backup_lane so solved ancestors skip backup!
  bool has_legal = false, not_known = false, has_win = false;
  int win_min = INT_MAX, loss_max = INT_MIN;
  for (int a = lane; a < A; a += 32) {
    const bool legal = t.invalid[e + a] == 0;
    const int eo = t.edge_out[e + a];
    const bool known = legal && eo != NOUT;
    has_legal |= legal;
    not_known |= legal && !known;
    if (known && eo == 1) { has_win = true; win_min = min(win_min, t.edge_pay[e + a]); }
    if (known && eo == 0) loss_max = max(loss_max, t.edge_pay[e + a]);
  }
  has_legal = __any_sync(FULL, has_legal);
  not_known = __any_sync(FULL, not_known);
  has_win = __any_sync(FULL, has_win);
  win_min = warp_min_int(win_min);
  loss_max = warp_max_int(loss_max);
  const bool all_categorical = has_legal && !not_known;
  const int candidate = has_win ? 1 : (all_categorical ? 0 : NOUT);
  const int distance = has_win ? win_min : loss_max;
  const bool publish_node = t.node_out[bN + n] == NOUT && candidate != NOUT;

  // Find incoming edge on parent (needed both for categorical publish and next ancestor's changed_action).
  const int parent = t.parents[bN + n];
  int incoming = -1;
  if (parent != -1) {
    const size_t pe = (bN + parent) * A;
    int found = A;
    for (int a0 = 0; a0 < A; a0 += 32) {
      const int a = a0 + lane;
      const unsigned m = __ballot_sync(FULL, a < A && t.children[pe + a] == n);
      if (m != 0u && found == A) found = a0 + __ffs(m) - 1;
    }
    if (found < A) incoming = found;
  }
  *out_incoming_on_parent = incoming;

  if (publish_node) {
    if (lane == 0) {
      if (parent != -1 && incoming >= 0) {
        const size_t pe_inc = (bN + parent) * A + incoming;
        if (t.edge_out[pe_inc] == NOUT) {
          const int old_count = t.edge_pay[pe_inc];
          t.node_pay[bN + parent] = t.node_pay[bN + parent] + 1 + n_down - old_count;
          t.edge_out[pe_inc] = (int8_t)align_cat(candidate, npl, t.to_play[bN + parent]);
          t.edge_pay[pe_inc] = distance + 1;
        }
      }
      t.node_pay[bN + n] = distance;
      t.node_out[bN + n] = (int8_t)candidate;
    }
    __syncwarp();
    return;
  }

  // Unresolved node: run fast weighted-sum backup!
  if (n_down > 0 && t.node_out[bN + n] == NOUT) {
    const float max_pi = backup_lane<KT, true>(
        S, t.invalid + e, S.pts, S.tq, S.wq, t.valpha + (bN + n) * K, g.d, lane);
    if (n == 0 && cfg.early_stop_prob > 0.f) {
      int32_t* pend = t.pending + (size_t)b * PEND;
      if (pend[6] >= cfg.min_stop_nodes && max_pi >= cfg.early_stop_prob) {
        if (lane == 0) pend[7] = 1;
      }
    }
  }
  __syncwarp();
}

template <int KT>
__device__ void fast_finish(
    const TreeRefs& t, const Geometry& g, const FastConfig& cfg,
    const FastInputs& in, const Smem& S, int b, int lane) {
  const int K = KT > 0 ? KT : g.K;
  const int A = g.A, N = g.N, C = g.C;
  const size_t bN = (size_t)b * N;
  int32_t* pend = t.pending + (size_t)b * PEND;
  const int parent = pend[0], action = pend[1], active = pend[2], tout = pend[3], child_player = pend[4];
  const int src = pend[5];
  if (!active) return;

  // Preload quadrature & atom tables into S once for all ancestors of this lane.
  for (int k = lane; k < K; k += 32) S.pts[k] = in.points[k];
  for (int q = lane; q < g.d.QJ; q += 32) {
    S.tq[q] = q < g.d.Q ? in.qnodes[q] : 0.f;
    S.wq[q] = q < g.d.Q ? in.qweights[q] : 0.f;
  }

  const size_t pa = (bN + parent) * A + action;
  const int existing_child = t.children[pa];
  const bool initialize = existing_child == -1;
  const int fresh = initialize ? pend[6] : existing_child;
  const int parent_player = t.to_play[bN + parent];
  const bool publish_terminal = tout != NOUT;
  const int old_edge_count = t.edge_pay[pa];
  const int parent_support = t.node_pay[bN + parent] + (1 - old_edge_count);
  __syncwarp();

  const float* leaf_value = nullptr;
  if (initialize && fresh < N) {
    if (lane == 0) {
      t.parents[bN + fresh] = parent;
      t.children[pa] = fresh;
      t.to_play[bN + fresh] = child_player;
      t.node_out[bN + fresh] = (int8_t)tout;
      pend[6] = fresh + 1;
    }
    if (!publish_terminal) {
      const float* v = (src >= 0) ? (in.net_value + (size_t)src * K) : (in.cache_v + (size_t)(-2 - src) * K);
      const float* q = (src >= 0) ? (in.net_q + (size_t)src * A * K) : (in.cache_q + (size_t)(-2 - src) * A * K);
      for (int k = lane; k < K; k += 32) {
        t.vprior[(bN + fresh) * K + k] = v[k];
        t.valpha[(bN + fresh) * K + k] = v[k];
      }
      float* dst = t.ealpha + (bN + fresh) * A * K;
      for (int i = lane; i < A * K; i += 32) dst[i] = q[i];
      leaf_value = v;
    }
    const int8_t* cb = t.board + (bN + fresh) * C;
    uint8_t* inv = t.invalid + (bN + fresh) * A;
    for (int a = lane; a < A; a += 32) inv[a] = publish_terminal ? 0 : (a < C ? cb[a] != 0 : 1);
  } else if (!publish_terminal && existing_child >= 0) {
    // Revisit at max_depth: reuse stored vprior of child without calling network!
    leaf_value = t.vprior + (bN + existing_child) * K;
  }

  if (publish_terminal && lane == 0) {
    t.node_pay[bN + parent] = parent_support;
    t.edge_out[pa] = (int8_t)align_cat(tout, child_player, parent_player);
    t.edge_pay[pa] = 1;
  }
  if (lane == 0) {
    pend[2] = 0;
    pend[5] = -1;
  }
  __syncwarp();

  int node = parent;
  int changed_edge = action;
  bool leaf_active = true;
  while (true) {
    int inc_on_parent = -1;
    fast_repair_node<KT>(
        t, g, cfg, S, b, node, changed_edge, leaf_active, leaf_value, child_player, &inc_on_parent, lane);
    if (node == 0) break;
    node = t.parents[bN + node];
    changed_edge = inc_on_parent;
    leaf_active = false;
  }
}

template <int KT>
__device__ void fast_descend(
    const TreeRefs& t, const Geometry& g, const FastConfig& cfg,
    const FastInputs& in, const FastOutputs& out, int b, int lane) {
  const int N = g.N, A = g.A, C = g.C;
  const size_t bN = (size_t)b * N;
  int32_t* pend = t.pending + (size_t)b * PEND;
  const int cur_size = pend[6];
  const int max_nodes = in.lane_caps ? min(in.lane_caps[b], N) : N;
  const bool root_active = (t.node_out[bN] == NOUT) && (cur_size < max_nodes) &&
                           (pend[7] == 0) && any_searchable(t, g, bN * A, lane);
  if (!root_active) {
    if (lane == 0) {
      pend[2] = 0;
      pend[3] = NOUT;
      pend[5] = -1;
    }
    return;
  }

  uint2 key = make_uint2(in.keys[2 * b], in.keys[2 * b + 1]);
  int node = 0, depth = 0, action = 0;
  while (true) {
    const uint2 next_key = split_key(key, 0u);
    const uint2 selection_key = split_key(key, 1u);
    const uint2 sample_key = split_key(selection_key, 0u);
    const uint2 tie_key = split_key(selection_key, 1u);
    action = sample_action<KT>(t, g, b, node, sample_key, tie_key, lane);
    const int child = t.children[(bN + node) * A + action];
    const bool visited = child != -1;
    const int safe = visited ? child : 0;
    depth += 1;
    const bool selectable = t.node_out[bN + safe] == NOUT && any_searchable(t, g, (bN + safe) * A, lane);
    if (!(visited && selectable && depth < g.max_depth)) break;
    key = next_key;
    node = child;
  }

  const int existing = t.children[(bN + node) * A + action];
  const bool visited = existing != -1;
  const int child_slot = visited ? existing : cur_size;
  Move m;
  int eval_src = -1;
  if (visited) {
    // Already expanded at max_depth: board, to_play and vprior are already stored!
    m.terminal_outcome = t.node_out[bN + existing];
    m.to_play = t.to_play[bN + existing];
    m.new_step = t.steps[bN + existing];
    eval_src = -1;
  } else {
    m = hex_step_board(t, g, in.order, b, node, action, child_slot, lane);
    if (m.terminal_outcome == NOUT) {
      eval_src = probe_or_enqueue(cfg, in, out, t.board + (bN + child_slot) * C, nullptr, C, m.new_step, lane);
    }
  }
  if (lane == 0) {
    pend[0] = node;
    pend[1] = action;
    pend[2] = 1;
    pend[3] = m.terminal_outcome;
    pend[4] = m.to_play;
    pend[5] = eval_src;
  }
}

template <int KT>
__device__ void fast_readout(
    const TreeRefs& t, const Geometry& g, const FastConfig& cfg,
    const FastInputs& in, const FastOutputs& out, const Smem& S, int b, int lane) {
  const int K = KT > 0 ? KT : g.K;
  const int A = g.A, N = g.N, KS = g.d.KS;
  const size_t bN = (size_t)b * N;
  const int v_out = t.node_out[bN];
  const int root_pl = t.to_play[bN];

  if (lane == 0) {
    out.out_v_cat_out[b] = (int8_t)v_out;
    out.out_v_cat_dist[b] = (v_out == NOUT) ? NDIST : t.node_pay[bN];
  }
  for (int k = lane; k < K; k += 32) {
    out.out_valpha[(size_t)b * K + k] = t.valpha[bN * K + k];
  }

  // Build effective root edge alphas and support mask.
  uint8_t* s_inv = reinterpret_cast<uint8_t*>(S.ordw);  // 256+ bytes free before Stage 3
  bool any_cand = false;
  for (int a = lane; a < A; a += 32) {
    const int eo = t.edge_out[bN * A + a];
    const int ep = t.edge_pay[bN * A + a];
    const bool legal = t.invalid[bN * A + a] == 0;
    const bool unresolved = eo == NOUT;
    const int ch = t.children[bN * A + a];
    const bool visited = ch != -1;
    const bool use_child_prior = visited && unresolved && ep <= 0;

    out.out_q_cat_out[(size_t)b * A + a] = (int8_t)eo;
    out.out_q_cat_dist[(size_t)b * A + a] = unresolved ? NDIST : ep;
    out.out_visits[(size_t)b * A + a] = unresolved ? (float)ep : 0.f;

    const float* src_row = use_child_prior ? (t.vprior + (bN + ch) * K) : (t.ealpha + (bN * A + a) * K);
    const bool flip = use_child_prior && (t.to_play[bN + ch] != root_pl);
    float* qa_dst = out.out_qalpha + ((size_t)b * A + a) * K;
    float* h = S.h + a * KS;
    const int hot = eo * (K - 1);
    for (int k = 0; k < K; ++k) {
      const float val = src_row[aligned_atom(k, K, flip)];
      qa_dst[k] = val;
      h[k] = unresolved ? val : (k == hot ? 1.f : 0.f);
    }

    const bool cand = legal && (cfg.support_search_evidence ? (!unresolved || ep > 0) : true);
    any_cand |= cand;
    s_inv[a] = cand ? 0 : 1;
  }
  any_cand = __any_sync(FULL, any_cand);
  if (!any_cand) {
    for (int a = lane; a < A; a += 32) s_inv[a] = t.invalid[bN * A + a];
  }
  __syncwarp();

  float* pol_out = out.out_policy + (size_t)b * A;
  if (v_out != NOUT) {
    // Solved root: uniform policy over distance-optimal certified edges, Gumbel tie-break for search_action.
    int opt_dist = (v_out == 1) ? INT_MAX : INT_MIN;
    for (int a = lane; a < A; a += 32) {
      if (s_inv[a] == 0 && t.edge_out[bN * A + a] == v_out) {
        const int d = t.edge_pay[bN * A + a];
        opt_dist = (v_out == 1) ? min(opt_dist, d) : max(opt_dist, d);
      }
    }
    opt_dist = (v_out == 1) ? warp_min_int(opt_dist) : warp_max_int(opt_dist);
    int n_opt = 0;
    for (int a0 = 0; a0 < A; a0 += 32) {
      const int a = a0 + lane;
      const bool match = a < A && s_inv[a] == 0 && t.edge_out[bN * A + a] == v_out && t.edge_pay[bN * A + a] == opt_dist;
      n_opt += __popc(__ballot_sync(FULL, match));
    }
    const float w_opt = n_opt > 0 ? (1.f / (float)n_opt) : 0.f;
    const uint2 tie_key = make_uint2(in.keys[2 * b], in.keys[2 * b + 1]);
    float best_g = -INFINITY;
    int best_a = 0;
    for (int a = lane; a < A; a += 32) {
      const bool match = n_opt > 0 ? (s_inv[a] == 0 && t.edge_out[bN * A + a] == v_out && t.edge_pay[bN * A + a] == opt_dist)
                                   : (s_inv[a] == 0);
      pol_out[a] = match ? w_opt : 0.f;
      if (match) {
        const float gv = gumbel_low(tie_key, (uint32_t)a);
        if (gv > best_g || (gv == best_g && a < best_a)) { best_g = gv; best_a = a; }
      }
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) {
      const float ov = __shfl_xor_sync(FULL, best_g, d);
      const int oa = __shfl_xor_sync(FULL, best_a, d);
      if (ov > best_g || (ov == best_g && oa < best_a)) { best_g = ov; best_a = oa; }
    }
    if (lane == 0) out.out_search_action[b] = best_a;
    return;
  }

  // Unresolved root: compute exact/fast posterior-best policy on support.
  compute_policy_lane<KT, true>(S, s_inv, in.points, in.qnodes, in.qweights, g.d, lane);

  if (cfg.policy_temp != 1.0f) {
    const float inv_t = 1.0f / cfg.policy_temp;
    float max_lp = -INFINITY;
    for (int a = lane; a < A; a += 32) {
      if (s_inv[a] == 0 && S.w[a] > 0.f) {
        max_lp = fmaxf(max_lp, logf(fmaxf(S.w[a], TINY)));
      }
    }
#pragma unroll
    for (int d = 16; d > 0; d >>= 1) max_lp = fmaxf(max_lp, __shfl_xor_sync(FULL, max_lp, d));
    float sum_p = 0.f;
    for (int a = lane; a < A; a += 32) {
      float p = 0.f;
      if (s_inv[a] == 0 && S.w[a] > 0.f) {
        p = expf((logf(fmaxf(S.w[a], TINY)) - max_lp) * inv_t);
      }
      S.w[a] = p;
      sum_p += p;
    }
    sum_p = warp_sum(sum_p);
    if (sum_p > 0.f) {
      for (int a = lane; a < A; a += 32) S.w[a] /= sum_p;
    }
    __syncwarp();
  }

  float best_p = -1.f;
  int best_a = 0;
  for (int a = lane; a < A; a += 32) {
    const float p = S.w[a];
    pol_out[a] = p;
    if (p > best_p || (p == best_p && a < best_a)) { best_p = p; best_a = a; }
  }
#pragma unroll
  for (int d = 16; d > 0; d >>= 1) {
    const float op = __shfl_xor_sync(FULL, best_p, d);
    const int oa = __shfl_xor_sync(FULL, best_a, d);
    if (op > best_p || (op == best_p && oa < best_a)) { best_p = op; best_a = oa; }
  }
  if (lane == 0) out.out_search_action[b] = best_a;
}

template <int KT>
__global__ void __launch_bounds__(32 * MAX_WARPS) fast_step_kernel(
    TreeRefs t, FastInputs in, FastOutputs out, Geometry g, FastConfig cfg, int flags) {
  extern __shared__ __align__(16) float smem[];
  const int lane = threadIdx.x & 31;
  const int wid = threadIdx.x >> 5;
  const int b = blockIdx.x * (blockDim.x >> 5) + wid;
  if (b >= g.B) return;
  const Smem S = carve(smem + (size_t)wid * g.warp_floats, g.off);
  if (flags & F_SUBTREE_INIT) fast_subtree_init<KT>(t, g, cfg, in, out, S, b, lane);
  if (flags & F_ROOT_FINISH) fast_root_finish<KT>(t, g, in, b, lane);
  if (flags & FINISH) fast_finish<KT>(t, g, cfg, in, S, b, lane);
  if (flags & DESCEND) fast_descend<KT>(t, g, cfg, in, out, b, lane);
  if (flags & F_READOUT) fast_readout<KT>(t, g, cfg, in, out, S, b, lane);
}

// ---- occupancy & launch helpers ---------------------------------------------

template <int KT, typename KernelFn>
ffi::Error choose_warps_generic(KernelFn kernel_fn, std::atomic<size_t>& configured, size_t per_warp, int* warps) {
  const size_t most = per_warp * MAX_WARPS;
  if (most > configured.load()) {
    int device = 0, limit = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    const size_t request = std::min<size_t>(most, (size_t)limit);
    cudaError_t e = cudaFuncSetAttribute(kernel_fn, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)request);
    if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
    configured.store(request);
  }
  int best = 0, best_resident = -1;
  for (int w = 1; w <= MAX_WARPS; w *= 2) {
    if (per_warp * w > configured.load()) break;
    int blocks = 0;
    cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, kernel_fn, 32 * w, per_warp * w);
    if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
    if (blocks * w >= best_resident) { best_resident = blocks * w; best = w; }
  }
  if (best_resident <= 0) return ffi::Error::Internal("search step does not fit in shared memory");
  *warps = best;
  return ffi::Error::Success();
}

template <int KT>
ffi::Error launch(cudaStream_t stream, const TreeRefs& t, const Inputs& in, const Geometry& g, int flags) {
  static std::atomic<size_t> configured{0};
  const size_t per_warp = (size_t)g.warp_floats * sizeof(float);
  int warps = 1;
  ffi::Error chosen = choose_warps_generic<KT>(step_kernel<KT>, configured, per_warp, &warps);
  if (chosen.failure()) return chosen;
  const int blocks = (g.B + warps - 1) / warps;
  step_kernel<KT><<<blocks, 32 * warps, per_warp * warps, stream>>>(t, in, g, flags);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  return ffi::Error::Success();
}

template <int KT>
ffi::Error launch_fast(cudaStream_t stream, const TreeRefs& t, const FastInputs& in, const FastOutputs& out,
                       const Geometry& g, const FastConfig& cfg, int flags) {
  static std::atomic<size_t> configured{0};
  const size_t per_warp = (size_t)g.warp_floats * sizeof(float);
  int warps = 1;
  ffi::Error chosen = choose_warps_generic<KT>(fast_step_kernel<KT>, configured, per_warp, &warps);
  if (chosen.failure()) return chosen;
  if (flags & (F_SUBTREE_INIT | DESCEND)) {
    cudaError_t me = cudaMemsetAsync(out.d_num_active, 0, sizeof(int32_t), stream);
    if (me != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(me));
  }
  const int blocks = (g.B + warps - 1) / warps;
  fast_step_kernel<KT><<<blocks, 32 * warps, per_warp * warps, stream>>>(t, in, out, g, cfg, flags);
  cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
  return ffi::Error::Success();
}

template <typename T>
T* data(ffi::AnyBuffer& buffer) { return reinterpret_cast<T*>(buffer.untyped_data()); }

ffi::Error Step(cudaStream_t stream, int64_t max_depth, int64_t flags,
                ffi::AnyBuffer parents, ffi::AnyBuffer children, ffi::AnyBuffer to_play, ffi::AnyBuffer node_out,
                ffi::AnyBuffer node_pay, ffi::AnyBuffer edge_out, ffi::AnyBuffer edge_pay, ffi::AnyBuffer vprior,
                ffi::AnyBuffer valpha, ffi::AnyBuffer ealpha, ffi::AnyBuffer invalid, ffi::AnyBuffer board,
                ffi::AnyBuffer steps, ffi::AnyBuffer pending, ffi::AnyBuffer net_value, ffi::AnyBuffer net_q,
                ffi::AnyBuffer keys, ffi::AnyBuffer order, ffi::AnyBuffer points, ffi::AnyBuffer qnodes,
                ffi::AnyBuffer qweights, ffi::AnyBuffer sim,
                ffi::Result<ffi::AnyBuffer> r_parents, ffi::Result<ffi::AnyBuffer> r_children,
                ffi::Result<ffi::AnyBuffer> r_to_play, ffi::Result<ffi::AnyBuffer> r_node_out,
                ffi::Result<ffi::AnyBuffer> r_node_pay, ffi::Result<ffi::AnyBuffer> r_edge_out,
                ffi::Result<ffi::AnyBuffer> r_edge_pay, ffi::Result<ffi::AnyBuffer> r_vprior,
                ffi::Result<ffi::AnyBuffer> r_valpha, ffi::Result<ffi::AnyBuffer> r_ealpha,
                ffi::Result<ffi::AnyBuffer> r_invalid, ffi::Result<ffi::AnyBuffer> r_board,
                ffi::Result<ffi::AnyBuffer> r_steps, ffi::Result<ffi::AnyBuffer> r_pending,
                ffi::Result<ffi::AnyBuffer> r_obs) {
  auto ed = ealpha.dimensions();
  if (ed.size() != 4) return ffi::Error::InvalidArgument("edge_alpha must be [B, N, A, K]");
  Geometry g;
  g.B = (int)ed[0];
  g.N = (int)ed[1];
  g.A = (int)ed[2];
  g.K = (int)ed[3];
  auto bd = board.dimensions();
  if (bd.size() != 3) return ffi::Error::InvalidArgument("boards must be [B, N, C]");
  g.C = (int)bd[2];
  g.size = (int)lround(sqrt((double)g.C));
  g.max_depth = (int)max_depth;
  if (g.size * g.size != g.C || g.A != g.C + 1 || g.C > MAX_CELLS)
    return ffi::Error::InvalidArgument("need a square Hex board with A = C + 1 moves and at most 127 cells");
  if (g.A > MAXA || g.K < 2 || g.K > MAXK) return ffi::Error::InvalidArgument("need A <= 128 and 2 <= K <= 129");
  if (ealpha.element_type() != ffi::F32 || vprior.element_type() != ffi::F32 || valpha.element_type() != ffi::F32 ||
      net_value.element_type() != ffi::F32 || net_q.element_type() != ffi::F32 || points.element_type() != ffi::F32)
    return ffi::Error::InvalidArgument("beliefs must be float32");
  if (node_out.element_type() != ffi::S8 || edge_out.element_type() != ffi::S8 || board.element_type() != ffi::S8)
    return ffi::Error::InvalidArgument("outcomes and boards must be int8");
  if (invalid.element_type() != ffi::PRED || r_obs->element_type() != ffi::PRED)
    return ffi::Error::InvalidArgument("invalid and obs must be bool");
  g.d = make_dims(g.A, g.K, false);
  if ((int)qnodes.element_count() != g.d.Q || (int)qweights.element_count() != g.d.Q)
    return ffi::Error::InvalidArgument("quadrature must have ceil(A / 2) nodes");
  if ((int64_t)net_q.element_count() != (int64_t)g.B * g.A * g.K || (int64_t)net_value.element_count() != (int64_t)g.B * g.K ||
      (int64_t)keys.element_count() != 2LL * g.B || (int64_t)order.element_count() != 2LL * g.B ||
      (int64_t)pending.element_count() != (int64_t)PEND * g.B || (int64_t)r_obs->element_count() != 4LL * g.B * g.C)
    return ffi::Error::InvalidArgument("operand shapes disagree");
  if (r_parents->untyped_data() != parents.untyped_data() || r_children->untyped_data() != children.untyped_data() ||
      r_to_play->untyped_data() != to_play.untyped_data() || r_node_out->untyped_data() != node_out.untyped_data() ||
      r_node_pay->untyped_data() != node_pay.untyped_data() || r_edge_out->untyped_data() != edge_out.untyped_data() ||
      r_edge_pay->untyped_data() != edge_pay.untyped_data() || r_vprior->untyped_data() != vprior.untyped_data() ||
      r_valpha->untyped_data() != valpha.untyped_data() || r_ealpha->untyped_data() != ealpha.untyped_data() ||
      r_invalid->untyped_data() != invalid.untyped_data() || r_board->untyped_data() != board.untyped_data() ||
      r_steps->untyped_data() != steps.untyped_data() || r_pending->untyped_data() != pending.untyped_data())
    return ffi::Error::InvalidArgument("tree operands must alias their results");
  if (g.B == 0) return ffi::Error::Success();
  g.off = make_offsets(g.d);
  g.warp_floats = round4(g.off.total);

  TreeRefs t{
      data<int32_t>(parents), data<int32_t>(children), data<int32_t>(to_play), data<int8_t>(node_out),
      data<int32_t>(node_pay), data<int8_t>(edge_out), data<int32_t>(edge_pay), data<float>(vprior),
      data<float>(valpha), data<float>(ealpha), data<uint8_t>(invalid), data<int8_t>(board),
      data<int32_t>(steps), data<int32_t>(pending)};
  Inputs in{
      data<const float>(net_value), data<const float>(net_q), data<const uint32_t>(keys),
      data<const int32_t>(order), data<const float>(points), data<const float>(qnodes),
      data<const float>(qweights), data<const int32_t>(sim), reinterpret_cast<uint8_t*>(r_obs->untyped_data())};
  switch (g.K) {
    case 9:  return launch<9>(stream, t, in, g, (int)flags);
    case 65: return launch<65>(stream, t, in, g, (int)flags);
    default: return launch<0>(stream, t, in, g, (int)flags);
  }
}

ffi::Error FastStep(
    cudaStream_t stream,
    int64_t max_depth, int64_t flags, int64_t min_stop_nodes, int64_t use_cache, int64_t support_search_evidence,
    double early_stop_prob, double policy_temp,
    ffi::AnyBuffer parents, ffi::AnyBuffer children, ffi::AnyBuffer to_play, ffi::AnyBuffer node_out,
    ffi::AnyBuffer node_pay, ffi::AnyBuffer edge_out, ffi::AnyBuffer edge_pay, ffi::AnyBuffer vprior,
    ffi::AnyBuffer valpha, ffi::AnyBuffer ealpha, ffi::AnyBuffer invalid, ffi::AnyBuffer board,
    ffi::AnyBuffer steps, ffi::AnyBuffer pending,
    ffi::AnyBuffer net_value, ffi::AnyBuffer net_q, ffi::AnyBuffer keys, ffi::AnyBuffer order,
    ffi::AnyBuffer points, ffi::AnyBuffer qnodes, ffi::AnyBuffer qweights,
    ffi::AnyBuffer prev_actions, ffi::AnyBuffer reset_mask, ffi::AnyBuffer root_boards,
    ffi::AnyBuffer root_steps, ffi::AnyBuffer root_to_play, ffi::AnyBuffer root_term_out,
    ffi::AnyBuffer root_invalid, ffi::AnyBuffer root_obs, ffi::AnyBuffer lane_caps,
    ffi::AnyBuffer cache_keys, ffi::AnyBuffer cache_locks, ffi::AnyBuffer cache_v, ffi::AnyBuffer cache_q,
    ffi::Result<ffi::AnyBuffer> r_parents, ffi::Result<ffi::AnyBuffer> r_children,
    ffi::Result<ffi::AnyBuffer> r_to_play, ffi::Result<ffi::AnyBuffer> r_node_out,
    ffi::Result<ffi::AnyBuffer> r_node_pay, ffi::Result<ffi::AnyBuffer> r_edge_out,
    ffi::Result<ffi::AnyBuffer> r_edge_pay, ffi::Result<ffi::AnyBuffer> r_vprior,
    ffi::Result<ffi::AnyBuffer> r_valpha, ffi::Result<ffi::AnyBuffer> r_ealpha,
    ffi::Result<ffi::AnyBuffer> r_invalid, ffi::Result<ffi::AnyBuffer> r_board,
    ffi::Result<ffi::AnyBuffer> r_steps, ffi::Result<ffi::AnyBuffer> r_pending,
    ffi::Result<ffi::AnyBuffer> r_obs_compact, ffi::Result<ffi::AnyBuffer> r_num_active,
    ffi::Result<ffi::AnyBuffer> r_miss_keys,
    ffi::Result<ffi::AnyBuffer> r_out_policy, ffi::Result<ffi::AnyBuffer> r_out_qalpha,
    ffi::Result<ffi::AnyBuffer> r_out_valpha, ffi::Result<ffi::AnyBuffer> r_out_visits,
    ffi::Result<ffi::AnyBuffer> r_out_q_cat_out, ffi::Result<ffi::AnyBuffer> r_out_q_cat_dist,
    ffi::Result<ffi::AnyBuffer> r_out_v_cat_out, ffi::Result<ffi::AnyBuffer> r_out_v_cat_dist,
    ffi::Result<ffi::AnyBuffer> r_out_search_action) {
  auto ed = ealpha.dimensions();
  if (ed.size() != 4) return ffi::Error::InvalidArgument("edge_alpha must be [B, N, A, K]");
  Geometry g;
  g.B = (int)ed[0];
  g.N = (int)ed[1];
  g.A = (int)ed[2];
  g.K = (int)ed[3];
  auto bd = board.dimensions();
  g.C = (int)bd[2];
  g.size = (int)lround(sqrt((double)g.C));
  g.max_depth = (int)max_depth;
  if (g.size * g.size != g.C || g.A != g.C + 1 || g.C > MAX_CELLS)
    return ffi::Error::InvalidArgument("need a square Hex board with A = C + 1 moves and at most 127 cells");
  if (g.B == 0) return ffi::Error::Success();

  g.d = make_dims(g.A, g.K, true);
  if ((int)qnodes.element_count() != g.d.Q || (int)qweights.element_count() != g.d.Q)
    return ffi::Error::InvalidArgument("fast quadrature must have min(ceil(A / 2), 9) nodes");
  g.off = make_offsets(g.d);
  g.warp_floats = round4(g.off.total);

  TreeRefs t{
      data<int32_t>(parents), data<int32_t>(children), data<int32_t>(to_play), data<int8_t>(node_out),
      data<int32_t>(node_pay), data<int8_t>(edge_out), data<int32_t>(edge_pay), data<float>(vprior),
      data<float>(valpha), data<float>(ealpha), data<uint8_t>(invalid), data<int8_t>(board),
      data<int32_t>(steps), data<int32_t>(pending)};
  FastInputs in{
      data<const float>(net_value), data<const float>(net_q), data<const uint32_t>(keys),
      data<const int32_t>(order), data<const float>(points), data<const float>(qnodes),
      data<const float>(qweights), data<const int32_t>(prev_actions), data<const uint8_t>(reset_mask),
      data<const int8_t>(root_boards), data<const int32_t>(root_steps), data<const int32_t>(root_to_play),
      data<const int8_t>(root_term_out), data<const uint8_t>(root_invalid), data<const uint8_t>(root_obs),
      data<const int32_t>(lane_caps),
      reinterpret_cast<const CacheKey*>(cache_keys.untyped_data()),
      data<const uint32_t>(cache_locks), data<const float>(cache_v), data<const float>(cache_q)};
  FastOutputs out{
      reinterpret_cast<uint8_t*>(r_obs_compact->untyped_data()),
      reinterpret_cast<int32_t*>(r_num_active->untyped_data()),
      reinterpret_cast<CacheKey*>(r_miss_keys->untyped_data()),
      reinterpret_cast<float*>(r_out_policy->untyped_data()),
      reinterpret_cast<float*>(r_out_qalpha->untyped_data()),
      reinterpret_cast<float*>(r_out_valpha->untyped_data()),
      reinterpret_cast<float*>(r_out_visits->untyped_data()),
      reinterpret_cast<int8_t*>(r_out_q_cat_out->untyped_data()),
      reinterpret_cast<int32_t*>(r_out_q_cat_dist->untyped_data()),
      reinterpret_cast<int8_t*>(r_out_v_cat_out->untyped_data()),
      reinterpret_cast<int32_t*>(r_out_v_cat_dist->untyped_data()),
      reinterpret_cast<int32_t*>(r_out_search_action->untyped_data())};
  FastConfig cfg{
      (int)max_depth, (int)min_stop_nodes, (int)use_cache, (int)support_search_evidence,
      (float)early_stop_prob, (float)policy_temp};

  switch (g.K) {
    case 9:  return launch_fast<9>(stream, t, in, out, g, cfg, (int)flags);
    case 65: return launch_fast<65>(stream, t, in, out, g, cfg, (int)flags);
    default: return launch_fast<0>(stream, t, in, out, g, cfg, (int)flags);
  }
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ScacchiBinnedSearchStep, Step,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("max_depth")
        .Attr<int64_t>("flags")
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>(),
    {ffi::Traits::kCmdBufferCompatible});

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ScacchiBinnedSearchFastStep, FastStep,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("max_depth")
        .Attr<int64_t>("flags")
        .Attr<int64_t>("min_stop_nodes")
        .Attr<int64_t>("use_cache")
        .Attr<int64_t>("support_search_evidence")
        .Attr<double>("early_stop_prob")
        .Attr<double>("policy_temp")
        // 14 in-place tree buffers (0..13)
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        // 20 read-only operands (14..33)
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        .Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>().Arg<ffi::AnyBuffer>()
        // 14 aliased tree results (0..13)
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        // 3 compaction outputs (14..16): obs_compact, d_num_active, miss_keys
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        // 9 readout outputs (17..25): policy, qalpha, valpha, visits, q_out, q_dist, v_out, v_dist, action
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>(),
    {ffi::Traits::kCmdBufferCompatible});
