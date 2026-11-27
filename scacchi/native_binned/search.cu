// Fused step of the binned Thompson tree search on Hex, one warp per lane.
//
// One call of `scacchi_binned_search_step` does, for every lane on its own
// (no lane waits for another, nothing is padded to the deepest lane):
//
//   finish (simulation s - 1)  write the evaluated leaf into the tree (expand)
//                              and repair every node on its path, leaf parent
//                              to root (search.backward with binned
//                              update_posterior and the binary certificate
//                              publication), with the weighted-sum backup of
//                              backup_lane.cuh;
//   descend (simulation s)     Thompson descent from the root
//                              (dirichlet_mctx.search.simulate with
//                              binned_search.action_selection), then the Hex
//                              move at the leaf (pgx.hex) and the observation
//                              that the network evaluates before the next call.
//
// The network stays outside (XLA), between calls. Every operation is the one
// of the JAX search, in the same order where it matters for the bits: threefry
// keys and bits as jax.random (partitionable threefry), uniform and Gumbel
// draws as jax.random.uniform / gumbel(mode="low"), sequential cumulative sums
// as XLA's reduce-window cumsum, integer payload arithmetic, and the backup of
// backup.cu. The tree arrays are the JAX Tree's own arrays, updated in place.
//
// Layout (all row-major): tree arrays [B, N, ...]; boards [B, N, C] int8 are
// pgx GameState.board (positive: stones of the player to move), steps [B, N]
// is GameState.step_count, pending [B, 8] = (parent, action, active,
// child terminal outcome, child to-play, -, -, -) of the last descent.

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "xla/ffi/api/ffi.h"
#include "backup_lane.cuh"

namespace ffi = xla::ffi;

namespace {

using namespace scacchi_backup;

constexpr int NOUT = -1;  // dirichlet_mctx.outcomes.NO_OUTCOME
constexpr int PEND = 8;
constexpr int MAX_CELLS = 127;  // int8 board ids (Hex 11 has 121 cells)
constexpr int MAX_WARPS = 4;    // lanes per block, at most; chosen for occupancy

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

// jax.random.split(key, n)[i]
__device__ __forceinline__ uint2 split_key(uint2 key, uint32_t i) { return threefry(key.x, key.y, 0u, i); }

// jax.random.bits(key, (n,), uint32)[i]
__device__ __forceinline__ uint32_t random_bits(uint2 key, uint32_t i) {
  const uint2 r = threefry(key.x, key.y, 0u, i);
  return r.x ^ r.y;
}

// jax.random.uniform(key, (n,))[i] in [0, 1)
__device__ __forceinline__ float uniform01(uint2 key, uint32_t i) {
  return __uint_as_float((random_bits(key, i) >> 9) | 0x3F800000u) - 1.0f;
}

// jax.random.gumbel(key, (n,), mode="low")[i] = -log(-log(uniform(minval=tiny, maxval=1)))
__device__ __forceinline__ float gumbel_low(uint2 key, uint32_t i) {
  const float f = uniform01(key, i);
  // f * (1 - tiny) + tiny, then max(tiny, .): 1 - tiny rounds to 1 and f + tiny
  // is f for every f > 0 (f is a multiple of 2^-23).
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

// outcomes.align_categorical_outcome for two outcomes.
__device__ __forceinline__ int align_cat(int outcome, int source_player, int target_player) {
  if (outcome == NOUT) return NOUT;
  return source_player == target_player ? outcome : 1 - outcome;
}

// Row k of outcomes.align_outcome(belief, source, target): atoms reversed when
// the players differ.
__device__ __forceinline__ int aligned_atom(int k, int K, bool flip) { return flip ? K - 1 - k : k; }

// ---- descent -------------------------------------------------------------------

// binned_search.action_selection at node n: sample_action over the effective
// edge beliefs (action_selection.effective_action_alpha), moves that are
// illegal or certified excluded.
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
      // jnp.cumsum(probs, axis=-1): sequential sums, as XLA's reduce-window.
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
  // jax.random.categorical(tie_key, where(tied, 0, -inf)): argmax of Gumbel
  // noise over the tied moves, first index on equal values.
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

// Any action that is legal and not certified (Tree.searchable_actions).
__device__ __forceinline__ bool any_searchable(const TreeRefs& t, const Geometry& g, size_t e, int lane) {
  bool any = false;
  for (int a = lane; a < g.A; a += 32) any |= (t.invalid[e + a] == 0) && (t.edge_out[e + a] == NOUT);
  return __any_sync(FULL, any);
}

// ---- Hex move (pgx.hex.Game.step, scacchi.envs._step, pgx core.Env.step) -------

struct Move {
  int terminal_outcome;  // of the child, from its player's view; NOUT if not terminal
  int to_play;           // child current player
};

__device__ Move hex_step(const TreeRefs& t, const Geometry& g, const Inputs& in, int b, int parent, int action,
                         int child_slot, int lane) {
  const int C = g.C, size = g.size, N = g.N;
  const size_t bN = (size_t)b * N;
  const int8_t* pb = t.board + (bN + parent) * C;
  int8_t* cb = t.board + (bN + child_slot) * C;
  const int step = t.steps[bN + parent];
  const int set_id = action + 1;
  // Ids of the mover's groups next to the new stone (pgx _neighbour order).
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
    const int nv = -(merge ? set_id : v);  // the next player's view
    cb[c] = (int8_t)nv;
    if (nv == target) {
      const int cx = c / size, cy = c % size;
      if (color == 0) { top |= cy == 0; bottom |= cy == size - 1; }
      else { top |= cx == 0; bottom |= cx == size - 1; }
    }
  }
  const bool terminal = __any_sync(FULL, top) && __any_sync(FULL, bottom);
  if (lane == 0) t.steps[bN + child_slot] = new_step;
  const int* order = in.order + 2 * b;
  const int child_player = order[color];
  Move m;
  m.to_play = child_player;
  m.terminal_outcome = NOUT;
  if (terminal) {
    // rewards = Game.rewards(x)[player_order]: -1 for colour `color`, +1 for the other.
    const int parent_player = t.to_play[bN + parent];
    const int reward = (order[parent_player] == color) ? -1 : 1;
    const int parent_outcome = (reward + 1) / 2;  // terminal_outcome_from_reward
    m.terminal_outcome = align_cat(parent_outcome, parent_player, child_player);
  }
  // Observation of the child for its player to move (pgx Game.observe).
  uint8_t* o = in.obs + (size_t)b * C * 4;
  __syncwarp();
  for (int c = lane; c < C; c += 32) {
    const int v = cb[c];
    o[4 * c + 0] = v > 0;
    o[4 * c + 1] = v < 0;
    o[4 * c + 2] = color != 0;
    o[4 * c + 3] = new_step == 1;
  }
  return m;
}

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
  const Move m = hex_step(t, g, in, b, node, action, sim + 1, lane);
  if (lane == 0) {
    pend[0] = node;
    pend[1] = action;
    pend[2] = 1;
    pend[3] = m.terminal_outcome;
    pend[4] = m.to_play;
  }
}

// ---- finish: expand and repair the path ----------------------------------------

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
  // _repair_inputs: direct leaf message, child refresh, effective beliefs.
  for (int a = lane; a < A; a += 32) {
    const int eo = t.edge_out[e + a];
    const int ep = t.edge_pay[e + a];
    const bool unresolved = eo == NOUT;
    const int ch = t.children[e + a];
    const bool visited = ch != -1;
    const bool direct = leaf_active && a == leaf_action && unresolved;
    int pay = ep + (direct ? 1 : 0);
    int source = direct ? 1 : 0;  // 0 stored edge, 1 leaf value, 2 child value, 3 child prior
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
      // binned.categorical_edges: a certified edge is a point mass on the
      // first (loss) or last (win) atom.
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
        stored[k] = v;  // _set_node_update: the repaired edge message
      }
    }
    if (unresolved && pay != ep) t.edge_pay[e + a] = pay;
  }
  count_delta = warp_sum_int(count_delta);
  const int node_pay_old = t.node_pay[bN + n];
  const int n_down = node_pay_old + count_delta;
  __syncwarp();
  if (n_down > 0) {
    backup_lane<KT>(S, t.invalid + e, in.points, in.qnodes, in.qweights, t.valpha + (bN + n) * K, g.d, lane);
  }
  if (lane == 0) t.node_pay[bN + n] = n_down;
  __syncwarp();

  // _categorize_node_and_publish, binary outcomes.
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
  if (!publish_node) return;  // no write below happens without it
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
  const int fresh = sim + 1;  // node slot of simulation `sim`
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
    // ~legal_action_mask of the child: empty cells, never the swap move; every
    // move once the game is over (pgx core.Env.step).
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
  // search.backward: leaf parent up to the root.
  int node = parent;
  bool leaf_active = true;
  while (true) {
    repair_node<KT>(t, g, in, S, b, node, leaf_active, action, child_player, lane);
    if (node == 0) break;
    node = t.parents[bN + node];
    leaf_active = false;
  }
}

enum : int { FINISH = 1, DESCEND = 2 };

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

// Lanes per block with the most lanes resident per SM (shared memory per lane
// grows with the number of moves: 5.9 KB on Hex 7, 14 KB on Hex 11).
template <int KT>
ffi::Error choose_warps(size_t per_warp, int* warps) {
  static std::atomic<size_t> configured{0};
  const size_t most = per_warp * MAX_WARPS;
  if (most > configured.load()) {
    int device = 0, limit = 0;
    cudaGetDevice(&device);
    cudaDeviceGetAttribute(&limit, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    const size_t request = std::min<size_t>(most, (size_t)limit);
    cudaError_t e = cudaFuncSetAttribute(step_kernel<KT>, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)request);
    if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
    configured.store(request);
  }
  int best = 0, best_resident = -1;
  for (int w = 1; w <= MAX_WARPS; w *= 2) {
    if (per_warp * w > configured.load()) break;
    int blocks = 0;
    cudaError_t e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, step_kernel<KT>, 32 * w, per_warp * w);
    if (e != cudaSuccess) return ffi::Error::Internal(cudaGetErrorString(e));
    if (blocks * w >= best_resident) { best_resident = blocks * w; best = w; }
  }
  if (best_resident <= 0) return ffi::Error::Internal("search step does not fit in shared memory");
  *warps = best;
  return ffi::Error::Success();
}

template <int KT>
ffi::Error launch(cudaStream_t stream, const TreeRefs& t, const Inputs& in, const Geometry& g, int flags) {
  const size_t per_warp = (size_t)g.warp_floats * sizeof(float);
  int warps = 1;
  ffi::Error chosen = choose_warps<KT>(per_warp, &warps);
  if (chosen.failure()) return chosen;
  const int blocks = (g.B + warps - 1) / warps;
  step_kernel<KT><<<blocks, 32 * warps, per_warp * warps, stream>>>(t, in, g, flags);
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
  g.d = make_dims(g.A, g.K);
  if ((int)qnodes.element_count() != g.d.Q || (int)qweights.element_count() != g.d.Q)
    return ffi::Error::InvalidArgument("quadrature must have ceil(A / 2) nodes");
  if ((int64_t)net_q.element_count() != (int64_t)g.B * g.A * g.K || (int64_t)net_value.element_count() != (int64_t)g.B * g.K ||
      (int64_t)keys.element_count() != 2LL * g.B || (int64_t)order.element_count() != 2LL * g.B ||
      (int64_t)pending.element_count() != (int64_t)PEND * g.B || (int64_t)r_obs->element_count() != 4LL * g.B * g.C)
    return ffi::Error::InvalidArgument("operand shapes disagree");
  // In-place: every tree operand must be its own result buffer.
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

  TreeRefs t;
  t.parents = data<int32_t>(parents);
  t.children = data<int32_t>(children);
  t.to_play = data<int32_t>(to_play);
  t.node_out = data<int8_t>(node_out);
  t.node_pay = data<int32_t>(node_pay);
  t.edge_out = data<int8_t>(edge_out);
  t.edge_pay = data<int32_t>(edge_pay);
  t.vprior = data<float>(vprior);
  t.valpha = data<float>(valpha);
  t.ealpha = data<float>(ealpha);
  t.invalid = data<uint8_t>(invalid);
  t.board = data<int8_t>(board);
  t.steps = data<int32_t>(steps);
  t.pending = data<int32_t>(pending);
  Inputs in;
  in.net_value = data<const float>(net_value);
  in.net_q = data<const float>(net_q);
  in.keys = data<const uint32_t>(keys);
  in.order = data<const int32_t>(order);
  in.points = data<const float>(points);
  in.qnodes = data<const float>(qnodes);
  in.qweights = data<const float>(qweights);
  in.sim = data<const int32_t>(sim);
  in.obs = reinterpret_cast<uint8_t*>(r_obs->untyped_data());
  switch (g.K) {
    case 9:  return launch<9>(stream, t, in, g, (int)flags);
    case 65: return launch<65>(stream, t, in, g, (int)flags);
    default: return launch<0>(stream, t, in, g, (int)flags);
  }
}

}  // namespace

XLA_FFI_DEFINE_HANDLER_SYMBOL(
    ScacchiBinnedSearchStep, Step,
    ffi::Ffi::Bind()
        .Ctx<ffi::PlatformStream<cudaStream_t>>()
        .Attr<int64_t>("max_depth")
        .Attr<int64_t>("flags")
        .Arg<ffi::AnyBuffer>()   // parents   [B, N]    i32
        .Arg<ffi::AnyBuffer>()   // children  [B, N, A] i32
        .Arg<ffi::AnyBuffer>()   // to_play   [B, N]    i32
        .Arg<ffi::AnyBuffer>()   // node_out  [B, N]    i8
        .Arg<ffi::AnyBuffer>()   // node_pay  [B, N]    i32
        .Arg<ffi::AnyBuffer>()   // edge_out  [B, N, A] i8
        .Arg<ffi::AnyBuffer>()   // edge_pay  [B, N, A] i32
        .Arg<ffi::AnyBuffer>()   // vprior    [B, N, K] f32
        .Arg<ffi::AnyBuffer>()   // valpha    [B, N, K] f32
        .Arg<ffi::AnyBuffer>()   // ealpha    [B, N, A, K] f32
        .Arg<ffi::AnyBuffer>()   // invalid   [B, N, A] bool
        .Arg<ffi::AnyBuffer>()   // boards    [B, N, C] i8
        .Arg<ffi::AnyBuffer>()   // steps     [B, N]    i32
        .Arg<ffi::AnyBuffer>()   // pending   [B, 8]    i32
        .Arg<ffi::AnyBuffer>()   // net value [B, K]    f32
        .Arg<ffi::AnyBuffer>()   // net q     [B, A, K] f32
        .Arg<ffi::AnyBuffer>()   // keys      [B, 2]    u32
        .Arg<ffi::AnyBuffer>()   // order     [B, 2]    i32
        .Arg<ffi::AnyBuffer>()   // points    [K]       f32
        .Arg<ffi::AnyBuffer>()   // qnodes    [Q]       f32
        .Arg<ffi::AnyBuffer>()   // qweights  [Q]       f32
        .Arg<ffi::AnyBuffer>()   // sim       []        i32
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>().Ret<ffi::AnyBuffer>()
        .Ret<ffi::AnyBuffer>(),  // obs [B, size, size, 4] bool
    {ffi::Traits::kCmdBufferCompatible});
