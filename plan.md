Below is a pasteable implementation brief. It assumes the coding agent has the revised `algorithms.tex` as context, especially the WDL Dirichlet posterior tree, CPU-side mutable search boundary, and no temporary reservation mass. Pgx is the right simulator substrate because it supports JAX-vectorized environment stepping and batched legal-action masks.  

````md
# Implementation brief: Redis-backed batched posterior tree for Dirichlet-Q AlphaZero

## Goal

Implement a fast Redis-backed posterior tree for the Dirichlet-Q AlphaZero search described in `algorithms.tex`.

Speed is priority #1. Everything must be batched.

The architecture should be:

```text
PGX state batch lives on device
JAX vmap(env.step) advances all active lanes
Redis stores compact canonical posterior nodes
local process keeps a hot decoded cache
NN evaluates unique leaf states in large batches
tree posteriors are updated in batched dirty-node flushes
````

Do **not** implement temporary posterior-mass reservation, virtual loss, or `P_{v,a}`. That was removed intentionally. Minimal in-flight bookkeeping is allowed only to avoid duplicate NN work; it must not enter the posterior.

---

## Non-negotiable design constraints

1. Do not store PGX states in Redis.
2. Do not use `child_key = hash(parent_key + action)` as the canonical state key.
3. Use canonical state keys computed from the actual resulting PGX state after `env.step`.
4. Redis should store posterior node blobs, not per-edge scalar fields.
5. Use batched Redis access: `MGET`, `MSET`, pipelines, and Lua/SETNX-style atomic claim scripts.
6. The hot traversal loop should hit local cache first and Redis only in batches.
7. The PGX stepping path must be vectorized with JAX:

   ```python
   step_fn = jax.jit(jax.vmap(env.step))
   ```
8. The tree should support transpositions: multiple parent edges may point to the same child node key.
9. Backup must follow the revised algorithm: direct evidence on the selected leaf edge, and search-weighted state posterior summaries for ancestor propagation.
10. WDL order is always:

    ```text
    [L, D, W]
    ```

---

## Key idea: node key vs edge key

Do not canonicalize nodes by path.

Wrong:

```text
child_key = H(parent_key, action)
```

Correct:

```text
parent edge:
  (parent_state_key, action) -> child_state_key

child node:
  node:{child_state_key} -> NodeBlob
```

The child key must be computed from the actual PGX child state:

```python
next_state = step_fn(state_batch, action_batch)
child_keys = state_key_fn(next_state)
```

This lets transpositions merge correctly:

```text
parent_1 --a--> state X
parent_2 --b--> state X

both edges point to node:{key(X)}
```

For deterministic perfect-information games this is sufficient. For stochastic/chance games, an edge may need to support multiple possible child keys per action. Start with deterministic two-player perfect-information support.

---

## State key requirements

Implement a 128-bit state key:

```python
StateKey = tuple[np.uint64, np.uint64]  # hi, lo
```

The hash must include every rule-relevant part of the PGX state:

```text
board / pieces / stones
current player
ko/pass/castling/en-passant/repetition-relevant state when applicable
hands/captured pieces for shogi
chance state if applicable
terminal/truncation state if rule-relevant
```

Do not hash only the observation unless the observation is guaranteed to contain all rule state.

Implementation requirement:

```python
state_key_fn = jax.jit(jax.vmap(canonical_state_key))
```

For v1, implement a generic JAX-compatible hash over selected PGX state leaves, using uint64 mixing, and add per-game overrides later. Prefer a deterministic 128-bit SplitMix64-style or Zobrist-style hash.

Add tests that verify:

```text
same state reached by different paths -> same key
different side to move -> different key
different ko/castling/pass/repetition rule state -> different key
```

---

## Redis role

Redis is the canonical shared node store / transposition table / checkpoint layer.

Redis is **not** the per-edge inner loop.

The hot loop should be:

```text
local cache lookup
batched Redis MGET for missing node blobs
decode blobs into local cache
JAX selection + PGX step
batched Redis claim for missing children
batched NN eval for unique leaves
batched node expansion + backup
dirty-node MSET flush
```

Avoid Redis hashes for the hot action arrays. Store one packed binary blob per node.

---

## Redis keys

Use a namespace that prevents collisions across games, rules, models, and experiments:

```text
dqaz:{run_id}:{game_id}:{rules_digest}:{model_id}:node:{hi:016x}{lo:016x}
```

Examples:

```text
dqaz:run42:go_9x9:rules_abcd:model_000123:node:4f3a...9c2d
```

Also allow optional keys:

```text
dqaz:{...}:root:{root_id}
dqaz:{...}:stats
dqaz:{...}:eval_stream
```

But v1 can keep NN eval queues in-process because PGX states are not stored in Redis.

---

## NodeBlob schema

Store one compact binary blob per expanded node.

Use this conceptual schema:

```python
class NodeBlob:
    # Header
    version: uint16
    status: uint8          # 0 = inflight, 1 = expanded, 2 = terminal
    game_id: uint16
    model_id: uint64
    key_hi: uint64
    key_lo: uint64
    current_player: int8
    num_actions: uint16
    terminal_outcome: int8 # -1 if not terminal; else L/D/W encoding
    dirty_version: uint64

    # State-level NN output
    value_alpha: float32[3]        # [L, D, W], from this node's player perspective

    # Legal action arrays, sparse over legal actions only
    legal_actions: uint32[num_actions]

    # NN policy/Q priors for this node's legal actions
    policy_logits: float16[num_actions]
    q_alpha: float32[num_actions, 3]

    # Edge-level posterior data for actions from this node
    edge_base_alpha: float32[num_actions, 3]
    edge_evidence_E: float32[num_actions, 3]
    edge_post_alpha: float32[num_actions, 3]  # cached = base + E

    # Child references
    child_key_hi: uint64[num_actions]  # zero means unknown
    child_key_lo: uint64[num_actions]

    # Search accounting
    visits: uint32[num_actions]

    # Backup/cache summaries
    pi_search: float16[num_actions]          # optional cached internal search policy
    state_summary_alpha: float32[3]          # sum_a pi_search[a] * edge_post_alpha[a]
```

For local decoded nodes, also build:

```python
action_to_index: dict[int, int]
```

Do not serialize `action_to_index`; rebuild it on decode.

### Initial values when expanding a fresh NN-evaluated node

For a non-terminal expanded node:

```python
value_alpha = network.value_alpha
legal_actions = actions where legal_action_mask is true
policy_logits = network.policy_logits[legal_actions]
q_alpha = network.q_alpha[legal_actions]

edge_base_alpha = q_alpha
edge_evidence_E = zeros_like(q_alpha)
edge_post_alpha = edge_base_alpha
child_key_hi = zeros
child_key_lo = zeros
visits = zeros
pi_search = compute_pi_search_fast(node)
state_summary_alpha = sum_a pi_search[a] * edge_post_alpha[a]
```

When a child later becomes expanded, update the parent edge base:

```python
edge_base_alpha[parent, action] =
    align_child_value_to_parent_perspective(child.value_alpha)

edge_post_alpha[parent, action] =
    edge_base_alpha[parent, action] + edge_evidence_E[parent, action]
```

This avoids fetching the child node during selection.

---

## Serialization

Implement:

```python
encode_node(node: NodeBlob) -> bytes
decode_node(blob: bytes) -> NodeBlob
```

Requirements:

```text
fast
little-endian
versioned
no pickle
no PGX state
single contiguous bytes object per node
round-trip exact for all integer fields
float32/float16 stable within dtype precision
```

A simple v1 can use:

```text
struct-packed fixed header
then raw numpy array bytes in a fixed order
```

Add a magic/version header so future schema migrations fail clearly.

---

## Redis store API

Implement a class like:

```python
class RedisNodeStore:
    def __init__(self, redis_client, namespace, cache_size=...):
        ...

    async def get_nodes(self, keys: list[StateKey]) -> dict[StateKey, NodeBlob | Missing]:
        """
        Batch local-cache lookup first.
        Batch Redis MGET only for cache misses.
        Decode blobs.
        Populate local cache.
        """

    async def put_nodes(self, nodes: list[NodeBlob]) -> None:
        """
        Encode and MSET/pipeline.
        Update local cache.
        """

    async def flush_dirty(self) -> None:
        """
        Encode dirty local nodes and MSET them.
        Clear dirty flags after successful write.
        """

    async def claim_missing_nodes(self, keys: list[StateKey], ttl_ms: int) -> ClaimResult:
        """
        Atomically mark missing nodes as inflight.
        Return which keys were claimed, already expanded, already inflight, or missing/error.
        """

    def mark_dirty(self, key: StateKey) -> None:
        ...
```

Use local cache aggressively:

```python
local_cache: OrderedDict[StateKey, NodeBlob]
dirty_keys: set[StateKey]
```

For v1, assume one writer owns a root search and writes back dirty blobs. Redis is the backing store / transposition table. Multi-writer concurrent evidence updates can be added later with CAS/versioning.

---

## Atomic claim script

Implement one Redis Lua script or equivalent `SET NX` logic for missing leaves:

```text
Input:
  node key
  inflight placeholder blob
  ttl_ms

Behavior:
  if key does not exist:
      SET key inflight_placeholder NX PX ttl_ms
      return CLAIMED
  else:
      return EXISTS
```

The inflight placeholder should contain:

```text
status = inflight
state_key
model_id
maybe eval_id
created_at_ms
```

When NN eval finishes, replace the inflight placeholder with the expanded node blob.

For v1, if a lane reaches a node that is already inflight, do not add posterior mass. Either:

```text
A. register this path as a waiter for the same leaf result, or
B. abandon/recycle the lane and start another traversal from root.
```

Prefer B for the first fast implementation unless waiter handling is already easy.

---

## Batching model

Let:

```text
G = number of active roots
L = lanes per root
B = G * L
```

Maintain batched arrays:

```python
pgx_state_batch      # PGX State PyTree, leading dimension B
current_keys[B]      # StateKey per lane
root_ids[B]
active[B]            # bool
depth[B]
path_keys[B, max_depth]
path_actions[B, max_depth]
path_players[B, max_depth]  # optional, for WDL alignment
```

Each lane starts at a root state/key. The root PGX state is repeated across its lanes.

---

## Wavefront traversal loop

Implement a batched wavefront traversal:

```python
while any(active) and max_depth_not_reached:
    # 1. Fetch current nodes
    unique_keys = unique(current_keys[active])
    nodes = await store.get_nodes(unique_keys)

    # 2. Pack node arrays to padded tensors
    packed = pack_nodes_for_selection(nodes, current_keys, active)
    # packed.legal_actions: [B, Amax]
    # packed.action_mask: [B, Amax]
    # packed.edge_post_alpha: [B, Amax, 3]
    # packed.policy_logits: [B, Amax]

    # 3. Thompson select actions in batch
    action_batch = thompson_select_jax(
        rng_key,
        packed.edge_post_alpha,
        packed.legal_actions,
        packed.action_mask,
    )

    # 4. Record path step
    path_keys[:, depth] = current_keys
    path_actions[:, depth] = action_batch

    # 5. Step PGX states on device
    next_state_batch = step_fn(pgx_state_batch, action_batch)

    # 6. Compute child canonical keys on device
    child_keys = state_key_fn(next_state_batch)

    # 7. Bring child keys / terminal masks to host at this boundary
    #    This is the required Redis synchronization point.
    child_keys_host = device_get(child_keys)
    terminal_host = device_get(next_state_batch.terminated)

    # 8. Update parent edge child pointers in local cache
    update_parent_edges_with_child_keys(
        parent_keys=current_keys,
        actions=action_batch,
        child_keys=child_keys_host,
    )

    # 9. Classify lanes
    #    terminal -> immediate terminal backup
    #    expanded child -> continue
    #    missing child -> claim for NN eval and stop lane
    #    inflight child -> recycle or wait
```

Do not block on every JAX operation except where host keys are required for Redis. Keep JAX dispatch asynchronous where possible.

---

## Thompson selection

Traversal selection uses posterior samples, not posterior means.

For each lane and legal action:

```python
phi[a] ~ Dirichlet(edge_post_alpha[a])
utility[a] = phi[a, W] - phi[a, L]
action = argmax_a utility[a]
```

JAX implementation:

```python
@jax.jit
def thompson_select_jax(rng, alpha, legal_actions, action_mask):
    # alpha: [B, Amax, 3]
    # legal_actions: [B, Amax]
    # action_mask: [B, Amax]
    gamma = jax.random.gamma(rng, alpha)
    phi = gamma / jnp.sum(gamma, axis=-1, keepdims=True)
    utility = phi[..., 2] - phi[..., 0]   # W - L
    utility = jnp.where(action_mask, utility, -jnp.inf)
    idx = jnp.argmax(utility, axis=-1)
    return jnp.take_along_axis(legal_actions, idx[:, None], axis=1)[:, 0]
```

Do not replace traversal with:

```python
argmax(mean_W - mean_L)
```

That is a greedy posterior-mean action, useful for final action selection, not for Thompson traversal.

---

## NN leaf evaluation

When lanes hit missing non-terminal child states:

1. Deduplicate by `child_state_key`.
2. Keep the corresponding PGX leaf states in an in-process batch.
3. Evaluate the unique leaf states with the NN.
4. Build expanded `NodeBlob`s from NN outputs and legal masks.
5. Store expanded nodes with batched Redis `MSET`.
6. Update parent edge base alphas to aligned child value alphas.
7. Run backup for the paths that produced those leaves.

The eval request should be local/in-process:

```python
class EvalRequest:
    leaf_key: StateKey
    leaf_state: PGXStateSliceOrBatchedIndex
    paths: list[PathRecord]
```

Do not put full PGX states in Redis.

If cross-process NN workers are needed later, send only compact NN inputs or observations, not the entire tree node.

---

## Terminal leaf handling

If PGX says a leaf is terminal, do not send it to the NN.

Construct one-hot WDL evidence:

```python
# WDL order [L, D, W]
loss = [1, 0, 0]
draw = [0, 1, 0]
win  = [0, 0, 1]
```

Use the configured terminal evidence strength:

```python
c_terminal
```

Then backup immediately.

---

## Perspective alignment

Every WDL vector or alpha is from the perspective of some player-to-move.

Implement:

```python
def flip_wdl(x):
    # [L, D, W] -> [W, D, L]
    return x[..., [2, 1, 0]]

def align_wdl(x, src_player, dst_player):
    if src_player == dst_player:
        return x
    else:
        return flip_wdl(x)
```

For standard alternating two-player zero-sum games this is enough.

Use this for:

```text
child.value_alpha -> parent edge base alpha
leaf evidence -> selected edge evidence
child state summary -> ancestor edge backup
```

---

## Backup rule

Follow `algorithms.tex`.

At a high level:

1. The final selected edge receives direct leaf evidence.
2. Ancestor edges receive calibrated state-summary evidence from the child search posterior.
3. State summaries are weighted by `pi_search`.

For a node `u`, maintain:

```python
state_summary_alpha[u] =
    sum_b pi_search_u[b] * edge_post_alpha[u, b]
```

where:

```python
edge_post_alpha[u, b] = edge_base_alpha[u, b] + edge_evidence_E[u, b]
```

Then for an ancestor edge `(v, a)` whose child is `u`, add:

```python
edge_evidence_E[v, a] += c_state * align_wdl(
    state_summary_alpha[u],
    src_player=u.current_player,
    dst_player=v.current_player,
)
```

For the final leaf edge, add direct leaf evidence using `c_leaf` or `c_terminal`:

```python
edge_evidence_E[parent, action] += c_leaf * aligned_leaf_mean
```

or terminal:

```python
edge_evidence_E[parent, action] += c_terminal * aligned_terminal_one_hot
```

After every evidence update:

```python
edge_post_alpha = edge_base_alpha + edge_evidence_E
visits += 1
recompute pi_search for the changed node
recompute state_summary_alpha
mark node dirty
```

Use small `c_state` initially, e.g.:

```text
c_state in [0.01, 0.25]
```

because backed-up neural search evidence is pseudo-evidence, not independent categorical evidence.

---

## Computing pi_search

Implement two modes.

### Fast internal mode

Used during search backup:

```python
q_mean[a] =
    (edge_post_alpha[a, W] - edge_post_alpha[a, L])
    / sum_z edge_post_alpha[a, z]

pi_search = softmax(policy_logits + q_scale * q_mean)
```

or:

```python
pi_search = softmax(q_scale * q_mean)
```

Make the exact formula configurable.

### Accurate target mode

Used at `finish_search` for training targets:

```python
for m in range(M):
    phi[a] ~ Dirichlet(edge_post_alpha[a])
    a_star = argmax_a(phi[a, W] - phi[a, L])
    counts[a_star] += 1

pi_target = counts / M
```

This is the posterior-best policy target.

---

## Final action selection

Traversal uses Thompson sampling.

Final play action should use greedy estimated value unless the training setup explicitly wants stochastic action sampling.

Compute posterior mean utility:

```python
q_mean[a] =
    (edge_post_alpha[a, W] - edge_post_alpha[a, L])
    / sum_z edge_post_alpha[a, z]

action = argmax_a q_mean[a]
```

Also return:

```python
pi_target          # posterior-best MC target
q_targets          # edge_post_alpha for explored/legal actions
visit_counts
root_edge_posteriors
```

---

## Redis/cache update policy

During traversal and backup:

```text
modify decoded local nodes
mark them dirty
flush dirty nodes at batch boundaries
```

Do not write to Redis after every edge update.

Flush points:

```text
after each completed NN eval batch
after each root search
before process shutdown
```

Use `MSET`/pipeline for dirty nodes.

---

## Handling duplicate leaves

Within a batch, deduplicate leaf evaluations by `leaf_key`.

Default v1 policy:

```text
evaluate each unique leaf once
expand node once
backup only one evidence contribution per unique leaf evaluation
recycle duplicate lanes if fixed simulation count is required
```

Do not add multiple full-strength copies of the same NN leaf result as if they were independent evidence unless explicitly configured.

Add a config:

```python
duplicate_leaf_policy: Literal[
    "recycle_lane",
    "waiter_no_extra_evidence",
    "backup_per_path"
]
```

Default:

```python
"recycle_lane"
```

---

## Public API

Implement a high-level search object:

```python
class RedisPosteriorSearch:
    def __init__(
        self,
        env,
        network,
        redis_store,
        state_key_fn,
        config,
    ):
        ...

    async def initialize_root(self, root_state) -> StateKey:
        """
        Evaluate root with NN if missing.
        Store expanded root node.
        Return root key.
        """

    async def search_batch(
        self,
        root_states,
        num_lanes_per_root: int,
        max_depth: int,
        num_simulations_or_batches: int,
    ) -> list[SearchResult]:
        """
        Run batched wavefront posterior search from many roots.
        """

    async def finish_search(self, root_key: StateKey) -> SearchResult:
        """
        Return final action, policy target, Q targets, posterior summaries.
        """
```

Suggested config:

```python
@dataclass
class SearchConfig:
    max_depth: int
    num_lanes_per_root: int
    leaf_batch_size: int
    c_leaf: float
    c_terminal: float
    c_state: float
    pi_search_mode: str
    q_scale: float
    posterior_target_mc_samples: int
    redis_inflight_ttl_ms: int
    duplicate_leaf_policy: str
    local_cache_size: int
    dtype_policy_logits: str = "float16"
    dtype_alpha: str = "float32"
```

---

## Modules to create

Suggested file layout:

```text
posterior_tree/
  __init__.py
  types.py              # StateKey, NodeBlob, SearchConfig, SearchResult
  codec.py              # encode_node/decode_node
  redis_store.py        # RedisNodeStore, Lua scripts, cache
  state_hash.py         # JAX state hashing
  selection.py          # Thompson selection, posterior-best target
  backup.py             # WDL flip/align, evidence updates, state summary
  pack.py               # pack decoded nodes into padded arrays
  search.py             # RedisPosteriorSearch wavefront loop
  tests/
    test_codec.py
    test_state_hash.py
    test_redis_claim.py
    test_selection.py
    test_backup.py
    test_transpositions.py
    test_wavefront_toy.py
```

---

## Tests

Add tests before optimizing.

### Serialization

```text
NodeBlob -> bytes -> NodeBlob round trip
all arrays same shape/dtype
legal action order preserved
no PGX state in blob
```

### State hashing

```text
same canonical state -> same key
different legal state -> different key
same state via two paths -> same key
current player affects key
```

### Redis claim

```text
missing key -> CLAIMED
same key again -> EXISTS/INFLIGHT
expanded key -> EXISTS/EXPANDED
claim batch deduplicates duplicate keys
```

### Thompson selection

```text
invalid actions never selected
shape [B] output
higher posterior utility selected more often statistically
```

### Backup

```text
flip([L,D,W]) == [W,D,L]
same-player alignment leaves vector unchanged
opponent alignment flips vector
edge_post_alpha == edge_base_alpha + edge_evidence_E
state_summary_alpha == sum pi_search * edge_post_alpha
```

### Transpositions

Use a toy deterministic environment where two paths reach the same state. Verify:

```text
two parent edges point to the same child_key
Redis stores one child node
NN evaluates that child once
```

### End-to-end toy search

Run small batched search on a toy PGX-like environment:

```text
initialize root
run search_batch
expand leaves
backup evidence
finish_search returns valid policy target
all pi_target probabilities sum to 1
```

---

## Benchmarks

Add simple benchmark scripts:

```text
bench_codec.py
bench_redis_mget_mset.py
bench_state_hash.py
bench_pgx_step.py
bench_wavefront.py
```

Report:

```text
nodes encoded/decoded per second
Redis MGET/MSET nodes/sec for batch sizes 1, 32, 256, 1024
state_key_fn states/sec for B=1024
PGX env.step states/sec for B=1024
end-to-end search leaf evals/sec
cache hit rate
average Redis round trips per search batch
```

The implementation is only acceptable if Redis operations are batched and cache hit rate is visible in logs.

---

## Logging/instrumentation

Track:

```python
num_cache_hits
num_cache_misses
num_redis_mget
num_redis_mset
num_nodes_claimed
num_nodes_inflight
num_leaf_evals
num_duplicate_leaf_keys
num_terminal_leaves
num_dirty_flushes
avg_depth
max_depth
search_time_ms
redis_time_ms
pgx_step_time_ms
nn_eval_time_ms
backup_time_ms
```

Make it easy to print one compact search summary per batch.

---

## First implementation order

1. Implement `NodeBlob`, codec, and tests.
2. Implement `RedisNodeStore` with local cache, `MGET`, `MSET`, and claim.
3. Implement JAX `state_key_fn`.
4. Implement WDL alignment and backup helpers.
5. Implement Thompson selection over padded node arrays.
6. Implement root initialization.
7. Implement one-root, many-lane wavefront traversal.
8. Add NN leaf expansion.
9. Add backup and dirty flush.
10. Add multi-root batching.
11. Add posterior-best target generation at finish.
12. Benchmark and optimize.

---

## Important implementation choices

Prefer:

```text
packed binary node blobs
sparse legal-action arrays
local decoded cache
batched Redis operations
JAX-vmapped PGX stepping
JAX-vmapped state hashing
deduplicated leaf evaluation
dirty-node flushes
```

Avoid:

```text
Redis HGET/HSET per edge
Redis round trip per action
path hashes as canonical node keys
storing PGX State in Redis
Python loops over lanes where a batch operation is easy
pending posterior mass / virtual loss
treating duplicate NN evals as independent evidence by default
```

---

## Done criteria for v1

A successful v1 should:

1. Run a batched posterior search from at least one PGX root state.
2. Store expanded posterior nodes in Redis as compact blobs.
3. Keep PGX states out of Redis.
4. Compute canonical child keys from actual PGX stepped states.
5. Merge transpositions by child state key.
6. Use Thompson selection for traversal.
7. Evaluate NN leaves in deduplicated batches.
8. Update WDL Dirichlet edge evidence and cached posteriors.
9. Propagate ancestor evidence through `pi_search`-weighted state summaries.
10. Return a valid posterior-best policy target and greedy posterior-mean final action.
11. Include tests and benchmarks for the above.

```
```


Use the following as a coding-agent brief. I anchored it to the algorithm file’s contract: edges carry WDL Dirichlet posteriors rather than scalar MCTS values, the CPU owns the mutable posterior tree, the GPU evaluates batches of leaf states, and the API shape is `NextRequest / ConsumeResult / FinishSearch`.  The implementation notes also call for arena-style node/edge IDs and a batched GPU boundary rather than waiting after single descents.  Pgx supports the required vectorized stepping pattern via `jax.vmap(env.step)`, batched actions, batched current-player vectors, observations, and legal-action masks, so the design below keeps live PGX states on device. 

Paste this to the coding agent:

````markdown
# Implementation brief: Redis-backed batched posterior tree for Dirichlet-Q AlphaZero

You have `algorithms.tex` as context. Implement the fast batched search architecture described here.

Speed is priority #1. Everything should be batched. Do not write a design that does Redis round-trips or neural inference one edge / one state at a time.

## 0. Source-of-truth algorithmic assumptions

We are implementing the Dirichlet-Q posterior tree search from `algorithms.tex`.

Core assumptions:

1. Tree edges store WDL Dirichlet posterior information, not scalar `Q`.
2. Outcome space is WDL: `(L, D, W)`.
3. Utility is:
   `U(phi) = phi_W - phi_L`.
4. Thompson traversal samples:
   `phi_{s,a} ~ Dirichlet(alpha_post(s,a))`
   and selects:
   `a = argmax_a (phi_W - phi_L)`.
5. Completed posterior is:
   `alpha_post(s,a) = alpha_base(s,a) + E(s,a)`.
6. `alpha_base(s,a)` is:
   - Q-head prior `alpha_Q(s,a)` if the child is not expanded,
   - aligned child value prior if the child is expanded.
7. Do not implement temporary pending posterior mass `P`. That was removed as premature optimization. In-flight bookkeeping is allowed only as scheduling metadata and must never enter the posterior.
8. Neural leaf evidence is pseudo-evidence. Terminal evidence can have larger strength than neural evidence.
9. Backups must support the improved search-weighted child-state summary:
   `summary_alpha(s) = sum_a pi_search_s(a) * (alpha_base(s,a) + E(s,a))`.

## 1. High-level architecture

Implement Redis as the canonical batched node store / transposition table / checkpoint layer, not as a per-edge inner-loop data structure.

The hot architecture should be:

```text
PGX state batch lives on device
Redis stores canonical node blobs keyed by state hash
Search worker keeps a local decoded node cache
Wavefront traversal runs in fixed-size batches
NN leaf evaluation is batched
Dirty Redis writes are flushed in batches
````

Avoid this:

```text
for each lane:
  for each tree step:
    redis GET one node
    choose one action
    redis SET one edge
```

Use this instead:

```text
for each wave:
  unique current state keys
  batch MGET missing node blobs
  pack node arrays
  vectorized Thompson action selection
  vmap(pgx.env.step)
  vectorized state hashing
  batch classify child states
  batch claim missing nodes
  batch NN eval leaves
  batch backup / dirty flush
```

## 2. State keys and transpositions

Do not use `child_key = H(parent_key, action)` as the canonical state key. That is a path hash and it will not merge transpositions.

Use two different concepts:

```text
edge identity:
  (parent_state_key, action)

canonical child node key:
  child_state_key = H(actual canonical child PGX state)
```

Parent edge stores:

```text
edge(parent_state_key, action) -> child_state_key
```

Redis node is stored under:

```text
node:{namespace}:{state_key}
```

where `state_key` is computed from the actual PGX state after `env.step`.

### Required state-key function

Implement:

```python
state_key_batch = jax.jit(jax.vmap(state_to_key))(pgx_state_batch)
```

`state_to_key(state)` must include every rule-relevant field:

* board / pieces / stones,
* current player,
* legal rule state such as ko/pass/castling/en-passant/repetition/hand pieces/chance state,
* any game-specific fields required to determine legal moves and terminal outcomes.

Do not hash only the observation unless the observation is guaranteed to contain all rule-relevant state.

Preferred representation:

```text
state_key = uint128
```

In JAX, represent as either:

```text
uint64[2]
```

or, if uint64 is inconvenient:

```text
uint32[4]
```

Use a Zobrist-style hash for board games. Pre-generate random tables per game/ruleset and compute by xor/reduction. Do not use Python object hashing or pickled PGX state bytes.

### Namespace design

Search evidence depends on the search/session/model snapshot. Do not accidentally share posterior evidence across unrelated searches.

Use:

```text
node:{game_id}:{rules_hash}:{model_id}:{search_session_id}:{state_key}
```

For a later optimization, split immutable NN eval cache from mutable search posterior state:

```text
eval:{game_id}:{rules_hash}:{model_id}:{state_key}
search:{search_session_id}:{state_key}
```

MVP can store both in one NodeBlob, but keep the code structured so this split is easy.

## 3. Redis value schema: NodeBlob

Redis should store one packed binary blob per node. Avoid thousands of Redis hash fields per node.

Use sparse legal-action arrays by default. Dense arrays are acceptable for Go/Hex, but chess/shogi need sparse legal actions.

Suggested `NodeBlob` schema:

```python
@dataclass
class NodeBlob:
    version: int
    status: NodeStatus  # MISSING | INFLIGHT | EXPANDED | TERMINAL

    # Identity / metadata
    state_key: UInt128
    model_id: int
    game_id: int
    rules_hash: int
    current_player: int
    ply: int

    # Terminal info
    terminal: bool
    terminal_wdl: float32[3]  # one-hot WDL from current-player perspective if terminal

    # Legal move list
    legal_count: int
    legal_actions: uint16[legal_count] or uint32[legal_count]

    # Network outputs for this state
    value_alpha: float32[3]
    policy_logits: float16/float32[legal_count]
    q_alpha: float16/float32[legal_count, 3]

    # Per-outgoing-edge posterior state
    edge_base_alpha: float32[legal_count, 3]
    edge_E: float32[legal_count, 3]
    edge_post_alpha: float32[legal_count, 3]  # cached = edge_base_alpha + edge_E
    edge_visits: uint32[legal_count]

    # Child references
    child_state_keys: UInt128[legal_count]  # zero/null if unknown
    child_status_hint: uint8[legal_count]   # optional cache only

    # Search summary for backing up this node as a state value
    pi_search_approx: float16/float32[legal_count]
    state_summary_alpha: float32[3]

    # Scheduling metadata only; must not enter posterior
    inflight_count: uint16
    dirty: bool
```

Important:

```text
edge_E is completed WDL evidence.
edge_post_alpha = edge_base_alpha + edge_E.
There is no pending posterior mass P.
inflight_count is scheduling metadata only.
```

For speed:

* Store `edge_E` and `edge_post_alpha` in float32.
* Network priors/logits can be stored as float16 only after verifying no instability.
* Clamp every alpha component to `eps > 0` before Dirichlet sampling.
* Use one packed binary blob with a version header.
* Implement `pack_node_blob(blob) -> bytes` and `unpack_node_blob(bytes) -> NodeBlob`.
* Benchmark raw struct packing vs msgpack/flatbuffers/capnp. Start simple but keep binary.

## 4. Redis APIs

Implement a store abstraction so search code does not know Redis details:

```python
class RedisNodeStore:
    def get_many(keys: list[StateKey]) -> dict[StateKey, NodeBlob | None]
    def put_many(blobs: list[NodeBlob]) -> None
    def claim_many_inflight(keys: list[StateKey], ttl_ms: int) -> ClaimResult
    def expand_many(blobs: list[NodeBlob]) -> None
    def update_many_edges(updates: list[EdgeUpdate]) -> None
```

Use pipelining for all multi-key operations.

Use an atomic Redis Lua script or Redis function for this operation:

```text
claim node as INFLIGHT if missing
else return existing status
```

Pseudo-logic:

```lua
-- claim_inflight(key, placeholder_blob, ttl)
if not EXISTS key:
    SET key placeholder_blob PX ttl
    return CLAIMED
else:
    return EXISTING_STATUS
```

Do not do separate `GET` then `SET` for claim; that races across workers.

If implementing a Redis NN queue, use it only for cross-process execution. For the first implementation, prefer an in-process NN queue because the active PGX state batch is already on device.

If a Redis NN queue is implemented, the queue payload must not contain full PGX state. It may contain:

```text
state_key
observation
legal_action_mask
current_player
terminal flags
representative_lane_id / request_id
```

The NN only needs observation/legal mask/current player. The tree expansion still happens in the search worker.

## 5. Local cache

Implement a local decoded node cache:

```python
local_node_cache: dict[StateKey, NodeBlob]
dirty_nodes: set[StateKey]
```

Search loop should do:

```text
unique_keys = unique(current_keys)
missing_from_cache = unique_keys - local_cache.keys()
redis MGET missing_from_cache
decode and insert into local_cache
```

Selection and backup should operate on local decoded blobs.

Flush dirty nodes to Redis at batch boundaries:

```text
after NN eval batch
after backup batch
at FinishSearch
```

Do not flush after every edge update.

## 6. Device-side PGX batch state

Redis does not store PGX state.

Maintain live PGX states only in the active traversal batch:

```python
pgx_state_batch        # pytree on device, shape [B, ...]
current_key[B]         # host/device state keys
root_id[B]
lane_id[B]
active_mask[B]
waiting_mask[B]
done_mask[B]
path_keys[B, max_depth]
path_actions[B, max_depth]
path_child_keys[B, max_depth]
path_len[B]
```

Use static shapes for JAX. Prefer masks over dynamic Python lists inside the device step.

Batch size:

```text
B = num_roots * lanes_per_root
```

Every lane starts at its root state. Duplicate the root PGX state into `lanes_per_root` copies.

## 7. Wavefront traversal

Implement wavefront traversal, not recursive per-lane traversal.

One wave:

```python
def search_wave(batch_state):
    # 1. Fetch current node blobs for all active lanes.
    current_keys = batch_state.current_key[active_mask]
    unique_keys = unique(current_keys)
    blobs = node_store.get_many_cached(unique_keys)

    # 2. Pack blobs into fixed-shape arrays for JAX selection.
    packed = pack_nodes_for_device(blobs, batch_state.current_key)
    # packed contains:
    #   legal_actions[B, Amax]
    #   legal_mask[B, Amax]
    #   edge_post_alpha[B, Amax, 3]
    #   policy_logits[B, Amax]
    #   child_keys[B, Amax]

    # 3. Select actions vectorized.
    actions = select_actions_thompson_jit(
        edge_post_alpha=packed.edge_post_alpha,
        legal_mask=packed.legal_mask,
        rng=batch_state.rng,
    )

    # 4. Step PGX state on device.
    next_pgx_state = step_fn(batch_state.pgx_state, actions)

    # 5. Compute canonical child keys on device.
    child_keys = state_key_fn(next_pgx_state)

    # 6. Copy only compact child_keys / terminal flags / masks to host.
    #    Do not copy full PGX state.
    host_child_keys = device_get(child_keys)
    host_terminal = device_get(next_pgx_state.terminated | next_pgx_state.truncated)
    host_legal_masks = device_get(next_pgx_state.legal_action_mask)  # only for lanes needing expansion/eval

    # 7. Update path arrays.
    append PathStep(parent_key, action, child_key)

    # 8. Classify child nodes:
    #    - terminal: backup terminal evidence
    #    - expanded in cache/Redis: continue traversal
    #    - missing: claim INFLIGHT and add representative lane to NN eval batch
    #    - inflight: park lane or mark waiting
```

Stop when every lane is one of:

```text
leaf_needs_nn_eval
terminal_done
waiting_on_inflight
max_depth
root_simulation_budget_done
```

Then run one batched NN eval for all unique claimed leaves.

## 8. Selection

Implement Thompson selection over WDL posteriors:

```python
@jax.jit
def select_actions_thompson(edge_post_alpha, legal_mask, rng):
    # edge_post_alpha: [B, Amax, 3]
    # legal_mask: [B, Amax]
    # sample Dirichlet via Gamma samples:
    gamma = jax.random.gamma(rng, edge_post_alpha)
    phi = gamma / jnp.sum(gamma, axis=-1, keepdims=True)
    utility = phi[..., W] - phi[..., L]
    utility = jnp.where(legal_mask, utility, -jnp.inf)
    idx = jnp.argmax(utility, axis=-1)
    return legal_actions[jnp.arange(B), idx]
```

Do not replace this with posterior-mean argmax for traversal.

Posterior-mean utility is useful for greedy final action or cheap `pi_search` approximation:

```python
q_mean = (alpha_W - alpha_L) / alpha_sum
```

But traversal is Thompson sampling.

## 9. Expansion and NN eval

When a lane reaches a missing child state:

1. It has already stepped PGX on device.
2. It has `child_state_key`.
3. The host checks local cache / Redis.
4. If missing, atomically claim `INFLIGHT`.
5. The representative lane is added to the NN eval batch.
6. Other lanes that hit the same key attach as waiters or are parked.

NN eval batch input:

```python
EvalBatch:
    state_keys[N]
    observations[N, ...]
    legal_action_masks[N, action_dim]
    current_players[N]
    representative_lane_ids[N]
```

NN eval output:

```python
EvalResult:
    state_key
    value_alpha[3]
    q_alpha[action_dim, 3] or q_alpha_legal[legal_count, 3]
    policy_logits[action_dim] or logits_legal[legal_count]
```

Expansion builds a `NodeBlob`:

```python
blob.status = EXPANDED
blob.legal_actions = legal actions from PGX legal mask
blob.value_alpha = result.value_alpha
blob.policy_logits = result.policy_logits[legal_actions]
blob.q_alpha = result.q_alpha[legal_actions]
blob.edge_base_alpha = q_alpha initially
blob.edge_E = zeros
blob.edge_post_alpha = edge_base_alpha
blob.child_state_keys = null
blob.edge_visits = zeros
blob.pi_search_approx = compute_pi_search_approx(blob)
blob.state_summary_alpha = compute_state_summary_alpha(blob)
```

For terminal child states:

```python
blob.status = TERMINAL
blob.terminal_wdl = one_hot_terminal_outcome_from_current_player_perspective
```

No NN eval is needed.

## 10. Parent edge child links

After stepping `(parent_key, action) -> child_key`, update the parent node locally:

```python
parent_blob.child_state_keys[action_index] = child_key
```

If child is expanded, update parent edge base alpha if following the file’s `EdgeBase` rule:

```python
parent.edge_base_alpha[action] = align_parent_from_child(child.value_alpha)
parent.edge_post_alpha[action] = parent.edge_base_alpha[action] + parent.edge_E[action]
```

Do not fetch the child on every future selection just to compute parent posterior. Materialize this update when the child is expanded or first observed.

For transpositions, many parent edges may point to the same `child_key`. Do not copy the child node. Store the same key in all parents.

Do not maintain reverse parent lists in MVP. Reverse indices are expensive. Parent edges can be updated lazily when traversed.

## 11. Search-weighted backup

Implement two levels of evidence:

### 11.1 Direct leaf evidence

For terminal leaf:

```python
d_leaf = one_hot_wdl
lambda_leaf = c_terminal
```

For non-terminal NN leaf:

```python
d_leaf = mean(value_alpha_leaf)
lambda_leaf = c_leaf
```

The final selected edge receives direct leaf evidence:

```python
E(parent, action) += lambda_leaf * align(parent <- leaf, d_leaf)
visits(parent, action) += 1
```

### 11.2 Search-weighted child-state summary for ancestors

For any expanded node `u`, compute:

```python
edge_post_alpha_u[a] = edge_base_alpha_u[a] + edge_E_u[a]

pi_search_u[a] = approximate_posterior_best_policy(edge_post_alpha_u)
```

Then:

```python
state_summary_alpha_u =
    sum_a pi_search_u[a] * edge_post_alpha_u[a]
```

This corresponds to:

```text
sum_a pi_search(a) * (E(a) + alpha_base(a))
```

Store `state_summary_alpha` in the node blob.

Default backup from child state `u` to incoming parent edge `(v,a)` should use a calibrated pseudo-evidence version:

```python
summary_d = state_summary_alpha_u / sum(state_summary_alpha_u)
E(v,a) += c_state * align(v <- u, summary_d)
```

Reason: raw `state_summary_alpha` already includes prior concentration and search evidence. Adding it at full strength can overconcentrate the parent posterior. Keep raw `state_summary_alpha` stored, but default backup should normalize it and apply a small `c_state`.

Add a config option:

```python
backup_config.propagate_raw_state_summary = False
```

If set true:

```python
E(v,a) += c_state * align(v <- u, state_summary_alpha_u)
```

but default should be normalized summary with `c_state`.

Recommended initial values:

```python
c_terminal > c_leaf
c_leaf around 0.25 to 1.0
c_state around 0.01 to 0.25
```

### 11.3 Leaf-to-root backup order

Process path from leaf to root.

For path:

```text
(v0, a0) -> v1
(v1, a1) -> v2
...
(v_{L-1}, a_{L-1}) -> vL
```

Backup procedure:

```python
# 1. Update last edge with direct leaf evidence.
update_edge_with_leaf_evidence(v_{L-1}, a_{L-1}, vL)

# 2. Recompute summary for v_{L-1}.
recompute_state_summary(v_{L-1})

# 3. For ancestors i = L-2 down to 0:
#    use child node v_{i+1}'s freshly updated summary.
for i in reversed(range(0, L-1)):
    child = v_{i+1}
    summary_d = normalize(child.state_summary_alpha)
    E(v_i, a_i) += c_state * align(v_i <- child, summary_d)
    visits(v_i, a_i) += 1
    recompute_state_summary(v_i)
```

This makes ancestor evidence depend on the child’s improved local search distribution, not just the raw leaf value.

## 12. Computing pi_search

For the root policy target, use posterior-best Monte Carlo:

```python
for m in range(M):
    phi[a] ~ Dirichlet(edge_post_alpha[a])
    a_star = argmax_a (phi[a,W] - phi[a,L])
pi_search[a] = count(a_star == a) / M
```

For internal backup summaries, speed matters. Implement both:

```python
mode = "softmax_mean"  # default for internal nodes
mode = "mc"            # optional more accurate
```

Fast default:

```python
q_mean[a] = (alpha_W[a] - alpha_L[a]) / sum_z alpha_z[a]
pi_search[a] = softmax(q_mean[a] / tau_internal)
```

Optional internal MC:

```python
M_internal = 8 or 16
```

Root target can use larger `M_root`, e.g. 64, 128, or 256 depending on cost.

## 13. FinishSearch and targets

At root:

```python
root_edge_post_alpha[a] = root.edge_base_alpha[a] + root.edge_E[a]
pi_search_root = posterior_best_mc(root_edge_post_alpha, M_root)
q_mean_root[a] = (alpha_W - alpha_L) / alpha_sum
```

Return:

```python
SearchResult:
    root_state_key
    policy_target = pi_search_root
    q_target_alpha[a] = root_edge_post_alpha[a] for explored actions
    q_mask[a] = 1 if sum(root.edge_E[a]) > 0 else 0
    action_to_play
```

For action to play, support config:

```python
commit_mode = "argmax_q_mean"       # greedy
commit_mode = "sample_pi_search"    # self-play exploration
commit_mode = "posterior_best_sample"
```

Default for fastest deterministic evaluation:

```python
action_to_play = argmax_a q_mean_root[a]
```

For self-play, use temperature/sample from `pi_search_root` as configured by the training loop.

## 14. Handling in-flight nodes without posterior reservation mass

Because we removed temporary pending posterior mass, use status-only scheduling:

```text
MISSING -> INFLIGHT -> EXPANDED
```

If a lane selects an action whose child is already `INFLIGHT`:

Options:

1. Park the lane until that key expands.
2. Restart that lane from the root for another simulation.
3. Resample within the same node with the inflight action masked.

MVP: park the lane. This is simplest and avoids reintroducing virtual-loss-like behavior.

Track:

```python
waiting_on_key[lane] = child_key
```

When `child_key` expands, unpark lanes.

Do not add artificial evidence or concentration for in-flight work.

## 15. Redis batching rules

All Redis access must be batch-oriented:

```text
MGET many node blobs
MSET many dirty node blobs
claim_many_inflight via pipeline or Lua batch
```

Flush policy:

```text
- after each NN eval batch,
- after each backup batch,
- before FinishSearch,
- before worker shutdown.
```

Metrics to log:

```text
redis_mget_count
redis_mset_count
redis_round_trip_ms
unique_keys_per_wave
local_cache_hit_rate
inflight_collision_rate
duplicate_leaf_rate
dirty_nodes_per_flush
blob_bytes_avg
```

## 16. Suggested files/modules

Implement roughly:

```text
dirichlet_tree/
  state_hash.py
  node_blob.py
  redis_store.py
  local_cache.py
  selection.py
  pi_search.py
  backup.py
  wavefront_search.py
  eval_queue.py
  targets.py
  tests/
```

### `state_hash.py`

```python
state_to_key(state) -> UInt128
state_to_key_batch = jax.jit(jax.vmap(state_to_key))
```

### `node_blob.py`

```python
@dataclass NodeBlob
pack_node_blob(blob) -> bytes
unpack_node_blob(data: bytes) -> NodeBlob
```

### `redis_store.py`

```python
class RedisNodeStore:
    get_many(keys)
    put_many(blobs)
    claim_many_inflight(keys)
    expand_many(blobs)
    flush_dirty(cache)
```

### `selection.py`

```python
select_actions_thompson_jit(...)
posterior_mean_utility(...)
```

### `pi_search.py`

```python
posterior_best_mc(...)
pi_search_softmax_mean(...)
compute_state_summary_alpha(...)
```

### `backup.py`

```python
backup_paths(...)
update_edge_leaf(...)
update_edge_state_summary(...)
recompute_node_summary(...)
```

### `wavefront_search.py`

```python
class BatchedPosteriorSearch:
    initialize_roots(root_pgx_states)
    run_search(num_simulations)
    run_wave()
    build_eval_batch()
    consume_eval_results()
    finish_search()
```

## 17. Test requirements

Implement tests before optimizing.

### 17.1 State key tests

* Same PGX state gives same key.
* Different legal states usually give different keys.
* State reached by two different parent paths maps to same key in a toy transposition environment.
* Key includes current player and rule-relevant state.

### 17.2 Redis blob tests

* Pack/unpack roundtrip is exact within dtype precision.
* `edge_post_alpha == edge_base_alpha + edge_E`.
* No PGX state is serialized into Redis.
* Blob version mismatch fails clearly.

### 17.3 Claim/inflight tests

* Two workers claiming same missing key produce exactly one `CLAIMED`.
* Existing `INFLIGHT` key is not overwritten.
* Existing `EXPANDED` key is not overwritten.

### 17.4 Batched traversal tests

* All lanes step with `vmap(env.step)`.
* No single-lane Redis call inside the inner loop.
* Terminal states back up without NN eval.
* Missing non-terminal states are deduplicated before NN eval.

### 17.5 Backup tests

* Direct leaf evidence updates only `edge_E`, not base alpha.
* Perspective flip maps `(L,D,W)` to `(W,D,L)`.
* `state_summary_alpha = sum_a pi_search[a] * edge_post_alpha[a]`.
* Ancestor backup uses the child summary and `c_state`.
* In-flight metadata never changes posterior alpha.

### 17.6 FinishSearch tests

* Root `pi_search` sums to 1.
* Q mask is 1 only for actions with completed evidence.
* Greedy action equals argmax posterior mean utility when `commit_mode="argmax_q_mean"`.

## 18. Performance milestones

Start with correctness, then measure.

Milestone 1: local-only, no Redis.

```text
Batched PGX stepping
Batched Thompson selection
Batched NN eval
Local dict node store
Correct backup and targets
```

Milestone 2: Redis backing store.

```text
NodeBlob pack/unpack
MGET/MSET
local cache
dirty flush
atomic inflight claim
```

Milestone 3: transpositions.

```text
child_state_key = hash(actual PGX state)
parent edge stores child key
multiple parents can point to same node
dedupe NN evals by child key
```

Milestone 4: wavefront batching.

```text
B = roots * lanes_per_root
static JAX shapes
mask/park lanes
all leaves evaluated in batches
```

Milestone 5: backup precision.

```text
state_summary_alpha
pi_search-weighted child summaries
c_state ancestor pseudo-evidence
root posterior-best target
```

Milestone 6: optimize.

```text
binary blob layout
float16 priors where safe
float32 evidence
cache hit rate
Redis pipeline batch size
NN batch saturation
device/host sync minimization
```

## 19. Non-goals for the first implementation

Do not implement these in MVP:

```text
temporary posterior reservation mass P
reverse parent propagation lists
global cross-training posterior sharing
storing PGX state in Redis
per-edge Redis fields
single-lane Redis calls
complex distributed NN workers
```

The first implementation should be a fast single-process search worker with Redis-backed canonical node blobs and an in-process NN eval batch. Distributed workers can come later once the local batched design is fast and correct.

## 20. Key design decisions to preserve

1. Canonical state key is computed from actual PGX state, not from parent hash plus action.
2. Parent edge stores action -> child state key.
3. Redis stores compact posterior node blobs, not PGX states.
4. PGX state batch stays on device.
5. Selection and stepping are vectorized.
6. Redis reads/writes are batched and cached.
7. In-flight scheduling does not affect posterior alpha.
8. Ancestor backup uses `pi_search`-weighted child summaries:
   `sum_a pi_search(a) * (alpha_base(a) + E(a))`.
9. Root target remains posterior-best policy from final root action posteriors.

```
```