Read it. Here is the rewritten spec.

I kept the core boundary from the older spec: Rust owns the mutable search forest, while Python only sends and receives opaque state batches and tensors. I also kept the posterior-best policy target semantics from the math reference: the policy target is the Monte Carlo estimate of the probability that each legal action is optimal under Dirichlet WDL samples, and the committed move mode is separate from the target used for training.

---

# Spec: Rust Batched-Forest Posterior Search Backend for Dirichlet-Q AlphaZero

## 0. Purpose

Implement an MVP Rust search backend for Dirichlet-Q AlphaZero.

The MVP is intentionally simpler than the older Rust-side tree-search spec:

```text
Rust owns the full mutable search forest.
Python owns PGX/environment stepping and neural network evaluation.
Python never manipulates tree nodes, paths, locks, or edge data.
Python only exchanges opaque state batches and batched tensors with Rust.
```

The backend searches a batch of trees. Each individual tree search is single-threaded. This removes the need for duplicate-request suppression, in-tree race handling, virtual loss, in-flight posterior mass, dirty concurrent repair, atomic edge snapshots, and similar complexity.

The MVP must natively support two node kinds:

```text
Decision node:
  A player-to-move node with legal actions and neural policy/value/Q outputs.

Terminal node:
  A completed game state with a positive terminal WDL Dirichlet alpha supplied
  by Python when it submits a fused transition/evaluation result.
```

Categorical/chance nodes are explicitly out of scope for the Scacchi MVP.
If a future stochastic environment needs them, Python must still own all
environment stepping and submit explicit outcome data; Rust should not grow a
game adapter.

The design must also be future-proof for subtree reuse. When an actual game move is played by the player or opponent, Rust should be able to prune the current tree to the already-created child subtree rather than rebuilding from scratch.

---

## 1. What changed from the older spec

Remove these from the MVP:

```text
num_threads
max_attempts
max_inflight_per_tree
terminal_epsilon
terminal_kappa
leaf_value_mode
kappa_leaf
dense_edges toggle
EvalStatus atomics
ValueStatus dirty/updating state
repair_token
dirty_flag
dirty queues
edge-local locks
CAS child insertion
CAS leaf claiming
Rayon traversal
per-tree duplicate suppression
virtual loss
in-flight posterior mass
```

Keep these concepts:

```text
action_size
observation_shape
simulations_per_root
posterior_best_samples
kappa_n
seed
debug
token-based fused transition/evaluation submission
posterior-best policy target
Dirichlet WDL edge posteriors
Q fallback before child expansion
child value / child cache after child expansion
clean interior-node training export
opaque state handles
padded transition batches
```

The old concurrent design required request records, in-flight state, duplicate suppression, and atomics because many workers could touch the same tree concurrently. In this MVP, every tree is mutated by only one Rust control path at a time, so duplicate requests and intra-tree races are structurally impossible.

---

## 2. Source of truth

Before coding, read:

```text
/mnt/data/algorithms.tex
/mnt/data/math.md
```

The implementation should follow the posterior-tree semantics in those files, with the simplifications in this MVP spec.

Core mathematical conventions:

```text
WDL order: [L, D, W]
Utility:   U(phi) = phi[W] - phi[L]
Flip:      flip([L, D, W]) = [W, D, L]
```

The network predicts, for decision nodes only:

```text
policy_logits: [A]
value_alpha:   [3]
q_alpha:       [A, 3]
```

The training policy target is not a visit-count target. It is:

```text
pi_search(a | s)
  = P(a is optimal under posterior WDL samples)
```

estimated by Monte Carlo Dirichlet sampling over legal actions. 

---

## 3. High-level architecture

Expose one Python module:

```python
import dqaz
```

Expose one main Python class:

```python
engine = dqaz.SearchEngine(config)
```

The engine owns a forest:

```text
Forest
  Tree 0
  Tree 1
  ...
  Tree B-1
```

Each tree is searched independently. Rust performs posterior traversal and
selects state-action transitions to evaluate. Python then executes one padded,
fused batch:

```text
child_states = env.step(parent_states, actions)
policy_logits, value_alpha, q_alpha = model(child_states.observation)
```

and submits both the child-state metadata and neural outputs back to Rust.

The Python loop should look like:

```python
root_eval = model_apply(root_states.observation)
tree_ids = engine.add_roots(
    root_states=root_states,
    observations=root_states.observation,
    legal_masks=root_states.legal_action_mask,
    current_players=root_states.current_player,
    policy_logits=root_eval.policy_logits,
    value_alpha=root_eval.value_alpha,
    q_alpha=root_eval.q_alpha,
)

while not engine.is_done(tree_ids):
    batch = engine.request_transitions(max_batch_size=256, pad_to=256)

    if batch.size == 0:
        continue

    child_states = env_step(batch.parent_states, batch.actions)
    child_eval = model_apply(child_states.observation)

    engine.submit_transitions(
        batch.token,
        child_states=child_states,
        observations=child_states.observation,
        legal_masks=child_states.legal_action_mask,
        current_players=child_states.current_player,
        terminated=child_states.terminated,
        terminal_alpha=terminal_alpha_from_rewards(child_states),
        policy_logits=child_eval.policy_logits,
        value_alpha=child_eval.value_alpha,
        q_alpha=child_eval.q_alpha,
    )

results = engine.finish(tree_ids, commit="posterior_sample")
targets = engine.export_targets(tree_ids)
```

No Python callback is allowed inside traversal, backup, repair, target
construction, pruning, or export. Python is called only by the outer driver when
Rust returns a `TransitionBatch`.

---

## 4. Config

Expose:

```python
config = dqaz.SearchConfig(
    action_size=A,
    observation_shape=obs_shape,
    simulations_per_root=64,
    posterior_best_samples=128,
    kappa_n=32.0,
    seed=0,
    debug=False,
)
```

Fields:

```text
action_size:
  Number of discrete decision actions.

observation_shape:
  Shape of one decision-node observation supplied by Python for root, leaf, and
  target export rows.

simulations_per_root:
  Target number of completed leaf evaluations under the current root.
  Existing reusable subtree evidence counts toward this budget after pruning.

posterior_best_samples:
  Monte Carlo samples used to estimate posterior-best policy targets.

kappa_n:
  Half-trust count for blending decision-node neural value alpha with child evidence:
      gamma = N_down / (kappa_n + N_down)

seed:
  RNG seed for Thompson sampling, posterior-best sampling, and posterior_sample
  committed actions.

debug:
  Enables extra validation and debug arrays.
```

Do not include terminal concentration or smoothing fields in the search config.
Terminal WDL alpha is supplied by Python in transition submissions and must
already be strictly positive for terminated rows.

---

## 5. Environment and Evaluation Boundary

Do not implement a Rust game adapter for Scacchi/PGX. Rust should not know how
to step Hex, encode observations, enumerate legal actions, or compute terminal
rewards. Python already has the PGX state and the model, so the only boundary
crossing needed during search is a padded fused transition/evaluation batch.

Rust owns:

```text
tree topology
node/edge posterior state
Thompson traversal
backup and cache recomputation
target export
subtree reuse for children that already exist
```

Python owns:

```text
PGX state objects
env.step / auto-reset policy outside the search tree
observation tensors
legal action masks
current-player metadata
terminal detection and terminal WDL alpha construction
neural network evaluation
```

A non-root leaf evaluation is therefore:

```text
Rust returns: parent_state, action, request metadata
Python runs:   child_state = env.step(parent_state, action)
Python runs:   logits, alpha_v, alpha_q = model(child_state.observation)
Python submits:
  child_state
  child observation
  child legal mask
  child current player
  terminated flag
  terminal alpha for terminated rows
  logits/value/q outputs for non-terminal rows
```

WDL perspective alignment is derived from submitted `current_player` metadata:
if the child current player differs from the parent current player, Rust flips
`[L, D, W] -> [W, D, L]`; otherwise it keeps the alpha unchanged.

Categorical/chance nodes are not part of the Scacchi MVP. If stochastic
environments are added later, Python should submit explicit chance outcomes and
probabilities through a separate transition-result shape; Rust still should not
contain a game adapter.

---

## 6. Python API

### 6.1 Add roots

Expose:

```python
tree_ids = engine.add_roots(
    root_states,
    observations,
    legal_masks,
    current_players,
    policy_logits,
    value_alpha,
    q_alpha,
    terminated=None,
    terminal_alpha=None,
)
```

`root_states` is a Python-owned opaque batch of environment states. Rust stores
opaque handles and returns them in later transition requests, but never inspects
or mutates their contents.

`add_roots` creates already-evaluated decision roots. Python is responsible for
running the root model evaluation before calling `add_roots`, because the root
does not require an environment transition.

For non-terminal rows:

```text
observations:     [B, *observation_shape], float32
legal_masks:      [B, action_size], bool
current_players:  [B], int32
policy_logits:    [B, action_size], float32
value_alpha:      [B, 3], float32, strictly positive
q_alpha:          [B, action_size, 3], float32, strictly positive
```

If `terminated` is provided and true for a row, `terminal_alpha[row]` must be
strictly positive. Terminal roots are tracked as complete but cannot be finished
into an action.

---

### 6.2 Request transitions

Expose:

```python
batch = engine.request_transitions(
    max_batch_size: int,
    pad_to: int | None = None,
) -> TransitionBatch
```

Returned object:

```python
@dataclass
class TransitionBatch:
    token: int
    size: int
    padded_size: int

    parent_states: object         # Python state batch, length padded_size
    actions: np.ndarray           # [padded_size], int32
    active_mask: np.ndarray       # [padded_size], bool

    # Debug only:
    tree_ids: np.ndarray          # [padded_size], uint64
    parent_node_ids: np.ndarray   # [padded_size], uint32
    request_ids: np.ndarray       # [padded_size], uint64
    tree_generations: np.ndarray  # [padded_size], uint32
```

Rules:

```text
1. Each active row is one selected decision transition `(parent_state, action)`.

2. A tree may have at most one pending transition request in the MVP.

3. request_transitions should iterate trees round-robin for fairness.

4. Traversal may complete terminal simulations without returning rows.

5. If `pad_to` is provided, Rust pads `parent_states/actions` to exactly
   `pad_to` rows by repeating any active row and sets `active_mask=False` for
   padding rows. Python may run the fused `env.step + model` over the full
   padded batch.

6. request_transitions returns when:
   - max_batch_size requests are collected, or
   - no eligible tree can currently produce a request, or
   - all selected trees are done.

7. The token is opaque and must be passed to submit_transitions.

8. Python must not see paths, edge IDs, child IDs, or mutable tree objects.
```

Because each tree is single-threaded and has at most one pending request,
duplicate leaf requests are impossible in the MVP.

---

### 6.3 Submit transitions

Expose:

```python
engine.submit_transitions(
    token: int,
    child_states,                 # opaque Python state batch, [padded_size]
    observations: np.ndarray,     # [padded_size, *obs_shape], float32
    legal_masks: np.ndarray,      # [padded_size, A], bool
    current_players: np.ndarray,  # [padded_size], int32
    terminated: np.ndarray,       # [padded_size], bool
    terminal_alpha: np.ndarray,   # [padded_size, 3], float32, positive for terminals
    policy_logits: np.ndarray,    # [padded_size, A], float32
    value_alpha: np.ndarray,      # [padded_size, 3], float32, positive for nonterminals
    q_alpha: np.ndarray,          # [padded_size, A, 3], float32, positive for nonterminals
)
```

Rows where `active_mask=False` in the matching `TransitionBatch` are ignored
after shape validation. This lets Python always run one statically-shaped fused
transition/evaluation call.

On submit, Rust must:

```text
1. Look up request records by token.
2. Validate tensor shapes.
3. Remove token from request table so it cannot be submitted twice.
4. Match outputs by row order.
5. Ignore stale requests whose tree generation no longer matches.
6. For terminal child rows:
   - create/store the terminal child state and terminal alpha;
   - back up terminal_alpha along the stored path;
   - count one completed simulation under the current root.
7. For non-terminal child rows:
   - create/store the child state, observation, legal mask, and current player;
   - store policy logits, value alpha, and Q alpha;
   - create a decision child with submitted model outputs;
   - initialize C^V to value_alpha;
   - back up value_alpha along the stored path;
   - count one completed simulation under the current root.
8. Recompute affected caches synchronously on the path to the root.
```

No `leaf_value_mode` exists. Neural leaves always back up `value_alpha`.

---

### 6.4 Done status

Expose:

```python
done = engine.is_done(tree_ids=None) -> bool
stats = engine.stats() -> dict
```

A tree is done when:

```text
current root is a decision node
root_completed_count >= simulations_per_root
tree has no pending transition request
```

Where:

```text
root_completed_count = root.n_down
```

for a clean current root.

Root model evaluation does not count as a completed simulation.

If the root is terminal, it is complete but has no action to finish.

---

### 6.5 Finish search

Expose:

```python
results = engine.finish(
    tree_ids=None,
    commit="posterior_sample",
) -> SearchResults
```

Supported commit modes:

```text
posterior_sample:
  sample action from posterior-best policy target.

posterior_argmax:
  choose argmax of posterior-best policy target.

mean_utility_argmax:
  choose argmax of posterior mean utility:
      U(alpha / sum(alpha))
  This is optional but useful for debugging the scalar-Q improvement view.
```

Returned object:

```python
@dataclass
class SearchResults:
    tree_ids: np.ndarray          # [G], uint64
    actions: np.ndarray           # [G], int32
    pi_search: np.ndarray         # [G, A], float32
    root_alpha: np.ndarray        # [G, A, 3], float32
    root_q_mean: np.ndarray       # [G, A], float32
    legal_masks: np.ndarray       # [G, A], bool
```

Before finishing each tree, Rust must:

```text
1. Require no pending transition request.
2. Require current root kind == Decision.
3. Require root_completed_count >= simulations_per_root,
   unless a future partial_finish mode is added.
4. Compute EdgePosterior(root, a) for every legal action.
5. Estimate pi_search using posterior_best_samples.
6. Select committed action according to commit mode.
```

The policy head is trained toward `pi_search` regardless of which committed-action mode is used. The math reference separates the posterior-best target from the committed action mode. 

---

### 6.6 Export training targets

Expose:

```python
targets = engine.export_targets(tree_ids=None) -> SearchTargets
```

Returned object:

```python
@dataclass
class SearchTargets:
    observations: np.ndarray       # [N, *observation_shape], float32
    legal_masks: np.ndarray        # [N, A], bool

    policy_target: np.ndarray      # [N, A], float32

    q_target_alpha: np.ndarray     # [N, A, 3], float32
    q_loss_weight: np.ndarray      # [N, A], float32

    v_target_alpha: np.ndarray     # [N, 3], float32

    row_mask: np.ndarray           # [N], bool

    # Debug optional:
    tree_ids: np.ndarray           # [N], uint64
    node_ids: np.ndarray           # [N], uint32
    depths: np.ndarray             # [N], uint32
```

Export only decision nodes satisfying:

```text
node kind == Decision
node has submitted model outputs
state is non-terminal
has_child_evidence == true
state cache C^V is available
```

Exclude:

```text
terminal nodes
pending transition leaves
newly evaluated decision leaves with no child evidence
nodes outside the current retained subtree
padding rows
```

For each exported decision node:

```text
policy_target[a] =
  posterior-best policy target over EdgePosterior(v, a)

q_target_alpha[a] =
  EdgePosterior(v, a)

q_loss_weight[a] =
  policy_target[a]

v_target_alpha =
  C^V_v
```

Illegal action rows:

```text
legal_mask[a] = False
policy_target[a] = 0
q_loss_weight[a] = 0
q_target_alpha[a] = [1, 1, 1] or any other positive dummy alpha
```

Terminal nodes do not produce policy or Q rows.

---

### 6.7 Clear/reset

Expose:

```python
engine.clear(tree_ids=None)
engine.clear_all()
```

This drops selected trees and their pending request records.

---

### 6.8 Root advancement and subtree reuse

Expose:

```python
engine.advance_roots(
    tree_ids: np.ndarray,         # [G], uint64
    actions: np.ndarray,          # [G], int32
)
```

This is used after an actual move is played in the real game, whether by the player or the opponent.

Rules:

```text
1. Current root must be a decision node.

2. The action must be legal at the current root.

3. The root must already have a child for that action.

4. Promote that child to be the new root.

5. Preserve all nodes, edge posteriors, network outputs, and caches in the promoted child subtree.

6. Drop or detach old ancestors and unreachable sibling subtrees unless the caller explicitly exports
   their targets before pruning.

7. Increment tree_generation so any stale pending request from the old root can be ignored.

8. Recompute root depth and parent pointers as needed.
```

If the real game advances along an action that was not searched and no child
exists, Python must create a fresh root with `add_roots` after evaluating that
state. Rust should not call back into the environment to synthesize missing
children.

After pruning, the new root may already have child evidence. The search budget for the new root should count reused evidence:

```text
root_completed_count = root.n_down
remaining = max(0, simulations_per_root - root_completed_count)
```

This is the key reuse behavior: already visited nodes under the played move are not recalculated.

For the MVP, either require no pending request before pruning, or cancel the
pending request by incrementing `tree_generation` and clearing
`tree.pending_request`. If the old fused result later arrives,
`submit_transitions` must ignore it as stale.

---

## 7. Internal data model

Use integer IDs, not pointer graphs.

```rust
type TreeId = u64;
type NodeId = u32;
type RequestId = u64;
type BatchToken = u64;
type Action = u16;
```

### 7.1 Forest

```rust
struct Forest {
    config: SearchConfig,

    trees: Vec<Tree>,
    free_tree_slots: Vec<usize>,

    next_tree_id: TreeId,
    next_request_id: RequestId,
    next_batch_token: BatchToken,

    request_table: HashMap<BatchToken, Vec<RequestRecord>>,
    round_robin_cursor: usize,
}
```

The MVP may protect public Python calls with a simple engine-level mutex. Inside each call, tree mutation is sequential per tree.

Future forest-level parallelism is allowed only if each worker owns disjoint trees for the duration of the call.

---

### 7.2 Tree

```rust
struct Tree {
    id: TreeId,
    generation: u32,

    nodes: Vec<Node>,
    root: NodeId,

    pending_request: Option<RequestId>,

    rng: ChaCha20Rng,
}
```

Do not store a permanent `T_done` lifetime counter. For the current root, use:

```text
root_completed_count = root.n_down
```

This allows subtree reuse after pruning.

---

### 7.3 Node

```rust
struct Node {
    id: NodeId,
    generation: u32,

    parent: Option<NodeId>,
    parent_link: Option<ParentLink>,
    depth: u32,

    state: PyObject,              // opaque Python/PGX state handle
    observation: Vec<f32>,         // [*observation_shape]
    legal_mask: Vec<bool>,         // [A]
    current_player: i32,
    kind: NodeKind,

    c_v: Option<[f32; 3]>,
    n_down: u32,
    cache_version: u32,
}
```

```rust
enum NodeKind {
    Decision(DecisionData),
    Terminal(TerminalData),
}
```

```rust
enum ParentLink {
    DecisionAction { action: Action },
}
```

---

### 7.4 DecisionData

```rust
struct DecisionData {
    policy_logits: Vec<f32>,       // [A]
    value_alpha: [f32; 3],
    q_alpha: Vec<[f32; 3]>,        // [A, 3]

    edges: Vec<DecisionEdge>,      // [A]
}
```

A decision node always has submitted model outputs. Transition requests are
pending on parent actions, not on unevaluated child nodes. The legal mask lives
on `Node` because it is supplied by Python for both roots and child states.

---

### 7.5 DecisionEdge

```rust
struct DecisionEdge {
    child: Option<NodeId>,

    completed: bool,
    b: [f32; 3],
    r_count: u32,

    child_cache_version: Option<u32>,
}
```

Interpretation:

```text
completed:
  Whether b is a completed edge posterior snapshot.

b:
  Full WDL Dirichlet posterior snapshot for this state-action edge.

r_count:
  Number of completed child summaries that have contributed to this edge.

child_cache_version:
  Version of child C^V that was last used to refresh this edge.
```

`b` is not accumulated component-wise. It is a full snapshot replacement.

---

### 7.6 TerminalData

```rust
struct TerminalData {
    alpha: [f32; 3],               // supplied by Python transition submission
}
```

The alpha must be strictly positive.

No `terminal_epsilon` or `terminal_kappa` exists in the engine config.

---

### 7.7 RequestRecord

```rust
struct RequestRecord {
    request_id: RequestId,
    tree_id: TreeId,
    tree_generation: u32,

    node_id: NodeId,
    node_generation: u32,

    path: Vec<PathStep>,
}
```

```rust
enum PathStep {
    DecisionAction {
        node_id: NodeId,
        action: Action,
    },
}
```

A batch token maps to:

```rust
Vec<RequestRecord>
```

For the MVP, each tree has at most one pending transition request. The request
table still exists because Python submits padded batched results by token.

---

## 8. Posterior semantics

### 8.1 Decision EdgeBase

For a decision node `v` and action `a`:

```text
DecisionEdgeBase(v, a):
  if edge(v,a) has child u and u is summarizable:
      return align_wdl(from_state=s_u, to_state=s_v, alpha=C^V_u)
  else if edge(v,a) has decision child u:
      return align_wdl(from_state=s_u, to_state=s_v, alpha=value_alpha[u])
  else:
      return q_alpha[v,a]
```

A child is summarizable when:

```text
Terminal:
  always summarizable using terminal alpha.

Decision:
  summarizable when it has child evidence and clean C^V.
```

A newly evaluated decision leaf with no child evidence is not summarizable. Its value alpha is used by the direct leaf backup, but it should not automatically overwrite the parent edge through child-cache refresh.

---

### 8.2 Decision EdgePosterior

```text
DecisionEdgePosterior(v, a):
  if edge(v,a).completed:
      return edge(v,a).b
  else:
      return DecisionEdgeBase(v, a)
```

---

### 8.3 Decision ThompsonSelect

For a decision node:

```text
ThompsonSelect(v):
  for each legal action a:
      alpha[a] = DecisionEdgePosterior(v, a)
      phi[a] ~ Dirichlet(alpha[a])
      utility[a] = phi[a,W] - phi[a,L]
  return argmax utility
```

No virtual loss or in-flight masking is required, because a tree has at most one pending request and is not concurrently searched.

---

### 8.4 PosteriorBestPolicyTarget

For a decision node:

```text
PosteriorBestPolicyTarget(v):
  count[a] = 0 for every action

  repeat M = posterior_best_samples times:
      for each legal action a:
          alpha[a] = DecisionEdgePosterior(v, a)
          phi[a] ~ Dirichlet(alpha[a])
          utility[a] = phi[a,W] - phi[a,L]

      a_star = argmax legal utility[a]
      count[a_star] += 1

  pi[a] = count[a] / M for legal actions
  pi[a] = 0 for illegal actions
```

---

## 9. State-posterior cache semantics

### 9.1 Decision node cache

For a decision node with child evidence:

```text
pi = PosteriorBestPolicyTarget(v)

E_v =
  sum over legal actions a:
      pi[a] * DecisionEdgePosterior(v, a)

N_down =
  sum over legal actions a:
      edge(v,a).r_count

gamma =
  N_down / (kappa_n + N_down)

C^V_v =
  (1 - gamma) * value_alpha[v] + gamma * E_v
```

If a decision node has no child evidence:

```text
C^V_v = value_alpha[v]
N_down = 0
```

Such a node is not exported as a training target row because it has no child evidence.

---

### 9.2 Terminal node cache

For a terminal node:

```text
C^V_terminal = terminal_alpha
N_down = 1
```

Terminal nodes are not exported.

---

## 10. Traversal and request generation

Implement internal:

```rust
fn next_transition_request(tree: &mut Tree, config: &SearchConfig)
    -> NextRequestResult;
```

```rust
enum NextRequestResult {
    TransitionRequest(RequestRecord),
    CompletedOneSimulation,
    BlockedByPendingRequest,
    TreeDone,
    NoProgress,
}
```

Core traversal:

```text
next_transition_request(tree):
  if tree has pending_request:
      return BlockedByPendingRequest

  if current root is a decision node and root_completed_count >= simulations_per_root:
      return TreeDone

  v = root
  path = []

  loop:
    match kind(v):

      Terminal:
        beta = terminal alpha at v
        backup(path, beta)
        return CompletedOneSimulation

      Decision:
        if decision has pending transition:
            return BlockedByPendingRequest

        action = ThompsonSelect(v)

        if edge(v, action).child exists:
            child = edge child
            path.push((v, action))
            v = child
            continue

        create transition request for parent state[v] and action
        mark tree pending_request = request_id
        return TransitionRequest(record)
```

For transition requests:

```text
path is non-empty and ends with the requested parent/action
submit_transitions creates the child from Python's fused result
submit_transitions backs up terminal_alpha or value_alpha along path
counts one completed simulation
```

---

## 11. Child creation

### 11.1 Decision action child

```text
create_decision_action_child(v, a, submitted_child):
  child = create_node_from_submitted_transition(submitted_child)
  edge(v,a).child = child
  return child
```

No CAS is needed in the MVP because one tree is not concurrently modified.
Rust never calls `env.step`; child creation happens only when Python submits a
transition result for a previously issued request.

---

## 12. Backup

Given:

```text
path = [
  PathStep(...),
  ...
]

beta_leaf: [f32; 3]
```

Backup must publish full edge snapshots. It must not do component-wise accumulation of `B`.

Rules:

```text
1. beta_leaf is a full Dirichlet alpha from:
   - neural value_alpha for a decision leaf, or
   - terminal alpha for a terminal leaf.

2. Walk path from leaf to root.

3. At each path step:
   - align beta/current child posterior to the parent node perspective;
   - publish it as the full edge posterior B;
   - set completed = true;
   - increment r_count;
   - recompute the parent node cache if possible.

4. If a recomputed parent cache becomes available and the parent itself has a parent edge,
   continue propagating the parent C^V upward.

5. Do not add EdgeBase + beta_leaf.

6. Do not atomic-add alpha_L, alpha_D, alpha_W.

7. Repeated evaluations increment r_count, but B remains a latest full posterior snapshot,
   not an accumulated Dirichlet count.
```

This preserves the snapshot semantics from `algorithms.tex`.

---

## 13. Synchronous cache recomputation

The older spec had dirty nodes, dirty queues, repair tokens, and cache versions to survive concurrent mutation. The MVP does not need them.

Implement:

```rust
fn recompute_upward_from_path(...)
```

Behavior:

```text
1. After a backup publishes an edge, recompute the parent node cache immediately.

2. If the parent is a decision node:
     - if it has child evidence, compute posterior-best policy target and C^V;
     - if it has no child evidence, C^V = value_alpha.

3. Increment cache_version whenever C^V changes.

4. Continue upward only when the recomputed node is summarizable.
```

A decision node is summarizable by its parent only if:

```text
has_child_evidence == true
C^V is available
```

This avoids overwriting a parent edge from a newly evaluated neural leaf with no child evidence.

A terminal node is always summarizable.

---

## 14. Finish semantics

For each selected tree:

```text
finish(tree):
  require no pending request
  require root kind == Decision
  require root_completed_count >= simulations_per_root

  root_alpha[a] = DecisionEdgePosterior(root, a) for legal actions
  root_alpha[a] = [1,1,1] for illegal actions

  root_q_mean[a] =
      (root_alpha[a][W] - root_alpha[a][L]) / sum(root_alpha[a])

  pi_search = PosteriorBestPolicyTarget(root)

  if commit == posterior_sample:
      action is sampled from pi_search

  if commit == posterior_argmax:
      action = argmax pi_search

  if commit == mean_utility_argmax:
      action = argmax root_q_mean over legal actions
```

`posterior_argmax` is greedy with respect to posterior optimal-action probability. `mean_utility_argmax` is greedy with respect to posterior mean utility.

---

## 15. Export semantics

Before exporting:

```text
require no pending request for selected trees
```

Then traverse retained nodes under each current root.

Export only decision nodes that satisfy:

```text
kind == Decision
has_child_evidence == true
C^V is available
node is still in retained current-root subtree
```

Do not export:

```text
terminal nodes
decision leaves with no child evidence
detached/pruned-away nodes
```

For each exported decision node:

```text
observation = node.observation

legal_mask = node.legal_mask

policy_target =
  PosteriorBestPolicyTarget(node)

q_target_alpha[a] =
  DecisionEdgePosterior(node, a) for legal actions

q_loss_weight[a] =
  policy_target[a]

v_target_alpha =
  node.C^V
```

## 16. Root pruning and reuse

### 16.1 Advance by decision action

```text
advance_roots(tree_ids, actions):
  for each tree:
      root = current root

      require root kind == Decision
      require action legal

      require root edge(action) has child
      new_root = root edge(action).child

      detach old parent/siblings
      set tree.root = new_root
      set new_root.parent = None
      set new_root.depth = 0
      increment tree.generation
      clear/cancel pending request if any
```

After pruning:

```text
If new_root is a decision node:
  reuse its policy logits, value alpha, Q alpha, edges, and caches.

If the chosen action was not searched and no child exists:
  Python must advance the environment externally, evaluate the resulting state,
  and create a fresh root with add_roots. Rust must not synthesize the child.

If new_root is terminal:
  no further action is available.
```

### 16.2 Search budget after pruning

The search budget after pruning must account for reused evidence:

```text
root_completed_count = root.n_down

if root_completed_count >= simulations_per_root:
    tree is already done for this root
else:
    remaining simulations are requested normally
```

This is the mechanism that avoids recalculating already visited nodes.

---

## 17. Scheduling over the forest

`request_transitions(max_batch_size, pad_to=None)` should be implemented as a
round-robin forest scheduler:

```text
request_transitions:
  records = []

  while records.len < max_batch_size:
      made_progress = false

      for each tree in round-robin order:
          if records.len == max_batch_size:
              break

          if tree has pending request:
              continue

          if tree is done:
              continue

          result = next_transition_request(tree)

          match result:
              TransitionRequest(record):
                  records.push(record)
                  made_progress = true

              CompletedOneSimulation:
                  made_progress = true
                  continue

              TreeDone:
                  continue

              BlockedByPendingRequest:
                  continue

              NoProgress:
                  continue

      if not made_progress:
          break

  if records empty:
      return TransitionBatch(size=0, padded_size=0)

  token = next_batch_token
  request_table[token] = records
  materialize parent_states and actions

  if pad_to is provided:
      require pad_to >= records.len
      padded_size = pad_to
      pad parent_states/actions by repeating an active row
      active_mask = true for active rows, false for padding rows
  else:
      padded_size = records.len
      active_mask = true for every row

  return TransitionBatch
```

Because each tree may have at most one pending request, batching comes from many trees, not from many concurrent lanes in one tree.

---

## 18. Determinism

The MVP should be reproducible with fixed seed in single-process mode.

Use per-tree RNG seeded from:

```text
global seed
tree id
tree generation
local counter
```

No bitwise determinism across future forest-level parallelism is required unless explicitly added later.

---

## 19. Validation and errors

Raise clear Python exceptions for:

```text
invalid config
unknown tree id
wrong transition batch token
submitting same token twice
transition output shape mismatch
pad_to smaller than collected active rows
opaque state batch length mismatch
current_player shape mismatch
alpha <= 0 for active rows in debug mode
terminal row missing positive terminal_alpha
non-terminal row missing positive value_alpha or q_alpha
invalid legal action in advance_roots
advance_roots called for an action without an existing child
finish called on non-decision root
finish called with pending request
finish called before simulations_per_root reached
export called with pending request
unsupported commit mode
terminal alpha invalid
```

Stale transition rows caused by root pruning should be ignored after token
lookup and generation checks. They should not mutate the tree.

In non-debug mode, validate shapes and critical invariants. Avoid expensive full-array validation unless debug is enabled.

---

## 20. Required tests

### 20.1 Posterior unit tests

Test:

```text
Dirichlet sampling returns legal argmax actions
posterior-best target sums to 1 over legal actions
illegal actions get zero policy probability
DecisionEdgeBase uses Q alpha before child expansion
DecisionEdgeBase uses aligned child value alpha after child expansion
newly evaluated decision leaf with no child evidence does not overwrite parent edge via refresh
terminal alpha must be strictly positive
align_wdl flip is correct
```

### 20.2 Fused transition request tests

Use a deterministic mock Python environment and model. Assert:

```text
request_transitions returns parent_states and actions, not observations
request_transitions returns active_mask and padded_size
padding rows repeat an active parent/action and are marked inactive
submit_transitions ignores inactive padded rows after shape validation
Python can run one fixed-size env.step + model call over padded_size rows
terminal child rows back up terminal_alpha without requiring NN outputs
non-terminal child rows store observations/legal masks/current players/NN outputs
```

### 20.3 MVP no-duplicate tests

Because each tree is single-threaded and has one pending request, assert:

```text
one tree cannot emit a second request while pending_request is set
request_transitions returns at most one pending row per tree
submitting the matching token clears pending_request
submitting stale token after prune is ignored safely
```

### 20.4 Backup tests

Assert:

```text
backup publishes full B snapshot
backup increments r_count
backup does not add EdgeBase + beta_leaf
repeated backups replace B rather than component-wise accumulating B
decision C^V recomputes synchronously after backup
```

### 20.5 Pruning/reuse tests

Construct a tree where root action `a` has an evaluated child with evidence.

Call:

```python
engine.advance_roots([tree_id], [a])
```

Assert:

```text
new root is the old child
existing child network outputs are preserved
existing child edges are preserved
existing child C^V is preserved
old siblings are detached or dropped
root_completed_count uses reused n_down
next request_transitions does not request a root model eval
advance_roots raises if the selected action has no existing child
```

### 20.6 Export tests

Assert exported rows satisfy:

```text
kind == Decision
has submitted model outputs
non-terminal
has_child_evidence
C^V available
```

Assert arrays have correct shapes:

```text
observations: [N, *obs_shape]
legal_masks: [N, A]
policy_target: [N, A]
q_target_alpha: [N, A, 3]
q_loss_weight: [N, A]
v_target_alpha: [N, 3]
```

Assert:

```text
terminal nodes are not exported
all exported target alphas are positive
```

### 20.7 Python integration test

Use a mock Python environment and model:

```python
def env_step(states, actions):
    return states.step(actions)

def model(obs, legal):
    B = obs.shape[0]
    A = legal.shape[1]

    policy_logits = np.zeros([B, A], np.float32)
    value_alpha = np.ones([B, 3], np.float32)
    q_alpha = np.ones([B, A, 3], np.float32)

    return policy_logits, value_alpha, q_alpha
```

Run:

```python
engine = dqaz.SearchEngine(config)

root_eval = model(root_states.observation, root_states.legal_action_mask)
tree_ids = engine.add_roots(
    root_states,
    root_states.observation,
    root_states.legal_action_mask,
    root_states.current_player,
    *root_eval,
)

while not engine.is_done(tree_ids):
    batch = engine.request_transitions(max_batch_size=128, pad_to=128)

    if batch.size:
        child_states = env_step(batch.parent_states, batch.actions)
        child_eval = model(child_states.observation, child_states.legal_action_mask)
        engine.submit_transitions(
            batch.token,
            child_states,
            child_states.observation,
            child_states.legal_action_mask,
            child_states.current_player,
            child_states.terminated,
            terminal_alpha_from_rewards(child_states),
            *child_eval,
        )

results = engine.finish(tree_ids)
targets = engine.export_targets(tree_ids)
```

Assert:

```text
no panics
no leaks
valid shapes
valid actions
all exported target alphas positive
```

---

## 21. Acceptance criteria

The MVP is acceptable when:

```text
1. Python imports dqaz successfully.

2. SearchEngine owns all tree memory in Rust.

3. Python never receives node pointers, paths, locks, or mutable tree objects.

4. request_transitions returns padded parent-state/action transition requests.

5. Each individual tree search is single-threaded.

6. Each tree has at most one pending transition request.

7. Duplicate leaf requests are structurally impossible in the MVP.

8. No virtual loss or in-flight posterior mass is used.

9. No terminal_epsilon, terminal_kappa, leaf_value_mode, kappa_leaf, max_attempts,
   max_inflight_per_tree, or num_threads config field exists.

10. Decision nodes use Thompson selection over Dirichlet WDL edge posteriors.

11. Rust contains no Scacchi/PGX game adapter and never calls env.step.

12. Python owns PGX states, env.step, legal masks, terminal alpha, and NN eval.

13. Terminal nodes use positive terminal alpha supplied by Python.

14. submit_transitions backs up neural value_alpha or terminal_alpha correctly.

15. Edge posterior B is a full snapshot replacement, not component-wise accumulated B.

16. finish returns posterior-best root policy and committed actions for decision roots.

17. export_targets returns flattened clean decision-node training rows.

18. advance_roots can prune to an existing child subtree after a played move.

19. advance_roots does not synthesize missing children; Python creates fresh roots for unsearched moves.

20. Reused subtree evidence counts toward the next root’s search budget.
```

---

## 22. Suggested file layout

```text
dqaz/
  pyproject.toml
  Cargo.toml

  src/
    lib.rs
    py_api.rs

    config.rs
    ids.rs

    forest.rs
    tree.rs
    node.rs
    edge.rs
    state_batch.rs

    request.rs
    scheduler.rs
    traversal.rs
    backup.rs
    posterior.rs
    targets.rs
    prune.rs
    rng.rs

  tests/
    test_python_api.py
    test_fused_transitions.py
    test_mock_search.py
    test_prune_reuse.py

  README.md
```

---

## 23. Minimal Python usage target

```python
import numpy as np
import dqaz

config = dqaz.SearchConfig(
    action_size=7,
    observation_shape=(6, 7, 2),
    simulations_per_root=32,
    posterior_best_samples=64,
    kappa_n=16.0,
    seed=0,
    debug=True,
)

engine = dqaz.SearchEngine(config)

root_states = make_initial_states(batch_size=64)

root_eval = model_apply(
    root_states.observation,
    root_states.legal_action_mask,
)

tree_ids = engine.add_roots(
    root_states,
    root_states.observation,
    root_states.legal_action_mask,
    root_states.current_player,
    root_eval.policy_logits,
    root_eval.value_alpha,
    root_eval.q_alpha,
)

while not engine.is_done(tree_ids):
    batch = engine.request_transitions(max_batch_size=128, pad_to=128)

    if batch.size == 0:
        continue

    child_states = env_step(batch.parent_states, batch.actions)
    child_eval = model_apply(
        child_states.observation,
        child_states.legal_action_mask,
    )

    engine.submit_transitions(
        batch.token,
        child_states,
        child_states.observation,
        child_states.legal_action_mask,
        child_states.current_player,
        child_states.terminated,
        terminal_alpha_from_rewards(child_states),
        child_eval.policy_logits,
        child_eval.value_alpha,
        child_eval.q_alpha,
    )

results = engine.finish(tree_ids, commit="posterior_sample")

# Play the actual moves in the environment.
actions = results.actions
next_env_states = env_step(root_states, actions)

# Reuse Rust tree subtrees only for actions that were searched.
engine.advance_roots(tree_ids, actions)

# Continue search from the pruned roots.
```
