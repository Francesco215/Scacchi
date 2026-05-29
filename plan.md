Read it. Here is the rewritten spec.

I kept the core boundary from the older spec: Rust owns the mutable search forest, while Python only sends and receives batched tensors. I also kept the posterior-best policy target semantics from the math reference: the policy target is the Monte Carlo estimate of the probability that each legal action is optimal under Dirichlet WDL samples, and the committed move mode is separate from the target used for training.  

---

# Spec: Rust Batched-Forest Posterior Search Backend for Dirichlet-Q AlphaZero

## 0. Purpose

Implement an MVP Rust search backend for Dirichlet-Q AlphaZero.

The MVP is intentionally simpler than the older Rust-side tree-search spec:

```text
Rust owns the full mutable search forest.
Python owns neural network evaluation.
Python never manipulates tree nodes, paths, locks, or edge data.
Python only exchanges batched tensors with Rust.
```

The backend searches a batch of trees. Each individual tree search is single-threaded. This removes the need for duplicate-request suppression, in-tree race handling, virtual loss, in-flight posterior mass, dirty concurrent repair, atomic edge snapshots, and similar complexity.

The MVP must natively support three node kinds:

```text
Decision node:
  A player-to-move node with legal actions and neural policy/value/Q outputs.

Categorical node:
  A stochastic/environment/chance node with explicit categorical outcomes and probabilities.
  It is not a policy node and is not trained as a policy/Q row.

Terminal node:
  A completed game state with a positive terminal WDL Dirichlet alpha supplied by the game adapter.
```

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
root eval requests through the normal request API
token-based submit_evaluations
posterior-best policy target
Dirichlet WDL edge posteriors
Q fallback before child expansion
child value / child cache after child expansion
clean interior-node training export
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

Each tree is searched independently. `request_evaluations(max_batch_size)` walks across trees, advances each eligible tree single-threadedly, and returns a batch of neural leaf requests to Python.

The Python loop should look like:

```python
tree_ids = engine.add_roots(root_states)

while not engine.is_done(tree_ids):
    batch = engine.request_evaluations(max_batch_size=256)

    if batch.size == 0:
        continue

    policy_logits, value_alpha, q_alpha = model_apply(
        batch.observations,
        batch.legal_masks,
    )

    engine.submit_evaluations(
        batch.token,
        policy_logits,
        value_alpha,
        q_alpha,
    )

results = engine.finish(tree_ids, commit="posterior_sample")
targets = engine.export_targets(tree_ids)
```

No Python callback is allowed inside traversal, backup, repair, target construction, pruning, or export.

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
  Shape of one decision-node observation returned to Python for neural evaluation.

simulations_per_root:
  Target number of completed leaf evaluations under the current root.
  Existing reusable subtree evidence counts toward this budget after pruning.

posterior_best_samples:
  Monte Carlo samples used to estimate posterior-best policy targets.

kappa_n:
  Half-trust count for blending decision-node neural value alpha with child evidence:
      gamma = N_down / (kappa_n + N_down)

seed:
  RNG seed for Thompson sampling, posterior-best sampling, categorical sampling, and
  posterior_sample committed actions.

debug:
  Enables extra validation and debug arrays.
```

Do not include terminal concentration or smoothing fields in the search config. Terminal WDL alpha is supplied by the game adapter and must already be strictly positive.

---

## 5. Game/environment boundary

Rust must be able to do all game operations without Python callbacks:

```text
classify node kind
legal actions
step decision action
enumerate categorical outcomes
terminal WDL alpha
observation encoding
WDL perspective alignment
```

Define a Rust trait like:

```rust
pub trait Game: Send + Sync + 'static {
    type State: Clone + Send + Sync + 'static;

    fn action_size(&self) -> usize;

    fn node_kind(&self, state: &Self::State) -> GameNodeKind<Self::State>;

    fn legal_mask(&self, state: &Self::State, out: &mut [bool]);

    fn step_action(
        &self,
        state: &Self::State,
        action: usize,
    ) -> GameTransition<Self::State>;

    fn encode_observation(&self, state: &Self::State, out: &mut [f32]);

    fn align_wdl(
        &self,
        from_state: &Self::State,
        to_state: &Self::State,
        alpha: [f32; 3],
    ) -> [f32; 3];
}
```

Supporting types:

```rust
pub enum GameNodeKind<S> {
    Decision,
    Categorical(Vec<CategoricalOutcome<S>>),
    Terminal { alpha: [f32; 3] },
}

pub enum GameTransition<S> {
    Decision(S),
    Categorical(Vec<CategoricalOutcome<S>>),
    Terminal { state: S, alpha: [f32; 3] },
}

pub struct CategoricalOutcome<S> {
    pub outcome_id: u32,
    pub probability: f32,
    pub state: S,
}
```

Rules:

```text
1. Decision nodes have legal actions and neural network outputs.

2. Categorical nodes have explicit outcomes with probabilities.
   They do not have policy logits, Q alpha, legal action masks, or policy targets.

3. Terminal nodes have a positive terminal Dirichlet alpha supplied by the game adapter.
   The engine must validate alpha > 0 in debug mode.

4. Categorical outcome probabilities must be nonnegative and sum to 1 within tolerance.

5. For ordinary alternating-turn zero-sum games, align_wdl usually flips:
      [L, D, W] -> [W, D, L]
   when moving between players.
   For chance nodes that do not change the player-to-move perspective, align_wdl may be identity.
```

For deterministic two-player board games, most states are `Decision` or `Terminal`. `Categorical` is used for dice, card draws, stochastic environment transitions, simultaneous hidden resolutions, or other chance-like events.

---

## 6. Python API

### 6.1 Add roots

Expose:

```python
tree_ids = engine.add_roots(root_states)
```

`root_states` is a batch of serialized states expected by the configured Rust game adapter.

`add_roots` creates roots but does not run the neural network. If a root is a decision node, its first neural evaluation must be requested through `request_evaluations`.

If a root is terminal, it is immediately known and cannot be finished into an action.

If a root is categorical, Rust may advance/evaluate its categorical outcome children, but `finish` is only valid once the current root is a decision node.

---

### 6.2 Request evaluations

Expose:

```python
batch = engine.request_evaluations(
    max_batch_size: int,
) -> EvalBatch
```

Returned object:

```python
@dataclass
class EvalBatch:
    token: int
    size: int
    observations: np.ndarray      # [size, *observation_shape], float32
    legal_masks: np.ndarray       # [size, action_size], bool

    # Debug only:
    tree_ids: np.ndarray          # [size], uint64
    node_ids: np.ndarray          # [size], uint32
    request_ids: np.ndarray       # [size], uint64
    tree_generations: np.ndarray  # [size], uint32
```

Rules:

```text
1. Each returned row is one unevaluated decision node.

2. A tree may have at most one pending neural request in the MVP.

3. request_evaluations should iterate trees round-robin for fairness.

4. request_evaluations may complete terminal simulations without returning rows.

5. request_evaluations returns when:
   - max_batch_size requests are collected, or
   - no eligible tree can currently produce a request, or
   - all selected trees are done.

6. The token is opaque and must be passed to submit_evaluations.

7. Python must not see paths, edge IDs, child IDs, or mutable tree objects.
```

Because each tree is single-threaded and has at most one pending request, duplicate leaf requests are impossible in the MVP.

---

### 6.3 Submit evaluations

Expose:

```python
engine.submit_evaluations(
    token: int,
    policy_logits: np.ndarray,    # [B, A], float32
    value_alpha: np.ndarray,      # [B, 3], float32, strictly positive
    q_alpha: np.ndarray,          # [B, A, 3], float32, strictly positive
)
```

`B` must equal the `size` from the matching `EvalBatch`.

On submit, Rust must:

```text
1. Look up request records by token.
2. Validate tensor shapes.
3. Remove token from request table so it cannot be submitted twice.
4. Match outputs by row order.
5. Ignore stale requests whose tree generation no longer matches.
6. For root requests:
   - store policy logits, value alpha, and Q alpha;
   - mark decision node expanded;
   - initialize C^V to value_alpha;
   - do not count this as a completed simulation.
7. For non-root leaf requests:
   - store policy logits, value alpha, and Q alpha;
   - mark decision node expanded;
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
root is expanded
root_completed_count >= simulations_per_root
tree has no pending neural request
```

Where:

```text
root_completed_count = root.n_down
```

for a clean current root.

Root neural evaluation does not count as a completed simulation.

If the root is categorical, the tree is not finishable into an action. It may still be searchable so that categorical outcome children can be evaluated or cached.

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
1. Require no pending neural request.
2. Require current root kind == Decision.
3. Require root expanded.
4. Require root_completed_count >= simulations_per_root,
   unless a future partial_finish mode is added.
5. Compute EdgePosterior(root, a) for every legal action.
6. Estimate pi_search using posterior_best_samples.
7. Select committed action according to commit mode.
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
node is expanded
state is non-terminal
has_child_evidence == true
state cache C^V is available
```

Exclude:

```text
terminal nodes
categorical nodes
pending neural leaves
newly expanded decision leaves with no child evidence
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

Categorical nodes do not produce policy or Q rows. If a decision action leads to a categorical child, the decision action’s Q target is the posterior of that action edge, which may summarize the categorical child’s clean `C^V`.

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

3. If the root already has a child for that action:
     promote that child to be the new root.

4. If the root does not have a child for that action:
     use Game::step_action to create the new root state.

5. Preserve all nodes, edge posteriors, network outputs, and caches in the promoted child subtree.

6. Drop or detach old ancestors and unreachable sibling subtrees unless the caller explicitly exports
   their targets before pruning.

7. Increment tree_generation so any stale pending request from the old root can be ignored.

8. Recompute root depth and parent pointers as needed.
```

If the resulting new root is a categorical node and the real environment immediately reveals the categorical outcome, expose:

```python
engine.advance_categorical_roots(
    tree_ids: np.ndarray,          # [G], uint64
    outcome_ids: np.ndarray,       # [G], uint32
)
```

Rules:

```text
1. Current root must be a categorical node.

2. outcome_id must match one of the root’s categorical outcomes.

3. If the outcome child exists:
     promote it to root.

4. If the outcome child does not exist:
     create it from the stored categorical outcome state.

5. Preserve the selected outcome subtree.

6. Drop or detach other outcome branches.

7. Increment tree_generation.
```

After pruning, the new root may already have child evidence. The search budget for the new root should count reused evidence:

```text
root_completed_count = root.n_down
remaining = max(0, simulations_per_root - root_completed_count)
```

This is the key reuse behavior: already visited nodes under the played move are not recalculated.

For the MVP, either require no pending request before pruning, or cancel the pending request by incrementing `tree_generation` and clearing `tree.pending_request`. If the old GPU result later arrives, `submit_evaluations` must ignore it as stale.

---

## 7. Internal data model

Use integer IDs, not pointer graphs.

```rust
type TreeId = u64;
type NodeId = u32;
type RequestId = u64;
type BatchToken = u64;
type Action = u16;
type OutcomeId = u32;
```

### 7.1 Forest

```rust
struct Forest<G: Game> {
    config: SearchConfig,
    game: G,

    trees: Vec<Tree<G::State>>,
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
struct Tree<S> {
    id: TreeId,
    generation: u32,

    nodes: Vec<Node<S>>,
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
struct Node<S> {
    id: NodeId,
    generation: u32,

    parent: Option<NodeId>,
    parent_link: Option<ParentLink>,
    depth: u32,

    state: S,
    kind: NodeKind,

    c_v: Option<[f32; 3]>,
    n_down: u32,
    cache_version: u32,
}
```

```rust
enum NodeKind {
    Decision(DecisionData),
    Categorical(CategoricalData),
    Terminal(TerminalData),
}
```

```rust
enum ParentLink {
    DecisionAction { action: Action },
    CategoricalOutcome { outcome_id: OutcomeId },
}
```

---

### 7.4 DecisionData

```rust
struct DecisionData {
    eval_status: DecisionEvalStatus,

    policy_logits: Vec<f32>,       // [A]
    value_alpha: [f32; 3],
    q_alpha: Vec<[f32; 3]>,        // [A, 3]

    legal_mask: Vec<bool>,         // [A]
    edges: Vec<DecisionEdge>,      // [A]
}
```

```rust
enum DecisionEvalStatus {
    Unexpanded,
    PendingEval { request_id: RequestId },
    Expanded,
}
```

A decision node is the only node kind that can be sent to the neural network.

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
  Number of completed leaf/categorical-child summaries that have contributed to this edge.

child_cache_version:
  Version of child C^V that was last used to refresh this edge.
```

`b` is not accumulated component-wise. It is a full snapshot replacement.

---

### 7.6 CategoricalData

```rust
struct CategoricalData {
    outcomes: Vec<CategoricalEdge>,
    complete: bool,
}
```

```rust
struct CategoricalEdge {
    outcome_id: OutcomeId,
    probability: f32,
    child: NodeId,

    completed: bool,
    b: [f32; 3],
    r_count: u32,

    child_cache_version: Option<u32>,
}
```

Categorical nodes are native. They do not use policy logits, legal masks, or Q alpha.

For the MVP, categorical outcome sets should be small enough to enumerate. Later versions may add sampled categorical approximation for large chance spaces.

---

### 7.7 TerminalData

```rust
struct TerminalData {
    alpha: [f32; 3],
}
```

The alpha must be strictly positive.

No `terminal_epsilon` or `terminal_kappa` exists in the engine config.

---

### 7.8 RequestRecord

```rust
struct RequestRecord {
    request_id: RequestId,
    tree_id: TreeId,
    tree_generation: u32,

    node_id: NodeId,
    node_generation: u32,

    path: Vec<PathStep>,

    is_root_request: bool,
}
```

```rust
enum PathStep {
    DecisionAction {
        node_id: NodeId,
        action: Action,
    },
    CategoricalOutcome {
        node_id: NodeId,
        outcome_id: OutcomeId,
    },
}
```

A batch token maps to:

```rust
Vec<RequestRecord>
```

For the MVP, each tree has at most one pending request. The request table still exists because Python submits batched results by token.

---

## 8. Posterior semantics

### 8.1 Decision EdgeBase

For a decision node `v` and action `a`:

```text
DecisionEdgeBase(v, a):
  if edge(v,a) has child u and u is summarizable:
      return align_wdl(from_state=s_u, to_state=s_v, alpha=C^V_u)
  else if edge(v,a) has expanded decision child u:
      return align_wdl(from_state=s_u, to_state=s_v, alpha=value_alpha[u])
  else:
      return q_alpha[v,a]
```

A child is summarizable when:

```text
Terminal:
  always summarizable using terminal alpha.

Categorical:
  summarizable when its categorical cache C^V is complete.

Decision:
  summarizable when it has child evidence and clean C^V.
```

A newly expanded decision leaf with no child evidence is not summarizable. Its value alpha is used by the direct leaf backup, but it should not automatically overwrite the parent edge through child-cache refresh.

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

### 8.3 Categorical EdgePosterior

For a categorical node `c` and outcome `i`:

```text
CategoricalEdgePosterior(c, i):
  if edge(c,i).completed:
      return edge(c,i).b
  else if child u is summarizable:
      return align_wdl(from_state=s_u, to_state=s_c, alpha=C^V_u)
  else:
      unavailable
```

A categorical node is complete only when every nonzero-probability outcome has an available posterior.

---

### 8.4 Decision ThompsonSelect

For an expanded decision node:

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

### 8.5 Categorical outcome selection

Categorical nodes are not selected by Thompson sampling.

For the MVP:

```text
CategoricalSelect(c):
  if any nonzero-probability outcome edge lacks an available posterior:
      choose one missing outcome to complete
      preferred order: descending probability, then stable outcome_id
  else:
      sample outcome according to categorical probabilities
```

This has two desirable properties:

```text
1. Small categorical nodes become exactly summarizable because all outcomes are eventually evaluated.
2. After all outcomes are known, additional simulations follow the environment probabilities.
```

Later versions may add sampled categorical approximations for large outcome spaces.

---

### 8.6 PosteriorBestPolicyTarget

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

Categorical nodes do not have posterior-best policy targets.

---

## 9. State-posterior cache semantics

### 9.1 Decision node cache

For an expanded decision node with child evidence:

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

If an expanded decision node has no child evidence:

```text
C^V_v = value_alpha[v]
N_down = 0
```

Such a node is not exported as a training target row because it has no child evidence.

---

### 9.2 Categorical node cache

For a complete categorical node:

```text
C^V_c =
  sum over outcomes i:
      probability[i] * CategoricalEdgePosterior(c, i)

N_down =
  sum over outcomes i:
      edge(c,i).r_count
```

No posterior-best target is computed.

No `kappa_n` blend is applied at categorical nodes. A categorical node is an environment expectation, not a policy-improvement node.

If any nonzero-probability outcome lacks an available posterior:

```text
C^V_c = None
complete = false
```

---

### 9.3 Terminal node cache

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
fn next_request(tree: &mut Tree, game: &G, config: &SearchConfig)
    -> NextRequestResult;
```

```rust
enum NextRequestResult {
    NeuralRequest(RequestRecord),
    CompletedOneSimulation,
    BlockedByPendingRequest,
    TreeDone,
    NoProgress,
}
```

Core traversal:

```text
next_request(tree):
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

      Categorical:
        outcome = CategoricalSelect(v)
        path.push((v, outcome_id))
        v = outcome child
        continue

      Decision:
        if decision eval_status == Unexpanded:
            create request for v with current path
            mark v PendingEval
            tree.pending_request = request_id
            return NeuralRequest(record)

        if decision eval_status == PendingEval:
            return BlockedByPendingRequest

        if decision eval_status == Expanded:
            action = ThompsonSelect(v)
            child = get_or_create_decision_action_child(v, action)
            path.push((v, action))
            v = child
            continue
```

For root decision evaluation:

```text
path is empty
is_root_request = true
submit_evaluations initializes the root but does not count a simulation
```

For non-root decision evaluation:

```text
path is non-empty
submit_evaluations backs up value_alpha along path
counts one completed simulation
```

---

## 11. Child creation

### 11.1 Decision action child

```text
get_or_create_decision_action_child(v, a):
  if edge(v,a).child exists:
      return child

  transition = game.step_action(state[v], a)

  child = create_node_from_transition(transition)
  edge(v,a).child = child
  return child
```

No CAS is needed in the MVP because one tree is not concurrently modified.

---

### 11.2 Categorical outcome children

When creating a categorical node, create all outcome edges immediately:

```text
create_categorical_node(outcomes):
  for each outcome:
      child = create_node_from_state_or_terminal(outcome.state)
      edge = CategoricalEdge {
          outcome_id,
          probability,
          child,
          completed=false,
          r_count=0,
      }
```

This makes categorical nodes explicit and pruneable by `outcome_id`.

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

2. If the parent is a categorical node:
     - if all outcome posteriors are available, compute C^V as probability-weighted mixture;
     - otherwise leave C^V unavailable.

3. If the parent is a decision node:
     - if expanded and has child evidence, compute posterior-best policy target and C^V;
     - if expanded and has no child evidence, C^V = value_alpha;
     - if unexpanded or pending, C^V unavailable except for direct leaf backup use.

4. Increment cache_version whenever C^V changes.

5. Continue upward only when the recomputed node is summarizable.
```

A decision node is summarizable by its parent only if:

```text
expanded == true
has_child_evidence == true
C^V is available
```

This avoids overwriting a parent edge from a newly expanded neural leaf with no child evidence.

A categorical node is summarizable by its parent when:

```text
complete == true
C^V is available
```

A terminal node is always summarizable.

---

## 14. Finish semantics

For each selected tree:

```text
finish(tree):
  require no pending request
  require root kind == Decision
  require root expanded
  require root_completed_count >= simulations_per_root

  root_alpha[a] = DecisionEdgePosterior(root, a) for legal actions
  root_alpha[a] = [1,1,1] for illegal actions

  root_q_mean[a] =
      (root_alpha[a][W] - root_alpha[a][L]) / sum(root_alpha[a])

  pi_search = PosteriorBestPolicyTarget(root)

  if commit == posterior_sample:
      action ~ Categorical(pi_search)

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
eval_status == Expanded
has_child_evidence == true
C^V is available
node is still in retained current-root subtree
```

Do not export:

```text
categorical nodes
terminal nodes
unexpanded decision nodes
pending decision nodes
expanded decision leaves with no child evidence
detached/pruned-away nodes
```

For each exported decision node:

```text
observation = game.encode_observation(node.state)

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

Categorical evidence influences exported decision targets only through decision edges that lead into categorical subtrees.

---

## 16. Root pruning and reuse

### 16.1 Advance by decision action

```text
advance_roots(tree_ids, actions):
  for each tree:
      root = current root

      require root kind == Decision
      require action legal

      if root edge(action) has child:
          new_root = child
      else:
          new_root = create child from game.step_action(root.state, action)

      detach old parent/siblings
      set tree.root = new_root
      set new_root.parent = None
      set new_root.depth = 0
      increment tree.generation
      clear/cancel pending request if any
```

After pruning:

```text
If new_root was already expanded:
  reuse its policy logits, value alpha, Q alpha, edges, categorical children, and caches.

If new_root was unexpanded:
  next request_evaluations will request its neural evaluation if it is a decision node.

If new_root is categorical:
  request_evaluations may evaluate its outcome subtrees, or the caller may provide an observed
  outcome via advance_categorical_roots.

If new_root is terminal:
  no further action is available.
```

### 16.2 Advance by categorical outcome

```text
advance_categorical_roots(tree_ids, outcome_ids):
  for each tree:
      root = current root

      require root kind == Categorical
      find outcome edge by outcome_id

      if outcome child exists:
          new_root = child
      else:
          create child from stored outcome state

      detach old categorical siblings
      set tree.root = new_root
      set new_root.parent = None
      set new_root.depth = 0
      increment tree.generation
      clear/cancel pending request if any
```

### 16.3 Search budget after pruning

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

`request_evaluations(max_batch_size)` should be implemented as a round-robin forest scheduler:

```text
request_evaluations:
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

          result = next_request(tree)

          match result:
              NeuralRequest(record):
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
      return EvalBatch(size=0)

  token = next_batch_token
  request_table[token] = records
  materialize observations and legal masks
  return EvalBatch
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
wrong batch token
submitting same token twice
output shape mismatch
alpha <= 0 in debug mode
invalid legal action in advance_roots
invalid outcome id in advance_categorical_roots
finish called on non-decision root
finish called with pending request
finish called before simulations_per_root reached
export called with pending request
unsupported commit mode
categorical probabilities invalid
terminal alpha invalid
```

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
newly expanded decision leaf with no child evidence does not overwrite parent edge via refresh
terminal alpha must be strictly positive
align_wdl flip is correct
```

### 20.2 Categorical node tests

Construct a toy game with a chance node:

```text
Decision action -> Categorical node with two outcomes
Outcome 0 probability 0.25
Outcome 1 probability 0.75
```

Assert:

```text
categorical node has no policy logits
categorical node has no Q alpha
categorical node is not exported
categorical outcome probabilities are stored
missing categorical outcomes are evaluated before node becomes complete
categorical C^V equals probability-weighted mixture of outcome posteriors
decision edge leading to categorical child uses categorical C^V once complete
```

### 20.3 MVP no-duplicate tests

Because each tree is single-threaded and has one pending request, assert:

```text
one tree cannot emit a second request while pending_request is set
request_evaluations returns at most one pending row per tree
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
categorical C^V recomputes synchronously after all outcomes are available
```

### 20.5 Pruning/reuse tests

Construct a tree where root action `a` has an expanded child with evidence.

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
next request_evaluations does not re-request already expanded root
```

For categorical pruning:

```text
root -> categorical outcome_id -> child
advance_categorical_roots promotes the selected child
unselected outcome branches are detached or dropped
```

### 20.6 Export tests

Assert exported rows satisfy:

```text
kind == Decision
expanded
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
categorical nodes are not exported
terminal nodes are not exported
all exported target alphas are positive
```

### 20.7 Python integration test

Use a mock Python model:

```python
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
tree_ids = engine.add_roots(root_states)

while not engine.is_done(tree_ids):
    batch = engine.request_evaluations(max_batch_size=128)

    if batch.size:
        outputs = model(batch.observations, batch.legal_masks)
        engine.submit_evaluations(batch.token, *outputs)

results = engine.finish(tree_ids)
targets = engine.export_targets(tree_ids)
```

Assert:

```text
no panics
no leaks
valid shapes
valid actions
all target alphas positive
```

---

## 21. Acceptance criteria

The MVP is acceptable when:

```text
1. Python imports dqaz successfully.

2. SearchEngine owns all tree memory in Rust.

3. Python never receives node pointers, paths, locks, or mutable tree objects.

4. request_evaluations returns batched decision-node neural requests.

5. Each individual tree search is single-threaded.

6. Each tree has at most one pending neural request.

7. Duplicate leaf requests are structurally impossible in the MVP.

8. No virtual loss or in-flight posterior mass is used.

9. No terminal_epsilon, terminal_kappa, leaf_value_mode, kappa_leaf, max_attempts,
   max_inflight_per_tree, or num_threads config field exists.

10. Decision nodes use Thompson selection over Dirichlet WDL edge posteriors.

11. Categorical nodes are represented natively and are not encoded as fake legal actions.

12. Categorical nodes are not exported as policy/Q training rows.

13. Terminal nodes use positive terminal alpha supplied by the game adapter.

14. submit_evaluations backs up neural value_alpha correctly.

15. Edge posterior B is a full snapshot replacement, not component-wise accumulated B.

16. finish returns posterior-best root policy and committed actions for decision roots.

17. export_targets returns flattened clean decision-node training rows.

18. advance_roots can prune to an existing child subtree after a played move.

19. advance_categorical_roots can prune to an existing categorical outcome child.

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

    game.rs
    forest.rs
    tree.rs
    node.rs
    edge.rs

    request.rs
    scheduler.rs
    traversal.rs
    backup.rs
    posterior.rs
    targets.rs
    prune.rs
    rng.rs

    games/
      mod.rs
      connect_four.rs
      toy_categorical.rs

  tests/
    test_python_api.py
    test_mock_search.py
    test_prune_reuse.py
    test_categorical_nodes.py

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
tree_ids = engine.add_roots(root_states)

while not engine.is_done(tree_ids):
    batch = engine.request_evaluations(max_batch_size=128)

    if batch.size == 0:
        continue

    policy_logits, value_alpha, q_alpha = model_apply(
        batch.observations,
        batch.legal_masks,
    )

    engine.submit_evaluations(
        batch.token,
        policy_logits,
        value_alpha,
        q_alpha,
    )

results = engine.finish(tree_ids, commit="posterior_sample")

# Play the actual moves in the environment.
actions = results.actions
next_env_states = env_step(root_states, actions)

# Reuse Rust tree subtrees instead of recalculating them.
engine.advance_roots(tree_ids, actions)

# Continue search from the pruned roots.
```
