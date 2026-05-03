# Plan: Dirichlet-Q AlphaZero — Staged Path

## Context

`/workspace/Scacchi/math.md` specifies a Dirichlet-Q AlphaZero variant where the network outputs WDL Dirichlet posteriors (`α^V` per state, `α^Q` per state-action) and the policy target is the *posterior probability that each action is optimal* (Monte-Carlo Thompson over per-action Dirichlets, optionally search-improved).

The pre-Dirichlet baseline (`scacchi/play.py`, `scacchi/loss.py`, `scacchi/network.py`) ran vanilla AlphaZero on `mctx.gumbel_muzero_policy` with a scalar value head and visit-count-based policy targets. Stage B adapts that path without changing mctx internals.

Full fidelity to math.md cannot be expressed inside stock mctx because root action selection is not Thompson sampling and the live root posterior is not updated as `α_a <- α_a + c y` during search. Stock mctx can still be used for a useful intermediate path: carry WDL/evidence metadata through embeddings, reconstruct post-search Dirichlet evidence, and train the same posterior/loss targets. We adopt a staged path where loss completion and Thompson root selection are deliberately separate.

- **Stage B** (this plan's first deliverable): Dirichlet heads + losses + Bayesian wrapper for the policy target. No mctx changes.
- **Stage A** (follow-up): Linear evidence aggregation through `tree.embeddings`, still using `mctx.gumbel_muzero_policy`. This changes the evidence/search trajectory distribution relative to math.md's Thompson search, but keeps the posterior-best target and loss forms compatible.
- **Stage Full-Losses** (follow-up): Add the full search mean losses and Dirichlet-KL losses from math.md while keeping Gumbel MuZero action selection.
- **Stage Full-Selection** (final): Replace Gumbel root selection with Thompson posterior sampling. This changes which actions/nodes are sampled and therefore the self-play data distribution; it should not require another loss rewrite.

---

## Stage B — Dirichlet heads + Bayesian policy wrapper (~80 LoC)

### B.1 Network outputs (`scacchi/network.py`)

Replace `AZNet.__call__` return signature `(logits, value)` → `(policy_logits [B,A], alpha_V [B,3], alpha_Q [B,A,3])`.

Parameterize each Dirichlet head with the math.md §3.4 stable form:
- `α = α_base + softplus(c_head) * softmax(r_head)` with `α_base = 1`.
- `alpha_V`: two new linear projections off the value-tower features → `r_V [B,3]`, `c_V [B]`.
- `alpha_Q`: two projections off a fresh action-conv tower (mirror existing `policy_conv` shape) → `r_Q [B,A,3]`, `c_Q [B,A]`.
- Drop the `tanh`-scalar value head entirely.

### B.2 Bridge Dirichlet → mctx (`scacchi/play.py`)

mctx still consumes scalar values in stage B. In `make_recurrent_fn` and the root construction in `make_selfplay`:
- Compute `value = U(mean(alpha_V)) = (alpha_V[...,W] − alpha_V[...,L]) / sum(alpha_V, -1)`.
- Pass `policy_logits` as `prior_logits` (legal-mask same as today, `play.py:34`).
- Keep mctx's existing `discount = -1` perspective flip (`play.py:47`); correct for scalar U because `U(flip(φ)) = −U(φ)`.

### B.3 Stuff `alpha_V_mean` into the embedding

`tree.embeddings` is an arbitrary pytree. Replace the bare `pgx.State` with:

```python
class NodeEmbedding(NamedTuple):
    state: pgx.State
    alpha_V_mean: jax.Array  # [B, 3] WDL mean from this node's α^V
```

In `make_recurrent_fn`, after `env.step`, run the network on the new observation, derive `alpha_V_mean = alpha_V / alpha_V.sum(-1, keepdims=True)`, and store it on the new node. After search, gather depth-1 children via `tree.children_index[:, ROOT_INDEX, :]` and look up `tree.embeddings.alpha_V_mean` at those indices to get one state-value WDL snapshot for each visited root action.

**Perspective flip on gather:** non-terminal child nodes are evaluated from the opponent's perspective. Apply `y_a = y_a[..., ::-1]` (swap W↔L, D fixed) before using as the root-action target.

**Terminal override:** if a visited root child is terminal (`children_discounts[:, ROOT_INDEX, a] == 0`), use the actual immediate reward as a one-hot WDL target from the root player's perspective: `one_hot(round(children_rewards[:, ROOT_INDEX, a]) + 1, 3)`. Do not use the network value on terminal children.

Unvisited actions keep their root prior `mean(alpha_Q[:, a])` as the fallback `y_a`, but receive zero search evidence (`c_a = 0`), so this fallback does not double-count the prior.

This is a snapshot — one WDL evaluation per child at expansion time, not search-improved. Stage A upgrades this.

### B.4 MC posterior-best policy target (§6/§7 approximation)

After `mctx.gumbel_muzero_policy` returns:

1. `N_a = search_tree.children_visits[:, ROOT_INDEX, :]` shape `[B, A]`.
2. `y_a = terminal_one_hot_or_flip_W_L(gather alpha_V_mean from root children)` shape `[B, A, 3]`.
3. `α_post[b,a] = alpha_Q[b,a] + ρ * sqrt(N_a[b,a]) * y_a[b,a]` — a deliberate Stage B approximation to math.md's linear evidence update, using completed visit counts so unvisited actions add no evidence.
4. `phi_samples = jax.random.dirichlet(key, alpha=α_post, shape=(M,))` → `[M, B, A, 3]`.
5. `U_samples = phi_samples[..., W] − phi_samples[..., L]` → `[M, B, A]`.
6. Mask invalid actions: `U_samples = jnp.where(invalid_actions[None, :, :], -jnp.inf, U_samples)`.
7. `argmax_a = U_samples.argmax(-1)` → `[M, B]`.
8. **Laplace-smooth** the histogram (keeps CE gradients well-behaved):
   `policy_target[b,a] = (count(a) + legal_mask[b,a]) / (M + num_legal[b])`

Default `M = 32`, configurable.

### B.5 Loss rewrite (`scacchi/loss.py`)

Drop the L2 scalar value loss. New Stage B losses:
- `L_pi`: cross-entropy of `policy_target` (stop-grad) vs `softmax(policy_logits)`.
- `L_V_outcome`: cross-entropy of WDL one-hot final outcome vs `mean(alpha_V)` (math.md §8.5).
- `L_Q_outcome`: cross-entropy of WDL one-hot final outcome vs `mean(alpha_Q[:, played_action])` — played action only (math.md §8.5).

Defer search mean and Dirichlet-KL terms (math.md §8.1-§8.4 and Appendix A); gate KL behind `dir_kl_weight: float = 0.0` until Stage Full-Losses enables it.

### B.6 Self-play data flow

`SelfplayOutput`:
- `obs`, `reward`, `terminated`, `discount` — unchanged.
- `policy_target [T, B, A]` — new MC posterior-best.
- `played_action [T, B]` — new (math.md §8.5 needs it).

`Sample`:
- `obs`, `policy_tgt`, `mask` — like today.
- `wdl_tgt [T, B, 3]` — `jax.nn.one_hot(value_tgt.astype(int) + 1, 3)` indexed `[L, D, W]`. The bootstrap loop in `make_compute_loss_input` (`loss.py:24`) already produces per-step values from the player-to-move perspective, so the WDL one-hot is already correct.
- `played_action [T, B]`.

### B.7 Config and weights

Add to `Config` in `scacchi/train.py` and mirror in `scacchi/configs/gardner_chess.yaml`:
```python
policy_mc_samples: int = 32
search_evidence_rho: float = 1.0
policy_loss_weight: float = 1.0
value_outcome_weight: float = 1.0
q_outcome_weight: float = 1.0
dir_kl_weight: float = 0.0          # Stage Full-Losses
evidence_schedule: str = "sqrt"     # optional fallback mode after Stage A
c_terminal: float = 8.0             # Stage A / Full-Losses
c_leaf: float = 2.0                 # Stage A / Full-Losses
```

Update `train.py`'s `dict_to_log` to log all three loss terms.

### B.8 Eval adapter (`scacchi/evaluations.py`)

The pgx baseline returns `(logits, value)` and we cannot change it.
- Replace `my_logits, _ = model(...)` → `my_logits, _, _ = model(...)` at lines 26, 68.
- For `make_mcts_evaluate`, convert `alpha_V` → scalar `U(mean(alpha_V))` to feed mctx.
- If the shared recurrent function needs a 3-tuple wrapper around a scalar baseline, synthesize `alpha_V` so that `U(mean(alpha_V))` preserves the baseline scalar value. Do not compress values to a smaller range such as `[-2/3, 2/3]`.
- The shared `make_recurrent_fn` (used at line 57) already gets the new flow.

### B.9 Files to modify (Stage B)

| File | Change |
|---|---|
| `scacchi/network.py` | Three-head output. |
| `scacchi/play.py` | `NodeEmbedding`; rewrite recurrent_fn and selfplay; MC posterior-best. |
| `scacchi/loss.py` | New `Sample`; rewrite `compute_loss_input` and `train`. |
| `scacchi/evaluations.py` | 3-tuple unpack + scalar U bridge. |
| `scacchi/train.py` | Extend `Config`; log new loss terms. |
| `scacchi/configs/gardner_chess.yaml` | Mirror config. |

### B.10 Reused functions (no rewrite)

- `mctx.gumbel_muzero_policy`, `mctx.qtransform_completed_by_mix_value` — unchanged callers.
- `pgx.experimental.auto_reset` — unchanged.
- `jax.random.dirichlet`, `jax.nn.one_hot`, `jax.nn.softplus`, `jax.nn.softmax`.
- `optax.softmax_cross_entropy` for `L_pi`.
- `logger.log` (`scacchi/logger.py:94`) — already accepts arbitrary scalar dicts.

---

## Stage A — Linear evidence summing via embedding aggregation (~80 LoC)

**Trigger:** ship if Stage B's snapshot `y_a` (depth-1 children only, no deeper search aggregation) proves to be the bottleneck (signal: flat policy loss with sane V/Q losses).

**Math change.** Stage B uses the sublinear approximation `α_post = α_prior + ρ √N · ȳ`. Stage A switches to the linear Dirichlet evidence update from math.md §4/§6:

$$
\alpha_a^{\mathrm{post}} =
\alpha_\theta^Q(s,a) +
\sum_{n \in \mathrm{subtree}(a)} c_n \cdot y_n^{\mathrm{aligned}}.
$$

This is the literal Bayesian update for a Dirichlet under independent categorical evidence: prior alphas plus a sum of `c · y` contributions. Per-leaf evidence weight `c` is `c_terminal` for terminal nodes and `c_leaf` for non-terminal — so this stage absorbs what was previously F.2 (terminal-vs-leaf weighting). `y_n^{\mathrm{aligned}}` is the node's WDL distribution, W↔L-flipped if the node sits at odd depth from root (opponent perspective).

- mctx's scalar `<U>` running mean cannot recover `Σ y_D` (one linear functional + simplex constraint = 2 equations, 3 unknowns — a leaf at `(0.5, 0, 0.5)` and one at `(0, 1, 0)` are indistinguishable to mctx). So we *cannot* reconstruct the per-action WDL sum from mctx's tree alone.
- *But* the embedding is an arbitrary pytree mctx never inspects. We can carry the WDL info there and reconstruct `Σ c · y` by post-hoc scatter-sum after search.
- This works only if root selection does **not** need a live per-action posterior during search — i.e., we drop Thompson root sampling. mctx's `gumbel_muzero_policy` (sequential halving over Gumbel-perturbed priors) replaces it. This is the one algorithmic concession; in exchange we delete ~600 LoC of vendored code.

### A.1 Embedding becomes the side channel

`NodeEmbedding` (Stage B carried `state` + `alpha_V_mean`) gains the metadata needed for post-hoc aggregation:

```python
class NodeEmbedding(NamedTuple):
    state: pgx.State
    y: jax.Array            # [B, 3] WDL distribution at this node
                            #   terminal: one-hot reward from this node's perspective
                            #   non-terminal: mean(α^V) from the network
    c: jax.Array            # [B] evidence weight: c_terminal if terminal else c_leaf
    root_action: jax.Array  # [B] which root action's subtree this node belongs to
                            #   (NO_PARENT = -1 at root; set on the depth-1 transition)
    depth_parity: jax.Array # [B] 0 = root player's perspective, 1 = opponent's
```

`make_recurrent_fn` populates these on every expansion. The invariant is that
`y` is always stored from the expanded node's local perspective; the
post-search scatter handles root-player alignment uniformly via `depth_parity`.

- `root_action = parent.root_action if parent.root_action != NO_PARENT else action_taken`
- `depth_parity = 1 - parent.depth_parity`
- non-terminal `y = mean(α^V)` from the network on the new state.
- terminal `y` is **not** the Stage B root/parent-aligned terminal override. `recurrent_fn` observes `reward = rewards[..., current_player]`, which is from the parent/action-taker perspective. Convert that to a one-hot WDL, then W↔L-flip before storing it on the child node:

```python
terminal_y_parent = jax.nn.one_hot(jnp.round(reward).astype(jnp.int32) + 1, 3)
terminal_y_child = terminal_y_parent[..., ::-1]
y = jnp.where(terminated[..., None], terminal_y_child, alpha_V_mean)
```

- `c = c_terminal` for terminal nodes, else `c_leaf`.

`recurrent_fn` still returns scalar `value = U(y) = y[W] − y[L]` to mctx. The existing `discount = -1` perspective trick (`play.py:47`) handles scalar perspective flips for selection — `U(flip(y)) = −U(y)`. **mctx's tree is unchanged.**

### A.2 Post-search scatter-sum

After `mctx.gumbel_muzero_policy(...)` returns:

```python
# Pull metadata from the tree's embedding pytree.
y            = search_tree.embeddings.y             # [B, N, 3]
c            = search_tree.embeddings.c             # [B, N]
root_action  = search_tree.embeddings.root_action   # [B, N]
depth_parity = search_tree.embeddings.depth_parity  # [B, N]
node_visits  = search_tree.node_visits              # [B, N]

# Align every node to root-player perspective.
y_aligned = jnp.where(depth_parity[..., None] == 1, y[..., ::-1], y)

# Mask out (a) the root itself (no root_action) and (b) any unexpanded slot.
valid = (root_action != NO_PARENT) & (node_visits > 0)
weight = jnp.where(valid, c, 0.0)

# Scatter-add per (batch, root_action) into [B, A, 3].
B, N = node_visits.shape
A = alpha_Q_prior.shape[1]
batch_idx = jnp.broadcast_to(jnp.arange(B)[:, None], (B, N))
safe_root_action = jnp.where(valid, root_action, 0)  # any in-range index, masked by weight=0
evidence_sum = jnp.zeros((B, A, 3), dtype=alpha_Q_prior.dtype)
evidence_sum = evidence_sum.at[batch_idx, safe_root_action].add(
    weight[..., None] * y_aligned)

alpha_Q_post = alpha_Q_prior + evidence_sum
```

Stage B's `_mc_posterior_best(alpha_Q_post, ...)` consumes this unchanged.

### A.3 What disappears from the previous Stage A draft

| Previously planned | Status |
|---|---|
| `scacchi/dirichlet_mctx.py` (vendored Tree, expand, backward, search) | **Deleted.** Stock mctx is sufficient. |
| `thompson_root_action_selection` | **Deleted.** Root selection is `gumbel_muzero_policy`'s sequential halving. |
| `wdl_qtransform` | **Deleted.** mctx's `qtransform_completed_by_mix_value` operates on scalar U directly. |
| WDL-shaped `node_values: [B,N,3]` etc. | **Deleted.** Tree carries scalar values; WDL lives in embeddings. |
| `mctx>=0.0.6,<0.0.7` pin to access `_src` | **Not needed.** All mctx use is via the public API. |

### A.4 What changes in the Stage B code

In `play.py`:
- Extend `NodeEmbedding` with the four new fields above; rename `alpha_V_mean → y` for clarity (semantic: the WDL distribution at this node, which is `mean(α^V)` for non-terminals).
- `make_recurrent_fn` populates `c`, `root_action`, `depth_parity` (purely arithmetic — no extra network calls).
- After `mctx.gumbel_muzero_policy(...)`, replace Stage B's depth-1 gather with the scatter-sum in §A.2.
- Drop Stage B's `search_evidence_rho` and `evidence_schedule` config knobs from the policy-target path. Add `c_terminal` and `c_leaf`.

In `loss.py`, `train.py`, `evaluations.py`: no changes from Stage B (config keys aside).

### A.5 Caveat: linear evidence vs. correlated neural evals

math.md §4 explicitly frames neural search evidence as calibrated pseudo-evidence rather than independent categorical observations. Linear summing of bootstrapped network evaluations can become overconfident because the evidence is correlated. At high simulation counts `Σ c · y_i ≈ N · c_leaf · ȳ`, which can drown the prior.

Mitigations:
- Set `c_leaf` small enough that `N_root_action · c_leaf` stays comparable to `α_Q_prior.sum(-1)` at the simulation budget you actually use. Reasonable starting points: `c_leaf = 1.0`, `c_terminal = 8.0`.
- Watch `α_Q_post.sum(-1)` during early training; if it climbs past ~`100 × prior_concentration`, lower `c_leaf`.
- Optional fallback: gate behind a config flag `evidence_mode: "linear" | "sqrt"` so we can revert to Stage B's sublinear schedule (computed from the same scatter-summed `evidence_sum` — divide by per-action weight totals to recover `ȳ`, then apply `ρ √N · ȳ`). Cheap insurance.

### A.6 Verification (Stage A)

1. **Shape sanity:** `evidence_sum.shape == (B, A, 3)`; no NaNs; `(α_Q_post / α_Q_post.sum(-1, keepdims=True)).sum(-1) ≈ 1`.
2. **Scatter correctness:** 1-batch, 4-simulation toy run. Manually trace which expanded nodes belong to which root action; hand-verify `evidence_sum[0, a, :]` matches `Σ c_n · y_n^aligned` over those nodes.
3. **Parity flip:** terminal child reachable in 1 step — confirm the stored child `y` is W↔L-flipped from the parent reward, then `depth_parity=1` flips it back during scatter so the final root-aligned evidence equals the root/player-to-move outcome.
4. **Equivalence to Stage B at depth 1:** with `num_simulations = 1` and only one root action visited, `evidence_sum[:, a, :]` should equal Stage B's `c · flip(y_child)` for that action — i.e., Stage A degenerates cleanly to Stage B for trivial trees.

---

## Stage Full-Losses — Complete loss suite + Dirichlet KL (~120 LoC)

**Goal:** add the remaining math.md losses while leaving Stage A's `mctx.gumbel_muzero_policy` action selection unchanged. This phase changes training signals, not search/action-selection mechanics.

### FL.1 Search targets to carry forward

Stage A should make these targets available from self-play:

```python
q_evidence_sum: [T, B, A, 3]  # Σ c_n · y_n^aligned per root action
q_evidence_w:   [T, B, A]     # q_evidence_sum.sum(-1)
q_search_mask:  [T, B, A]     # legal and q_evidence_w > 0
```

For searched actions:

```python
q_search_mean = q_evidence_sum / jnp.maximum(q_evidence_w[..., None], eps)
beta_Q = alpha_base + q_evidence_sum
```

Use `beta_Q = alpha_base + q_evidence_sum`, not `alpha_Q_prior + q_evidence_sum`. The latter is the root posterior used for policy improvement; the loss target should describe the evidence search produced.

For the state-value head, use a clearly defined root-state target. A practical first target is the posterior-best-policy mixture over searched root-action evidence:

```python
v_evidence_sum = (stopgrad(policy_target)[..., None] * q_evidence_sum).sum(axis=-2)
v_evidence_w = v_evidence_sum.sum(-1)
v_search_mean = v_evidence_sum / jnp.maximum(v_evidence_w[..., None], eps)
beta_V = alpha_base + v_evidence_sum
```

If `v_evidence_w == 0`, skip the search value loss for that row and keep the grounded outcome loss.

### FL.2 Mean losses

Add the search mean losses from math.md §8.3/§8.4:

- `L_V_mean`: cross-entropy of `v_search_mean` vs `mean(alpha_V)`, masked by `v_evidence_w > 0`.
- `L_Q_mean`: weighted cross-entropy of `q_search_mean` vs `mean(alpha_Q[:, a])`, over `q_search_mask`.

Keep the Stage B grounded outcome losses (`L_V_outcome`, `L_Q_outcome`) as separate terms. They train against final game outcomes and are not a substitute for the search mean losses.

### FL.3 Dirichlet-KL losses

Implement Appendix A KL with `jax.scipy.special.gammaln` and `jax.lax.digamma`:

```python
L_V_Dir = KL(Dir(stopgrad(beta_V)) || Dir(alpha_V))
L_Q_Dir = KL(Dir(stopgrad(beta_Q)) || Dir(alpha_Q))
```

Mask `L_Q_Dir` to searched/legal actions only. Start with `dir_kl_weight = 0.0`, confirm finite values, then enable a small calibrated weight.

### FL.4 What does not change

- Network architecture.
- Root/search action selection: still `mctx.gumbel_muzero_policy`.
- MC posterior-best policy target computation, except it now consumes Stage A's `alpha_Q_prior + q_evidence_sum`.
- Evaluation adapters, except for any new metric logging.

---

## Stage Full-Selection — Thompson root sampling (~150-260 LoC)

**Goal:** make the search trajectory match math.md §6 more closely by replacing Gumbel root action selection with Thompson sampling from the live root action posteriors.

At each root simulation:

```python
phi_a ~ Dirichlet(alpha_a_current)
a_t = argmax_a U(phi_a)  # or p_W if explicitly using exploratory win-probability mode
evaluate a_t
alpha_a_current[a_t] += c_t * y_t
```

This phase changes which actions and nodes are expanded, and therefore changes:

- `q_evidence_sum`
- `alpha_Q_post`
- `policy_target`
- the played self-play actions
- the downstream replay/data distribution

It should not change the loss formulas from Stage Full-Losses.

### FS.1 Implementation constraint

Post-hoc embedding scatter is sufficient for Stage A evidence aggregation and Stage Full-Losses, but it is not sufficient for true Thompson root selection. Thompson root selection needs the posterior to update during the simulation loop. That requires either:

- a small custom root-search loop that keeps live root `alpha_a_current`, or
- a refreshed minimal mctx fork/wrapper where only root selection and root evidence updates are custom.

Interior node selection can still reuse scalar `U(mean WDL)` with existing mctx-style selection unless we decide to make the whole tree WDL-native later.

### FS.2 Verification

1. With `num_simulations=1`, verify the selected root action is `argmax U(phi_a)` from one Dirichlet sample under the root prior.
2. With a deterministic fake evaluator, verify only the selected root action receives `+c*y` after each simulation.
3. Compare Gumbel vs Thompson runs with the same network seed and confirm policy/loss code paths are unchanged while visited root-action histograms differ.

### Recap

| Stage | New / changed LoC | Where |
|---|---:|---|
| B | ~80 | network/play/loss/eval/train/yaml |
| A | ~80 | embedding metadata + post-search scatter-sum |
| Full-Losses | ~120 | self-play outputs + loss.py KL/mean terms + logging |
| Full-Selection | ~150-260 | custom root selection/search wrapper or refreshed fork |

---

## Verification (end-to-end, applies after each stage)

1. **Smoke test, no NaNs:** Drop `selfplay_batch_size=8`, `num_simulations=4`, `max_num_steps=16`, `policy_mc_samples=8`. Run `uv run python -m scacchi.train` 2 iterations. Confirm shapes and no NaN/inf in any loss term.
2. **Loss decrease:** 20 iterations on default Gardner config, `wandb_enabled: false`, `eval_interval: 5`. The active losses for that stage should trend down. Policy loss should not collapse to ~0 in <5 iters (sign of degenerate one-hot targets — if it does, raise `policy_mc_samples` or lower the effective evidence weight).
3. **Behavioral sanity vs baseline:** `make_evaluate` (cheap sampling) against `gardner_chess_v0`. Win rate should drift positive within ~50 iterations.
4. **Posterior calibration spot-check:** One-shot debug print at `iteration == 0`: `mean(alpha_V).sum() ≈ 1`, `mean(alpha_Q).sum(-1) ≈ 1`, `policy_target.sum(-1) ≈ 1`. Remove after verifying.
5. **Imports clean:** `uv run python -c "import scacchi.train"`.

Stage A adds: confirm `q_evidence_sum.shape == (T, B, A, 3)`, `alpha_Q_post = alpha_Q_prior + q_evidence_sum` is positive, and MC `policy_target.sum(-1) ≈ 1`. Stage Full-Losses adds: confirm Dirichlet-KL terms are finite before enabling nonzero `dir_kl_weight`. Stage Full-Selection adds: confirm loss code paths are unchanged while Gumbel and Thompson runs produce different visited-action histograms.

## Recommendation

Implement Stage B first as a single PR. Validate end-to-end (verification §1-§4). Then open Stage A for evidence aggregation. After Stage A is stable, do Stage Full-Losses as a focused loss/target PR. Only then change trajectory sampling in Stage Full-Selection.
