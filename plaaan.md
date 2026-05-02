# Plan: Dirichlet-Q AlphaZero — Three-Stage Path

## Context

`/workspace/Scacchi/math.md` specifies a Dirichlet-Q AlphaZero variant where the network outputs WDL Dirichlet posteriors (`α^V` per state, `α^Q` per state-action) and the policy target is the *posterior probability that each action is optimal* (Monte-Carlo Thompson over per-action Dirichlets, optionally search-improved).

The current code (`scacchi/play.py`, `scacchi/loss.py`, `scacchi/network.py`) runs vanilla AlphaZero on `mctx.gumbel_muzero_policy` with a scalar value head and visit-count-based policy targets.

Full fidelity to math.md cannot be expressed inside stock mctx — the `Tree` dataclass holds `children_values` as scalars, the backup at `mctx/_src/search.py:265` is hard-coded scalar averaging, and the `gumbel_muzero_root_action_selection` is not Thompson sampling. We adopt a **three-stage incremental path**: each stage is a meaningful, testable improvement; none of the work in earlier stages is throwaway.

- **Stage B** (this plan's first deliverable): Dirichlet heads + losses + Bayesian wrapper for the policy target. No mctx changes.
- **Stage A** (follow-up): Vendor a minimal WDL tree/search fork to get *search-improved* running WDL means at every node and Thompson root selection (§10). Reuse `mctx.muzero_action_selection` for interior nodes.
- **Stage Full** (further follow-up): Dirichlet-KL losses (§14/§15) and terminal-vs-leaf evidence weighting (§8).

---

## Stage B — Dirichlet heads + Bayesian policy wrapper (~80 LoC)

### B.1 Network outputs (`scacchi/network.py`)

Replace `AZNet.__call__` return signature `(logits, value)` → `(policy_logits [B,A], alpha_V [B,3], alpha_Q [B,A,3])`.

Parameterize each Dirichlet head with the math.md §2 stable form:
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

### B.4 MC posterior-best policy target (§6, §11, §12)

After `mctx.gumbel_muzero_policy` returns:

1. `N_a = search_tree.children_visits[:, ROOT_INDEX, :]` shape `[B, A]`.
2. `y_a = terminal_one_hot_or_flip_W_L(gather alpha_V_mean from root children)` shape `[B, A, 3]`.
3. `α_post[b,a] = alpha_Q[b,a] + ρ * sqrt(N_a[b,a]) * y_a[b,a]` (§11 sublinear schedule, using completed visit counts so unvisited actions add no evidence).
4. `phi_samples = jax.random.dirichlet(key, alpha=α_post, shape=(M,))` → `[M, B, A, 3]`.
5. `U_samples = phi_samples[..., W] − phi_samples[..., L]` → `[M, B, A]`.
6. Mask invalid actions: `U_samples = jnp.where(invalid_actions[None, :, :], -jnp.inf, U_samples)`.
7. `argmax_a = U_samples.argmax(-1)` → `[M, B]`.
8. **Laplace-smooth** the histogram (keeps CE gradients well-behaved):
   `policy_target[b,a] = (count(a) + legal_mask[b,a]) / (M + num_legal[b])`

Default `M = 32`, configurable.

### B.5 Loss rewrite (`scacchi/loss.py`)

Drop the L2 scalar value loss. New losses (§13, §17):
- `L_pi`: cross-entropy of `policy_target` (stop-grad) vs `softmax(policy_logits)`.
- `L_V_outcome`: cross-entropy of WDL one-hot final outcome vs `mean(alpha_V)`.
- `L_Q_outcome`: cross-entropy of WDL one-hot final outcome vs `mean(alpha_Q[:, played_action])` — played action only (§17).

Defer §14/§15 KL terms; gate behind `dir_kl_weight: float = 0.0` (Stage Full enables them).

### B.6 Self-play data flow

`SelfplayOutput`:
- `obs`, `reward`, `terminated`, `discount` — unchanged.
- `policy_target [T, B, A]` — new MC posterior-best.
- `played_action [T, B]` — new (§17 needs it).

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
dir_kl_weight: float = 0.0          # Stage Full
evidence_schedule: str = "sqrt"     # "linear" | "sqrt" | "log" — Stage Full uses this
c_terminal: float = 8.0             # Stage Full
c_leaf: float = 2.0                 # Stage Full
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

## Stage A — WDL tree/search fork + Thompson root (~260 LoC)

**Trigger:** ship if Stage B's snapshot `y_a` proves to be the bottleneck (signal: flat policy loss with sane V/Q losses).

Create `scacchi/dirichlet_mctx.py` — one new file with the minimum code needed where stock mctx's scalar value assumption leaks. This is **not** a Gumbel fork.

### A.1 What remains custom

| Component | Origin | Lines | Edit |
|---|---|---|---|
| `Tree` dataclass | `mctx/_src/tree.py:28-115` | ~90 | `node_values: [B,N,3]`, `children_values: [B,N,A,3]`. Update `_unbatched_qvalues` (broadcast). `summary.qvalues` collapses to scalar via U. |
| `RootFnOutput` | local small container | ~5 | Same fields as mctx root output, but `value: [B,3]` WDL mean. Could reuse `mctx.RootFnOutput`, but a local type avoids scalar-value ambiguity. |
| `RecurrentFnOutput` | local small container | ~8 | `reward: [B]`, `is_flip: [B]`, `prior_logits: [B,A]`, `value: [B,3]`. `is_flip` replaces stock scalar `discount` semantics for WDL backup. |
| `instantiate_tree_from_root` | `mctx/_src/search.py:345-385` | ~40 | Allocate WDL-shaped value arrays. Stash `alpha_Q_prior` and root invalid-action mask on the tree for Thompson root selection. |
| `expand` | `mctx/_src/search.py:190-244` | ~55 | Drop `chex.assert_shape(step.value, [batch_size])`. Otherwise unchanged. |
| `backward` | `mctx/_src/search.py:247-292` | ~45 | Replace `discount * leaf_value` with `jnp.where(is_flip, flip_wdl(leaf_value), leaf_value)` — vector flip is W↔L swap, *not* negation. `discount` becomes `is_flip: bool` in `RecurrentFnOutput`. |
| `search` | `mctx/_src/search.py:31-114` | ~85 | Mostly copied loop, but calls the WDL `instantiate_tree_from_root`, `expand`, and `backward`. Stock `mctx.search` cannot be reused because those calls are hard-wired to scalar internals. |
| `thompson_root_action_selection` | new | ~25 | Implements math.md §10/§11: sample per-action Dirichlets from the current root posterior and select `argmax U(phi)`. |
| `dirichlet_policy` | local small wrapper | ~35 | Masks illegal root actions, runs WDL search, returns `mctx.PolicyOutput` with the final MC posterior-best action weights. No Gumbel, no sequential halving. |
| `wdl_qtransform` | new | ~15 | Collapse WDL → scalar U only at the qtransform boundary so `mctx.muzero_action_selection` can be reused for interior nodes. |

### A.2 What is reused untouched

Imported from `mctx._src`:
- `action_selection.muzero_action_selection` for non-root nodes. It only needs tree-like visit/prior fields and a scalar `qtransform`; our `wdl_qtransform` supplies scalar `U`.
- Optionally `action_selection.switching_action_selection_wrapper` for root-vs-interior dispatch.
- `action_selection.masked_argmax`, if useful.
- `mctx.PolicyOutput` as the return container.

Not reused:
- `mctx.gumbel_muzero_policy`, `gumbel_muzero_root_action_selection`, `gumbel_muzero_interior_action_selection`, `GumbelMuZeroExtraData`, and `seq_halving`.
- `mctx.search`, `mctx.expand`, `mctx.backward`, and `mctx.instantiate_tree_from_root`, because they allocate and update scalar value arrays.

Pin `mctx>=0.0.6,<0.0.7` in `pyproject.toml` since we import from `_src`.

### A.3 Thompson root posterior

Root action selection follows math.md §10/§11 in Stage A, not Stage Full:

```python
def thompson_root_action_selection(rng, tree, node_index, *, rho, schedule):
    alpha_prior = tree.extra_data.alpha_Q_prior[node_index]   # [A, 3]
    y_bar = tree.children_values[node_index]                  # [A, 3]
    visits = tree.children_visits[node_index]                 # [A]
    evidence = evidence_schedule(visits, rho, schedule)       # [A]
    alpha_post = alpha_prior + evidence[..., None] * y_bar
    phi = jax.random.dirichlet(rng, alpha_post)               # [A, 3]
    utility = phi[..., W] - phi[..., L]
    utility = jnp.where(tree.root_invalid_actions, -jnp.inf, utility)
    return jnp.argmax(utility)
```

For completed visit counts, `evidence_schedule(N, ρ, "sqrt") = ρ * jnp.sqrt(N)`, so unvisited actions add no search evidence. Also support `"linear"` (`ρ * N`) and `"log"` (`ρ * log1p(N)`).

### A.4 Effect on Stage B code

In `play.py`:
- Drop `NodeEmbedding`; embedding goes back to bare `pgx.State`. The running WDL mean now lives at `tree.children_values[:, ROOT_INDEX, :, :]` directly.
- `make_recurrent_fn` returns the local `dirichlet_mctx.RecurrentFnOutput` with `value: [B, 3]` (the WDL mean from `α^V` of the leaf state) and `is_flip: [B]`.
- For terminal children, return terminal one-hot WDL from the parent/root player's perspective and set `is_flip=False`; for non-terminal children, return the child state's WDL mean and set `is_flip=True`.
- The Stage B "gather child embedding then flip" logic disappears — the forked `backward` maintains root-action WDL means directly.
- Replace `mctx.gumbel_muzero_policy` with `dirichlet_mctx.dirichlet_policy`.
- Build the final `policy_target` from the final root posterior (`alpha_Q + evidence * tree.children_values[:, ROOT, :, :]`) using the existing MC posterior-best helper.

In `loss.py` and `train.py`: no changes from Stage B.

In `evaluations.py`:
- Sampling eval remains unchanged.
- MCTS eval uses `dirichlet_mctx.dirichlet_policy` for both sides. The pgx baseline wrapper still synthesizes `alpha_V` so `U(mean(alpha_V))` preserves its scalar value.

### A.5 Verification (Stage A)

- Confirm `tree.children_values[:, ROOT_INDEX, :, :].sum(-1)` is approximately 1 across batch and actions (running mean of normalized WDL).
- Run a unit-style script: 1 batch element, 8 simulations, terminal child reachable in 1 step. Check that the terminal's WDL one-hot back-propagates with correct flip across odd/even depths.

---

## Stage Full — Dirichlet KL + evidence weighting (~100 LoC)

### F.1 Dirichlet-KL losses (§14, §15)

In `scacchi/loss.py`:
- Implement §16 KL formula via `jax.scipy.special.gammaln` and `jax.lax.digamma`. ~15 LoC.
- Construct `β_V = α_base + c_V^search * y_V^search` and `β_a = α_base + c_a^search * y_a^search` from search outputs. Use the search-improved running mean from `tree.node_values[:, ROOT_INDEX, :]` (V) and `tree.children_values[:, ROOT_INDEX, :, :]` (Q). ~20 LoC.
- Add weighted `L_V_Dir`, `L_Q_Dir` to total loss. ~15 LoC.

Flip `dir_kl_weight` from `0.0` to a calibrated value (try 0.5, then sweep).

### F.2 Terminal vs leaf evidence weighting (§8)

Plan A's running mean treats every backup-step as 1 unit of evidence; §8 says terminals carry more weight (`c_terminal > c_leaf`).

- Add `evidence: [B]` to `RecurrentFnOutput` (or derive in `recurrent_fn`: `c_terminal` if `terminated` else `c_leaf`).
- Replace mctx's update formula in our forked `backward`:
  ```
  parent_value = (prev * accumulated + leaf * w) / (accumulated + w)
  ```
- Add a parallel `accumulated_evidence: [B, N]` field to the forked `Tree` (preserve `node_visits` for logging/debug).

### F.3 What does not need change in Stage Full

- Network architecture (heads stay).
- `SelfplayOutput` shape.
- `evaluations.py`.
- `train.py` outer loop.
- Stage A's Thompson root selection and interior `mctx.muzero_action_selection`.
- Stage A's forked `search`, `expand`, `instantiate_tree_from_root` (only `backward` changes for weighted evidence).

### F.4 Cumulative LoC (recap)

| Stage | New / changed LoC | Where |
|---|---|---|
| B | ~80 | network/play/loss/eval/train/yaml |
| A | +260 | one new file `dirichlet_mctx.py` + targeted play/eval integration |
| Full | +100 | inside the fork (~40) + loss.py KL terms (~50) + config (~10) |
| **Total** | **~440** | one fork file + targeted edits in 6 existing files |

---

## Verification (end-to-end, applies after each stage)

1. **Smoke test, no NaNs:** Drop `selfplay_batch_size=8`, `num_simulations=4`, `max_num_steps=16`, `policy_mc_samples=8`. Run `uv run python -m scacchi.train` 2 iterations. Confirm shapes and no NaN/inf in any loss term.
2. **Loss decrease:** 20 iterations on default Gardner config, `wandb_enabled: false`, `eval_interval: 5`. `policy_loss`, `value_outcome_loss`, `q_outcome_loss` should all trend down. Policy loss should not collapse to ~0 in <5 iters (sign of degenerate one-hot targets — if it does, raise `policy_mc_samples` or lower `search_evidence_rho`).
3. **Behavioral sanity vs baseline:** `make_evaluate` (cheap sampling) against `gardner_chess_v0`. Win rate should drift positive within ~50 iterations.
4. **Posterior calibration spot-check:** One-shot debug print at `iteration == 0`: `mean(alpha_V).sum() ≈ 1`, `mean(alpha_Q).sum(-1) ≈ 1`, `policy_target.sum(-1) ≈ 1`. Remove after verifying.
5. **Imports clean:** `uv run python -c "import scacchi.train"`.

Stage A adds: confirm `tree.children_values[:, ROOT, :, :].sum(-1) ≈ 1`. Stage Full adds: confirm Dirichlet-KL term is finite and decreases when `dir_kl_weight` is enabled mid-training.

## Recommendation

Implement Stage B first as a single PR. Validate end-to-end (verification §1–§4). Only after confirming the heads/losses train without pathology should we open Stage A. Stage Full follows from A in another focused PR.
