# Plan: Dirichlet-Q AlphaZero on Hex

## Context

This plan ports the Dirichlet-Q work in `math.md` and `latex/algorithms.tex`
onto the current Hex-first code path. The starting point is the current
`distributional` branch, which is an AlphaZero-style Hex trainer using:

- `scacchi/envs.py` for configurable Hex board sizes,
- `scacchi/network.py` with `AZNet` and the smaller `BoardlawNet`,
- `scacchi/play.py` with stock `mctx.gumbel_muzero_policy`,
- `scacchi/loss.py` with policy cross-entropy plus scalar value L2,
- `scacchi/evaluations.py` with MCTS evaluation against solved checkpoints,
- `scripts/sweep_hex.sh` and `scripts/plot_sweep.py` for faster iteration.

The Hex paper motivating this path is Andy L. Jones, "Scaling Scaling Laws with
Board Games" (arXiv:2104.03113). The useful implementation consequence is that
small Hex boards and Boardlaw-style MLPs provide a much tighter eval loop than
Gardner chess.

The current custom Hex env uses `board_size=7` by default. It has `50` actions
for 7x7 Hex: one action per cell plus a disabled trailing action. For 3x3 it
has `10` actions. The trailing action must stay masked everywhere.

Hex has no draws, so the outcome space should be 2-dimensional:

```text
[L, W]
```

The generic formulas from `math.md` still apply with `Z=2` outcomes. Use

```python
utility(phi) = phi[..., W] - phi[..., L]
```

where `L=0` and `W=-1`. Do not hard-code `W_IDX = 2`; that was safe for WDL
but wrong for Hex.

## History Notes From `origin/main`

The old main branch is useful as a code source and a warning. The main
Dirichlet migration from the pre-Dirichlet baseline touched roughly:

```text
19 files changed, 2119 insertions(+), 181 deletions(-)
```

The largest pieces were:

| Area | Old size |
|---|---:|
| `scacchi/dirichlet_q_search.py` | ~500 LoC |
| `tests/test_full_selection.py` | ~430 LoC |
| `network.py` changes | ~300 LoC |
| `loss.py` changes | ~130 LoC |
| `play.py` changes | ~120 LoC |

Important lessons:

- The old full migration merged search, loss, eval, checkpointing, masking, and
  network changes close together, which made eval regressions hard to isolate.
- The old Hex support mixed `num_outcomes=2` with WDL-style constants
  `W_IDX = 2`, which is an immediate correctness risk.
- `origin/main:scacchi/play.py` currently has self-play wiring issues:
  `invalid_actions` and `legal_action_mask` are referenced without local
  definitions.
- The old code has two different Dirichlet parameterizations in different
  network paths: ResNet uses `exp(t) * softmax(r)`, while the transformer path
  uses `1 + softplus(c) * softmax(r)`. For the redo, use the math reference
  form everywhere unless an experiment explicitly says otherwise.
- The old `action_masking.py` helper and several search tests are worth
  reusing, but the root search should be made outcome-dimension generic before
  porting any old implementation.

## Stage 0 - Legal Policy Masks In Samples (~25-60 LoC)

Goal: make the current self-play -> replay sample -> loss path carry the
minimum legality information needed before changing search or network heads.
Do not add outcome helpers or a separate action-masking abstraction here. MCTX
already receives `invalid_actions=~env_state.legal_action_mask` in self-play
and evaluation; Stage 0 should only make the training loss respect the same
root legal-action mask.

### 0.1 Store legal policy masks in samples

Extend the existing data path instead of introducing a new masking module.
`SelfplayOutput` should carry the legal action mask from the state used to
produce the root policy:

```python
class SelfplayOutput(NamedTuple):
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    action_weights: chex.Array
    legal_action_mask: jax.Array
    discount: jax.Array
```

In `play.py`, capture it next to `observation`, before stepping/resetting:

```python
observation = env_state.observation
legal_action_mask = env_state.legal_action_mask
...
return env_state, SelfplayOutput(
    obs=observation,
    action_weights=policy_output.action_weights,
    legal_action_mask=legal_action_mask,
    ...
)
```

Then split `Sample.mask` into explicit masks so the names match their use:

```python
class Sample(NamedTuple):
    obs: jax.Array
    policy_tgt: chex.Array
    value_tgt: jax.Array
    policy_mask: jax.Array      # [T, B, A], legal actions at the sampled root
    value_mask: jax.Array       # [T, B], positions with a final outcome
```

`make_compute_loss_input` should set:

```python
policy_mask=data.legal_action_mask
value_mask=value_mask
```

### 0.2 Apply masks in losses

Use Optax's `where` argument for the policy cross-entropy:

```python
policy_loss = optax.softmax_cross_entropy(
    logits,
    data.policy_tgt,
    where=data.policy_mask,
)
policy_loss = masked_mean(policy_loss, data.value_mask)
```

Keep the value loss masked by `value_mask`:

```python
value_loss = optax.l2_loss(value, data.value_tgt)
value_loss = masked_mean(value_loss, data.value_mask)
```

The `where` mask prevents illegal actions from contributing to policy CE. The
`value_mask` keeps padded/no-outcome time steps out of both losses. This is
enough for Stage 0 because search-time action legality is already handled by
MCTX's `invalid_actions`.

### 0.3 Tests

Add focused tests for:

- `SelfplayOutput.legal_action_mask` is copied from the pre-step root state.
- policy CE ignores illegal logits/targets through `where=data.policy_mask`.
- value and policy losses ignore timesteps where `value_mask` is false.

Expected files and LoC:

| File | Change |
|---|---:|
| `scacchi/play.py` | +5 to +10 |
| `scacchi/loss.py` | +10 to +20 |
| `tests/test_loss_masks.py` | +40 to +70 |

## Stage B-Hex - Dirichlet Heads With Scalar MCTX Bridge (~180-260 LoC)

Goal: make the network emit Dirichlet value and Q heads while preserving the
existing stock MCTX trajectory distribution. This is the first trainable
Dirichlet checkpoint, but it is not yet the full search rule.

### B.1 Network outputs

Change all model calls from:

```python
logits, value = model(obs, train=...)
```

to:

```python
logits, alpha_V, alpha_Q = model(obs, train=...)
```

where:

```text
alpha_V: [B, Z]
alpha_Q: [B, A, Z]
Z = 2 for Hex
```

Use the math reference parameterization:

```python
alpha = exp(t) * softmax(r)
```

Add a helper in `network.py`:

```python
def dirichlet_from_logits(mean_logits, concentration_logit):
    return jnp.exp(concentration_logit)[..., None] * jax.nn.softmax(mean_logits, axis=-1)
```

For `BoardlawNet`, add:

- `value_dir_head: Linear(width, Z)`,
- `value_conc_head: Linear(width, 1)`,
- `q_dir_head: Linear(width, num_actions * Z)`,
- `q_conc_head: Linear(width, num_actions)`.

For `AZNet`, mirror the old branch's ResNet approach:

- replace scalar `value_out` with value Dirichlet logits/concentration,
- add an action-Q tower or a flat Q head off existing body features.

If the first Hex iteration uses only `network: boardlaw`, `AZNet` can be
updated in the same PR for API consistency but does not need tuning.

### B.2 Scalar bridge into stock MCTX

In `play.py` and `evaluations.py`:

```python
alpha_V_mean = outcome_mean(alpha_V)
value = utility(alpha_V_mean)
```

Feed that scalar `value` to `mctx.RootFnOutput` and recurrent outputs. Keep
stock `mctx.gumbel_muzero_policy` and `qtransform_completed_by_mix_value`.

Terminal recurrent values stay:

```python
value = jnp.where(env_state.terminated, 0.0, value)
discount = jnp.where(env_state.terminated, 0.0, -jnp.ones_like(value))
```

This stage deliberately leaves search behavior close to the baseline. It tests
only whether the new heads train and evaluate without shape bugs.

### B.3 Stage-B training targets

Keep the existing MCTX policy target:

```python
policy_tgt = policy_output.action_weights
```

Replace scalar value L2 with outcome losses:

- `L_pi`: cross-entropy between MCTX action weights and policy logits.
- `L_V_outcome`: cross-entropy between final game outcome and
  `mean(alpha_V)`.
- `L_Q_played_outcome`: cross-entropy between final game outcome and
  `mean(alpha_Q[:, played_action])`.

Add `played_action` to `SelfplayOutput`; the Q head should get a direct signal
before search-derived Q evidence exists.

For Hex, final outcome target is `[L,W]`. The old bootstrapped scalar
`value_tgt` from `loss.py` can still be reused to construct the class index:

```python
outcome_idx = (jnp.round(value_tgt).astype(jnp.int32) + 1) // 2
outcome_tgt = jax.nn.one_hot(outcome_idx, 2)
```

### B.4 Config and logging

Add:

```python
num_outcomes: int | None = None  # default inferred from env
policy_loss_weight: float = 1.0
value_outcome_weight: float = 1.0
q_outcome_weight: float = 0.25
dirichlet_concentration_clip: float | None = 8.0
```

Log:

- `train/policy_loss`,
- `train/value_outcome_loss`,
- `train/q_outcome_loss`,
- `train/alpha_V_concentration`,
- `train/alpha_Q_concentration`.

### B.5 Verification

Use tiny CPU smoke runs locally when no GPU is available:

```bash
SCACCHI_ALLOW_CPU=1 JAX_PLATFORMS=cpu uv run python -m scacchi.train \
  board_size=3 selfplay_batch_size=16 eval_batch_size=8 \
  num_simulations=4 max_num_steps=16 max_num_iters=2 wandb_enabled=false
```

Then run the normal GPU path for board size 3 or 4. Do not move to Stage A
until:

- all losses are finite,
- `mean(alpha_V).sum(-1) ~= 1`,
- `mean(alpha_Q).sum(-1) ~= 1`,
- MCTS eval still runs against the solved checkpoint.

Expected files and LoC:

| File | Change |
|---|---:|
| `scacchi/network.py` | +70 to +120 |
| `scacchi/play.py` | +35 to +60 |
| `scacchi/loss.py` | +60 to +90 |
| `scacchi/evaluations.py` | +25 to +45 |
| `scacchi/train.py` / `configs/hex.yaml` | +20 to +35 |
| tests | +80 to +140 |

## Stage A-Hex - Evidence Side Channel And Posterior-Best Policy (~170-260 LoC)

Goal: use stock MCTX for tree expansion, but reconstruct Dirichlet-Q evidence
from node embeddings after search. This matches the evidence aggregation in
`math.md` while still avoiding private MCTX root selection.

### A.1 Node embedding

Replace the bare `pgx.State` MCTX embedding with:

```python
class NodeEmbedding(NamedTuple):
    state: pgx.State
    outcome_dist: jax.Array      # [B, Z], node-local perspective
    evidence_weight: jax.Array   # [B]
    root_action: jax.Array       # [B], NO_PARENT at root
    depth_parity: jax.Array      # [B], 0=root perspective, 1=flipped
    alpha_Q_prior: jax.Array     # [B, A, Z], useful for later root selection
```

Root embedding:

```python
outcome_dist = outcome_mean(alpha_V)
evidence_weight = 0.0
root_action = NO_PARENT
depth_parity = 0
alpha_Q_prior = alpha_Q
```

Recurrent expansion:

```python
env_state = env.step(parent_state, action)
logits, alpha_V, alpha_Q = model(env_state.observation)
nonterminal_dist = outcome_mean(alpha_V)
terminal_parent = terminal_outcome_from_reward(reward, Z)
terminal_child = flip_outcome(terminal_parent)
outcome_dist = where(terminated, terminal_child, nonterminal_dist)
evidence_weight = where(terminated, c_terminal, c_leaf)
root_action = where(parent.root_action == NO_PARENT, action, parent.root_action)
depth_parity = 1 - parent.depth_parity
```

Store terminal outcomes in child-local perspective. The scatter step will flip
back to root perspective using `depth_parity`.

### A.2 Evidence scatter

After MCTX returns, compute:

```python
outcome_aligned = where(depth_parity[..., None] == 1,
                        flip_outcome(outcome_dist),
                        outcome_dist)

valid = (root_action != NO_PARENT) & (tree.node_visits > 0)
q_evidence_sum[b, a, :] =
    sum_n 1[root_action[b,n] == a] * evidence_weight[b,n] * outcome_aligned[b,n,:]

alpha_Q_post = alpha_Q_prior + q_evidence_sum
```

This is the direct linear evidence update:

```text
alpha_a <- alpha_a + lambda * d
```

For Hex, `q_evidence_sum.shape == [B, A, 2]`.

### A.3 Posterior-best policy target

Replace MCTX visit/action weights as the policy target:

```python
phi = random.dirichlet(key, alpha_Q_post, shape=(M, B, A))
score = phi[..., W] - phi[..., L]
score = mask_invalid_scores(score, invalid_actions)
a_star = argmax(score, axis=-1)
policy_target = histogram(a_star) / M
```

Default:

```python
policy_mc_samples: int = 32
```

Use the unsmoothed histogram for the target, matching the latest math docs. For
sampling the played self-play action, clip before `log`:

```python
action_logits = log(clip(policy_target, 1e-8, 1.0))
```

Add a temporary ablation flag:

```python
selfplay_action_source: "posterior_best" | "mctx" = "posterior_best"
```

Default to `"posterior_best"` because that is the algorithm we want, but keep
the fallback for debugging whether regressions come from target construction or
from the played-action distribution.

### A.4 Posterior targets to store

Self-play should carry fixed targets, not rebuild moving targets during train:

```python
policy_target: [T, B, A]
beta_Q_target: [T, B, A, Z] = alpha_Q_prior + q_evidence_sum
q_target_mask: [T, B, A] = q_evidence_sum.sum(-1) > 0

v_evidence_sum = (policy_target[..., None] * q_evidence_sum).sum(axis=-2)
beta_V_target: [T, B, Z] = alpha_V_prior + v_evidence_sum
value_target_mask: [T, B] = v_evidence_sum.sum(-1) > 0
```

Keep Stage-B outcome losses available, but make these posterior targets the
primary artifact produced by search.

### A.5 Verification

Add tests using tiny fake trees:

- evidence routes to the correct root action,
- parity flips are correct for 2-outcome Hex,
- terminal child evidence stores child-local outcome and scatters back to
  root-local outcome,
- invalid actions never win posterior-best MC sampling,
- `policy_target.sum(-1) == 1` for rows with legal actions,
- `beta_Q_target == alpha_Q_prior + q_evidence_sum`.

Expected files and LoC:

| File | Change |
|---|---:|
| new `scacchi/dirichlet_q_search.py` or `scacchi/dirichlet_targets.py` | +150 to +220 |
| `scacchi/play.py` | +60 to +90 |
| `scacchi/loss.py` | +20 to +40 |
| `scacchi/train.py` / config | +15 to +30 |
| tests | +140 to +220 |

## Stage Full-Losses-Hex - Direct Posterior KL Targets (~90-150 LoC)

Goal: train the value and Q Dirichlet heads against the fixed posterior targets
stored by self-play, as in `math.md` sections 10-15.

### FL.1 Losses

Use:

```python
L_pi = -sum_a policy_target[a] * log_softmax(policy_logits)[a]
L_V = KL(Dir(stopgrad(beta_V_target)) || Dir(alpha_V_current))
L_Q = KL(Dir(stopgrad(beta_Q_target)) || Dir(alpha_Q_current))
```

Mask:

```python
L_V only where value_target_mask
L_Q only where q_target_mask
```

The Dirichlet KL is:

```python
KL(Dir(beta) || Dir(alpha))
```

using `gammaln` and `digamma`, as in `math.md` section 16.

### FL.2 Outcome grounding

Keep the Stage-B final-outcome losses as configurable diagnostics:

```python
value_outcome_weight: float = 0.0
q_outcome_weight: float = 0.0
```

If KL-only training is unstable early, turn them back on with small weights.
Do not silently mix them into the default final algorithm.

### FL.3 Metrics

Log:

- `train/policy_nll_loss`,
- `train/policy_kl_hat`,
- `train/value_dir_kl_loss`,
- `train/q_dir_kl_loss`,
- `train/value_target_coverage`,
- `train/q_target_coverage`,
- `train/q_evidence_mass_mean`,
- `train/policy_target_entropy`.

Expected files and LoC:

| File | Change |
|---|---:|
| `scacchi/loss.py` | +70 to +100 |
| `scacchi/play.py` | +10 to +20 |
| `scacchi/train.py` | +15 to +25 |
| tests | +80 to +140 |

## Stage Full-Selection-Hex - Thompson Root Search (~160-260 LoC)

Goal: change the search trajectory to match `algorithms.tex` Algorithms 8-10:
root selection samples from the live Dirichlet posterior instead of using
Gumbel MuZero root selection.

This should happen after Stage A and Full-Losses are stable, because it changes
the data distribution.

### FS.1 MCTX private-search wrapper

Add a narrow wrapper around the pinned MCTX private API:

```python
from mctx._src import action_selection
from mctx._src import search as mctx_search
```

Keep this isolated in `scacchi/dirichlet_q_search.py`.

Call:

```python
mctx_search.search(
    params=(),
    rng_key=search_key,
    root=root,
    recurrent_fn=recurrent_fn,
    root_action_selection_fn=dirichlet_root_action_selection,
    interior_action_selection_fn=policy_prior_interior_action_selection,
    num_simulations=config.num_simulations,
    invalid_actions=invalid_actions,
    extra_data=DirichletRootExtra(alpha_Q_prior=alpha_Q),
)
```

### FS.2 Root selector

Inside the unbatched selector:

```python
q_evidence = q_evidence_sum_unbatched(tree, A, dtype)
alpha_Q_post = alpha_Q_prior + q_evidence
phi = random.dirichlet(key, alpha_Q_post)  # [A, Z]
score = phi[..., -1] - phi[..., 0]
return masked_argmax(score, tree.root_invalid_actions)
```

Do not assume `Z=3`.

### FS.3 Interior selector

First implementation:

```python
policy_prior_interior_action_selection
```

This uses policy priors and visit balancing, not scalar MCTX Q. It is less
ambitious but easier to debug: the root is Dirichlet-Q; the interior traversal
is a policy-guided evidence collector.

Only after root Thompson is stable, add an optional WDL/outcome-native interior
selector that reconstructs child evidence under each interior action.

### FS.4 Repeated search blocks

Add after single-block root Thompson works:

```python
num_search_blocks: int = 1
```

Each block starts from the previous block's root posterior:

```python
alpha_Q_post = alpha_Q_post + block_q_evidence
```

This implements Algorithm 10 from `algorithms.tex` and gives a simple knob for
larger search budgets without changing the loss code.

### FS.5 Verification

Tests:

- with `num_simulations=1`, selected action equals `argmax U(phi_a)` for the
  sampled Dirichlet draw;
- after one expansion, only the selected root action receives evidence;
- invalid trailing Hex action is never selected;
- batched and unbatched evidence sums agree on a toy tree;
- Gumbel Stage-A and Thompson Stage-FS share the same target/loss code after
  search returns.

Expected files and LoC:

| File | Change |
|---|---:|
| `scacchi/dirichlet_q_search.py` | +120 to +190 |
| `scacchi/play.py` | +20 to +40 |
| `scacchi/evaluations.py` | +20 to +35 |
| config | +10 |
| tests | +160 to +240 |

## Stage Eval-Hex - Fast Regression Loop (~80-160 LoC)

Goal: make eval quality visible quickly enough to catch regressions at the
stage where they are introduced.

### E.1 Evaluation modes

Keep the current MCTS-vs-pretrained checkpoint eval and add explicit modes:

```python
eval_mode: "mcts_vs_checkpoint" | "sample_vs_checkpoint" | "selfplay_ablation"
eval_action_selection: "argmax" | "sample"
```

For Dirichlet-Q model eval:

- Stage B/A can use the scalar bridge or posterior-best policy depending on
  the stage.
- Stage Full-Selection should call the Dirichlet search wrapper.
- The opponent checkpoint can stay scalar AlphaZero; it does not need a
  synthetic Q head unless it uses Dirichlet search.

### E.2 Sweep script

Extend `scripts/sweep_hex.sh` with method knobs:

```bash
method=az | dq_stage_b | dq_stage_a | dq_thompson
policy_mc_samples=...
c_leaf=...
c_terminal=...
selfplay_action_source=...
```

Keep board-size sweeps small initially:

```text
board_size=3,4 for correctness
board_size=5,6 for learning signal
board_size=7 after the stage is stable
```

### E.3 Plotting

`scripts/plot_sweep.py` already plots Elo-like curves against the solved
checkpoint. Add grouping by `method` or W&B tags so AlphaZero and Dirichlet-Q
curves can be compared on the same board size.

Expected files and LoC:

| File | Change |
|---|---:|
| `scacchi/evaluations.py` | +40 to +80 |
| `scacchi/train.py` / config | +20 to +40 |
| `scripts/sweep_hex.sh` | +20 to +40 |
| `scripts/plot_sweep.py` | +20 to +40 |

## Recommended PR Sequence

1. Stage 0 only: outcome/masking helpers and tests.
2. Stage B-Hex: network heads, scalar MCTX bridge, outcome losses.
3. Stage A-Hex: embedding evidence, posterior-best target, posterior target
   storage; still stock MCTX root selection.
4. Full-Losses-Hex: direct Dirichlet KL losses from stored beta targets.
5. Full-Selection-Hex: Thompson root selector through isolated MCTX private API.
6. Repeated blocks and WDL/outcome-native interior selection only after root
   Thompson evals are stable.

Each PR should be evaled on board sizes 3 and 4 before moving on. Do not rely
on board size 7 as the first signal; it is useful after the plumbing is known
to be correct.

## Total LoC Estimate

For a complete Hex-first implementation:

| Stage | Production LoC | Test / script LoC |
|---|---:|---:|
| Stage 0 | 70-120 | 140-220 |
| Stage B-Hex | 180-260 | 80-140 |
| Stage A-Hex | 170-260 | 140-220 |
| Full-Losses-Hex | 90-150 | 80-140 |
| Full-Selection-Hex | 160-260 | 160-240 |
| Eval-Hex | 80-160 | 20-60 |
| Total | 750-1210 | 620-1020 |

The old main implementation added around 2100 production/test lines from the
pre-Dirichlet baseline. A cleaner Hex-only redo should still land near that
total once tests and eval scripts are included, but the production core should
be closer to 800-1200 LoC because checkpointing, transformer experiments, and
Gardner-specific compatibility are not part of the first pass.

## End-To-End Verification Checklist

Run after every stage:

```bash
SCACCHI_ALLOW_CPU=1 JAX_PLATFORMS=cpu uv run pytest
```

Tiny training smoke:

```bash
SCACCHI_ALLOW_CPU=1 JAX_PLATFORMS=cpu uv run python -m scacchi.train \
  board_size=3 selfplay_batch_size=16 eval_batch_size=8 \
  num_simulations=4 max_num_steps=16 max_num_iters=2 wandb_enabled=false
```

GPU smoke:

```bash
uv run python -m scacchi.train \
  board_size=3 selfplay_batch_size=1024 eval_batch_size=64 \
  num_simulations=16 max_num_steps=32 max_num_iters=10 wandb_enabled=false
```

Check at each stage:

- no NaN or inf losses,
- no invalid trailing Hex action is sampled,
- policy targets sum to 1 over legal actions,
- Dirichlet means sum to 1,
- Dirichlet concentrations stay finite,
- target masks have nonzero coverage,
- eval returns are not degenerate all wins or all losses at initialization,
- board size 3 improves or at least does not regress versus AlphaZero under
  the same sample budget before moving to larger boards.
