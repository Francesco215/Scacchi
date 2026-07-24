# Scacchi

Scacchi is a JAX/PGX research codebase for AlphaZero-style self-play. The
current experimental path is Dirichlet-Q AlphaZero: instead of treating search
values as only scalar estimates, the search stores and trains against
Dirichlet posteriors over win/draw/loss outcomes.

## Math Reference

[`math.md`](math.md) is the high-level math reference. In brief, the model
predicts:

- a policy head for move probabilities,
- a state-value Dirichlet head for uncertainty over the current state's WDL
  value,
- an action Dirichlet-Q head for uncertainty over each legal move's WDL value.

Search refines uncertain WDL beliefs by passing Dirichlet messages through the
tree and propagates exact terminal results as native categorical
outcome/distance certificates. Categorical branches are absorbing and pruned;
the search-improved policy compares their exact utility with Thompson samples
from unresolved branches. Value/Q heads use posterior KL for unresolved targets
and epsilon-interior Dirichlet density NLL for categorical targets.

Q supervision selects legal actions with positive search evidence, together
with legal solved actions:

\[
M_{s,a}=\mathbf 1[\operatorname{legal}(s,a)\land
(\operatorname{evidence}_{s,a}>0\lor\operatorname{solved}_{s,a})].
\]

The default Q loss is the mean over those selected state-action pairs,
\[
L_Q=\frac{\sum_{s,a}M_{s,a}\ell_{s,a}}{\sum_{s,a}M_{s,a}}.
\]
Search evidence determines whether an action receives Q supervision. Its
magnitude does not scale the loss. The mean is over selected state-action
pairs, not states, so states with more selected actions contribute more pairs.

Each posterior-update variant has one scalar repair constant, `kappa`, used
only in the structural mixing weight
`gamma = n_down / (kappa + n_down)`. Terminal expansion instead returns an
exact `terminal_outcome` tag. It does not manufacture a terminal Dirichlet or
inject a fixed concentration: model alphas remain the learned representation
for unresolved leaves and caches, while categorical outcome/distance sidecars
own solved semantics.
For a mixed node-cache update, a categorical edge is projected temporarily as
$(\sum_i A_i)e_z$: the exact tag supplies its direction and its existing
effective alpha supplies the learned mass. This projection is neither stored
nor used as a categorical target.

The Thompson tree-search backend lives in `scacchi/dirichlet_mctx/`;
`scacchi/dirichlet_q_search.py` contains the shared leaf expansion,
terminal-outcome extraction, and posterior-target helpers.

## Guarded Q21 mode

For binary games such as Hex, the optional `prefix_cdf` estimator computes
posterior-best action probabilities on an adaptive 21-point grid. Search and
action commitment select their posterior updater independently:

```yaml
selfplay:
  action_commitment:
    kind: posterior_sample
    posterior_update: numerical
    posterior_sample_temperature: 0.3333333333333333
  search:
    dirichlet_thompson:
      posterior_update:
        kind: numerical
        numerical:
          half_width: 10  # Q = 2 * half_width + 1 = 21
```

The search selection controls internal repair and the replay target. The
action-commitment selection builds a fresh action-only policy from the searched
root posteriors; `null` reuses the search selection. Both selectors use the
Monte Carlo and numerical parameter blocks under the Dirichlet search config.
Numerically unsafe estimates fall back to the unchanged winner-sampling path
for the whole batch. The action-only policy is not written to replay. With
`posterior_sample`, temperature \(T\) samples on the positive support
from \(q_T(a)\propto\operatorname{clip}(q(a),10^{-8},1)^{1/T}\); exact zeros
stay zero. Prefix-CDF requires a two-outcome head.
The Hex6 recipe enables Q21, cubic (`T=1/3`) commitment, and W&B logging.
The numerical guards detect specified integration failures, not arbitrary
quadrature error outside the Hex6 envelope used to select Q21.

## Codebase Structure

- `scacchi/`: main Python package.
- `scacchi/train.py`: Hydra entry point, config validation, model setup,
  checkpointing, evaluation, and training loop.
- `scacchi/network.py`: neural network definitions, including the
  policy/value/Q Dirichlet model.
- `scacchi/play.py`: training and evaluation play loops.
- `scacchi/play_search.py`: evaluator, search, player, and action commitment
  boundaries.
- `scacchi/dirichlet_mctx/`: lightweight MCTX-shaped Dirichlet Thompson search.
- `scacchi/pipeline.py`: replay/minibatch handling and per-iteration training.
- `scacchi/loss.py`: policy, scalar value, typed Dirichlet/categorical, and
  outcome losses.
- `scacchi/configs/`: Hydra YAML configs, currently centered on Hex.
- `scripts/`: benchmarks, sweeps, and plotting utilities.
- `tests/`: unit tests for config validation, losses, network behavior, and
  search utilities.

## Common Commands

```bash
uv sync
uv run pytest
uv run scacchi-train
```
