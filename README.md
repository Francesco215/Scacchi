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

The native search has one scalar posterior-repair constant, `kappa`, used only
in the structural mixing weight
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
