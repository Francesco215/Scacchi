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

Search refines these WDL beliefs by accumulating evidence in the tree. The
search-improved policy target is the posterior probability that each action is
optimal, and the value/Q heads are trained toward posterior Dirichlet targets.

The Thompson tree-search backend lives in `scacchi/dirichlet_mctx/`;
`scacchi/dirichlet_q_search.py` contains the shared leaf expansion,
posterior-target helpers, and the small adapter used by Dirichlet-Gumbel MCTX.

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
- `scacchi/loss.py`: policy, scalar value, Dirichlet KL, and outcome losses.
- `scacchi/dirichlet_tree/`: categorical target metadata helpers.
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
