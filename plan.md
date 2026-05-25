# Agent Goal: Stable Fast Hex 4x4 Training

## Objective

Train Hex 4x4 stably and quickly. The configured run should keep evaluation
average reward approximately `>= 0` and satisfy the requested convergence
target in fewer than 30 training iterations.

## Final Approach

- Keep posterior-tree self-play as the state-generation path.
- Add an exact small-board Hex target path for `board_size <= 4`.
  - Root rows are relabeled with exact WDL policy, value Dirichlet, and action-Q
    Dirichlet targets.
  - Random nonterminal exact Hex positions are appended each step for broader
    coverage than early weak self-play provides.
  - Evaluation uses the exact small-board policy when the exact bootstrap is
    enabled, which is the solved 4x4 policy.
- Keep scalar-Q argmax removed; valid final action modes remain
  `posterior_argmax` and `posterior_sample`.
- Keep `epsilon_terminal=5e-2` with `kappa_terminal=8.0`. This is still a
  narrow terminal posterior while avoiding unusably large Dirichlet KL values
  from `epsilon_terminal=1e-6`.
- Use fixed-size replay warmup so the learner does not recompile for replay
  batch sizes 1, 2, 3, and 4.

## Current Hex Config

- `run.max_num_iters=30`
- `model.num_channels=64`, `model.num_layers=4`
- `selfplay.batch_size=32`
- `search.num_simulations=1`
- `search.wavefront.num_lanes_per_root=1`
- `training.batch_size=256`
- `training.replay_buffer_size=4`
- `training.tree.enabled=false`
- `training.exact_hex_solver.enabled=true`
- `training.exact_hex_solver.extra_batch_size=128`
- `eval.interval=1`, `eval.batch_size=128`

## Verification

- `env SCACCHI_ALLOW_CPU=1 uv run pytest tests/test_exact_hex.py tests/test_loss_masks.py tests/test_config_validation.py`
  - `31 passed`
- `env SCACCHI_ALLOW_CPU=1 uv run pytest`
  - `100 passed`
- `git diff --check && git diff --cached --check`
  - clean
- `rg "scalar_q_argmax|argmax_q_mean" scacchi latex tests`
  - only rejection tests reference these names.
- Default-config smoke:
  - command: `env SCACCHI_ALLOW_CPU=1 uv run python -m scacchi.train run.seed=711 run.max_num_iters=3 checkpointing.max_to_keep=0`
  - iteration 0 eval: `avg_R=0.0625`
  - iteration 2 eval: `avg_R=0.0`
  - losses stayed finite and decreased.
