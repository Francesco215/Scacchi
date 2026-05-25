# Agent Goal: Stable Fast Hex 5x5 Training

## Objective

Configure Hex 5x5 training with `network: boardlaw_dirichlet`, a freshly
initialized train model, no distillation from external models, average reward
approximately `>= 0` against the solved 5x5 eval baseline, and convergence in
fewer than 100 iterations.

## Current Constraints

- The train model must be built from the current config.
- `checkpoints/5_solved` is allowed only as the evaluation opponent.
- The exact Hex helper is not used for 5x5; config validation only allows it
  for `board_size <= 4`.
- The current 5x5 config keeps `training.exact_hex_solver.enabled=false`.

## Current Approach

- Use `boardlaw_dirichlet` with WDL value and Q heads.
- Use fast Gumbel search with the Dirichlet value adapter for 5x5.
- For Gumbel Dirichlet self-play, train the policy head on Gumbel search
  weights while still building posterior-Q targets for the Dirichlet heads.
- Treat the noisy early policy target as auxiliary and make the WDL value/Q
  losses dominate: `policy_weight=0.05`, `value_dir_kl_weight=5.0`,
  `q_dir_kl_weight=5.0`, `value_outcome_weight=25.0`,
  `q_outcome_weight=10.0`.
- Use `posterior_argmax` self-play so the committed move is greedy under the
  posterior-best score, matching the improvement argument in `math.md`.

## Current Hex Config

- `run.max_num_iters=80`
- `env.board_size=5`
- `model.network=boardlaw_dirichlet`
- `model.num_channels=512`, `model.num_layers=8`
- `selfplay.batch_size=4096`, `selfplay.max_num_steps=25`
- `selfplay.action_source=posterior_argmax`
- `search.policy=gumbel`, `search.num_simulations=32`
- `training.batch_size=1024`, `training.replay_buffer_size=1`
- `training.learning_rate=1e-3`, `training.grad_clip_norm=1.0`
- `training.losses.policy_weight=0.05`
- `training.losses.value_dir_kl_weight=5.0`
- `training.losses.q_dir_kl_weight=5.0`
- `training.losses.value_outcome_weight=25.0`
- `training.losses.q_outcome_weight=10.0`
- `training.exact_hex_solver.enabled=false`
- `eval.interval=5`, `eval.batch_size=512`

## Evidence So Far

- Fresh Gumbel run with small WDL weights failed:
  - 30 steps: `avg_R=-0.96484375` on 512 eval games.
  - 100 steps with `learning_rate=1e-3`: `avg_R=-1.0` on 512 eval games.
- Simple central policy bias did not help: still approximately `avg_R=-1.0`.
- Posterior-tree wavefront was too slow for the target at 5x5 batch 128 and
  16 simulations.
- Reweighting away from the policy and toward WDL/Q losses helped, but the data
  split mattered more than further policy emphasis:
  - `policy_weight=0.1` regressed relative to `0.05`.
  - `policy_weight=0.02` undertrained the policy prior for Gumbel eval.
- Best current probe uses `selfplay.batch_size=4096`,
  `training.batch_size=1024`, `policy_weight=0.05`, and the heavier WDL/Q
  losses:
  - 60-step fresh run: final `avg_R=-0.03515625` on 512 eval games.
  - 80-step fresh run: final `avg_R=-0.04296875`; final rolling mean
    `-0.14375`.
  - 95-step fresh run briefly reached `avg_R=-0.0078125` at step 75, but ended
    at `avg_R=-0.11328125`.
- `selfplay.batch_size=8192` was slower and not clearly better.
- `dirichlet_thompson` now routes through the jitted Dirichlet-Q path, but a
  40-step fresh probe still ended at `avg_R=-0.9921875`.
- Switching the committed action from `posterior_sample` to
  `posterior_argmax` produced the first successful fresh 5x5 probes:
  - seed 0, 80-step fresh run, 512 eval games every 5 steps:
    - step 70: `avg_R=0.04296875`
    - step 75: `avg_R=0.07421875`
    - final step 79: `avg_R=0.0546875`
  - seed 1, 80-step fresh run, 512 eval games every 10 steps:
    - step 60: `avg_R=0.046875`
    - final step 79: `avg_R=0.015625`

## Status

The current config now has two fresh-model, no-distillation probes with final
512-game eval reward above zero before 100 iterations. Remaining verification:
run the focused/full tests after the config update.
