# Hex 5x5 Training Notebook

## 2026-05-25

Goal: make `scacchi/configs/hex.yaml` train Hex 5x5 stably and quickly with a
freshly initialized `boardlaw_dirichlet` model. The solved model is only an eval
baseline, not a source of train parameters or policy distillation.

### Implementation Notes

- `train.main` builds the train model from the current config.
- `train.main` loads `checkpoints/{board_size}_solved` only as the eval
  baseline, selected from `env.board_size`.
- `checkpointing.init_from` is not present in the Hex config.
- Exact Hex is disabled for 5x5.
- For Gumbel Dirichlet self-play, policy loss uses Gumbel search weights and
  the Dirichlet heads use posterior-Q targets.
- Added `policy_target_mode` for experiments, but the default remains `search`.

### Weighting Findings

- The original policy-heavy weighting was too aggressive for a fresh model.
- Lowering `policy_weight` and increasing value/Q losses improved the early
  reward curve, but did not reach approximately nonnegative reward.
- Best previous current-direction probe:
  - `policy_weight=0.1`
  - `value_dir_kl_weight=1.0`
  - `q_dir_kl_weight=1.0`
  - `value_outcome_weight=5.0`
  - `q_outcome_weight=2.0`
  - 512-game evals: step 10 `-0.92578125`, step 30 `-0.84375`, step 40
    `-0.82421875`.
- Updated config after reviewing the loss ratios:
  - `policy_weight=0.05`
  - `value_dir_kl_weight=5.0`
  - `q_dir_kl_weight=5.0`
  - `value_outcome_weight=25.0`
  - `q_outcome_weight=10.0`
- Increasing `policy_weight` to `0.1` regressed, so the policy target remains
  auxiliary.
- The strongest current direction was increasing the fresh self-play batch and
  reducing the learner minibatch, then committing the posterior-best action:
  - `selfplay.batch_size=4096`
  - `training.batch_size=1024`
  - `selfplay.action_source=posterior_argmax`
  - 60-step fresh run: final `avg_R=-0.03515625` on 512 eval games.
  - 80-step fresh run: final `avg_R=-0.04296875`; final rolling mean
    `-0.14375`.
  - 95-step fresh run: reached `avg_R=-0.0078125` at step 75 but ended at
    `avg_R=-0.11328125`.
- With `posterior_argmax`, the fresh 5x5 setup reached the target:
  - seed 0, 80 steps, 512 eval games every 5 steps: final
    `avg_R=0.0546875`.
  - seed 1, 80 steps, 512 eval games every 10 steps: final
    `avg_R=0.015625`.
- `selfplay.batch_size=8192` was slower and not clearly better.

### Negative Results

- Fresh Gumbel search with small WDL weights stayed near `avg_R=-1.0` through
  100 steps.
- Central-cell hand bias did not improve scratch eval.
- Winner-action policy targets made the model confident in bad self-play
  actions.
- Dirichlet-Thompson no longer goes through the posterior-tree CPU path, but a
  40-step fresh jitted probe still ended at `avg_R=-0.9921875`.
- Posterior-tree wavefront was too slow for this target under 5x5 batch 128 and
  16 simulations.

### Status

The no-distillation/fresh-model constraints are respected, and the current
posterior-argmax config has reached nonnegative reward in two fresh probes.
