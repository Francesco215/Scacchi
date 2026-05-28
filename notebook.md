# Hex Training Notebook

## 2026-05-25

Goal: train Hex 5x5 from a newly initialized `boardlaw_dirichlet` model, with
bootstrapped self-play/search learning only. Solved checkpoints are allowed only
as evaluation opponents. Target evidence is average reward approximately `>= 0`
with convergence in `<= 150` training steps, then repeat for 6x6.

Theory constraints checked against `latex/algorithms.tex`:

- Use WDL Dirichlet value/Q heads and posterior-best policy targets.
- Terminal/categorical outcomes must stay native categorical targets. They are
  not converted into synthetic Dirichlet targets.
- JAX-side categorical loss is Dirichlet density NLL at `CatPoint(z)`; any
  realized-outcome mean-probability NLL is diagnostic/auxiliary only.
- Categorical edges/nodes are absorbing and must not be searched below.
- The solved model is an eval baseline, not a source of train parameters,
  policy labels, or distillation.

Current code/config state:

- `scacchi/configs/hex.yaml` uses `network: boardlaw_dirichlet`,
  `search.policy: posterior_tree_wavefront`, and WDL3 outcomes for 5x5.
- Training initializes the train model from `build_model(... rngs=config.seed)`.
  The only solved checkpoint load is `baseline_model = from_pretrained(...)` for
  evaluation.
- To prove fresh starts, use unused seeds and verify the log line
  `No checkpoint found, starting from scratch.`

Existing W&B/local evidence before this continuation:

- 5x5 Gumbel Dirichlet probes with `selfplay.batch_size=4096`,
  `training.batch_size=1024`, `policy_weight=0.05`,
  `value_dir_kl_weight=5`, `q_dir_kl_weight=5`, and
  `selfplay.action_source=posterior_argmax` reached nonnegative final eval:
  seed 0 final `avg_R=0.05859375`, another seed-0 fresh run final
  `avg_R=0.08203125`.
- Those Gumbel probes are useful evidence about loss weighting and batch scale,
  but they do not satisfy the current paper-compliant `posterior_tree_wavefront`
  config constraint.
- 6x6 Gumbel probes at 95-100 steps were still negative, e.g. final
  `avg_R=-0.28515625` for seed 6233 with `selfplay.action_source=search_action`.
- A tiny 4x4 `posterior_tree_wavefront` smoke run succeeded mechanically, but
  it is not acceptance evidence for 5x5.

Next experimental question:

Can the native `posterior_tree_wavefront` path train 5x5 fast enough with the
same conservative Dirichlet weighting, or does it need a smaller but still stable
batch/search schedule? First run smoke tests and short fresh-start wavefront
probes, then scale only from evidence.

Wavefront diagnostics:

- 5x5 `posterior_tree_wavefront` smoke with `selfplay.batch_size=8`,
  `training.batch_size=32`, and the default c512/l8 model completed one fresh
  iteration, but took about 101 seconds. The full default batch/search schedule
  is therefore not fast enough for the requested iteration budget.
- Smaller c32 diagnostics showed valid active/terminal rows and nonnegative raw
  Dirichlet KL rows. Negative aggregate value/Q losses were explained by native
  categorical Dirichlet-density NLL, which can be negative because it is a
  density, not a probability mass. This is not reward hacking and does not imply
  a sign error in `_dirichlet_kl`.

Dirichlet-Thompson 5x5 probes:

- `kuzmg70n` (`hex5_dqthompson_seed9100_probe`) was a fresh run:
  `board_size=5`, `seed=9100`, `dirichlet_thompson`, c128/l8,
  `selfplay.batch_size=4096`, `posterior_sample`, `num_simulations=4`,
  `num_blocks=8`, `policy_samples=64`, `lr=0.003`, no outcome auxiliaries.
  It was stopped after a weak curve; W&B summary at crash was
  `avg_R=-0.7421875`, rolling-10 `-0.8953125`.
- `azcxc9xp` (`hex5_dqthompson_outcome_seed9101`) was a fresh run:
  `board_size=5`, `seed=9101`, `dirichlet_thompson`, c128/l8,
  `selfplay.batch_size=8192`, `posterior_best`, `num_simulations=4`,
  `num_blocks=8`, `policy_samples=32`, `training.batch_size=1024`,
  `lr=0.001`, `value_outcome_weight=1.0`, `q_outcome_weight=0.25`.
  Final W&B summary after 100 steps was `avg_R=0.03125`,
  rolling-10 `-0.0078125`, win rate `0.515625`, lose rate `0.484375`,
  and `train/hours=0.07754`. The run crossed nonnegative eval by the
  step-30 evaluation and stayed around parity afterward, with some noise.

Outcome-auxiliary note:

- The outcome losses above use realized self-play environment outcomes from a
  freshly initialized model. They do not use the solved model as train labels or
  a distillation source. They are, however, an auxiliary signal in addition to
  the native Dirichlet-Q bootstrapped search targets in `latex/algorithms.tex`;
  keep them explicit in configs/results rather than treating them as part of the
  pure posterior-tree algorithm.

6x6 continuation:

- `3b7bsoez` (`hex6_dqthompson_outcome_seed9102`) was a fresh 150-step run:
  `board_size=6`, `seed=9102`, `dirichlet_thompson`, c128/l8,
  `selfplay.batch_size=4096`, `posterior_sample`, `max_num_steps=64`,
  `num_simulations=4`, `num_blocks=8`, `policy_samples=32`,
  `training.batch_size=1024`, `lr=0.003`, `value_outcome_weight=0.25`,
  `q_outcome_weight=0.1`. It failed badly: final `avg_R=-0.9453125`,
  rolling-10 `-0.94140625`, win rate `0.02734375`, lose rate `0.97265625`,
  `train/hours=0.19932`.
- This negative result means the current 6x6 recipe is not stable across seeds.
  Next 6x6 experiments should either reproduce a historically successful seed
  under current code to check drift, or increase batch/evaluation confidence
  while retaining fresh initialization and no solved-model training labels.

6x6 same-seed cooked recipe test:

- `6xb4x5dx` (`hex6_dqthompson_outcome1_best_seed9103`) was an interrupted
  early copy of the outcome-heavy recipe on a new seed. It used
  `board_size=6`, `seed=9103`, `dirichlet_thompson`, c128/l8,
  `selfplay.batch_size=8192`, `posterior_best`, `max_num_steps=36`,
  `num_simulations=4`, `num_blocks=8`, `policy_samples=32`,
  `training.batch_size=1024`, `lr=0.001`, `value_outcome_weight=1.0`,
  `q_outcome_weight=0.25`. It was stopped too early at step 34; this was a bad
  experimental decision because historical non-Gumbel 6x6 curves can remain
  very negative early.
- `hoif9g7h` (`hex6_dqthompson_outcome1_best_seed9103_full`) reran the same
  recipe from scratch and was allowed to cook for all 150 steps. It confirmed
  `No checkpoint found, starting from scratch.` The curve improved from near
  `-1.0` to final `avg_R=-0.25`, rolling-10 `-0.28203125`, win rate `0.375`,
  lose rate `0.625`, but it did not meet the 6x6 target.
- Do not treat switching to historically good seeds as evidence. The next 6x6
  experiment should keep seed `9103` and change recipe variables that have a
  first-principles rationale, then let the run finish.
- `ohn3i1n9` (`hex6_dqthompson_outcome1_best_seed9103_16k64_ps64`) kept seed
  `9103` and changed data/target quality rather than seed: `batch_size=16384`,
  `max_num_steps=64`, `policy_samples=64`, `eval.batch_size=512`, same
  c128/l8 `dirichlet_thompson`, `posterior_best`, `lr=0.001`, outcome weights
  `1.0/0.25`. It was allowed to finish all 150 steps. This was a near miss, not
  a success: final and best `avg_R=-0.015625`, rolling-10 `-0.12734375`, win
  rate `0.4921875`, lose rate `0.5078125`, `train/hours=2.01277`.
- Interpreting the original target literally as "approximately >=0", this run
  is acceptable approximate 6x6 parity: it is only 8 net games below parity in a
  512-game eval. Keep the exact numbers and rolling-mean caveat visible, and do
  not describe it as proven optimal play.
- Next same-seed recipe change: enable terminal categorical targets for native
  self-play terminal edges/parents. This follows the categorical-terminal
  treatment in `latex/algorithms.tex` and uses only environment outcomes from
  self-play, not the solved baseline.
- `lsw8krxf`
  (`hex6_dqthompson_outcome1_terminal_seed9103_16k64_ps64`) enabled
  `terminal_edge_targets=true` and `terminal_parent_targets=true` on the same
  seed and otherwise kept the near-miss recipe. It finished all 150 steps and
  was worse than the no-terminal run: final and best `avg_R=-0.03125`,
  rolling-10 `-0.1109375`, win rate `0.484375`, lose rate `0.515625`,
  `train/hours=2.01087`. Terminal categorical targets are therefore not the
  next config choice for this 6x6 profile.
- Next same-seed recipe change: return to the no-terminal near-miss and use a
  learning-rate schedule, `lr=0.003` decayed to about `0.001` after iteration
  75. Rationale: the no-terminal run was still improving and reached its best
  only at the final eval; a larger early step size may move the same recipe into
  the parity band before the 150-step cutoff while late decay limits instability.
- `37mejadx`
  (`hex6_dqthompson_outcome1_lr3e3decay_seed9103_16k64_ps64`) tested that LR
  schedule on the same seed and no-terminal near-miss recipe. It improved early
  learning (`avg_R` about `-0.14` by the step-30 eval), but destabilized the
  middle of the run and finished worse than the constant-`0.001` near miss:
  final `avg_R=-0.07421875`, rolling-10 `-0.131640625`, win rate `0.462890625`,
  lose rate `0.537109375`. Do not use this LR schedule as the final config.
- Next same-seed recipe change: target self-play distribution instead of
  optimizer speed. Use `posterior_sample` with smaller outcome auxiliary weights
  (`0.25/0.1`) and `lr=0.003`, keeping the 16k/64 data scale. Rationale:
  deterministic `posterior_best` plus strong outcome auxiliaries may lock onto
  narrow self-play outcomes; sampled actions should preserve exploration while
  lower outcome weights keep the native Dirichlet-Q targets primary.
- `rvm5gevs`
  (`hex6_dqthompson_sample_lowout_seed9103_16k64_ps64`) tested that sampled,
  lower-outcome recipe on the same seed and finished all 150 steps. It was a
  clear negative ablation: final `avg_R=-0.69531`, rolling-10 `-0.74883`, win
  rate `0.15234`, lose rate `0.84766`. Do not use this recipe as the 6x6
  config choice.
- Gumbel-policy historical curves are out-of-bounds for this goal. They may
  explain why an older "good curve" looked good, but they are not acceptance
  evidence for the required non-Gumbel Boardlaw-Dirichlet experiment.
- Next same-seed recipe change: return to the best no-terminal near-miss
  (`ohn3i1n9`) and increase native Dirichlet-Q search evidence instead of
  changing seeds, action source, or optimizer schedule. Keep `seed=9103`,
  `posterior_best`, `lr=0.001`, outcome weights `1.0/0.25`, 16k/64 self-play,
  and `policy_samples=64`, but raise `search.num_blocks` from `8` to `16`.
- `bcirz9rs`
  (`hex6_dqthompson_outcome1_best_seed9103_16k64_ps64_blocks16`) was started as
  that follow-up but intentionally stopped by user request at iteration 42/150
  after accepting the approximate-parity 6x6 result. It is not completed
  evidence. The partial W&B summary at termination was `avg_R=-0.21875`,
  rolling-10 `-0.58941`, win rate `0.39062`, lose rate `0.60938`.

7x7 continuation:

- Start 7x7 from the accepted 6x6 recipe rather than the interrupted blocks16
  variant: fresh seed `9103`, `dirichlet_thompson`, `posterior_best`, c128/l8,
  `selfplay.batch_size=16384`, `max_num_steps=98`, `num_simulations=4`,
  `num_blocks=8`, `policy_samples=64`, `training.batch_size=1024`,
  `lr=0.001`, outcome weights `1.0/0.25`, and no solved-model training labels.
- `atbhc05y`
  (`hex7_dqthompson_outcome1_best_seed9103_16k98_ps64`) ran that direct 7x7
  transfer recipe from scratch for all 150 steps. It was mechanically stable
  but did not approach parity: final `avg_R=-0.62109375`, rolling-10 `-0.6375`,
  win rate `0.189453125`, lose rate `0.810546875`, `train/hours=3.05582`.
  Best point estimate was `avg_R=-0.53515625` with win rate `0.232421875`.
  This is a completed negative baseline for direct 6x6 recipe transfer.
- Next same-seed 7x7 recipe change: keep the 7x7 baseline fixed except raise
  `search.num_blocks` from `8` to `16`. Rationale: the direct transfer is
  mechanically stable and learns off the floor, but plateaus far below parity;
  increasing native Dirichlet-Q search evidence tests whether the larger board
  is target/search-quality limited without switching seeds or using Gumbel.
- `f3qdsbuw`
  (`hex7_dqthompson_outcome1_best_seed9103_16k98_ps64_blocks16`) was stopped by
  user request before completion while the objective was being reconsidered.
  Do not count it as completed evidence. The partial W&B summary was
  `avg_R=-0.9140625`, rolling-10 `-0.96630859375`, win rate `0.04296875`,
  lose rate `0.95703125`, `train/hours=0.97921`.
