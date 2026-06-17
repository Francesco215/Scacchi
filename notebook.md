# Dirichlet-Thompson Training Notebook

## Objective

- Train with `selfplay.search.kind=dirichlet_thompson`.
- Reach `eval/vs_baseline/avg_R >= 0` by `_step <= 100` against the current Go
  9x9 baseline.
- Keep TPU workers busy, but treat runs as scientific probes rather than a
  random hyperparameter lottery.
- Evaluation is cheap enough to run every training step.

## Guardrails

- Do not reward hack. A run below about `20%` win rate is not meaningful
  strength evidence, even if `avg_R` moves.
- Do not trust a metric until its code path is checked. `policy_target_entropy`
  must represent the full policy target distribution, not one sampled action or
  one search block's visit weights.
- If the training target is intentionally a proxy, log the canary from the full
  posterior-best distribution separately. For DT policy entropy/KL canaries,
  use a 128-sample posterior-best estimate over the full action posterior. Treat
  32-sample entropy as biased/noisy and do not spend 256 samples unless a later
  fixed-state audit proves 128 is still too noisy.
- Interpret entropy against the action scale. `ln(16) ~= 2.77`; values around
  that level mean the target is nearly uniform over about 16 plausible moves and
  should be treated as no-search-signal evidence.
- If a run looks different, first understand the mechanism. Check search
  semantics, target construction, metric definition, masking, loss weighting,
  and real game strength before optimizing around it.
- Before launching or pruning, write down the question: what mechanism is this
  run testing, what observation would change the next decision, and what result
  would falsify the idea? Do not run experiments whose only purpose is "maybe
  this seed/config works."
- Do not change seed as a cheap knob.
- Do not kill a run without recording window, W&B ID, remote PID, current step,
  best step-100 eval, win rate, entropy, and kill reason.
- When pruning a methodological branch, keep one or two representative runs as
  controls unless the branch is invalid, crashed, NaN, or the user explicitly
  asks to stop all of it.

## First Principles Learned

- Do not confuse "DT has no Go signal" with "DT cannot beat the PGX Go9 policy
  early." Hex did not break under shallow DT plus the AZ-style net, and Go9 DT
  learned strongly against a random evaluator. The hard failure is specifically
  early strength against the current PGX Go9 baseline.
- Sharper is not automatically better. Lower entropy or lower target prior is
  useful only if win rate rises above the random-play floor. Low entropy with
  less than about `20%` win rate should be treated as sharp-wrong evidence, not
  progress.
- Hex falsified the cheap explanations. Low depth alone and the AZ-style
  Dirichlet net alone were not sufficient to reproduce the Go collapse, so
  repeating those as Go sweeps is not informative.
- The current failure mode is most likely one of: Go9 action-space/horizon
  dilution, PGX baseline strength, DT posterior target semantics, or action
  commitment/eval semantics. New experiments should isolate one of those
  mechanisms explicitly.
- Alpha concentration growth is not the whole story. Some mature failed DT runs
  remain below `10%` win rate with `alpha_Q` around `1.5-2.8`, while the
  `5iluv1nv`-style runs saturate near the clip. Therefore "lower the
  concentration" has already been falsified as a sufficient fix. A cleaner test
  is to remove bootstrapped Dirichlet KL targets entirely and train alpha means
  from realized outcomes while keeping DT self-play and DT policy targets.
- Do not treat DT `search/path_depth_mean=0` as proof that the fixed-depth
  search is not being used. The current wrapper returns zero depth diagnostics
  for the batched DT backend because the sequential block scan does not retain a
  search tree for metrics. Code inspection shows each block still passes
  `max_depth` into MCTX search, and `num_blocks` are sequential posterior-update
  blocks.
- Do not relaunch visit/search-weight policy targets as a "new" idea. Mature
  visit-target branches already failed by step 100, so the next target-semantics
  probes must change something more fundamental than using the last block's
  visit weights.
- Pure outcome-only alpha supervision is a real test of the bootstrapping
  hypothesis, but it may be optimizer-sensitive. The first posterior-argmax
  outcome-only run went NaN around step 8. Keep the stable posterior-sample
  version running, and compare it with a lower-LR clipped version before
  concluding the idea itself is dead.
- The softened-PGX ladder is now the cleanest evidence that DT is learning some
  PGX-correlated signal before it reaches hard-baseline strength. At step 7,
  PGX temperature `5.0` is above the 20% floor while temperature `2.0` is still
  below it, with the same broad target entropy around `2.66`. This is not a
  solved policy; it says the signal exists but is weak.
- When a weak signal only appears against softened baselines, the next ablation
  should target signal amplification, not another random sweep. The live
  worker-0 test changes only self-play commitment from posterior sampling to
  posterior argmax while keeping hard-PGX sampled eval.
- `policy_loss` must be interpreted as cross entropy:
  `CE(target, model) = H(target) + KL(target || model)`. Recent DT runs have
  `policy_loss ~= 3.2`, `policy_target_entropy ~= 2.6`, and therefore
  `policy_kl_hat ~= 0.6`. The model is not merely seeing a broad target; it is
  failing to fit even that broad target on the logged batch. Added
  `train/policy_pred_entropy` so future runs distinguish a still-uniform policy
  from a sharp-wrong policy.
- The direct ablation for the CE gap is not "change seed"; it is
  optimization-vs-target-consistency. Keep DT targets fixed and reduce
  `training.batch_size` from `4096` to `1024`, increasing optimizer steps per
  self-play batch. If `policy_kl_hat` drops, fit was a bottleneck. If it stays
  high, the likely issue is target inconsistency or a moving bootstrap target.
- First patched rows from the batch-size fit ablation show
  `policy_pred_entropy ~= 3.64` while `policy_target_entropy ~= 2.88` and
  `policy_kl_hat ~= 0.77`. Early on, this is a too-uniform model, not a
  sharp-wrong one. The next useful observation is whether `policy_pred_entropy`
  moves toward the target entropy and whether `policy_kl_hat` falls over real
  training steps.
- The next diagnostic should inspect trained targets directly, not infer target
  quality only from rewards. Worker 8 now runs a base hard-PGX DT recipe with
  raw model snapshots every 10 iterations, and `scripts/diagnose_dt_signal.py`
  can load those snapshots via `--raw-snapshot` to measure PGX agreement, target
  rank, top gap, and effective support on fixed Go states.

## Theory

- The Gumbel paper proves expected-value policy improvement by coupling the
  baseline action and improved action with the same Gumbel vector, then adding a
  monotone transform of correctly evaluated action values. The theorem is about
  value, not entropy.
- For Dirichlet-Thompson search, `posterior_argmax` is greedy with respect to
  posterior optimality probability `p_T(a) = pi_search(a)`. This improves the
  score "probability this action is posterior-best" relative to any proposed
  action, but it is not automatically a scalar expected-utility guarantee.
- For Dirichlet-Thompson search, `mean_utility_argmax` is greedy with respect to
  posterior mean utility `q_hat_T(a) = U(mean(alpha_search(a)))`. This gives the
  pointwise estimated-Q guarantee `q_hat_T(A_+) >= q_hat_T(A_0)` for any
  proposed action in the candidate set, hence also in expectation under
  `q_hat_T`.
- The posterior-optimality and posterior-mean-utility rankings coincide only
  when they rank actions the same way. Do not claim scalar expected-utility
  improvement for `posterior_argmax` unless that ranking condition is checked or
  assumed.
- Completed-Q targets are currently excluded from the main loop. They are a
  different operator from the present DT posterior-best method and should not
  consume Go speedrun slots unless explicitly requested.
- Entropy is a canary, not the theorem. A value-improving policy can become more
  diffuse if the prior was overconfident on a bad action; a sharper policy can
  also be worse.

## Main Hypothesis

- The original target was `model_prior_alpha + search_evidence`.
- Because `model_prior_alpha` is produced by the same network being trained,
  this can create a confidence feedback loop: concentration rises, targets get
  too strong, and the policy/value signal collapses before strength improves.
- The fixed-target-prior intervention preserves the posterior mean of
  `model_prior + evidence` but projects concentration to
  `fixed_prior_concentration + evidence_mass`.
- For WDL targets, `3.0` is the neutral starting point: one unit of
  concentration per bucket.

## Current Go Loop

- Base: `go9x9_3`, fixed target-prior concentration mostly around `3.0`, eval
  every step.
- Policy target remains the DT posterior-best/search target.
- Active replacements should compare:
  - `mean_utility_argmax` self-play commitment as the scalar estimated-Q
    improvement action.
  - `posterior_argmax` and `posterior_sample` controls.
  - target-prior concentration `0.3`, `1.0`, and `3.0`.
  - higher posterior-best sample count where target Monte Carlo noise is a
    plausible bottleneck.
- Because the conservative loss/LR overrides have not produced a hard-PGX
  strength signal, keep the base `go9x9_3` posterior-sample branch and the
  soft-PGX ladder alive as controls before inventing more target operators.
- 2026-06-16 conclusion: stop treating this as a hyperparameter sweep. Across
  many Go variants, the best rewards stay around single-digit win rate and
  target entropy stays around `2.6`. The repeated result is that the current Go
  DT root target is not producing a useful action-ranking signal. The next work
  is fixed-position search/target diagnostics, not launching more variants.
- 2026-06-17 fixed-position diagnostic: with a fresh random model and 32 total
  root evidence units, `hex5` has target entropy `2.92` at ply 0 and `2.65` at
  ply 8, while `go9x9_3` has `3.66` and `3.64`. In effective-action terms this
  is roughly Hex `19 -> 14` versus Go `39 -> 38`. Go also explores only about
  `35%` of legal root actions at 32 blocks because the root has about `82`
  legal moves. This supports action-space/evidence dilution as the immediate
  failure mode.
- Increasing Go evidence to 128 total root evaluations did not help at
  initialization: it explored every legal action but stayed diffuse
  (`H ~= 3.66`). More weak symmetric evidence just spreads over the action
  space. A softer DT search prior is the next controlled test: projecting the
  search prior concentration to `0.3` reduced initial Go effective actions to
  about `29`; `0.1` reduced it to about `16`, but may be noisy. Test this in
  training rather than optimizing entropy directly.

## Hex Diagnostic Loop

- Hex is a control task because `hex5.yaml` and `hex6.yaml` are known-working
  DT Boardlaw-Dirichlet recipes.
- The point is not to optimize Hex. The point is to try to reproduce the Go
  failure mode in a cleaner setting.
- Keep at least one near-baseline Hex control when running Hex diagnostics.
- Compare Hex control against one-axis perturbations: `mean_utility_argmax`,
  fixed target prior, altered search shape, and policy target/metric variants.
- If Hex breaks like Go, compare `policy_target_entropy`, alpha concentration,
  and win-rate traces to isolate the shared mechanism.
- First Hex evidence: the `hex5` control reached parity by step 100, while
  `hex5 + mean_utility_argmax + fixed_prior_c3` did not. This does not prove
  mean-utility commitment is bad by itself, because fixed-prior projection was
  also changed. The next Hex probes should isolate `mean_utility_argmax` without
  fixed prior and fixed prior with the default posterior-argmax commitment.
- Second Hex evidence: `hex5 + fixed_prior_c3 + posterior_argmax` still reached
  parity, while `hex5 + mean_utility_argmax` without fixed prior did not. This
  points at `mean_utility_argmax` as a poor self-play commitment for this code
  path, despite its estimated-Q guarantee. Keep the guarantee distinction, but
  prefer posterior-best commitments for the Go speedrun until there is contrary
  evidence.
- Current Hex breakage test: try to recreate the Go failure in Hex instead of
  repeating Go sweeps. Use three axes:
  - lower search depth only: if this breaks Boardlaw Hex, terminal reach/search
    depth is the likely shared mechanism.
  - `aznet_dirichlet` only: if this breaks at normal depth, the issue is the
    network/head/trunk interaction rather than the game.
  - lower search depth plus `aznet_dirichlet`: if only this breaks, the failure
    is likely an interaction between weak network priors and shallow evidence.
- If none of these Hex probes breaks, the current Go issue is probably more
  Go-specific: large root action space, longer horizon, or baseline mismatch.
- Result: none of the three Hex probes broke. `hex5-depth1` reached best
  `avg_R=0.28125`; `hex5-az` reached best `0.140625`; `hex5-az-depth1`
  reached best `0.28125`, all by step 100. This falsifies "low depth alone"
  and "AZ net alone" as sufficient explanations for the Go collapse.
- 2026-06-17 replication: rerun the direct user-suggested breakage test rather
  than launch another Go sweep: `hex5` with `aznet_dirichlet` and DT
  `max_depth=1`, plus a same-time `boardlaw_dirichlet` depth-1 control. The
  question is whether shallow DT evidence plus the AZ-style network is enough
  to recreate the Go failure. If the AZ-depth1 Hex rerun again reaches parity,
  stop blaming low depth or AZ architecture in isolation and return to
  Go-specific action-space/horizon/baseline analysis.
- Replication result: `hex5-az-depth1` rerun `djglw9zp` finished with best step
  92 `avg_R=0.28125`, win `0.640625`; the same-time Boardlaw depth-1 control
  `rsfsa5he` finished with best step 49 `avg_R=0.28125`, win `0.640625`.
  This confirms the earlier Hex result. The Go collapse is not explained by
  shallow DT depth or the AZ-style Dirichlet net in isolation.
- Next diagnostic should stay in Go and isolate whether the failure is action
  space/horizon (`go5x5_dirichlet`) or the strong PGX eval baseline
  (`go9x9_3` evaluated against random).
- 2026-06-17 Go control after Hex/PXG sanity:
  - PGX policy-vs-policy eval is approximately symmetric, so the PGX eval loop
    is not the obvious bug.
  - Run scalar AZ Gumbel (`go9x9_gumbel`) beside Dirichlet-network Gumbel
    (`go9x9_3` with `selfplay.search.kind=gumbel`). This is a mechanism
    control, not a replacement objective. If scalar Gumbel works and Dirichlet
    Gumbel fails, suspect the Dirichlet head/loss path. If Dirichlet Gumbel
    works and DT fails, suspect the DT target/search operator. If neither works
    by step 100, the current PGX speedrun target is probably not reachable from
    scratch under these small-run conditions.
- Current Go-specific diagnostics:
  - `go5x5_dirichlet` with DT and AZ net: if this learns, Go rules are not the
    issue by themselves and 9x9 action-space/horizon remains suspect.
  - `go9x9_3` evaluated against random: if this learns while PGX eval stays
    dead, the PGX baseline is masking early progress; if it also fails, the
    9x9 training target/search signal is broken before baseline strength.
  - `go9x9_3` random eval plus search-prior concentration `0.1`: compares the
    action-space dilution intervention under an easier evaluator.
- Result so far: Go learns under DT when the evaluator is not the PGX baseline.
  `go5x5_dirichlet` is near-perfect against random by step 45, and `go9x9_3`
  beats random by step 8-11. Therefore the repeated PGX failure is not evidence
  that the DT target has no Go signal at all; it is evidence that the signal is
  too weak/slow to approach the PGX 9x9 policy by step 100.
- The random-eval comparison still supports the search-prior intervention:
  search prior `0.1` reached higher early 9x9 random-eval reward than the base
  random-eval control while also lowering target entropy.
- New eval-semantics diagnostic: current Go PGX configs evaluate `kind=policy`,
  while the stated method objective is Dirichlet-Thompson search. Run PGX eval
  with `eval.player_search.kind=dirichlet_thompson` and
  `eval.player_action_commitment_type=posterior_sample`, preserving the
  policy-eval PGX runs as controls. This is not a replacement success metric
  until explicitly accepted; it tests whether failure is policy-head strength or
  search-improved action strength.
- Eval-semantics result: DT-search PGX eval was much slower and did not rescue
  early strength; rows stayed around `1-2%` win rate. Do not spend long jobs on
  per-step DT-search eval unless there is a stronger reason.
- Lowering search prior below `0.1` made targets sharper but not stronger
  against PGX. Treat this as sharp-wrong evidence unless later rows contradict
  it.
- `5iluv1nv`-anchored method check: reproduce the user-pointed Gumbel recipe
  beside DT swaps rather than keep sweeping from the failing small-batch setup.
  Compare current-code Gumbel, DT with `search_action`, and DT with
  `posterior_argmax` while holding batch sizes, losses, depth, and eval cadence
  fixed. If the Gumbel anchor moves and both DT variants remain dead, the
  bottleneck is the DT search/target operator. If only one DT commitment mode
  moves, the action commitment semantics are the immediate target. If none move
  by step 100, the `5iluv1nv` recipe is not a step-100 solution in the current
  codebase even though the historical run became useful later.
- Scalar Gumbel action-selection check: the scalar AZ Gumbel control is the only
  live PGX run above the 20% win-rate floor so far, but it is evaluated by
  sampled policy action. Run the same training with greedy/argmax player eval
  against the same sampled PGX baseline. If greedy eval is much stronger, the
  model has a useful mode before the sampled policy is good, and policy
  sharpening/action-selection semantics matter. If greedy eval is not stronger,
  the scalar Gumbel control is still genuinely below PGX strength.
- 2026-06-17 fixed-state DT target audit:
  - Increasing Go9 DT root evidence from 32 to 128 explored almost all legal
    root actions but did not make the posterior-best target align with PGX.
    At 128 one-step blocks, target entropy stayed around `3.81-3.97`
    (`45-53` effective actions), and PGX top-action agreement remained `0`.
  - Using deeper per-block search (`num_simulations=4`, `num_blocks=32`) was
    similarly broad: entropy around `3.78-3.92`, PGX top-action agreement `0`.
  - Therefore the next Go intervention should not be another budget or
    concentration sweep. The sharper question is whether training on the
    posterior-best policy target is the bottleneck.
  - Later small-batch rerun against `go_9x9_v0` confirmed the random-init
    picture: `H(target)=3.82..3.98`, top gap `<0.01`, and PGX top agreement
    `0..12.5%`. This is not trained-checkpoint evidence; the available
    `checkpoints/9_solved` are Hex scalar checkpoints, not compatible Go9 DT
    checkpoints.
- New mathematically aligned target test:
  - Added `training.losses.policy_target_mode=mean_utility_argmax`, which
    trains the policy on the posterior mean-utility argmax action. This matches
    the scalar estimated-Q improvement view more directly than posterior-best
    probabilities and does not use completed-Q smoothing.
  - Launched w2 `dt-muarg-5ilu-w2` / `g4sfoj08`: same 5ilu DT `search_action`
    recipe as w4, but with `policy_target_mode=mean_utility_argmax`.
  - Launched w7 `dt-muarg-outcome-w7` / `zhqt20au`: outcome-only alpha
    supervision plus the same mean-utility policy target.
  - Decision: if w2 separates from w4, the policy target object was the
    bottleneck. If only w7 moves, target object and bootstrapped KL interact.
    If neither moves, the scalar argmax target is not enough and the target
    quality/search operator remains suspect.
- Hex8 bridge test:
  - Hex5 did not recreate the Go collapse; Hex8 has 64 root actions and a
    cleaner rule set than Go9. This is a better environment-independent test of
    action-space dilution.
  - Recycled mature Go failures on workers `9,14` and launched
    `hex8-dt-random-w9w14` on topology pair `topo2-8` with
    `eval.baseline=random`, policy-only random baseline search, and reduced
    single-run batch sizes.
  - Decision: if Hex8 learns random quickly, large action count alone is not
    enough to explain Go. If Hex8 stalls, compare its entropy/alpha traces to
    Go before launching more Go PGX sweeps.
- 2026-06-17 metric correction:
  - The first `mean_utility_argmax` runs accidentally logged
    `policy_target_entropy` from the one-hot proxy target. That is invalid as a
    canary because it says the training proxy is sharp, not that the full
    posterior-best distribution is sharp.
  - The code now keeps the one-hot mean-utility argmax as the policy training
    target, while `policy_metric_tgt` remains the full posterior-best
    distribution. Old W&B IDs `g4sfoj08` and `zhqt20au` are invalid for metric
    conclusions; corrected runs are `06hjdp8v` and `fikpoeb1`.
  - Added a third corrected target-object control on worker 6:
    `clh6t6vo`, the base `go9x9_3` posterior-sample recipe plus
    `mean_utility_argmax`. This answers whether the scalar target helps under
    the least-bad mature base DT branch, not only under the 5ilu or outcome-only
    recipes.
- 2026-06-17 Hex8 result:
  - `hex8-dt-random-w9w14` / `brc1dycu` was already around
    `avg_R=0.64`, win `0.82` by step 15 against random.
  - This falsifies the simple hypothesis "64 legal actions alone recreate the
    Go failure." Go still may fail from action-space plus horizon plus PGX
    baseline strength, but action count by itself is not enough.
- 2026-06-17 scalar Gumbel control:
  - `rnwartle` eventually crossed parity after the decision horizon: step 119
    `avg_R=0.052734`, win `0.526367`.
  - It missed the objective because best<=100 stayed negative at step 96
    `avg_R=-0.234375`, win `0.382812`, and it is not a DT-search training run.
  - Scientific implication: the PGX evaluator is not unreachable in this
    runtime, but DT needs a target/search improvement rather than another
    small loss-weight sweep.
- 2026-06-17 soft-PGX ladder:
  - Recycled mature posterior-sample runs that had hundreds of steps and still
    stayed below `9%` win by the step-100 decision rule.
  - Launched `dt-pgxT2-w10` and `dt-pgxT5-w12`, which keep the base DT training
    recipe but evaluate against PGX policy sampled at temperatures `2.0` and
    `5.0`.
  - Also launched `muarg-random-w8`, a Go9 random-eval sanity check for the
    corrected `mean_utility_argmax` target.
  - Question: does the current DT policy become PGX-correlated but too weak, or
    does it fail even against a softened PGX policy? The ladder is diagnostic;
    hard PGX remains the objective.
- 2026-06-17 greedy-eval check:
  - Recycled the last two mature posterior-sample tails, which were still below
    `10%` win by step 100 and still weak after hundreds of steps.
  - Launched base DT and `mean_utility_argmax` DT with hard PGX evaluation but
    greedy player policy action selection.
  - Question: is the learned policy hiding a useful top action behind sampled
    eval, or is the policy head itself failing to rank PGX-level moves? Greedy
    eval can diagnose action-selection semantics, but hard PGX parity by step
    100 remains the success condition.
- 2026-06-17 alpha-growth update:
  - Hex8 random eval reached almost perfect reward by step 43 while
    `alpha_Q` grew to about `15`.
  - Therefore alpha concentration growth is not, by itself, a sufficient
    explanation for failure. It can still be part of a bad feedback loop on
    Go9, but the deeper question is whether the target/search operator ranks
    PGX-relevant actions before the model becomes confident.
- 2026-06-17 Hex8 baseline-strength ablation:
  - `hex8-dt-random-w9w14` / `brc1dycu` finished with near-perfect random
    baseline reward by step 50, and best step-100 reward was `1.0` at step 46.
    This falsifies "large clean action space alone" as the explanation.
  - The next Hex question is baseline strength, not board size. Reuse the same
    Hex8 DT recipe against the solved `checkpoints/8_solved` baseline.
  - Because the solved checkpoint is scalar, the baseline search must be Gumbel
    with `baseline_action_commitment_type=search_action`; evaluating that
    scalar checkpoint with DT search is an invalid config, not a failed
    scientific experiment.
  - Decision rule: if Hex8 learns the solved checkpoint baseline while Go9
    still stalls against PGX, focus on Go/PGX-specific target alignment. If
    Hex8 also stalls against the solved checkpoint, baseline strength is an
    environment-independent mechanism and a curriculum/ladder becomes a more
    principled next intervention than another Go loss-weight sweep.

## Decision Rules

- A run succeeds only if W&B shows `eval/vs_baseline/avg_R >= 0` at some
  `_step <= 100`.
- If a run hits the target, preserve its exact config and W&B ID before
  changing anything.
- If every live Go run remains below the win-rate floor and entropy is still
  near `ln(16)`, stop sweeping loss weights and return to search/target
  semantics.
- If a run has low entropy but remains below the win-rate floor, assume the
  target may be sharp-but-wrong until proven otherwise.
- Recycle mature failures into either a tighter Go DT probe or the Hex control
  set. Avoid killing young runs just because early reward is noisy.
- Do not wipe an entire live cohort based only on an early negative read; leave
  a small tail to show whether the hypothesis was merely slow.
