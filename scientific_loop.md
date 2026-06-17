# Scientific Loop For DT Go9 Training

## Read This First

Every time this file is opened, ask these questions before acting:

- What is the most important information i want to discover right now?
- What are the things i can't explain?
- What hypothesis would explain the current evidence with the fewest moving
  parts?
- What observation would falsify that hypothesis?
- Which experiment changes one mechanism while preserving the relevant
  controls?
- Have I already run similar experiments? what were their outcome? Is it worth running this one? will this experiment give me new information? if not, come up with a different experiemnt.
- What metric could be reward-hacked or misread?
- What would make me stop this branch and recycle the worker?
- Is this experiment teaching something new, or is it the same experiment with a
  new name?
- A good sanity check for any hypothesis is to ablate it in more than one environment. 
- If an algorithm is not working in one environment and working in the other, figure out why. On what conditions would the algo break in the working environment? if you can go from working -> not working in one env, maybe you can go from not working -> working in the other env.

## Objective

- Train with `selfplay.search.kind=dirichlet_thompson`.
- Reach `eval/vs_baseline/avg_R >= 0` by `_step <= 100` against the current
  PGX Go9 baseline.
- Keep TPU workers busy, but do not fill them with duplicate sweeps.
- Treat `policy_target_entropy` as a canary, not an objective. Aim for targets
  that become sharp because they rank good actions, not because entropy was
  optimized directly.

## Scientific Loop

1. State the hypothesis in falsifiable form.
2. Identify the causal mechanism it claims.
3. Choose one intervention that isolates that mechanism.
4. Preserve controls that rule out simpler explanations.
5. Define the decision rule before launching.
6. Run long enough for the decision rule to be meaningful.
7. Record the result, including negative evidence.
8. Update the hypothesis before launching the next experiment.

## Valid Hypotheses

A valid hypothesis should name a mechanism, not a vibe.

Good:

- "DT posterior-best targets are too diffuse in Go9 because root evidence is
  diluted over many legal actions."
- "Bootstrapped Dirichlet KL targets self-reinforce wrong alpha means."
- "The model learns cheap anti-random heuristics but DT targets do not rank
  PGX-level moves."

Bad:

- "Maybe this seed works."
- "Maybe more weight fixes it."
- "Entropy is lower, so it must be better."

## Valid Experiments

An experiment is valid only if it answers a question.

- One primary mechanism should change at a time.
- Controls should stay alive unless they are invalid, crashed, NaN, or already
  past the decision boundary.
- The result must have an interpretation for both success and failure.
- Prefer fixed-state diagnostics before spending TPU hours when the question is
  about target quality.
- Do not rerun branches already falsified by mature runs: visit/search-weight
  policy targets, target-prior concentration alone, search-prior concentration
  alone, and shallow-depth Hex breakage.

## Measurement Guardrails

- A run below about `20%` win rate against PGX is not meaningful strength
  evidence.
- `ln(16) ~= 2.77`; target entropy around that value means roughly 16 effective
  actions and should be treated as weak search signal unless win rate improves.
- Low entropy with bad win rate is sharp-wrong, not progress.
- `search/path_depth_* = 0` on DT runs is currently a diagnostics limitation:
  the sequential block wrapper does not retain search trees for depth metrics.
  It is not evidence that `max_depth` is ignored.
- W&B eval every step is cheap enough; use exact `_step <= 100` for success.

## Environment-Independent Principles

- Beating random is weak evidence. Random baselines can be beaten by shallow
  heuristics that do not imply strength against a competent policy.
- Target quality and model fit are separate failure modes.
- Sharper targets are useful only if they rank good actions.
- Self-bootstrapping can create confidence feedback when the model's own
  posterior supplies the target evidence.
- Agreement audits are general: compare search targets against stronger
  policies, rollouts, one-step value rankings, or later successful policies.
- Budget scaling is general: if fixed-state target quality improves with budget,
  evidence dilution is likely; if not, the value/prior/search operator is
  suspect.

## Environment-Dependent Facts

- The current hard baseline is PGX Go9.
- Go9 has a large legal root action space; Hex5 did not reproduce the failure
  under low depth or the AZ-style Dirichlet net.
- The `20%` win-rate floor is empirical for this PGX Go9 setup.
- Pass behavior, komi, captures, game length, and terminal frequency are Go
  mechanisms and must not be assumed to transfer directly to Hex.

## Current Main Hypotheses

| hypothesis | current evidence | decisive next evidence |
| --- | --- | --- |
| PGX baseline gap, not no Go signal | DT learns Go9 strongly against random; scalar Gumbel improves against PGX but missed step 100 | Ladder eval against intermediate opponents shows smooth improvement |
| DT target quality is poor on Go9 | Initial fixed-state audit: DT target has about 36-53 effective actions and low agreement with PGX/value top actions | Fixed-state audit across budgets/checkpoints shows whether agreement improves |
| Alpha self-bootstrapping poisons learning | Some runs saturate alpha near clip, but mature failures also exist at low alpha concentration | Outcome-only alpha runs improve or fail while concentration stays controlled |
| Action commitment matters | `posterior_argmax` variants sometimes look slightly different early, but still below floor | Same recipe differs materially by commitment mode under stable training |

## Active Experiments

Snapshot: 2026-06-17 around 11:35 UTC.

| worker | run | W&B | role | latest status | decision rule |
| --- | --- | --- | --- | --- | --- |
| 0 | `go-gumbel-scalar-w0` | `rnwartle` | Scalar AZ Gumbel control against PGX | step 109 `avg_R=-0.214844`, win `0.392578`; best<=100 step 96 `avg_R=-0.234375` | Confirms PGX target is hard but learnable; not a DT success |
| 11 | `go-gumbel-argmaxeval-w11` | `rr56jqv3` | Same scalar Gumbel family with greedy player eval | step 29 `avg_R=-0.826172`, win `0.086914` | If greedy eval beats sampled eval, action selection matters; currently no rescue |
| 1 | `go5ilu-gumbel-w1` | `gsobsq6d` | Current-code reproduction of user-pointed `5iluv1nv` recipe | step 32 `avg_R=-0.985352`, win `0.007324`, entropy `2.71444`, `alpha_Q=7.90072` | If this stays dead, historical recipe is not a current step-100 anchor |
| 4 | `go5ilu-dt-w4` | `7agypa11` | `5iluv1nv` recipe with DT `search_action` | step 32 `avg_R=-0.984375`, win `0.007812`, entropy `2.69772`, `alpha_Q=7.95077` | Compare against w1 and w3 to isolate DT swap and commitment |
| 3 | `go5ilu-dt-parg-w3` | `xt0iahzu` | `5iluv1nv` DT with `posterior_argmax` | step 22 `avg_R=-0.945312`, win `0.027344`, entropy `2.52344`, `alpha_Q=7.87519` | If it separates from w4, commitment matters; still below floor |
| 5 | `go-outcome-dt-psamp-w5` | `2v2u9q7x` | Outcome-only alpha supervision, DT posterior-sample | step 18 `avg_R=-0.998047`, win `0.000977`, entropy `2.75810`, `alpha_Q=2.97528` | Tests whether removing bootstrapped KL fixes learning; stable but no signal yet |
| 6 | `go-outcome-lr1e3-clip-w6` | `3ido45kf` | Stabilized outcome-only alpha supervision | step 5 `avg_R=-0.994141`, win `0.002930`, entropy `2.45847`, `alpha_Q=3.00225` | If stable and stronger than w5, outcome-only is optimizer-sensitive |

## Answered Or Recycle-Eligible Branches

These runs are still occupying workers but have passed the step-100 decision
boundary and remain below the win-rate floor. They should not be extended as a
scientific branch unless a new mechanism is added.

| worker | run | W&B | best<=100 | latest |
| --- | --- | --- | --- | --- |
| 2 | `psamp-base-c1-lr1e3-w2` | `h0cy57r6` | step 38 `avg_R=-0.835938`, win `0.082031` | step 330 `avg_R=-0.886719`, win `0.056641` |
| 7 | `psamp-2x32-c1-w7` | `fj3gihh0` | step 57 `avg_R=-0.867188`, win `0.066406` | step 151 `avg_R=-0.935547`, win `0.032227` |
| 8 | `psamp-base-c0p3-w8` | `5zvikpyx` | step 38 `avg_R=-0.822266`, win `0.088867` | step 350 `avg_R=-0.841797`, win `0.079102` |
| 9 | `go-spc0p3-b64-w9` | `qyxxjqy1` | step 89 `avg_R=-0.845703`, win `0.077148` | step 130 `avg_R=-0.880859`, win `0.059570` |
| 10 | `psamp-base-c1-bs2k-w10` | `nj2899pg` | step 38 `avg_R=-0.826172`, win `0.086914` | step 173 `avg_R=-0.878906`, win `0.060547` |
| 12 | `psamp-base-c1-ps128-w12` | `ktvs1w9z` | step 56 `avg_R=-0.837891`, win `0.081055` | step 290 `avg_R=-0.851562`, win `0.074219` |
| 13 | `psamp-base-c3-w13` | `kbveqsxs` | step 38 `avg_R=-0.814453`, win `0.092773` | step 357 `avg_R=-0.818359`, win `0.090820` |
| 14 | `go-spc0p1-b64-w14` | `28efa2sx` | step 33 `avg_R=-0.812500`, win `0.093750` | step 129 `avg_R=-0.917969`, win `0.041016` |
| 15 | `psamp-base-c1-b64-w15` | `4vubg1j9` | step 38 `avg_R=-0.822266`, win `0.088867` | step 171 `avg_R=-0.875000`, win `0.062500` |

## Current Diagnostic Priority

Run fixed-state target audits before launching more Go sweeps:

- DT target entropy and top action gap.
- DT agreement with PGX top action.
- DT rank/mass assigned to PGX top action.
- DT agreement with one-step value-greedy action.
- Budget scaling: `num_blocks`, `num_simulations`, `search_prior_concentration`.
- Checkpoints if available: target quality should improve before policy strength
  improves.

If fixed-state DT targets do not align with PGX/value references even at larger
budgets, then the next code-level work should focus on the DT target/search
operator, not another training hyperparameter.
