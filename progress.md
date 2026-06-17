# Current Progress

## Goal

Train a Go 9x9 model with `selfplay.search.kind=dirichlet_thompson` that reaches
`eval/vs_baseline/avg_R >= 0` by `_step <= 100` against the current baseline.

## Current Status

- No W&B run has verified the target yet.
- TPUs are managed through tmux session `0`.
- The current main direction is DT posterior-best/search targets with
  `mean_utility_argmax`, `posterior_argmax`, and `posterior_sample` commitment
  controls.
- Completed-Q Go jobs were stopped and should not be relaunched unless
  explicitly requested.
- Future pruning should keep one or two representative runs from a branch unless
  the runs are invalid, crashed, NaN, or explicitly stopped by the user.

## Important Code Changes

- Added optional fixed target-prior concentration for Dirichlet posterior
  targets. This preserves posterior means while avoiding inherited model
  concentration in targets.
- Added separate `policy_metric_tgt` so metrics can measure the full
  posterior-best distribution even when the training target is a proxy.
- For DT `policy_samples=0`, `policy_target_entropy` now uses a 32-sample
  posterior-best estimate over the full action posterior.
- Added `mean_utility_argmax` commitment:
  `argmax_a U(mean(alpha_search(a)))`.
- Added focused tests for the fixed-prior target, metric-target separation,
  completed-policy metric semantics, and mean-utility commitment.

## Validation

- `python -m py_compile scacchi/types.py scacchi/play.py scacchi/play_search.py scacchi/loss.py`
- Focused pytest suites for config validation, loss metric target separation,
  DT search output semantics, and mean-utility commitment passed.
- CPU smoke passed for `go9x9_3` with `mean_utility_argmax` and fixed target
  prior.
- CPU smoke passed for `hex5` with `mean_utility_argmax`.

## Key Negative Evidence

- Fixed target-prior concentration controlled alpha concentration but did not
  by itself produce strong Go policy targets. Many runs stayed around
  `policy_target_entropy ~= 2.5..2.6`, close to `ln(16)`.
- Visit/proxy targets can look sharp while reward remains near zero wins. This
  exposed a metric bug: entropy was sometimes measuring a proxy rather than the
  full posterior-best target.
- Reward below about `20%` win rate has not been treated as strength evidence.
- Completed-Q runs were stopped for methodological reasons: they test a
  different operator from the current DT posterior-best method.

## Killed Completed-Q Runs

Stopped on 2026-06-16 after the user objected to using completed-Q as the main
method. All were young Go runs; kill reason was methodological mismatch.
In hindsight, killing all ten was too aggressive; future methodological cleanup
should leave a small exploratory tail unless the user explicitly asks to stop
everything.

| window | name | W&B | PID | last evidence |
| --- | --- | --- | --- | --- |
| 0 | `cq-s0p5-samp-w0` | `2fpcj3zc` | `1750432` | step 4, best `avg_R=-0.982422`, win `0.008789`, entropy `3.397996` |
| 1 | `cq-s1-samp-w1` | `8wyc3c5l` | `714993` | step 4, best `avg_R=-0.982422`, win `0.008789`, entropy `3.305237` |
| 2 | `cq-s2-samp-w2` | `8lruar63` | `769329` | step 4, best `avg_R=-0.892578`, win `0.053711`, entropy `3.577807` |
| 3 | `cq-s5-samp-w3` | `pwplv4g7` | `757230` | step 4, best `avg_R=-0.875000`, win `0.062500`, entropy `3.243258` |
| 4 | `cq-s10-samp-w4` | `og5b6jvg` | `674184` | step 4, best `avg_R=-0.880859`, win `0.059570`, entropy `3.228669` |
| 5 | `cq-s0p5-arg-w5` | `l3jbwu91` | `690930` | step 4, best `avg_R=-0.865234`, win `0.067383`, entropy `4.122842` |
| 6 | `cq-s1-arg-w6` | `fqlbvgy7` | `695002` | step 4, best `avg_R=-0.833984`, win `0.083008`, entropy `2.205148` |
| 7 | `cq-s2-arg-w7` | `lagu2kcx` | `686936` | step 4, best `avg_R=-0.970703`, win `0.014648`, entropy `2.199901` |
| 8 | `cq-s5-arg-w8` | `djd1wu6m` | `734890` | step 4, best `avg_R=-0.916016`, win `0.041992`, entropy `2.192782` |
| 12 | `cq-s10-arg-w12` | `db2x8alg` | `742442` | step 4, best `avg_R=-0.933594`, win `0.033203`, entropy `3.203135` |

The tmux launcher cleanup killed those exact PIDs and each worker reported
`READY`.

## Active Replacement Grid

Launched 2026-06-16 in tmux session `0`.

| window | name | W&B | PID | purpose |
| --- | --- | --- | --- | --- |
| 0 | `meanq-c3-w0` | `cav8inbo` | `1770952` | Go, fixed prior `3.0`, `mean_utility_argmax` |
| 1 | `meanq-c1-w1` | `cy8yixbx` | `720548` | Go, fixed prior `1.0`, `mean_utility_argmax` |
| 2 | `meanq-c0p3-w2` | `0047umkl` | `775017` | Go, fixed prior `0.3`, `mean_utility_argmax` |
| 3 | `meanq-ps128-c3-w3` | `csijteuk` | `762709` | Go, fixed prior `3.0`, 128 posterior-best policy samples |
| 4 | `meanq-8x8-c3-w4` | `xgi8yrxn` | `679789` | Go, fixed prior `3.0`, 8 simulations x 8 blocks |
| 5 | `parg-c3-w5` | `jb65ak1c` | `696470` | Go posterior-argmax control |
| 6 | `psamp-c3-w6` | `83p19sgv` | `700641` | Go posterior-sample control |
| 7 | `meanq-vout-c3-w7` | `hafw9h2c` | `692467` | Go, stronger outcome losses with `mean_utility_argmax` |
| 8 | `hex5-control-w8` | `nsqzd0gd` | `742141` | Hex 5x5 known-working DT control |
| 12 | `hex5-meanq-w12` | `t58el4lm` | `748214` | Hex 5x5 `mean_utility_argmax` diagnostic |

## Next Check

- Confirm all replacement runs have W&B IDs and remote Python PIDs.
- Do not judge Go replacements before they have real step/eval history.
- Recycle only mature failures, and keep the notes concise.

## Status 2026-06-16 14:00 UTC

- No W&B hit yet.
- Active replacement grid is live in tmux. W&B history has not fully populated
  for the replacement IDs yet, but pane tails show Go jobs at step `0..1` and
  Hex jobs at step `0..1`.
- Old visit-target Go runs are still below step 100 and remain weak:
  - w13 `icwypjek`: step 90, best `avg_R=-0.960938`, win `0.019531`.
  - w14 `mk9iff3v`: step 94, best `avg_R=-0.925781`, win `0.037109`.
  - w15 `f4xb84rv`: step 81, best `avg_R=-0.966797`, win `0.016602`.
- Decision: do not recycle again until at least the near-horizon old visit runs
  cross step 100, unless a worker crashes or a run NaNs.

## Status 2026-06-16 14:32 UTC

- No Go W&B hit yet.
- Hex control provided useful positive evidence:
  - `hex5-control-w8` / W&B `nsqzd0gd` finished 100 iters and crossed parity
    multiple times, best step 63 `avg_R=0.117188`, win rate `0.558594`,
    entropy `2.045521`.
  - `hex5-meanq-w12` / W&B `t58el4lm` finished 100 iters but did not reach
    parity, best step 31 `avg_R=-0.242188`, win rate `0.378906`, entropy
    `2.051467`.
- Interpretation: Hex confirms the base DT recipe can work on a different env,
  while `mean_utility_argmax + fixed_prior_c3` underperformed there. This
  confounds action commitment and fixed-prior projection, so run disentangling
  probes before drawing a conclusion.
- Mature Go visit-target failures recycled:
  - w13 `visit32-out-c3-w13` / W&B `icwypjek`, PID `695334`, step 100, best
    `avg_R=-0.960938`, win `0.019531`, entropy `2.629237`.
  - w14 `visit4x8-out-c3-w14` / W&B `mk9iff3v`, PID `676586`, step 104, best
    `avg_R=-0.925781`, win `0.037109`, entropy `1.242045`.
  Both were below the 20% win-rate floor and missed the step-100 target. The
  launcher killed the exact PIDs and workers returned `READY`.

## Active Additions 2026-06-16 14:35 UTC

| window | name | W&B | PID | purpose |
| --- | --- | --- | --- | --- |
| 8 | `hex5-meanq-nofix-w8` | `72t66c35` | `748335` | Hex isolate `mean_utility_argmax` without fixed prior |
| 12 | `hex5-c3-parg-w12` | `gv0n8tj7` | `754395` | Hex isolate fixed prior `3.0` with default posterior-argmax |
| 13 | `meanq-nofix-w13` | `rn38bg4l` | `720715` | Go `mean_utility_argmax` without fixed-prior projection |
| 14 | `psamp-c1-w14` | `pie1v3x5` | `702482` | Go posterior-sample with softer fixed prior `1.0` |

## Status 2026-06-16 15:07 UTC

- No Go W&B hit yet.
- Hex disentanglers finished:
  - `hex5-meanq-nofix-w8` / `72t66c35`: best step 92
    `avg_R=-0.054688`, win `0.472656`, entropy `2.142132`.
  - `hex5-c3-parg-w12` / `gv0n8tj7`: best step 66
    `avg_R=0.132812`, win `0.566406`, entropy `1.920015`.
- Interpretation: fixed-prior projection does not break Hex when paired with
  posterior-argmax. `mean_utility_argmax` underperforms on Hex even without
  fixed-prior projection. For Go, posterior-best commitments are now the main
  line; keep mean-utility Go runs only as ongoing negative/contrast evidence.
- Recycled mature Go failure:
  - w15 `visit4x8-arg-c3-w15` / `f4xb84rv`, PID `671928`, step 102, best
    `avg_R=-0.966797`, win `0.016602`, entropy `1.247093`. Below the 20%
    win-rate floor and missed step-100 target. The launcher killed PID
    `671928` and worker 15 returned `READY`.
- Reused finished Hex slots plus w15 for Go posterior-best probes:

| window | name | W&B | PID | purpose |
| --- | --- | --- | --- | --- |
| 8 | `parg-c1-w8` | `8r943om0` | `754078` | Go posterior-argmax with fixed prior `1.0` |
| 12 | `parg-nofix-w12` | `fo8e02kz` | `760334` | Go posterior-argmax without fixed-prior projection |
| 15 | `parg-ps128-c3-w15` | `rapaco4n` | `696653` | Go posterior-argmax, fixed prior `3.0`, 128 posterior-best samples |

## Status 2026-06-16 15:45 UTC

- No Go W&B hit yet.
- Current best live Go evidence remains below the win-rate floor:
  - `psamp-c3-w6` / `83p19sgv`: best step 31 `avg_R=-0.830078`, win
    `0.084961`, entropy `2.607780`.
  - `psamp-c1-w14` / `pie1v3x5`: best step 20 `avg_R=-0.855469`, win
    `0.072266`, entropy `2.614440`.
- Mature old visit-target failures:
  - killed w9 `visit2x16-arg-c3-w9` / `idjf2kwq`, PID `707867`, step 102,
    best step 22 `avg_R=-0.978516`, win `0.010742`, entropy `0.634309`.
  - killed w11 `visit4x8-arg-p4-c3-w11` / `v95zq452`, PID `714585`, step
    102, best step 40 `avg_R=-0.947266`, win `0.026367`, entropy `1.249280`.
  - kept w10 `visit8x4-arg-c3-w10` / `e8rg7b1r` running as the representative
    old-branch tail control; it is weak but was the least bad mature visit run
    so far.
- Reused w9 and w11 for posterior-best Go probes:

| window | name | W&B | PID | purpose |
| --- | --- | --- | --- | --- |
| 9 | `parg-c0p3-w9` | `6oxmbu4g` | `734367` | Go posterior-argmax with fixed prior `0.3` |
| 11 | `psamp-nofix-w11` | `yvrvdpe6` | `739228` | Go posterior-sample without fixed-prior projection |

## Status 2026-06-16 16:15 UTC

- No Go W&B hit yet.
- No recycle on this pass. The only mature weak run is the deliberately kept
  old visit-target tail control w10 `e8rg7b1r`; the rest of the current Go grid
  is still below step 100.
- Best live Go rows are still below the 20% win-rate floor:
  - `psamp-c3-w6` / `83p19sgv`: best step 31 `avg_R=-0.830078`, win
    `0.084961`, entropy `2.607780`.
  - `psamp-c1-w14` / `pie1v3x5`: best step 20 `avg_R=-0.855469`, win
    `0.072266`, entropy `2.614440`.
  - `psamp-nofix-w11` / `yvrvdpe6`: young, best step 5
    `avg_R=-0.863281`, win `0.068359`, entropy `2.643490`.
- Interpretation unchanged: posterior-sample variants are the least bad Go
  probes so far, but they are still weak and too diffuse to count as real
  strength evidence.

## Status 2026-06-16 17:16 UTC

- No Go W&B hit yet.
- No recycle on this pass. Several main-cohort runs are at steps `60..75`, so
  the next useful pruning decision should wait until they cross step 100.
- Best live Go rows are still below the 20% win-rate floor:
  - `psamp-c3-w6` / `83p19sgv`: best step 31 `avg_R=-0.830078`, win
    `0.084961`, entropy `2.607780`.
  - `psamp-nofix-w11` / `yvrvdpe6`: young, best/latest step 31
    `avg_R=-0.835938`, win `0.082031`, entropy `2.610410`.
  - `psamp-c1-w14` / `pie1v3x5`: best step 38 `avg_R=-0.845703`, win
    `0.077148`, entropy `2.609320`.
- The current pattern still points to posterior-sample being less bad than
  posterior-argmax or mean-utility on Go, but all of it remains far from a
  useful strength signal.

## Status 2026-06-16 18:17 UTC

- No Go W&B hit yet.
- No recycle on this pass. Several runs are just short of the step-100 decision
  horizon, so wait for completed step-100 evidence rather than pruning on
  nearly mature curves.
- Best live Go rows remain below the 20% win-rate floor:
  - `psamp-c3-w6` / `83p19sgv`: best step 31 `avg_R=-0.830078`, win
    `0.084961`, entropy `2.607780`.
  - `meanq-c3-w0` / `cav8inbo`: best step 93 `avg_R=-0.845703`, win
    `0.077148`, entropy `2.530760`. This late, lower-entropy point is worth
    watching, but it is still not a trustworthy strength signal.
  - `psamp-c1-w14` / `pie1v3x5`: best step 38 `avg_R=-0.845703`, win
    `0.077148`, entropy `2.609320`.

## Recycle 2026-06-16 18:52 UTC

- Killed mature weak runs while preserving representative controls:
  - w1 `meanq-c1-w1` / `cy8yixbx`, PID `720548`, step 108, best step 28
    `avg_R=-0.923828`, win `0.038086`, entropy `2.668870`; latest alpha
    metrics were NaN after missing the horizon.
  - w5 `parg-c3-w5` / `jb65ak1c`, PID `696470`, step 110, best step 21
    `avg_R=-0.888672`, win `0.055664`, entropy `2.614480`.
  - w7 `meanq-vout-c3-w7` / `hafw9h2c`, PID `692467`, step 102, best step 7
    `avg_R=-0.927734`, win `0.036133`, entropy `2.675740`.
- Kept `meanq-c3-w0` as the late mean-utility control, `psamp-c3-w6` as the
  best mature posterior-sample control, and `visit8x4-arg-c3-w10` as the old
  visit-target tail control.
- Refilled with baseline-loss posterior-sample probes:

| window | name | W&B | PID | purpose |
| --- | --- | --- | --- | --- |
| 1 | `psamp-base-w1` | `bclctcdb` | `742488` | Go posterior-sample using the base `go9x9_3` loss/LR recipe |
| 5 | `psamp-base-c1-w5` | `drpxuvza` | `719363` | Base loss/LR recipe plus fixed target prior `1.0` |
| 7 | `psamp-2x32-c1-w7` | `fj3gihh0` | `715407` | Base loss/LR recipe, fixed prior `1.0`, two DT simulations |

## Status And Recycle 2026-06-16 19:55 UTC

- No Go W&B hit yet.
- Baseline-loss posterior-sample probes are young but not immediately worse
  than the conservative-loss branch:
  - `psamp-base-w1` / `bclctcdb`: step 19, best step 5
    `avg_R=-0.867188`, win `0.066406`, entropy `2.680100`.
  - `psamp-base-c1-w5` / `drpxuvza`: step 19, best step 12
    `avg_R=-0.857422`, win `0.071289`, entropy `2.614950`.
- Killed additional mature weak or invalid controls:
  - w0 `meanq-c3-w0` / `cav8inbo`, PID `1770952`, step 129, best step 93
    `avg_R=-0.845703`, win `0.077148`, entropy `2.530760`; latest alpha/root
    metrics became NaN and `policy_target_entropy` reported `0`.
  - w3 `meanq-ps128-c3-w3` / `csijteuk`, PID `762709`, step 113, best step 27
    `avg_R=-0.914062`, win `0.042969`, entropy `3.059350`.
  - w8 `parg-c1-w8` / `8r943om0`, PID `754078`, step 105, best step 12
    `avg_R=-0.869141`, win `0.065430`, entropy `2.612520`.
  - w13 `meanq-nofix-w13` / `rn38bg4l`, PID `720715`, step 113, best step 49
    `avg_R=-0.917969`, win `0.041016`, entropy `2.692160`.
- Refilled with the baseline-loss grid:

| window | name | W&B | PID | purpose |
| --- | --- | --- | --- | --- |
| 0 | `parg-base-w0` | `9t3a2rbf` | `1817037` | Base loss/LR recipe with posterior-argmax, no fixed prior |
| 3 | `parg-base-c1-w3` | `1b8elfyf` | `789773` | Base loss/LR recipe with posterior-argmax and fixed prior `1.0` |
| 8 | `psamp-base-c0p3-w8` | `5zvikpyx` | `775578` | Base loss/LR recipe with posterior-sample and fixed prior `0.3` |
| 13 | `psamp-base-c3-w13` | `kbveqsxs` | `745080` | Base loss/LR recipe with posterior-sample and fixed prior `3.0` |

## Status And Recycle 2026-06-16 21:00 UTC

- No Go W&B hit yet.
- Baseline-loss posterior-sample is the least bad branch so far, still below
  the 20% win-rate floor:
  - `psamp-base-c1-w5` / `drpxuvza`: best step 38 `avg_R=-0.818359`, win
    `0.090820`, entropy `2.611620`.
  - `psamp-base-w1` / `bclctcdb`: best step 31 `avg_R=-0.822266`, win
    `0.088867`, entropy `2.607860`.
- Killed stale mature controls:
  - w12 `parg-nofix-w12` / `fo8e02kz`, PID `760334`, step 138, best step 64
    `avg_R=-0.847656`, win `0.076172`, entropy `2.602280`; latest metrics
    had NaN/root invalidity.
  - w10 `visit8x4-arg-c3-w10` / `e8rg7b1r`, PID `711551`, step 199, best
    step 97 `avg_R=-0.935547`, win `0.032227`, entropy `1.794740`.
  - w2 `meanq-c0p3-w2` / `0047umkl`, PID `775017`, step 149, best step 13
    `avg_R=-0.886719`, win `0.056641`, entropy `2.907480`.
  - w15 `parg-ps128-c3-w15` / `rapaco4n`, PID `696653`, step 116, best step
    27 `avg_R=-0.849609`, win `0.075195`, entropy `3.000410`.
- Refilled around the `psamp-base-c1` center:

| window | name | W&B | PID | purpose |
| --- | --- | --- | --- | --- |
| 2 | `psamp-base-c1-lr1e3-w2` | `h0cy57r6` | `804762` | Base loss recipe with lower LR `1e-3` |
| 10 | `psamp-base-c1-bs2k-w10` | `nj2899pg` | `754292` | Base loss recipe with self-play batch `2048` |
| 12 | `psamp-base-c1-ps128-w12` | `ktvs1w9z` | `787163` | Base loss recipe with `policy_samples=128` |
| 15 | `psamp-base-c1-b64-w15` | `4vubg1j9` | `721778` | Base loss recipe with `num_blocks=64` |

## Status 2026-06-16 22:00 UTC

- No Go W&B hit yet.
- No recycle on this pass. The baseline-loss branch is still mostly below step
  100 and is the best current direction.
- Best live Go rows remain below the 20% win-rate floor:
  - `psamp-base-c3-w13` / `kbveqsxs`: best step 38 `avg_R=-0.814453`, win
    `0.092773`, entropy `2.595610`.
  - `psamp-base-c1-w5` / `drpxuvza`: best step 38 `avg_R=-0.818359`, win
    `0.090820`, entropy `2.611620`.
  - `psamp-base-w1` / `bclctcdb`: best step 31 `avg_R=-0.822266`, win
    `0.088867`, entropy `2.607860`.
  - `psamp-base-c0p3-w8` / `5zvikpyx`: best step 38
    `avg_R=-0.822266`, win `0.088867`, entropy `2.616270`.
- Interpretation: reverting to the base `go9x9_3` loss/LR recipe improved the
  early reward curve versus the conservative-loss grid, but the target entropy
  is still around `2.6`, so the search target remains diffuse.

## Status 2026-06-16 23:01 UTC

- No Go W&B hit yet.
- Baseline-loss posterior-sample remains the best direction, but it has not
  broken out:
  - `psamp-base-c3-w13` / `kbveqsxs`: best step 38 `avg_R=-0.814453`, win
    `0.092773`, entropy `2.595610`.
  - `psamp-base-c1-w5` / `drpxuvza`: best step 38 `avg_R=-0.818359`, win
    `0.090820`, entropy `2.611620`.
  - `psamp-base-w1` / `bclctcdb`: best step 31 `avg_R=-0.822266`, win
    `0.088867`, entropy `2.607860`.
- These are close enough to the step-100 horizon that the next pruning decision
  should wait until the core baseline-loss runs cross step 100.

## Status 2026-06-16 23:32 UTC

- No Go W&B hit yet.
- Two core baseline-loss runs crossed the step-100 horizon and missed:
  - `psamp-base-w1` / `bclctcdb`: step 103, best step 31
    `avg_R=-0.822266`, win `0.088867`, entropy `2.607860`; latest regressed
    to `avg_R=-0.923828`.
  - `psamp-base-c1-w5` / `drpxuvza`: step 101, best step 38
    `avg_R=-0.818359`, win `0.090820`, entropy `2.611620`; latest regressed
    to `avg_R=-0.966797`.
- The remaining baseline-loss variants are still young enough to keep running:
  lower LR, larger self-play batch, higher target sample count, more blocks,
  posterior-argmax controls, and fixed-prior concentration variants.
- Interpretation: baseline losses improve the early curve but do not solve the
  signal problem; entropy remains around `2.6`, still near the no-search-signal
  range.

## Pivot 2026-06-17 00:05 UTC

- Fixed-position DT signal diagnostic added in `scripts/diagnose_dt_signal.py`.
  Focused validation passed:
  - `python -m py_compile scacchi/types.py scacchi/play_search.py scripts/diagnose_dt_signal.py`
  - `JAX_PLATFORMS=cpu pytest` for the new config-validation and backend-prior
    projection tests.
- Diagnostic result with fresh models:
  - `hex5`, 32 evidence units: ply 0 `H=2.9215`, `exp(H)=18.71`; ply 8
    `H=2.6536`, `exp(H)=14.20`.
  - `go9x9_3`, 32 evidence units: ply 0 `H=3.6640`, `exp(H)=39.04`; ply 8
    `H=3.6362`, `exp(H)=37.95`.
  - `go9x9_3`, 128 evidence units plus search-prior concentration `0.3`:
    `H` stayed around `3.6`; covering every legal action with symmetric weak
    evidence does not create a ranking signal.
  - `go9x9_3`, search-prior concentration `0.1`: ply 0 `H=2.8263`,
    `exp(H)=16.88`; ply 8 `H=2.7397`, `exp(H)=15.57`. This is still not good,
    but it is the first first-principles intervention that moves the canary.
- Code change: added
  `selfplay.search.dirichlet_thompson.search_prior_concentration`, which
  preserves each action's WDL mean while projecting the concentration used by
  DT search. This tests whether model alpha confidence is overpowering weak Go
  evidence before the loss even sees a target.

Killed mature/repetitive runs:

| window | name | W&B | PID | evidence and reason |
| --- | --- | --- | --- | --- |
| 1 | `psamp-base-w1` | `bclctcdb` | `742488` | step 112, best<=100 `avg_R=-0.822266`, win `0.088867`, entropy `2.607860`; mature miss |
| 4 | `meanq-8x8-c3-w4` | `xgi8yrxn` | `679789` | step 89, latest `avg_R=-0.873047` in W&B but pane had later near-zero-win evals; mean-utility branch already failed Hex/Go |
| 5 | `psamp-base-c1-w5` | `drpxuvza` | `719363` | step 110, best<=100 `avg_R=-0.818359`, win `0.090820`, entropy `2.611620`; mature miss |
| 6 | `psamp-c3-w6` | `83p19sgv` | `700641` | step 225, best<=100 `avg_R=-0.830078`, win `0.084961`, entropy `2.607780`; mature miss |
| 9 | `parg-c0p3-w9` | `6oxmbu4g` | `734367` | step 184, best<=100 `avg_R=-0.863281`, win `0.068359`, entropy `2.605510`; mature miss |
| 11 | `psamp-nofix-w11` | `yvrvdpe6` | `739228` | step 183, best<=100 `avg_R=-0.835938`, win `0.082031`, entropy `2.610410`; mature miss |
| 14 | `psamp-c1-w14` | `pie1v3x5` | `702482` | step 206, best<=100 `avg_R=-0.843750`, win `0.078125`, entropy `2.620850`; mature miss |

Launched search-prior tests:

| window | name | W&B | purpose |
| --- | --- | --- | --- |
| 1 | `go-spc0p1-tpc1-w1` | `se3qmqqq` | Go, search prior `0.1`, target prior `1.0`, posterior-sample |
| 4 | `go-spc0p3-tpc1-w4` | `qo8socr2` | Go, search prior `0.3`, target prior `1.0`, posterior-sample |
| 5 | `go-spc1-tpc1-w5` | `fy1zhbgs` | Go, search prior `1.0`, target prior `1.0`, posterior-sample |
| 6 | `go-spc0p3-tpc0p3-w6` | `v2frfgtn` | Go, search prior `0.3`, target prior `0.3`, posterior-sample |
| 9 | `go-spc0p3-b64-w9` | `qyxxjqy1` | Go, search prior `0.3`, target prior `1.0`, 64 blocks |
| 11 | `hex5-spc0p3-w11` | `10rkf4ix` | Hex regression: does softened DT search prior preserve the working Hex recipe? |
| 14 | `go-spc0p1-b64-w14` | `28efa2sx` | Go, search prior `0.1`, target prior `1.0`, 64 blocks |

## Status 2026-06-17 00:07 UTC

- New search-prior runs are live in tmux and W&B but have not logged an eval row
  yet. Treat them as too young; do not infer from empty summaries.
- Older live controls are still weak and diffuse:
  - w0 `9t3a2rbf`: step 93, best<=100 `avg_R=-0.832031`, win `0.083984`,
    latest entropy `2.676020`.
  - w3 `1b8elfyf`: step 93, best<=100 `avg_R=-0.851562`, win `0.074219`,
    latest entropy `2.619770`.
  - w8 `5zvikpyx`: step 90, best<=100 `avg_R=-0.822266`, win `0.088867`,
    latest entropy `2.619290`.
  - w13 `kbveqsxs`: step 92, best<=100 `avg_R=-0.814453`, win `0.092773`,
    latest entropy `2.621550`.
- Decision: no pruning from this snapshot. Wait for either logged rows from the
  search-prior cohort or for the near-horizon controls to cross step 100.

## Hex Diagnostic Pivot 2026-06-17 00:39 UTC

- Question: can the Go failure be recreated in Hex with a cleaner intervention?
  The three-way test is:
  - lower Hex search depth only;
  - Hex with the Go-style `aznet_dirichlet` network;
  - lower Hex search depth plus `aznet_dirichlet`.
- Interpretation plan:
  - low-depth Boardlaw Hex breaks -> search depth/terminal reach is sufficient
    to reproduce the failure;
  - AZ Hex breaks at normal depth -> architecture/head/trunk interaction is
    sufficient;
  - only AZ+low-depth breaks -> interaction between weak priors and shallow
    evidence;
  - none break -> Go-specific action-space/horizon/baseline issue is more
    likely.
- Recycled old posterior-argmax controls to free Hex diagnostic slots:
  - w0 `parg-base-w0` / `9t3a2rbf`: latest step 102
    `avg_R=-0.861328`, win `0.069336`, entropy `2.617796`; best<=100 step 31
    `avg_R=-0.832031`, win `0.083984`, entropy `2.595079`.
  - w3 `parg-base-c1-w3` / `1b8elfyf`: latest step 102
    `avg_R=-0.914062`, win `0.042969`, entropy `2.615129`; best<=100 step 74
    `avg_R=-0.851562`, win `0.074219`, entropy `2.611167`.
- Launched Hex diagnostics:

| window | name | W&B | purpose |
| --- | --- | --- | --- |
| 0 | `hex5-depth1-w0` | `t3g007tj` | Hex Boardlaw net, DT max depth `1`, eval every step |
| 3 | `hex5-az-w3` | `n6pr639g` | Hex with `aznet_dirichlet`, normal search depth, eval every step |
| 11 | `hex5-az-depth1-w11` | `doenj6mc` | Hex with `aznet_dirichlet` and DT max depth `1`, eval every step |

- Startup status:
  - w3 reached training and started eval step 1; tmux showed step 0
    `avg_R=-1.000000`, policy loss `2.8394`, q loss weight mean `0.0703`.
  - w0 and w11 reached W&B and started evaluation at step 0; W&B history had
    not yet synced useful eval rows at this snapshot.
- Search-prior Go cohort is still too young and below the reward floor:
  best early win rates are all below `0.06`, so no strength inference should be
  made from them yet.

## Status 2026-06-17 00:40 UTC

- Hex AZ-only run `n6pr639g` is producing W&B rows. Latest seen at step 4:
  `avg_R=-0.734375`, win `0.132812`, entropy `2.335509`, policy loss
  `2.747608`, q loss weight mean `0.073874`.
- This is still below the `0.20` win-rate floor, so do not call it a good run.
  The useful observation is that AZ-only Hex is climbing early rather than
  being obviously Go-like.
- Depth-1 Hex runs `t3g007tj` and `doenj6mc` have started evaluations, but W&B
  has not synced usable eval rows yet. Wait for those instead of probing every
  few seconds.

## Status 2026-06-17 01:11 UTC

- Hex diagnostic cohort finished all 100 eval steps:
  - `hex5-depth1-w0` / `t3g007tj`: latest step 99 `avg_R=0.117188`, win
    `0.558594`, entropy `2.047104`; best step 49 `avg_R=0.281250`, win
    `0.640625`, entropy `1.986403`.
  - `hex5-az-w3` / `n6pr639g`: latest step 99 `avg_R=-0.039062`, win
    `0.480469`, entropy `2.028030`; best step 66 `avg_R=0.140625`, win
    `0.570312`, entropy `2.027732`.
  - `hex5-az-depth1-w11` / `doenj6mc`: latest step 99 `avg_R=0.093750`, win
    `0.546875`, entropy `2.150481`; best step 92 `avg_R=0.281250`, win
    `0.640625`, entropy `2.136873`.
- Conclusion: the Go failure did not reproduce in Hex. Low depth alone,
  `aznet_dirichlet` alone, and the combination all still learned Hex by the
  step-100 criterion. The next useful tests are Go-specific:
  action-space/horizon and PGX-baseline difficulty.
- Search-prior Go cohort still has no trustworthy strength signal:
  best early win rates remain below `0.07` through the latest snapshot.

## Go-Specific Diagnostics 2026-06-17 01:20 UTC

- Since Hex did not break under low depth or `aznet_dirichlet`, recycled the
  finished Hex workers into Go-specific diagnostics:

| window | name | W&B | purpose |
| --- | --- | --- | --- |
| 0 | `go5-dt-az-w0` | `xxynm90b` | Go 5x5 DT with AZ net, random baseline, tests whether smaller Go action-space/horizon learns |
| 3 | `go9-rand-base-w3` | `cylpenvk` | Go 9x9 current DT recipe evaluated against random, tests whether PGX baseline is masking early learning |
| 11 | `go9-rand-spc0p1-w11` | `1k5uxvzm` | Same random-eval Go 9x9 probe with DT search-prior concentration `0.1` and target prior `1.0` |

- Setup notes:
  - `go5x5_dirichlet.yaml` contains stale `training.losses.outcome_dir_targets`;
    launch removed it with `~training.losses.outcome_dir_targets`.
  - Go 5x5 random baseline must use `eval.baseline_search.kind=policy`; the
    YAML's Gumbel baseline search expects a value head that the random baseline
    does not provide.
  - Added config fields absent from YAML need Hydra `+` overrides.
- Startup status: all three reached W&B and started step-0 evaluation. Failed
  setup runs before these IDs should be ignored.

## Status 2026-06-17 01:25 UTC

- Corrected Go-specific diagnostics are alive and logging:
  - `go5-dt-az-w0` / `xxynm90b`: W&B step 0 `avg_R=0.222656`, win
    `0.611328`, entropy `2.047745`. This is already above the random-baseline
    floor; the question is whether it stays high and improves.
  - `go9-rand-base-w3` / `cylpenvk`: W&B step 0 `avg_R=-0.039062`, win
    `0.480469`, entropy `2.844844`.
  - `go9-rand-spc0p1-w11` / `1k5uxvzm`: W&B step 0 `avg_R=-0.039062`, win
    `0.480469`, entropy `2.411969`.
- Pane logs already show both 9x9 random-eval probes moving through step 1/2;
  no immediate crash. Wait before drawing conclusions.

## Status 2026-06-17 01:56 UTC

- Go-specific diagnostics now answer the mechanism question:
  - `go5-dt-az-w0` / `xxynm90b`: latest step 44 `avg_R=0.990234`, win
    `0.995117`, entropy `1.766411`; best step 9 `avg_R=1.000000`, win
    `1.000000`.
  - `go9-rand-base-w3` / `cylpenvk`: latest step 11 `avg_R=0.402344`, win
    `0.701172`, entropy `2.649505`; best step 8 `avg_R=0.453125`, win
    `0.726562`.
  - `go9-rand-spc0p1-w11` / `1k5uxvzm`: latest/best step 11
    `avg_R=0.550781`, win `0.775391`, entropy `2.338148`.
- Interpretation:
  - DT can learn Go under this code path; Go 5x5 and Go 9x9 versus random both
    get real win rates quickly.
  - The repeated bad result is specific to the strong PGX 9x9 eval baseline, not
    to "no Go signal at all."
  - The softer search prior remains promising in mechanism terms: on random
    9x9 eval it has lower target entropy and higher early reward than the base
    random-eval control.
- PGX-baseline search-prior runs are still weak through the latest snapshot:
  best early win rates remain under `0.084`. Do not call random-baseline success
  a solution to the original objective.

## Status 2026-06-17 02:27 UTC

- Go-specific diagnostics continue to separate "Go signal" from "PGX parity":
  - `go5-dt-az-w0` / `xxynm90b`: latest step 90 `avg_R=0.998047`, win
    `0.999023`, entropy `1.730668`; best step 9 `avg_R=1.000000`.
  - `go9-rand-base-w3` / `cylpenvk`: latest/best step 23
    `avg_R=0.533203`, win `0.766602`, entropy `2.589297`.
  - `go9-rand-spc0p1-w11` / `1k5uxvzm`: latest step 22 `avg_R=0.529297`, win
    `0.764648`, entropy `2.344076`; best step 12 `avg_R=0.568359`.
- Interpretation update:
  - 9x9 DT clearly learns enough to beat random quickly.
  - The softer search prior is still lower entropy, but the base random-eval
    run has caught up in reward, so do not overclaim `0.1` as a fix.
  - PGX-eval search-prior runs remain far below the 20% win-rate floor: best
    live row is still `fy1zhbgs` at `0.083008`.

## Status And Recycle 2026-06-17 03:31 UTC

- After another hour, no PGX policy-eval run is close:
  - best live PGX row remains `go-spc1-tpc1-w5` / `fy1zhbgs` step 38
    `avg_R=-0.833984`, win `0.083008`.
  - search-prior runs at steps `71..72` regressed toward `1-3%` win rate.
- Random-eval controls confirmed the Go signal:
  - `go5-dt-az-w0` / `xxynm90b` finished: latest and best
    `avg_R=1.000000`, win `1.000000`.
  - `go9-rand-base-w3` / `cylpenvk`: latest step 46 `avg_R=0.539062`, win
    `0.769531`; best step 37 `avg_R=0.582031`, win `0.791016`.
  - `go9-rand-spc0p1-w11` / `1k5uxvzm`: latest step 45 `avg_R=0.570312`, win
    `0.785156`; best step 27 `avg_R=0.585938`, win `0.792969`.
- Next mechanism question: is PGX failure caused by weak raw policy, while
  Dirichlet-Thompson search-improved actions are stronger? This matters because
  the method objective is DT search, but current PGX eval uses `kind=policy`.
- Recycle plan:
  - reuse completed w0;
  - stop w3 random-eval base control because it already answered the
    random-baseline question and w11 keeps the softer-prior random-eval tail.
- Launch DT-search PGX eval diagnostics:
  - base training with `eval.player_search.kind=dirichlet_thompson`;
  - search-prior `0.1` training with matching DT eval prior.

Launched:

| window | name | W&B | purpose |
| --- | --- | --- | --- |
| 0 | `go9-pgx-dteval-base-w0` | `tgq83bpy` | Base Go 9x9 DT training, PGX eval with DT player search |
| 3 | `go9-pgx-dteval-spc0p1-w3` | `u7eselet` | Search-prior `0.1` Go 9x9 DT training, PGX eval with matching DT player search prior |

- Recycled w3 random-eval base control after recording its result; killed PID
  `820833` on worker 3 before relaunch.

## Recycle 2026-06-17 04:08 UTC

- DT-search PGX eval was too expensive and not promising:
  - `go9-pgx-dteval-base-w0` / `tgq83bpy`: latest step 3
    `avg_R=-0.966797`, win `0.016602`, entropy `2.819776`; killed worker 0
    PID `1902263`.
  - `go9-pgx-dteval-spc0p1-w3` / `u7eselet`: latest step 3
    `avg_R=-0.972656`, win `0.013672`, entropy `2.356148`; killed worker 3
    PID `835394`.
- Decision: do not spend 8-14 hours on per-step DT-search eval when early rows
  do not show a rescue. Return to policy-eval PGX speedrun runs.
- New concentration bracket:
  - w0 `go-spc0p03-tpc1-w0` / `lgvz8e13`: Go 9x9, search prior `0.03`,
    target prior `1.0`.
  - w3 `go-spc0p01-tpc1-w3` / `gl1hwd0z`: Go 9x9, search prior `0.01`,
    target prior `1.0`.
- Mechanism: if action-space dilution is the issue, lowering prior
  concentration should sharpen targets further. If these remain under the
  win-rate floor, the problem is not just prior concentration.

## Status 2026-06-17 04:42 UTC

- Lower concentration bracket:
  - `go-spc0p03-tpc1-w0` / `lgvz8e13`: latest/best step 9
    `avg_R=-0.978516`, win `0.010742`, entropy `2.089655`.
  - `go-spc0p01-tpc1-w3` / `gl1hwd0z`: latest step 9 `avg_R=-0.992188`, win
    `0.003906`, entropy `1.987846`; best step 7 `avg_R=-0.976562`, win
    `0.011719`.
- Interpretation: lowering search-prior concentration does sharpen targets, but
  it does not improve PGX strength. This is sharp-wrong evidence, not a
  solution.
- Main PGX policy-eval runs are maturing and still miss:
  - `go-spc1-tpc1-w5` / `fy1zhbgs` reached step 100; best<=100 remains step 38
    `avg_R=-0.833984`, win `0.083008`.
  - Other search-prior rows remain below `0.094` win rate.
- Random-eval tail remains strong: `go9-rand-spc0p1-w11` / `1k5uxvzm` best
  step 70 `avg_R=0.607422`, win `0.803711`.

## Status 2026-06-17 04:58 UTC

- Main search-prior PGX cohort crossed the step-100 horizon and missed:
  - `go-spc0p1-tpc1-w1` / `se3qmqqq`: best<=100 step 52
    `avg_R=-0.943359`, win `0.028320`, entropy `2.357167`.
  - `go-spc0p3-tpc1-w4` / `qo8socr2`: best<=100 step 76
    `avg_R=-0.916016`, win `0.041992`, entropy `2.459719`.
  - `go-spc1-tpc1-w5` / `fy1zhbgs`: best<=100 step 38
    `avg_R=-0.833984`, win `0.083008`, entropy `2.540435`.
  - `go-spc0p3-tpc0p3-w6` / `v2frfgtn`: best<=100 step 81
    `avg_R=-0.888672`, win `0.055664`, entropy `2.482129`.
- 64-block variants are younger but still under the floor:
  - `go-spc0p3-b64-w9` / `qyxxjqy1`: best<=100 win `0.076172`.
  - `go-spc0p1-b64-w14` / `28efa2sx`: best<=100 win `0.093750`.
- Lower-prior bracket remains sharp-wrong:
  - `0.03` / `lgvz8e13`: best win `0.010742`.
  - `0.01` / `gl1hwd0z`: best win `0.011719`.
- Conclusion: prior concentration alone does not solve PGX parity. The search
  target can learn Go versus random but does not produce PGX-level action
  quality by step 100 under these recipes.

## Recycle And Hex Rerun 2026-06-17 05:06 UTC

- Recycled the sharp-wrong lower-prior Go bracket:
  - w0 `go-spc0p03-tpc1-w0` / `lgvz8e13`, remote PID `1911289`.
    Latest W&B step 15: `avg_R=-0.986328`, win `0.006836`, entropy
    `2.100935`; best<=100 step 9 `avg_R=-0.978516`, win `0.010742`.
    Killed with `eopod kill-tpu --force --worker 0`; worker returned `READY`.
  - w3 `go-spc0p01-tpc1-w3` / `gl1hwd0z`, remote PID `842411`.
    Latest W&B step 15: `avg_R=-0.990234`, win `0.004883`, entropy
    `2.020867`; best<=100 step 7 `avg_R=-0.976562`, win `0.011719`.
    Killed with `eopod kill-tpu --force --worker 3`; worker returned `READY`.
- Reason: both runs had already answered the concentration question. Lower
  search-prior concentration sharpened targets but produced a sharp-wrong Go
  policy against PGX.
- New user-suggested Hex breakage replication:

| window | name | W&B | purpose |
| --- | --- | --- | --- |
| 0 | `hex5-az-depth1-w0` | `djglw9zp` | Hex 5x5, `aznet_dirichlet`, DT `max_depth=1`, tests AZ-net plus shallow search |
| 3 | `hex5-depth1-w3` | `rsfsa5he` | Hex 5x5, `boardlaw_dirichlet`, DT `max_depth=1`, same-time control |

- Setup note: `max_depth` is absent from `hex5.yaml`, so Hydra requires
  `+selfplay.search.dirichlet_thompson.max_depth=1`. The first attempt without
  `+` failed before training and produced no W&B run. The corrected launches
  reached W&B and the training loop.
- Decision rule: if the AZ-depth1 Hex rerun again reaches parity by step 100,
  do not keep blaming low search depth or AZ architecture in isolation. Return
  to Go-specific action-space/horizon/baseline analysis.

## Hex Result 2026-06-17 05:39 UTC

- Both Hex reruns finished and crossed parity:
  - w0 `hex5-az-depth1-w0` / `djglw9zp`: latest step 99
    `avg_R=0.093750`, win `0.546875`, entropy `2.150481`; best<=100 step 92
    `avg_R=0.281250`, win `0.640625`, entropy `2.136873`.
  - w3 `hex5-depth1-w3` / `rsfsa5he`: latest step 99 `avg_R=0.117188`, win
    `0.558594`, entropy `2.047104`; best<=100 step 49 `avg_R=0.281250`, win
    `0.640625`, entropy `1.986403`.
- Conclusion: the direct Hex breakage test still does not reproduce Go. Shallow
  DT depth plus `aznet_dirichlet` is not sufficient to cause the Go collapse.
  The next probe should stay in Go and test PGX baseline/eval semantics or
  action-space/horizon effects.

## Go Eval Sanity And Controls 2026-06-17 05:46 UTC

- Local CPU PGX policy-vs-policy sanity check over 1024 games per seed:
  - seed 0: `avg_R=-0.013672`, win `0.493164`.
  - seed 1: `avg_R=0.007812`, win `0.503906`.
  - seed 2: `avg_R=0.027344`, win `0.513672`.
- Interpretation: PGX self-eval is essentially symmetric. The repeated Go
  collapse is unlikely to be a simple eval-color/action-mask bug.
- Next control A/B on the freed workers:
  - scalar AZ Gumbel from `go9x9_gumbel`, max 120 iters, eval every step;
  - Dirichlet-network Gumbel from `go9x9_3` with
    `selfplay.search.kind=gumbel`, max 120 iters, eval every step.
- Mechanism: if scalar Gumbel works and Dirichlet Gumbel fails, suspect the
  Dirichlet head/loss path. If Dirichlet Gumbel works and DT fails, suspect the
  DT target/search operator. If neither works by step 100, the current PGX
  speedrun target is probably too hard from scratch under these small-run
  settings.
- Launched controls:

| window | name | W&B | purpose |
| --- | --- | --- | --- |
| 0 | `go-gumbel-scalar-w0` | `rnwartle` | Scalar AZ Gumbel from `go9x9_gumbel`, max 120 iters |
| 3 | `go-gumbel-dirnet-w3` | `exvs0rwu` | `aznet_dirichlet` Gumbel from `go9x9_3`, max 120 iters |

## Recycle For 5iluv1nv-Anchored A/B 2026-06-17 07:55 UTC

- Historical check: the user-pointed run `5iluv1nv` was not DT; it was
  `aznet_dirichlet` with Gumbel self-play, larger `2048/8192` self-play/train
  batches, `lr=1e-3`, evidence-mass Q weighting, masked-mean Q KL, terminal
  targets, and policy eval against PGX. It missed the step-100 objective
  (`avg_R=-0.352539`, win `0.323730`) but became strong later
  (`avg_R=0.510742`, win `0.755371` by step 224).
- Recycled mature failed search-prior Go runs:
  - w1 `go-spc0p1-tpc1-w1` / `se3qmqqq`, remote PID `765183`.
    Latest step 167: `avg_R=-0.970703`, win `0.014648`, entropy `2.333235`.
    Best<=100 step 52: `avg_R=-0.943359`, win `0.028320`. Killed with
    `eopod kill-tpu --force --worker 1`; worker returned `READY`.
  - w4 `go-spc0p3-tpc1-w4` / `qo8socr2`, remote PID `720859`.
    Latest step 168: `avg_R=-0.966797`, win `0.016602`, entropy `2.473567`.
    Best<=100 step 76: `avg_R=-0.916016`, win `0.041992`. Killed with
    `eopod kill-tpu --force --worker 4`; worker returned `READY`.
- New A/B:
  - reproduce the `5iluv1nv` Gumbel recipe as a current-code anchor;
  - swap only self-play search to `dirichlet_thompson`, with explicit
    `max_depth=64` and the same `num_simulations=4,num_blocks=8` search budget.
- Mechanism: if the reproduced Gumbel anchor improves but the DT swap does not,
  the DT operator/search target is the bottleneck. If both are slow, the
  `5iluv1nv` recipe itself is not a step-100 solution even after the depth fix.
- Launched:

| window | name | W&B | purpose |
| --- | --- | --- | --- |
| 1 | `go5ilu-gumbel-w1` | `gsobsq6d` | Current-code reproduction of `5iluv1nv` Gumbel recipe |
| 4 | `go5ilu-dt-w4` | `7agypa11` | Same recipe with self-play search swapped to DT, `max_depth=64` |

## Recycle For DT Commitment Isolation 2026-06-17 09:02 UTC

- Recycled w3 `go-gumbel-dirnet-w3` / `exvs0rwu`, remote PID `857502`, because
  it was mature enough to answer the mechanism question and remained far below
  the win-rate floor. At the recycle check it was around step 56, with latest
  `avg_R=-0.984375`, win `0.0078125`; best<=100 was step 14
  `avg_R=-0.886719`, win `0.056641`. Killed with
  `eopod kill-tpu --force --worker 3`; worker returned `READY`.
- Launched w3 `go5ilu-dt-parg-w3` / `xt0iahzu`: same `5iluv1nv`-anchored DT
  recipe as `go5ilu-dt-w4`, but with
  `selfplay.action_commitment_type=posterior_argmax` instead of `search_action`.
- Mechanism: keep the recipe, depth, batch sizes, losses, and eval cadence fixed
  while changing only the DT self-play commitment mode. This isolates whether
  posterior-best commitment is materially different from sampling/committed
  search action in the Go PGX speedrun setting.
- Launch hygiene: an earlier w3 command was aborted because shell quoting
  dropped the shared `5iluv1nv` overrides before a valid W&B training run was
  established. Do not count that aborted start as an experiment.

## One-Hour Read 2026-06-17 10:05 UTC

- Active mechanism runs:
  - w0 `go-gumbel-scalar-w0` / `rnwartle`: running, latest step 78
    `avg_R=-0.519531`, win `0.240234`, policy entropy `0`. This is the only
    live PGX run currently above the 20% win-rate floor, but it has not reached
    parity.
  - w1 `go5ilu-gumbel-w1` / `gsobsq6d`: running, latest step 17
    `avg_R=-0.993164`, win `0.003418`, entropy `2.701814`. Too young to judge
    the slow `5iluv1nv` recipe, but not promising yet.
  - w4 `go5ilu-dt-w4` / `7agypa11`: running, latest step 17
    `avg_R=-0.991211`, win `0.004395`, entropy `2.708996`.
  - w3 `go5ilu-dt-parg-w3` / `xt0iahzu`: running, latest step 7
    `avg_R=-0.956055`, win `0.021973`, entropy `2.589192`.
- Completed control:
  - w11 `go9-rand-spc0p1-w11` / `1k5uxvzm`: finished 100 iters. Best<=100 was
    step 90 `avg_R=0.666016`, win `0.833008`; latest step 99
    `avg_R=0.335938`, win `0.667969`. This confirms again that DT can learn
    Go 9x9 against a random evaluator; the PGX baseline is the hard case.
- Old posterior-sample/search-prior family is no longer informative:
  - all checked runs are past step 100 and still around `4-9%` win rate:
    `fj3gihh0`, `h0cy57r6`, `5zvikpyx`, `nj2899pg`, `ktvs1w9z`,
    `kbveqsxs`, `4vubg1j9`, `fy1zhbgs`, `v2frfgtn`, `qyxxjqy1`,
    `28efa2sx`.
  - Do not extend that family unless a new mechanism is added. It is sharp or
    diffuse wrong, not near the objective.
- Worker 11 became free after the random-baseline control completed. Launched
  w11 `go-gumbel-argmaxeval-w11` / `rr56jqv3`: same scalar Gumbel training as
  the moving w0 control, but evaluate the player with
  `eval.player_action_commitment_type=posterior_argmax` while keeping the PGX
  baseline at `posterior_sample`.
- Mechanism: if argmax eval is much stronger than sampled-policy eval, the
  model may already contain a useful mode and the immediate bottleneck is
  policy sharpening/action selection. If it is not stronger, the scalar Gumbel
  improvement is genuinely still below PGX strength.

## Recycle For Outcome-Only Alpha Probe 2026-06-17 10:47 UTC

- Fresh read before recycle:
  - w0 `go-gumbel-scalar-w0` / `rnwartle`: running, latest step 90
    `avg_R=-0.496094`, win `0.251953`; best<=100 step 83
    `avg_R=-0.292969`, win `0.353516`.
  - w11 `go-gumbel-argmaxeval-w11` / `rr56jqv3`: running, latest step 9
    `avg_R=-0.996094`, win `0.001953`; too young, but no early argmax-eval
    rescue yet.
  - w1 `go5ilu-gumbel-w1` / `gsobsq6d`: latest step 23
    `avg_R=-0.992188`, win `0.003906`, `alpha_Q=7.900`.
  - w4 `go5ilu-dt-w4` / `7agypa11`: latest step 23
    `avg_R=-0.988281`, win `0.005859`, `alpha_Q=7.949`.
  - w3 `go5ilu-dt-parg-w3` / `xt0iahzu`: latest step 13
    `avg_R=-0.916992`, win `0.041504`, `alpha_Q=7.776`.
- Important inference: the old failed DT family is not explained only by alpha
  concentration explosion. Several mature failures have `alpha_Q` around
  `1.5-2.8` and still remain below `10%` win rate. The `5iluv1nv`-style recipe
  does saturate near the clip, but that is not the only failure mode.
- Recycled mature failed old-family runs:
  - w5 `go-spc1-tpc1-w5` / `fy1zhbgs`, remote PID `742409`. Latest step 234:
    `avg_R=-0.898438`, win `0.050781`, entropy `2.557650`,
    `alpha_Q=2.284`. Best<=100 step 38: `avg_R=-0.833984`, win `0.083008`.
    Killed with `eopod kill-tpu --force --worker 5`; worker returned `READY`.
  - w6 `go-spc0p3-tpc0p3-w6` / `v2frfgtn`, remote PID `742194`. Latest step
    230: `avg_R=-0.933594`, win `0.033203`, entropy `2.474090`,
    `alpha_Q=1.515`. Best<=100 step 81: `avg_R=-0.888672`, win `0.055664`.
    Killed with `eopod kill-tpu --force --worker 6`; worker returned `READY`.
- New mechanism probe: train with DT self-play and DT policy targets, but remove
  bootstrapped Dirichlet KL supervision from alpha heads. The alpha means are
  supervised only by realized trajectory outcomes:
  `value_dir_kl_weight=0`, `q_dir_kl_weight=0`, `value_outcome_weight=1`,
  `q_outcome_weight=1`, `terminal_edge_targets=false`,
  `terminal_parent_targets=false`.
- Launched:

| window | name | W&B | distinction |
| --- | --- | --- | --- |
| 5 | `go-outcome-dt-psamp-w5` | `2v2u9q7x` | outcome-only alpha supervision, `selfplay.action_commitment_type=posterior_sample` |
| 6 | `go-outcome-dt-parg-w6` | `281c5o0q` | same, but `posterior_argmax` commitment |

- Mechanism: if these runs move while KL-supervised DT does not, the
  self-bootstrapping Dirichlet target is the culprit. If both fail with low
  concentration, the issue is more likely the DT posterior policy target/search
  operator or PGX action-space/horizon mismatch, not just concentration growth.

## Search Semantics Check 2026-06-17 10:58 UTC

- Inspected `scacchi/dirichlet_q_search.py`.
- The fixed-depth path is wired as intended: `dirichlet_q_policy` runs
  `num_search_blocks` sequential posterior-update blocks, and each block passes
  `max_depth` through to `mctx.search`.
- The current `search/path_depth_* = 0` metrics on DT runs are a diagnostics
  limitation, not direct evidence of depth-zero search. The batched DT wrapper
  does not retain the search tree from the sequential block scan, so path-depth
  diagnostics are zero-filled while node/evidence metrics are still logged.
- Checked previous notes for visit/search-weight policy targets. That branch
  already produced mature step-100 failures, so do not spend more workers on
  policy_samples=0 / visit-target relaunches unless the underlying target
  operator changes too.

## Outcome-Only First Read 2026-06-17 11:22 UTC

- Scalar Gumbel control `rnwartle` crossed the step-100 horizon without hitting
  the objective. Best<=100 was step 96 `avg_R=-0.234375`, win `0.382812`;
  latest step 101 was `avg_R=-0.199219`, win `0.400391`. It is learning
  against PGX, but not enough for the target.
- `go-outcome-dt-psamp-w5` / `2v2u9q7x`: running, latest step 9
  `avg_R=-0.988281`, win `0.005859`, entropy `2.464070`,
  `alpha_Q=3.000950`. Stable so far, but no strength signal yet.
- `go-outcome-dt-parg-w6` / `281c5o0q`: invalid. It went NaN around step 8:
  policy/value/outcome losses became `nan`, `q_loss_weight_mean` later dropped
  to zero, and eval was still below the win-rate floor. Killed remote PID
  `786174` with `eopod kill-tpu --force --worker 6`; worker returned `READY`.
- New stabilized outcome-only variant on worker 6:
  `go-outcome-lr1e3-clip-w6` / `3ido45kf`, same posterior-sample outcome-only
  alpha supervision as worker 5, but with `training.learning_rate=1e-3` and
  `training.grad_clip_norm=1.0`.
- Mechanism: if the lower-LR clipped variant avoids NaNs and improves while w5
  does not, the pure outcome-only idea is optimization-sensitive. If both stay
  below the floor with entropy still around `2.4-2.5`, then removing
  bootstrapped KL alone is not enough.

## Mean-Utility Target And Hex8 Probe 2026-06-17 12:03 UTC

- Fixed-state audit result before launching:
  - Go9 default DT target with 32 root evidence samples was broad and poorly
    aligned with PGX top actions.
  - Raising root evidence to 128 explored nearly all legal actions but target
    entropy stayed high: about `3.9709`, `3.8558`, `3.8103` at plies
    `0,8,16`, with PGX top-action agreement `0`.
  - Using deeper blocks (`num_simulations=4`, `num_blocks=32`) also stayed
    broad: entropy about `3.9173`, `3.8691`, `3.7831`, PGX agreement `0`.
  - Conclusion: budget alone is not the next informative Go run.
- Code change:
  - Added `PolicyTargetMode.mean_utility_argmax`.
  - `_dirichlet_search_output_from_backend` can now train the policy on a
    one-hot posterior mean-utility argmax target, while preserving posterior
    alpha targets.
  - Focused CPU tests passed:
    `tests/test_config_validation.py::test_mean_utility_argmax_policy_target_mode_is_allowed`,
    `tests/test_config_validation.py::test_policy_target_mode_must_be_known`,
    `tests/test_play_search_tictactoe.py::test_mean_utility_argmax_policy_target_uses_search_posterior_mean`,
    `tests/test_play_search_tictactoe.py::test_posterior_mean_utility_action_uses_search_posterior_mean`.
  - Broader `tests/test_play_search_tictactoe.py` still has an unrelated
    existing DQAZ tic-tac-toe failure: the test constructs `posterior_argmax`
    where `_selfplay_action_source` currently asserts `posterior_sample`.
- Recycled mature failed workers:
  - w2 `h0cy57r6`, remote PID `804762`, killed and replaced.
  - w7 `fj3gihh0`, remote PID `715407`, killed and replaced.
  - w9 `qyxxjqy1`, remote PID `769760`, killed and replaced for Hex8.
  - w14 `28efa2sx`, remote PID `741968`, killed and replaced for Hex8.
- New Go runs:
  - w2 `dt-muarg-5ilu-w2` / W&B `g4sfoj08`: same 5ilu DT `search_action`
    recipe as w4, plus `training.losses.policy_target_mode=mean_utility_argmax`.
    W&B registered, no eval rows yet.
  - w7 `dt-muarg-outcome-w7` / W&B `zhqt20au`: outcome-only alpha supervision,
    `search_action`, and `mean_utility_argmax` policy target. W&B registered,
    no eval rows yet.
- New Hex run:
  - w9,w14 `hex8-dt-random-w9w14`: `hex8` DT on topology `topo2-8`, reduced
    batch sizes, `eval.baseline=random`, `eval.baseline_search.kind=policy`,
    `eval.baseline_action_commitment_type=posterior_sample`.
  - First launch attempt was invalid: `hex8.yaml` did not contain
    `eval.baseline_action_commitment_type`, so Hydra required the append form.
    Relaunched with `+eval.baseline_action_commitment_type=posterior_sample`.
  - Corrected launch reached two-process distributed setup:
    worker 9 -> process `0/2`, worker 14 -> process `1/2`. W&B ID still
    pending at this read.

## Startup Check 2026-06-17 12:07 UTC

- Read `scientific_loop.md` before acting. Current key uncertainty: whether the
  scalar mean-utility policy target changes Go PGX learning, and whether Hex8
  reproduces the stall in a clean 64-action game.
- W&B active-run refresh:
  - `g4sfoj08` (`dt-muarg-5ilu-w2`): running, no synced eval row yet.
  - `zhqt20au` (`dt-muarg-outcome-w7`): running, no synced W&B eval row yet;
    tmux shows local step-0 eval completed with `avg_R=-0.9902`.
  - `brc1dycu` (`hex8-dt-random-w9w14`): running, no synced eval row yet;
    tmux shows JAX process count 2 and eval step 0 started.
  - `3ido45kf` (`go-outcome-lr1e3-clip-w6`): entropy reached `1.9972`, but
    latest PGX eval is still `avg_R=-0.992188`, win `0.003906`. This is
    sharp-wrong under the measurement guardrail, not progress.
- Decision: do not recycle more workers from this evidence. Wait for real rows
  from the mean-utility and Hex8 probes before launching another related run.

## Metric Fix And Recycle 2026-06-17 12:17 UTC

- The first mean-utility target launches were invalid as metric evidence:
  - w2 `dt-muarg-5ilu-w2` / `g4sfoj08`: no useful synced eval rows before
    replacement.
  - w7 `dt-muarg-outcome-w7` / `zhqt20au`: local step-0 eval existed, but
    `policy_target_entropy=0` was measuring the one-hot proxy training target.
- Fixed the metric path: `mean_utility_argmax` still trains policy on a one-hot
  posterior mean-utility argmax action, but `policy_metric_tgt` remains the
  full posterior-best distribution. This preserves the entropy canary.
- Focused tests after the metric fix passed:
  `tests/test_config_validation.py::test_mean_utility_argmax_policy_target_mode_is_allowed`,
  `tests/test_play_search_tictactoe.py::test_mean_utility_argmax_policy_target_uses_search_posterior_mean`,
  and
  `tests/test_play_search_tictactoe.py::test_policy_samples_zero_keeps_full_posterior_metric_target`.
- Replaced the invalid mean-utility runs:
  - w2 corrected run `dt-muarg-5ilu-w2` / W&B `06hjdp8v`.
  - w7 corrected run `dt-muarg-outcome-w7` / W&B `fikpoeb1`.
  Both were launched from scratch and had no synced eval rows at the 12:14 UTC
  W&B snapshot.
- Worker 6 `go-outcome-lr1e3-clip-w6` / W&B `3ido45kf` became invalid:
  policy/value/Q losses were `NaN` by step 16 while PGX win rate remained below
  `1%`. The later eval bounce to about `5%` win at step 17 occurred after NaNs
  and should not be treated as strength evidence. Killed remote PID `792310`;
  worker returned `READY`.
- Refilled worker 6 with a distinct target-object control:
  `dt-muarg-base-w6`, base `go9x9_3` posterior-sample recipe plus
  `training.losses.policy_target_mode=mean_utility_argmax`. This tests the
  scalar target under the least-bad mature base DT family instead of the 5ilu
  or outcome-only recipes. W&B ID: `clh6t6vo`.
- Hex8 result is already decisive for the current environment ablation:
  `hex8-dt-random-w9w14` / `brc1dycu` reached step 15
  `avg_R=0.642578`, win `0.821289`, entropy `2.87156`. A 64-action clean
  environment does not by itself reproduce the Go PGX stall.
- Scalar Gumbel control `rnwartle` finished with late positive reward:
  step 119 `avg_R=0.052734`, win `0.526367`, but best<=100 remained negative
  at step 96 `avg_R=-0.234375`, win `0.382812`. PGX is learnable under scalar
  Gumbel in this setup, but it does not satisfy the DT objective.

## Recycle And Ladder Diagnostics 2026-06-17 12:26 UTC

- W&B refresh before recycling:
  - Old posterior-sample branches w8 `5zvikpyx`, w10 `nj2899pg`, and w12
    `ktvs1w9z` were mature failures, still running at steps `363`, `180`, and
    `301` respectively, with best<=100 win rates all below `9%`.
  - These runs were not new evidence anymore; they were the same
    posterior-sample target result with controlled alpha concentration but no
    PGX strength.
- Killed and recycled:
  - w8 `5zvikpyx`, remote PID `775578`.
  - w10 `nj2899pg`, remote PID `754292`.
  - w12 `ktvs1w9z`, remote PID `787163`.
  All three workers returned `READY`.
- New diagnostics:
  - w8 `muarg-random-w8` / W&B `2gwzbv2t`: base `go9x9_3` DT recipe plus
    `policy_target_mode=mean_utility_argmax`, evaluated against random. This
    tests whether the scalar target can learn Go9 at all before attributing
    failure only to PGX strength.
  - w10 `dt-pgxT2-w10` / W&B `xbr49ti1`: base `go9x9_3` DT recipe, but eval
    baseline is PGX policy sampled at temperature `2.0`.
  - w12 `dt-pgxT5-w12` / W&B `sgjfhl59`: same, with PGX baseline temperature
    `5.0`.
- Ladder interpretation:
  - If the soft-PGX runs improve while hard-PGX runs fail, DT may be learning
    PGX-correlated but insufficiently strong policy signal.
  - If even temperature `5.0` remains near the random floor, the current DT
    policy target is probably not learning PGX-relevant action rankings.
  - These ladder runs are diagnostics, not replacements for the hard PGX
    objective.

## Greedy Eval Diagnostics 2026-06-17 12:31 UTC

- W&B refresh showed the remaining old posterior-sample workers w13 `kbveqsxs`
  and w15 `4vubg1j9` were still mature failures:
  - w13: latest step 373 `avg_R=-0.896484`, win `0.051758`; best<=100 step 38
    `avg_R=-0.814453`, win `0.092773`.
  - w15: latest step 179 `avg_R=-0.843750`, win `0.078125`; best<=100 step 38
    `avg_R=-0.822266`, win `0.088867`.
- Killed and recycled:
  - w13 `kbveqsxs`, remote PID `745080`.
  - w15 `4vubg1j9`, remote PID `721778`.
  Both workers returned `READY`.
- New eval-semantics diagnostics:
  - w13 `dt-base-greedyeval-w13` / W&B `653e6t79`: base `go9x9_3` DT recipe,
    hard PGX baseline, `eval.player_action_commitment_type=posterior_argmax`.
  - w15 `dt-muarg-greedyeval-w15` / W&B `92a5fbh8`: same greedy player eval,
    but with `training.losses.policy_target_mode=mean_utility_argmax`.
- Question: if greedy policy eval improves while sampled/search-action policy
  eval fails, the learned policy may have a useful mode that sampling hides. If
  greedy eval also fails, the policy head is not simply under-sharpened; the
  target/search signal is still not ranking PGX-level actions.

## Sparse Check 2026-06-17 12:32 UTC

- No DT hard-PGX run has hit the objective.
- The new soft-PGX ladder (`xbr49ti1`, `sgjfhl59`), random mean-utility sanity
  (`2gwzbv2t`), and greedy-eval controls (`653e6t79`, `92a5fbh8`) are all
  registered but still have no synced eval rows. Do not judge them yet.
- Corrected mean-utility hard-PGX runs are still too young:
  - `06hjdp8v`: step 0 `avg_R=-0.991211`, entropy `2.97794`.
  - `fikpoeb1`: step 4 `avg_R=-1.000000`, entropy `2.39189`.
  - `clh6t6vo`: step 2 `avg_R=-0.996094`, entropy `2.93284`.
- Scalar Gumbel greedy eval `rr56jqv3` is above the Go PGX random floor:
  step 43 `avg_R=-0.492188`, win `0.253906`, but it is not DT and still far
  from parity.
- Hex8 random eval `brc1dycu` is now essentially solved by step 43:
  `avg_R=0.998047`, win `0.999023`, entropy `2.72040`, `alpha_Q=14.91`.
  This is useful negative evidence against the simple alpha-growth explanation:
  concentration growth alone is not sufficient to cause failure.
- Decision: wait. The active diagnostic branches are too young; recycling now
  would create duplicate noise.

## Sparse Check And Hex8 Strong-Baseline Launch 2026-06-17 12:46 UTC

- No DT hard-PGX run has hit the objective.
- Live summary:
  - `rr56jqv3` scalar Gumbel greedy eval reached step 48
    `avg_R=-0.355469`, win `0.322266`. This is useful PGX-control evidence,
    but it is not DT.
  - `gsobsq6d`, `7agypa11`, `xt0iahzu`, `06hjdp8v`, `clh6t6vo`, `653e6t79`,
    and `92a5fbh8` remain far below the PGX win-rate floor.
  - `2v2u9q7x` and `fikpoeb1` have W&B entropy summaries at `0`; treat those
    summaries cautiously and do not use them as valid canaries without checking
    the exact logged target path.
  - `2gwzbv2t` is learning against random at step 4 (`avg_R=0.333984`, win
    `0.666992`). This says the corrected scalar target can learn Go9 signal
    against random, not that it can approach PGX.
  - Soft-PGX ladder is separating by opponent strength: temperature `2.0`
    `xbr49ti1` is step 4 `avg_R=-0.840820`, win `0.079590`; temperature `5.0`
    `sgjfhl59` is step 4 `avg_R=-0.614258`, win `0.192871`, near but still
    below the 20% floor.
- Hex8 random result finalized:
  - `brc1dycu` finished at step 50 with `avg_R=0.998047`, win `0.999023`; best
    by step 100 was step 46 `avg_R=1.0`.
  - It crashed only at checkpoint finalization because that run overrode
    checkpoint retention on a two-process local-disk topology. The learning
    result before that crash is still decisive for the random-baseline question.
- Reused topology pair `9,14` for the next Hex diagnostic:
  - Intended run: `hex8-dt-checkpoint-gumbelbase-w9w14`.
  - Hypothesis: if Hex8 DT learns random but stalls against a solved checkpoint
    baseline, baseline strength is an environment-independent explanation for
    the Go PGX stall. If Hex8 also learns the solved checkpoint baseline, the
    Go failure is more Go-specific or PGX-specific.
  - First launch `amkhzrmn` crashed at step 0 because the scalar checkpoint
    baseline was evaluated with `baseline_search.kind=dirichlet_thompson`.
  - Second launch failed before training because Hydra required
    `+eval.baseline_search.gumbel.*` when changing a DT baseline search config
    to Gumbel.
  - Third launch failed before training because `hex8.yaml` also needed
    `+eval.baseline_action_commitment_type=search_action`.
  - Relaunched with `eval.baseline_search.kind=gumbel`,
    `+eval.baseline_search.gumbel.*`, and
    `+eval.baseline_action_commitment_type=search_action`.
  - Final live run: `t667fbu0` / W&B `swept-dragon-632`. It reached
    two-process JAX assignment on workers `9,14` and started iteration 0 eval.
    No eval row had synced yet at the first check.

## Worker 0 Self-Play Commitment Ablation 2026-06-17 12:56 UTC

- Refreshed W&B before recycling:
  - Soft-PGX ladder is separating by opponent temperature under the same base
    DT recipe:
    - `xbr49ti1` temperature `2.0`: step 7 `avg_R=-0.718750`, win
      `0.140625`, `train/policy_target_entropy=2.65943`.
    - `sgjfhl59` temperature `5.0`: step 7 `avg_R=-0.450195`, win
      `0.274902`, `train/policy_target_entropy=2.65943`.
  - Hard-PGX DT runs remain below the useful win-rate floor so far.
  - The current entropy values are still broad; this is weak real signal, not a
    sharp solved policy.
- Recycled worker `0` because scalar Gumbel control `rnwartle` had already
  finished. Eopod reported `READY` and no TPU processes.
- New run:
  - `dt-base-parg-w0` / W&B `z379f24e` (`splendid-lake-633`).
  - Base `go9x9_3` DT recipe, hard PGX eval, eval player stays
    `posterior_sample`.
  - Only intended mechanism change versus the base posterior-sample recipe:
    `selfplay.action_commitment_type=posterior_argmax`.
- Hypothesis:
  - If this separates from posterior-sample self-play, the learned signal was
    being diluted by sampled self-play trajectories.
  - If it remains below the floor while the T5 ladder improves, the bottleneck
    is probably policy strength/target ranking rather than self-play sampling
    alone.
- Startup status: worker `0` reached JAX TPU initialization, registered W&B
  run `z379f24e`, and started iteration 0 eval.
- Sanity check on Hex8 solved-checkpoint run `t667fbu0`: not hung. The slow
  first eval completed and the run reached step 3, still at `avg_R=-1.000000`.
  This is early evidence that a strong clean Hex baseline can recreate the
  "learns random, fails strong policy" pattern, but it is not mature enough to
  prune.

## Policy-Fit Instrumentation 2026-06-17 12:59 UTC

- User question: why is `policy_loss` around `ln(32)` while
  `policy_target_entropy` is around `ln(16)`? If targets were simple and
  coherent, the model should fit them.
- Checked current logs:
  - Soft-PGX `xbr49ti1`/`sgjfhl59`: `policy_loss=3.2211`,
    `policy_target_entropy=2.6295`, `policy_kl_hat=0.5916`.
  - `7agypa11`: `policy_loss=3.1835`, `policy_target_entropy=2.5934`,
    `policy_kl_hat=0.5901`.
  - `xt0iahzu`: `policy_loss=3.0319`, `policy_target_entropy=2.4932`,
    `policy_kl_hat=0.5387`.
- Interpretation: the model is not merely limited by broad targets; it is not
  fitting even the broad target distribution on the logged batch. Because the
  logged loss is cross entropy, the gap over target entropy is the empirical
  `KL(target || model)`.
- Added `train/policy_pred_entropy` to future runs. This metric is the entropy
  of the model policy after the legal-action softmax on the same masked rows.
  It separates "still uniform/high entropy" from "sharp but wrong."
- Verification: `JAX_PLATFORMS=cpu uv run pytest tests/test_loss_masks.py -q`
  passed: `22 passed, 1 skipped`.

## Policy-Fit Ablation Launch 2026-06-17 13:07 UTC

- Refreshed W&B before recycling:
  - Hard-PGX DT runs still have no success.
  - Base soft-PGX ladder remains separated but weak:
    - `xbr49ti1` T2 step 11 `avg_R=-0.738281`, win `0.130859`,
      `policy_target_entropy=2.64951`, `policy_kl_hat=0.59268`.
    - `sgjfhl59` T5 step 11 `avg_R=-0.430664`, win `0.284668`,
      `policy_target_entropy=2.64951`, `policy_kl_hat=0.59268`.
  - The persistent `~0.6` policy KL gap makes policy fit a valid bottleneck
    hypothesis.
- Recycled non-DT controls to avoid spending TPU time on branches outside the
  objective:
  - Worker `1`: `gsobsq6d` current-code Gumbel reproduction, step 43
    `avg_R=-0.976562`, best<=100 step 42 `avg_R=-0.967773`.
  - Worker `11`: `rr56jqv3` scalar Gumbel greedy eval, step 53
    `avg_R=-0.648438`; it had already served as a non-DT PGX-control branch.
- New DT fit ablations:
  - Worker `1`: `dt-fit-b1024-hard-w1` / W&B `a4xvb4ms`
    (`sandy-star-635`). Base `go9x9_3`, hard PGX sampled eval,
    `training.batch_size=1024`.
  - Worker `11`: `dt-fit-b1024-pgxT5-w11` / W&B `5qgqx299`
    (`smooth-cosmos-634`). Same but PGX baseline policy temperature `5.0`.
- Mechanism under test:
  - Reducing train minibatch from `4096` to `1024` increases optimizer steps per
    self-play batch while keeping the DT target/search operator fixed.
  - If `policy_kl_hat` falls and `policy_pred_entropy` shows a non-uniform
    fitted policy, then model fit/optimization is a real bottleneck.
  - If `policy_kl_hat` remains high, the target is likely inconsistent or
    moving too fast rather than merely under-optimized.
- Startup status: both runs reached JAX TPU initialization, registered W&B IDs,
  and started iteration 0 eval.

## Sparse Check 2026-06-17 13:09 UTC

- No DT hard-PGX run has hit the objective.
- The new policy-fit ablations are registered but still have no synced eval or
  train row:
  - `a4xvb4ms` (`dt-fit-b1024-hard-w1`) has W&B state `running`, no `_step`.
  - `5qgqx299` (`dt-fit-b1024-pgxT5-w11`) has W&B state `running`, no `_step`.
  Do not recycle them before `train/policy_pred_entropy` and
  `train/policy_kl_hat` have real rows.
- Existing ladder status:
  - T2 `xbr49ti1` step 12 `avg_R=-0.710938`, win `0.144531`,
    `policy_target_entropy=2.62156`, `policy_kl_hat=0.57728`.
  - T5 `sgjfhl59` step 12 `avg_R=-0.349609`, win `0.325195`,
    `policy_target_entropy=2.62156`, `policy_kl_hat=0.57728`.
  The ladder still supports weak PGX-correlated signal, but this is not hard
  PGX progress.
- Hex8 solved-checkpoint ablation `t667fbu0` reached step 16 and remains
  `avg_R=-1.000000`, win `0`; this strengthens the baseline-strength
  hypothesis but is not a Go solution.
- Decision: wait. The active experiments are now asking distinct questions and
  none has crossed a decision boundary.

## Sparse Check 2026-06-17 13:11 UTC

- No DT hard-PGX run has hit the objective.
- The policy-fit ablations still have no usable rows:
  - `a4xvb4ms` (`dt-fit-b1024-hard-w1`): W&B state `running`, no `_step`.
  - `5qgqx299` (`dt-fit-b1024-pgxT5-w11`): W&B state `running`, no `_step`.
  These should not be judged until they log `train/policy_pred_entropy`,
  `train/policy_kl_hat`, and at least a few eval points.
- Soft-PGX ladder became noisy:
  - T2 `xbr49ti1`: step 13 `avg_R=-0.964844`, win `0.017578`; best<=100 still
    step 5 `avg_R=-0.691406`, win `0.154297`.
  - T5 `sgjfhl59`: step 13 `avg_R=-0.778320`, win `0.110840`; best<=100 still
    step 12 `avg_R=-0.349609`, win `0.325195`.
  Interpretation: the T5 run has shown an above-floor point, but the signal is
  not stable enough to treat as strength progress.
- Hex8 solved-checkpoint `t667fbu0` remains a strong-baseline failure at step
  18, `avg_R=-1.000000`.
- Decision: wait. No branch has a clean new decision boundary yet, and probing
  again before the fit runs log rows would be noise.

## Fit Ablation Pane Check 2026-06-17 13:13 UTC

- W&B still showed no synced rows for `a4xvb4ms` and `5qgqx299`, but tmux panes
  confirmed both runs passed startup and completed the first eval/train display
  update:
  - `a4xvb4ms` hard PGX: iteration 0 display `avg_R=-0.9844`,
    `policy_loss=3.6345`, `q_dir_kl_loss=0.0272`.
  - `5qgqx299` PGX T5: iteration 0 display `avg_R=-0.8652`,
    `policy_loss=3.6345`, `q_dir_kl_loss=0.0272`.
- This only proves the jobs are healthy. The decision metric
  `train/policy_pred_entropy` is not visible in the pane and has not synced to
  W&B yet, so the policy-fit hypothesis is still unevaluated.

## Sparse Check 2026-06-17 13:14 UTC

- No DT hard-PGX run has hit the objective.
- Policy-fit ablations still have no W&B rows:
  - `a4xvb4ms` hard PGX: W&B state `running`, no `_step`.
  - `5qgqx299` PGX T5: W&B state `running`, no `_step`.
  The panes previously showed both runs past iteration 0 startup, so treat this
  as W&B sync lag unless a later pane check shows an error.
- Soft-PGX ladder remains noisy but the T5 branch is again above the floor:
  - T2 `xbr49ti1`: step 14 `avg_R=-0.688477`, win `0.155762`.
  - T5 `sgjfhl59`: step 14 `avg_R=-0.368164`, win `0.315918`.
- Hard-PGX sampled posterior-argmax self-play `z379f24e` is still far below the
  floor at step 4, `avg_R=-0.886719`, win `0.056641`.
- Decision: wait. The only run family that can answer the immediate policy-fit
  hypothesis has not synced its decisive metrics yet.

## Sparse Check 2026-06-17 13:20 UTC

- No DT hard-PGX sampled run has hit the objective.
- The policy-fit ablations now have W&B rows and the new
  `train/policy_pred_entropy` metric:
  - Hard PGX `a4xvb4ms`: step 1 `avg_R=-0.980469`, win `0.009766`,
    `policy_loss=3.64526`, `policy_target_entropy=2.87916`,
    `policy_pred_entropy=3.64443`, `policy_kl_hat=0.76610`.
  - PGX T5 `5qgqx299`: step 1 `avg_R=-0.888672`, win `0.055664`,
    same train metrics because the training recipe is the same and only eval
    baseline temperature differs.
- Interpretation:
  - At initialization / very early training, the model policy is still much
    broader than the target. This is the "too uniform / not fit yet" failure
    mode, not a sharp-wrong model.
  - One step is not enough to decide whether smaller batch fixes the gap. The
    decision metric is the trajectory of `policy_pred_entropy` and
    `policy_kl_hat`, not the step-1 value.
- Other useful signals:
  - Soft PGX T5 `sgjfhl59` remains above the 20% floor at step 16
    (`avg_R=-0.401367`, win `0.299316`), but this is not hard-PGX objective
    success.
  - Hard-PGX greedy-eval `653e6t79` has best<=100 step 12
    `avg_R=-0.544922`, win `0.227539`; this suggests a weak top-action signal
    may exist, but changing eval action selection alone is not a valid
    objective win.
  - Hex8 solved-checkpoint `t667fbu0` is still essentially losing at step 30
    (`avg_R=-0.998047`, win `0.000977`), supporting the strong-baseline
    failure hypothesis.
- Decision:
  - Do not recycle workers `1` or `11` yet; they are now answering the
    fit-vs-target-consistency question.
  - Do not launch another broad Go sweep until the fit rows show whether the
    CE/KL gap closes.

## Fixed-State Target Audit 2026-06-17 13:22 UTC

- Ran:
  `JAX_PLATFORMS=cpu uv run python scripts/diagnose_dt_signal.py --configs go9x9_3 --plies 0 8 16 --batch-size 8 --policy-samples 128 --pgx-baseline-id go_9x9_v0`
- Results:
  - Ply 0: `H(target)=3.9751`, `exp(H)=53.33`, top gap `0.0039`,
    PGX top agreement `0.0000`, PGX rank `30.38`.
  - Ply 8: `H(target)=3.9126`, `exp(H)=50.15`, top gap `0.0049`,
    PGX top agreement `0.1250`, PGX rank `23.75`.
  - Ply 16: `H(target)=3.8244`, `exp(H)=45.85`, top gap `0.0088`,
    PGX top agreement `0.0000`, PGX rank `23.88`.
- Interpretation:
  - At random initialization, the DT posterior-best target is not a simple
    target that a model should trivially fit. It is extremely diffuse, with
    roughly 46-53 effective actions and sub-1% top-action gaps.
  - This does not explain the live CE/KL gap by itself, because live training
    targets have lower entropy around `2.6-2.9`. It does show that the search
    operator starts with weak ranking evidence and likely needs the model to
    improve before the target becomes coherent.
  - Available `checkpoints/9_solved` are Hex checkpoints, not Go9 PGX
    checkpoints, and they are scalar Boardlaw models. The current diagnostic
    requires Dirichlet value/action heads, so do not pretend this script has
    audited trained Go9 DT checkpoints.

## Entropy And Soft-Baseline Check 2026-06-17 13:49 UTC

- No DT hard-PGX run has hit the objective.
- The better-looking `sgjfhl59` and `5qgqx299` rewards are explained by eval
  baseline temperature:
  - `sgjfhl59`: `eval.baseline_search.kind=policy`,
    `eval.baseline_search.policy.temperature=5`, best<=100 step 26
    `avg_R=-0.326172`, win `0.336914`.
  - `5qgqx299`: same T5 baseline, best<=100 step 12
    `avg_R=-0.302734`, win `0.348633`.
  - Its hard-PGX training twin `a4xvb4ms` has identical train metrics at step
    12 but hard baseline temperature `1`, best<=100 step 10
    `avg_R=-0.857422`, win `0.071289`.
  Therefore T5 runs are ladder diagnostics, not objective progress.
- Entropy / cross-entropy finding:
  - `a4xvb4ms` step 12: `policy_loss=3.13982`,
    `policy_target_entropy=2.58677`, `policy_pred_entropy=3.13958`,
    `policy_kl_hat=0.55305`.
  - The model is not fully uniform anymore, but its effective prediction
    support is still roughly `exp(3.14) ~= 23` actions against a target of
    `exp(2.59) ~= 13` actions.
  - Smaller training minibatches did not remove the KL gap; the gap is still
    about the same as base runs.
- Next hypothesis:
  - The persistent gap plus rising alpha confidence is more consistent with
    bootstrapped concentration feedback / target drift than with a simple
    optimizer-step bottleneck.
  - Use the existing mathematically aligned concentration-projection knobs:
    `training.losses.dirichlet_target_prior_concentration` and
    `selfplay.search.dirichlet_thompson.search_prior_concentration`.
  - These preserve posterior means and evidence while preventing the network's
    learned alpha concentration from being reused as unbounded confidence.
- Planned ablations:
  - Target-prior projection only: hard PGX, base Go9 DT, set
    `training.losses.dirichlet_target_prior_concentration=3.0`.
  - Search+target projection: same, additionally set
    `selfplay.search.dirichlet_thompson.search_prior_concentration=3.0`.
- Decision rule:
  - Useful if `policy_target_entropy` moves below the current `2.5..2.7`
    plateau, `policy_kl_hat` does not increase, and hard-PGX win rate crosses
    the `20%` floor by step 100.
  - Falsified if it leaves hard-PGX best<=100 near the old `5..10%` range and
    keeps target entropy around `ln(16)`.

## Concentration-Projection Launch 2026-06-17 13:54 UTC

- Recycled answered soft-baseline ladder workers:
  - Worker `10` / old W&B `xbr49ti1` (`T=2`) was interrupted at about step 30.
  - Worker `12` / old W&B `sgjfhl59` (`T=5`) was interrupted at about step 30.
  Both were diagnostic-only because their reward lift came from softened PGX,
  not different training metrics.
- First launch attempt failed because Hydra requires `+` for keys absent from
  the YAML file. Local override validation then passed for:
  - `+training.losses.dirichlet_target_prior_concentration=3.0`
  - `+selfplay.search.dirichlet_thompson.search_prior_concentration=3.0`
- New hard-PGX DT runs:
  - Worker `10`: `dt-targetprior-c3-w10`, W&B `61lbvlyg`
    (`solar-surf-636`). Verified config: `selfplay.search.kind=dirichlet_thompson`,
    `eval.baseline_search.policy.temperature=1`,
    `training.losses.dirichlet_target_prior_concentration=3`,
    `selfplay.search.dirichlet_thompson.search_prior_concentration=None`.
  - Worker `12`: `dt-searchtarget-c3-w12`, W&B `5u1nnnew`
    (`mild-hill-637`). Verified config: same hard PGX setup, plus
    `selfplay.search.dirichlet_thompson.search_prior_concentration=3`.
- Both runs reached JAX TPU startup, registered W&B, and started iteration 0
  eval. No train/eval rows yet.
- Verification:
  `JAX_PLATFORMS=cpu uv run pytest tests/test_dirichlet_q_search.py::test_posterior_targets_can_project_to_fixed_prior_concentration tests/test_play_search_tictactoe.py::test_dirichlet_backend_can_project_search_prior_concentration tests/test_config_validation.py::test_dirichlet_target_prior_concentration_must_be_positive_when_set tests/test_config_validation.py::test_dirichlet_search_prior_concentration_must_be_positive_when_set -q`
  passed: `4 passed`.

## Trained-Target Audit Setup 2026-06-17 14:05 UTC

- First concentration-projection rows are early and not decision-grade:
  - `61lbvlyg` (`dt-targetprior-c3-w10`) step 1:
    `avg_R=-0.986328`, win `0.006836`, `policy_loss=3.67016`,
    `policy_target_entropy=2.87972`, `policy_pred_entropy=3.66996`,
    `policy_kl_hat=0.79044`.
  - `5u1nnnew` (`dt-searchtarget-c3-w12`) step 0:
    `avg_R=-0.990234`, win `0.004883`, `policy_loss=3.63526`,
    `policy_target_entropy=2.84469`, `policy_pred_entropy=3.63302`,
    `policy_kl_hat=0.79057`.
  These do not yet support the concentration hypothesis; wait for real training
  steps before pruning.
- Added `--raw-snapshot` support to `scripts/diagnose_dt_signal.py` so fixed
  state target audits can load `SCACCHI_RAW_SNAPSHOT_DIR/model_*.pkl` snapshots.
  CPU smoke tests passed for both random-init and raw-snapshot restore paths:
  - `JAX_PLATFORMS=cpu uv run python scripts/diagnose_dt_signal.py --configs hex5 --plies 0 --batch-size 1 --policy-samples 2`
  - `JAX_PLATFORMS=cpu uv run python scripts/diagnose_dt_signal.py --configs hex5 --plies 0 --batch-size 1 --policy-samples 2 --raw-snapshot /tmp/tmpmwh6yg1q.pkl`
- Added `scripts/audit_tpu_raw_snapshot.sh` to fetch a raw snapshot from a TPU
  worker and immediately run the fixed-state Go9/PGX audit. Dry-run checked:
  `bash scripts/audit_tpu_raw_snapshot.sh --step 10 --dry-run`.
  When worker 8 has written step 10, run:
  `bash scripts/audit_tpu_raw_snapshot.sh --step 10`.
- Recycled worker `8`:
  - Killed W&B `2gwzbv2t` (`muarg-random-w8`) at step 32.
  - Current value: it already proved DT can learn Go9 against random
    (`avg_R=0.462891`, win `0.731445` at step 32).
  - Kill reason: random-baseline success does not answer hard-PGX target
    quality, and the worker is more useful producing trained Go DT snapshots.
- Launched worker `8` as `dt-base-snap-w8`, W&B `pfyum6xl`
  (`hopeful-breeze-638`):
  - Command:
    `SCACCHI_RAW_SNAPSHOT_DIR=/home/francescosacco/Scacchi-pod/raw_snapshots/go9_dt_base_w8 bash scripts/train_tpu_batch_parallel.sh --worker 8 --config-name go9x9_3 run.max_num_iters=120 eval.interval=1 eval.baseline_search.policy.temperature=1.0 checkpointing.save_interval_steps=10`
  - Role: base hard-PGX Go9 DT run that writes raw model snapshots every 10
    iterations for trained-target audits.
  - Question: do DT root posterior-best targets become better aligned with PGX
    top actions / one-step value rankings as the model trains, or do they stay
    diffuse/misaligned even when the network has learned anti-random features?
  - Decision rule: after snapshots at steps 10/20/30, copy them back and run
    `scripts/diagnose_dt_signal.py --raw-snapshot ... --pgx-baseline-id go_9x9_v0`.
    If PGX agreement, PGX rank, target top gap, or effective support do not
    improve materially, stop treating training hyperparameters as the main
    lever and focus on the DT target/search operator.

## 128-Sample Policy Metric Rule 2026-06-17

- Hypothesis: part of the confusing entropy story may be measurement bias.
  A 32-sample empirical posterior-best histogram can under-estimate true entropy
  and make a broad Go9 target look artificially sharp.
- Decision: use `128` posterior-best samples for DT policy entropy/KL canaries.
  Do not use `256` routinely; it is already too expensive unless a fixed-state
  audit shows 128 is still too noisy.
- Code change:
  - Added `search.dirichlet_thompson.policy_metric_samples`.
  - `policy_metric_tgt` can now use more samples than the training target.
  - The `policy_samples=0` search-target path also honors
    `policy_metric_samples` instead of hard-coding 32 metric samples.
- Verification:
  - `JAX_PLATFORMS=cpu uv run pytest tests/test_config_validation.py::test_dirichlet_thompson_allows_policy_metric_samples_when_set tests/test_config_validation.py::test_dirichlet_thompson_policy_metric_samples_must_be_positive_when_set tests/test_play_search_tictactoe.py::test_policy_samples_zero_keeps_full_posterior_metric_target tests/test_play_search_tictactoe.py::test_policy_metric_samples_can_use_more_samples_than_training_target -q`
    passed: `4 passed`.
  - Broader `tests/test_config_validation.py tests/test_play_search_tictactoe.py`
    still has three unrelated failures from current worktree/config expectations
    (`hex6`/`hex8` checkpoint baseline search kind and a DQAZ
    `posterior_argmax` training assertion). Do not treat those as evidence
    against the 128-sample metric change.

## AZDirichlet Init + 128-Sample Launch 2026-06-17 14:21 UTC

- User changed AZDirichlet layer initialization by removing zero initializers
  from the heads in `scacchi/network.py`.
- Recycled worker `9` from the Hex8 checkpoint probe. Kill reason: W&B
  `t667fbu0` had crossed the step-100 decision boundary at about `2%` win rate
  against its checkpoint baseline, so it was below the trust floor and had
  answered the "Hex8 can also stall against a strong baseline" question.
- Launched worker `9` as tmux `dt-init-ps128-w9`, W&B `y4f7h75j`
  (`pious-terrain-639`):
  - Command:
    `SCACCHI_SYNC=0 bash scripts/train_tpu_batch_parallel.sh --worker 9 --config-name go9x9_3 run.max_num_iters=120 eval.interval=1 eval.baseline_search.policy.temperature=1.0 +selfplay.search.dirichlet_thompson.policy_samples=128 +selfplay.search.dirichlet_thompson.policy_metric_samples=128`
  - Config verified in W&B:
    `model.network=aznet_dirichlet`,
    `eval.baseline_search.policy.temperature=1`,
    `selfplay.search.dirichlet_thompson.policy_samples=128`,
    `selfplay.search.dirichlet_thompson.policy_metric_samples=128`.
  - Two earlier launch attempts failed only because Hydra needed `+` for
    absent YAML keys (`policy_samples`, then `policy_sample_chunk_size`). The
    successful run omits the chunk-size override and uses the runtime default
    of `32`.
- Scientific question: does nonzero AZDirichlet head initialization change the
  early DT target/model entropy relationship or `policy_kl_hat` enough to move
  hard-PGX win rate above the `20%` trust floor by step 100?
- Decision rule: compare the first 10-30 steps against base hard-PGX DT runs.
  Useful if `policy_pred_entropy` is no longer forced to mirror a flat policy,
  `policy_kl_hat` drops without making the target sharp-wrong, and hard-PGX win
  rate crosses `20%` before step 100. If reward stays under the floor and
  entropy/KL remain in the same regime, initialization is not the main lever.
