current good runs:
- https://wandb.ai/pal/scacchi-az/runs/5iluv1nv
- https://wandb.ai/pal/scacchi-az/runs/43uohwan

## 2026-06-14 DT speedrun restart

Goal: train with `selfplay.search.kind=dirichlet_thompson` and reach
`eval/vs_baseline/avg_R >= 0` by step 100 against `go_9x9_v0`, without changing
the eval baseline or reward signal.

Evidence used:
- `43uohwan` is a fast scalar Gumbel run that crossed zero at step 94. Its
  useful shape is `posterior_sample` eval, `selfplay.batch_size=1024`,
  `training.batch_size=4096`, `lr=1e-3`, policy-only losses.
- `5iluv1nv` is the useful Dirichlet-net Gumbel run. It did not cross by step
  100, but later learned strongly, so its Dirichlet loss scale is a useful
  controlled variant.
- Current worktree asserts training uses `selfplay.action_commitment_type:
  posterior_sample`; the new DT runs follow that instead of the earlier
  `search_action` experiments.

Base config added: `scacchi/configs/go9x9_dirichlet_thompson_ps.yaml`.
It uses `aznet_dirichlet`, DT self-play and DT eval, seed 1, eval every step,
checkpointing disabled, single-worker scale, and `posterior_sample`
commitment.

Worker 0 initially failed with `open(/dev/accel*): Operation not permitted`.
Inspection showed PID 1231080 was a long-running `pytest
tests/test_play.py tests/test_evaluations.py` process holding the device, not a
training run. I terminated that specific pytest process and relaunched worker 0.

Launched one single-worker run per TPU worker:

| worker | W&B run | variant |
| --- | --- | --- |
| 0 | `s78wszjy` | base DT32, blocks1, policy_samples32 |
| 1 | `jn3175xx` | DT16 |
| 2 | `1c7snrfq` | DT8 |
| 3 | `m4ulqt2u` | DT32 with `num_blocks=4` |
| 4 | `w41dbqpm` | posterior target samples 8 |
| 5 | `iqd5vvyh` | Dirichlet/value losses from good Dirichlet-Gumbel scale |
| 6 | `kvft2gi4` | same loss family, smaller weights |
| 7 | `18od56jt` | outcome losses only |
| 8 | `rzs7znay` | Q Dirichlet KL only |
| 9 | `4py870nh` | policy-only with `loss_mask_mode=pgx` |
| 10 | `fivrm9i5` | policy-only with `learning_rate=3e-4` |
| 11 | `piwp248v` | policy-only with `grad_clip_norm=1.0` |
| 12 | `pw2vm2d0` | larger batch: self-play 2048, train 8192 |
| 13 | `5z9w0cl7` | `kappa_terminal=4` |
| 14 | `hiqfo1i3` | `kappa_terminal=16` |
| 15 | `xlb1mu3w` | `policy_samples=0`, use backend action weights target |

Initial W&B check: all 16 are `running`, all report
`selfplay.search.kind=dirichlet_thompson`, `eval.player_search.kind=dirichlet_thompson`,
and `selfplay.action_commitment_type=posterior_sample`. No eval rows had been
logged yet at the time of launch verification.

First short follow-up after launch:
- No tmux panes show immediate crash signatures.
- Worker 2 / `1c7snrfq` (`sim8`) already logged eval rows and is very weak:
  step 0 `avg_R=-0.976562`, step 1 `avg_R=-0.996094`.
- The other workers were still running startup/first eval when checked. This is
  expected for the 16/32-simulation variants on one worker; next useful read is
  after they have logged several eval rows rather than during first compilation.

30-minute follow-up:
- All `dtps-*` tmux windows are still alive; no crash signatures in the pane
  tails.
- No run has reached `avg_R >= 0` by any step, and all current evals are very
  negative.
- Current best early result is worker 7 / `18od56jt` (`outcome`), step 4
  `avg_R=-0.943359`. This is only "best" relative to a poor batch and is not
  close to the goal.
- Other representative results: `s78wszjy` base step 3 `-0.986328`,
  `jn3175xx` sim16 best step 8 `-0.976562`, `1c7snrfq` sim8 best step 15
  `-0.974609`, `w41dbqpm` policy_samples8 best step 2 `-0.974609`.
- Interpretation: DT + posterior-sample evaluation is initially much worse
  than the scalar-Gumbel baseline trajectory. The experiments are still too
  early to distinguish learning from startup variance, so keep them running for
  more eval rows before replacing slots.

One-hour follow-up and slot replacement:
- Still no run reached `avg_R >= 0` by step 100. Best current DT-eval result is
  still worker 7 / `18od56jt`, best step 7 `avg_R=-0.931641`.
- DT-eval appears to be measuring a very weak stochastic search policy. To test
  the model-training objective more directly, I reclaimed three bad/slow slots
  and kept self-play DT while changing only player eval to the policy head
  (`eval.player_search.kind=policy`, `eval.player_action_commitment_type=posterior_sample`).
- Stopped:
  - worker 2 / `1c7snrfq` (`sim8`), step 46, best100 `-0.972656`.
  - worker 3 / `m4ulqt2u` (`blocks4`), step 2 after long runtime, best100 `-0.976562`.
  - worker 12 / `pw2vm2d0` (`bigbatch`), step 8, best100 `-0.980469`.
- Worker 3 had an orphaned remote training process after the tmux stop:
  `python -u -m scacchi.train ... dtps_3_blocks4`, PID 514317. I killed that
  exact stale process before relaunching.
- Launched replacements:
  - worker 2 / `t7lwpyk2`: base DT self-play, policy-head eval.
  - worker 3 / `xa5lwecf`: outcome-loss variant, policy-head eval.
  - worker 12 / `j33thog7`: Dirichlet-loss scale variant, policy-head eval.

Policy-eval replacement first read:
- `t7lwpyk2` base policy eval: step 3 `avg_R=-0.982422`.
- `xa5lwecf` outcome-loss policy eval: step 3 `avg_R=-0.990234`, best step 1
  `-0.988281`.
- `j33thog7` Dirichlet-loss policy eval: step 3 `avg_R=-0.982422`.
- Interpretation: evaluating the policy head instead of the DT player does not
  immediately recover the scalar-Gumbel trajectory. The issue is likely in the
  training data/search target quality at initialization, not just the eval
  commitment.

Second replacement batch:
- After another hour, no hit. Best active result remained worker 7 /
  `18od56jt`, best100 `-0.931641`; the rest clustered around `-0.95` to
  `-0.99`.
- Reclaimed three weak slots:
  - worker 8 / `rzs7znay` (`qonly`), best100 `-0.974609`.
  - worker 10 / `fivrm9i5` (`lr3e4`), best100 `-0.970703`, last `-0.998047`.
  - worker 13 / `5z9w0cl7` (`kterm4`), best100 `-0.974609`.
- Worker 13 had an orphaned old training process, PID 485063. I killed that
  exact stale process before relaunching.
- Launched replacements:
  - worker 8 / `74pq1s9f`: DT self-play, policy eval, `loss_mask_mode=pgx`.
  - worker 10 / `ft5sqx2u`: DT self-play, policy eval, `policy_samples=0`.
  - worker 13 / `wtj54ap3`: DT self-play, policy eval, `policy_samples=8`.

Current diagnostic:
- Still no hit (`avg_R >= 0` by step <= 100). The most useful comparison is
  target entropy:
  - scalar Gumbel good run `43uohwan` crossed at step 94; by step 60 it was
    already `avg_R=-0.572266`.
  - Dirichlet-Gumbel reference `5iluv1nv` had policy-target entropy around
    `0.75-1.2` after the first few dozen steps and crossed only at step 125.
  - Current DT variants mostly keep policy-target entropy around `2.5-2.8`;
    `policy_samples=8` lowers this to about `1.76` but has not improved reward.
- Interpretation: DT self-play is producing too-diffuse early policy targets,
  so the next small test is a sharper single posterior-best sample target.

Third replacement:
- Reclaimed worker 1 / `jn3175xx` (`sim16`), step 62, last eval `-0.992188`,
  best100 `-0.970703`, policy-target entropy about `2.61`. This was behind the
  scalar-Gumbel reference trajectory by step 60 and showed no upward signal.
- Stopped exact remote process on worker 1:
  `/home/francescosacco/Scacchi-pod/.venv/bin/python -u -m scacchi.train ... checkpoints/dtps_1_sim16`,
  PID `496393`. `SIGTERM` did not stop it after 5 seconds, so I used `kill -9`
  on that exact PID and verified no matching `dtps_1_sim16` process remained.
- Launched worker 1 / `0d42msim`:
  DT self-play, `policy_samples=1`, policy-head eval, checkpoint directory
  `checkpoints/dtps_1_ps1`.

Fourth replacement:
- Status pass: no active run has reached `avg_R >= 0` by step <= 100. Best
  remains worker 7 / `18od56jt`, best100 `-0.931641`.
- Reclaimed worker 11 / `piwp248v` (`clip1`), step 34, last eval `-0.984375`,
  best100 `-0.972656`. This slot was effectively a duplicate of the base run:
  same target entropy (`~2.56`) and same eval trajectory as `s78wszjy`/`4py870nh`.
- Stopped exact remote process on worker 11:
  `/home/francescosacco/Scacchi-pod/.venv/bin/python -u -m scacchi.train ... checkpoints/dtps_11_clip1`,
  PID `485807`. `SIGTERM` did not stop it after 5 seconds, so I used `kill -9`
  on that exact PID and verified no matching `dtps_11_clip1` process remained.
- Launched worker 11 / `guaasdn5`:
  DT self-play, `policy_samples=4`, policy-head eval, checkpoint directory
  `checkpoints/dtps_11_ps4`.
- Current sharp-target runs `0d42msim` (`policy_samples=1`) and `guaasdn5`
  (`policy_samples=4`) are both running but still inside their first eval, so
  there is no outcome evidence from them yet.

Latest status pass:
- All intended tmux worker windows are present. The two known duplicate old
  shells for workers 10 and 13 remain, but active replacement windows are still
  `21:dtps-10-policeval-target` and `22:dtps-13-policeval-ps8`.
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Best active result is still worker 7 / `18od56jt`, best100 `-0.931641`;
  the only other comparatively less bad variants are worker 6 / `kvft2gi4`
  best100 `-0.949219` and worker 5 / `iqd5vvyh` best100 `-0.953125`.
- `policy_samples=8` variants lower policy-target entropy to about `1.75-1.78`
  but remain around `-0.97` to `-0.99` reward early. `policy_samples=0` has
  even higher entropy (`~2.79`) and no reward signal.
- `0d42msim` (`policy_samples=1`) and `guaasdn5` (`policy_samples=4`) are
  running but still have no logged eval/train rows. Do not replace more slots
  until these sharp-target ablations have at least several rows.

Sharp-target startup check:
- W&B API still has no synced rows for `0d42msim` or `guaasdn5`, but tmux pane
  output shows worker 1 / `0d42msim` completed step 0 after compile/eval:
  `eval/vs_baseline/avg_R=-0.990234`, `train/policy_loss=3.2111`,
  `train/q_loss_weight_mean=1.0`. This confirms the `policy_samples=1` variant
  is running and producing the expected one-sample target weighting, but it has
  no positive reward signal yet.
- Worker 11 / `guaasdn5` is still in first eval. Remote process PID `504326` is
  alive and using CPU, so treat it as compiling/evaluating, not dead.
- No slot replacement now: all workers are occupied, no run has hit the goal,
  and the sharp-target ablations need several rows before they can inform the
  next change.

Fifth replacement:
- W&B still shows no hit. `0d42msim` (`policy_samples=1`) synced step 0
  `avg_R=-0.990234`; tmux shows step 1 `avg_R=-0.988281`. `guaasdn5`
  (`policy_samples=4`) completed step 0 in tmux with `avg_R=-0.990234`.
- Reclaimed worker 9 / `4py870nh` (`pgxmask`), step 36, last eval
  `-0.990234`, best100 `-0.972656`. It was effectively a stale base-like
  duplicate, with the same high-entropy target trajectory as the base run.
- Stopped exact remote process on worker 9:
  `/home/francescosacco/Scacchi-pod/.venv/bin/python -u -m scacchi.train ... checkpoints/dtps_9_pgxmask`,
  PID `483066`. `SIGTERM` did not stop it after 5 seconds, so I used `kill -9`
  on that exact PID and verified no matching `dtps_9_pgxmask` process remained.
- Launched worker 9 / `xnh5t2vz`:
  DT self-play, `policy_samples=1`, policy-head eval, plus the outcome-loss
  settings from the least-bad `outcome` family
  (`value_outcome_weight=0.25`, `q_outcome_weight=0.1`,
  `loss_mask_mode=value`, terminal edge/parent targets enabled). Checkpoint
  directory: `checkpoints/dtps_9_ps1_outcome`.

Sixth replacement:
- Status pass still shows no hit (`avg_R >= 0` by step <= 100). Sharp-target
  runs remain negative: `0d42msim` step 1 `-0.988281`, `guaasdn5` step 0
  `-0.990234`; `xnh5t2vz` is launched but has not logged eval rows yet.
- Reclaimed worker 2 / `t7lwpyk2` (`policeval_base`), step 33, last eval
  `-0.986328`, best100 `-0.978516`, policy-target entropy still high
  (`~2.59`). It was another stale base-like policy-eval slot.
- Stopped exact remote process on worker 2:
  `/home/francescosacco/Scacchi-pod/.venv/bin/python -u -m scacchi.train ... checkpoints/dtps_2_policeval_base`,
  PID `531824`. `SIGTERM` did not stop it after 5 seconds, so I used `kill -9`
  on that exact PID and verified no matching `dtps_2_policeval_base` process
  remained.
- Launched worker 2 / `4slwcxag`:
  DT self-play, policy-head eval, `training.losses.policy_target_mode=winner_action`.
  This tests whether training on the actually played action from winning
  frames can escape the diffuse/random posterior-target problem without
  changing the eval baseline or reward.

Latest no-replacement status:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Current sharp-target rows are still negative:
  - `0d42msim` (`policy_samples=1`) step 3 last `-0.990234`, best100
    `-0.988281`; target entropy is `0` by construction and
    `q_loss_weight_mean=1.0`.
  - `guaasdn5` (`policy_samples=4`) step 1 last/best100 `-0.988281`, target
    entropy around `1.24`.
  - `xnh5t2vz` (`policy_samples=1` + outcome losses) has not synced W&B rows
    yet, but its tmux pane shows step 0 `avg_R=-0.990234`,
    `policy_loss=3.1278`, `q_loss_weight_mean=1.0`, value/q outcome losses
    active.
  - `4slwcxag` (`winner_action`) has not completed first eval yet; remote PID
    `544101` is alive and active.
- Decision: no further slot replacement this pass. All intended worker windows
  are present, newest ablations are active, and they need several rows before
  the next change is informative.

Corrected status pass:
- Corrected the W&B status helper locally to treat step `0` as a real
  `<=100` step instead of dropping it via a falsey check. Still no hit.
- All intended tmux worker windows are present.
- Best active result is unchanged: worker 7 / `18od56jt`, best100
  `-0.931641`.
- Latest sharp/alternative target rows:
  - `0d42msim` (`policy_samples=1`): step 3 last `-0.990234`, best100
    `-0.988281`.
  - `guaasdn5` (`policy_samples=4`): step 2 last `-0.994141`, best100
    `-0.988281`.
  - `xnh5t2vz` (`policy_samples=1` + outcome losses): step 0 synced,
    `-0.990234`.
  - `4slwcxag` (`winner_action`): running but no synced eval row yet.
- No replacement this pass. The newest variants are active but too early to
  judge; replacing more would mostly add churn rather than information.

Latest status pass:
- W&B still shows no hit (`avg_R >= 0` by step <= 100). All intended tmux
  worker windows are present.
- Best active result is unchanged: worker 7 / `18od56jt`, best100
  `-0.931641`.
- Sharp/alternative-target variants remain very weak:
  - `0d42msim` (`policy_samples=1`): W&B step 4 last `-0.992188`, best100
    `-0.988281`.
  - `guaasdn5` (`policy_samples=4`): W&B step 2 last `-0.994141`, best100
    `-0.988281`.
  - `xnh5t2vz` (`policy_samples=1` + outcome losses): W&B step 0 `-0.990234`;
    tmux pane shows step 2 still `-0.990234`, with outcome losses active and
    `q_loss_weight_mean=1.0`.
  - `4slwcxag` (`winner_action`): no W&B eval row yet, but tmux pane shows step
    0 `-0.990234`; remote PID `544101` is active.
- Decision: no replacement this pass. The newest ablations are running and still
  only have startup rows; another kill would not be information-driven.

Latest no-replacement status:
- W&B still shows no hit (`avg_R >= 0` by step <= 100), using corrected step-0
  handling. All intended tmux worker windows are present.
- Best active result remains worker 7 / `18od56jt`, best100 `-0.931641`.
- Current sorted best100 leaders:
  - `18od56jt` outcome: `-0.931641`
  - `kvft2gi4` weights025: `-0.949219`
  - `iqd5vvyh` weights1: `-0.953125`
- Newer ablations are still not promising but are too early to reclaim:
  - `0d42msim` (`policy_samples=1`) step 4 last `-0.992188`, best100
    `-0.988281`.
  - `guaasdn5` (`policy_samples=4`) step 3 last `-0.990234`, best100
    `-0.988281`.
  - `xnh5t2vz` (`policy_samples=1` + outcome losses) step 1 last `-0.990234`;
    pane shows it continuing normally.
  - `4slwcxag` (`winner_action`) has no synced W&B row yet, but pane shows step
    0 `-0.990234`; remote PID `544101` is alive.
- Decision: no replacement this pass. Keep all slots running until the newest
  variants have several rows.

Replacement status:
- W&B status pass still shows no hit (`avg_R >= 0` by step <= 100).
- Best active before replacement was worker 7 / `18od56jt` (`outcome`),
  best100 `-0.931641`. The current useful-but-not-enough leaders were:
  `kvft2gi4` (`weights025`) best100 `-0.939453` at step 44 and `iqd5vvyh`
  (`weights1`) best100 `-0.951172` at step 44.
- Readback from reference runs:
  - Scalar Gumbel `43uohwan` crosses zero at step 94 and has policy CE around
    `1.3-1.7` in the useful phase.
  - Dirichlet-Gumbel `5iluv1nv` does not cross by step 100 but improves much
    faster than current DT; its policy target entropy is roughly `0.7-1.2` and
    its Dirichlet heads sit near concentration `8` with `value/q_dir_kl_weight`
    `4/4`.
  - Current high-entropy DT targets around `2.5+` are too diffuse, but plain
    zero-entropy `policy_samples=1` without stronger value/Q learning is also
    not enough.
- Reclaimed three objectively stale slots:
  - Killed worker 15 remote PID `469894`, W&B `xlb1mu3w`, checkpoint
    `dtps_15_target_weights`; TERM did not exit, so used `kill -9`.
  - Killed worker 14 remote PID `477557`, W&B `hiqfo1i3`, checkpoint
    `dtps_14_kterm16`; TERM did not exit, so used `kill -9`.
  - Killed worker 4 remote PID `463719`, W&B `w41dbqpm`, checkpoint
    `dtps_4_ps8`; TERM did not exit, so used `kill -9`.
- Launched replacements, all still `selfplay.search.kind=dirichlet_thompson`
  and eval every step:
  - Worker 4 / W&B `0zc4tcr7` / remote PID `483614`: `dtps_4_w4ps4`,
    `policy_samples=4`, policy eval, Dirichlet loss scale `4/4`, outcome
    losses `0.25/0.1`, evidence-mass Q weighting, masked-mean Dirichlet KL,
    value mask, terminal targets, concentration clip `100`.
  - Worker 14 / W&B `x7aavdzs` / remote PID `499164`: `dtps_14_w4ps8`,
    same loss setup with `policy_samples=8`.
  - Worker 15 / W&B `yf3bmdkz` / remote PID `490392`:
    `dtps_15_w4ps4_b1024`, same as worker 4 but `training.batch_size=1024`
    to test smaller optimizer batches on the same self-play rows.
- Hypothesis: DT needs value/Q learning to become useful early, and policy
  targets need to be closer to the entropy range that worked for the
  Dirichlet-Gumbel bridge. These variants keep compute/search scale roughly
  fixed while testing that hypothesis.

Follow-up status:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Active best100 remains worker 7 / `18od56jt` (`outcome`) at step 7,
  `-0.931641`. The next best active rows are worker 6 / `kvft2gi4`
  (`weights025`) at step 44, `-0.939453`, and worker 5 / `iqd5vvyh`
  (`weights1`) at step 46, `-0.945312`.
- New replacement runs are alive but have no synced eval rows yet:
  - worker 4 / `0zc4tcr7` (`dtps_4_w4ps4`) is at initial eval/compile.
  - worker 14 / `x7aavdzs` (`dtps_14_w4ps8`) is at initial eval/compile.
  - worker 15 / `yf3bmdkz` (`dtps_15_w4ps4_b1024`) is at initial
    eval/compile.
- No further replacement this pass. The new sharper-target/loss-scale slots
  need first rows before they can be judged, and all intended worker slots are
  still occupied.

Latest action:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Reclaimed stale worker 0 base run:
  - Killed worker 0 remote Python PID `1248693`, W&B `s78wszjy`, checkpoint
    `dtps_0_base32`; TERM did not exit, so used `kill -9`.
  - Rationale: plain high-entropy base32 DT run was around step 40 with
    best100 `-0.972656` and no trend, superseded by sharper-target/loss-scale
    hypotheses.
- Launched worker 0 / W&B `7vxaa74t` / remote PID `1295752`:
  `dtps_0_w4ps4_dteval`, `policy_samples=4`, DT self-play and DT eval,
  Dirichlet loss scale `4/4`, outcome losses `0.25/0.1`, evidence-mass Q
  weighting, masked-mean Dirichlet KL, value mask, terminal targets,
  concentration clip `100`.
- This is the DT-eval counterpart to worker 4 `0zc4tcr7`; it tests the same
  stronger value/Q learning hypothesis while evaluating with the intended
  search instead of only the raw policy head.

Current status:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Current active best100 leaders:
  - worker 7 / `18od56jt` (`outcome`, DT eval): step 7 `-0.931641`;
    latest step 42 `-0.970703`.
  - worker 5 / `iqd5vvyh` (`weights1`, DT eval): step 47 `-0.939453`.
  - worker 6 / `kvft2gi4` (`weights025`, DT eval): step 44 `-0.939453`.
  - worker 12 / `j33thog7` (`policeval_w1`): step 44 `-0.957031`.
- Newer sharper/loss-scale runs are alive but still have no synced eval row:
  worker 0 / `7vxaa74t`, worker 4 / `0zc4tcr7`, worker 14 / `x7aavdzs`,
  worker 15 / `yf3bmdkz`.
- Decision: no further replacement this pass. The new slots need first-row
  evidence before judging the stronger value/Q learning hypothesis, and all
  TPU worker slots remain occupied.

Latest replacement:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- First rows arrived for the strong-loss policy-eval variants:
  - worker 4 / `0zc4tcr7` (`policy_samples=4`): step 0 `-0.990234`;
    target entropy `1.2229`, `q_loss_weight_mean=1.93`.
  - worker 14 / `x7aavdzs` (`policy_samples=8`): step 0 `-0.990234`;
    target entropy `1.7553`, `q_loss_weight_mean=1.93`.
  - worker 15 / `yf3bmdkz` (`policy_samples=4`, batch 1024): step 0
    `-0.990234`; target entropy `1.2228`, `q_loss_weight_mean=1.93`.
- Reclaimed stale worker 1:
  - Killed remote Python PID `514365`, W&B `0d42msim`, checkpoint
    `dtps_1_ps1`; TERM did not exit, so used `kill -9`.
  - Rationale: plain `policy_samples=1` reached step 10 with best100 only
    `-0.982422`; it showed that zero-entropy targets alone are not enough.
- Launched worker 1 / W&B `94cwovky` / remote PID `521404`:
  `dtps_1_w4ps1`, `policy_samples=1`, policy eval, Dirichlet loss scale
  `4/4`, outcome losses `0.25/0.1`, evidence-mass Q weighting, masked-mean
  Dirichlet KL, value mask, terminal targets, concentration clip `100`.
- Hypothesis: if the issue is that ps4/ps8 targets are still too soft but ps1
  failed only because the value/Q heads were not trained strongly enough, this
  should improve over the old `0d42msim` control without changing search
  compute.

Status pass 2026-06-14 16:42 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- All active runs report `state=running` in W&B, and tmux session `0` still has
  training windows for all 16 worker slots.
- Current active best100 leaders:
  - worker 7 / `18od56jt` (`outcome`, DT eval): step 7 `-0.931641`;
    latest step 43 `-0.962891`.
  - worker 5 / `iqd5vvyh` (`weights1`, DT eval): step 47 `-0.939453`;
    latest step 49 `-0.966797`.
  - worker 6 / `kvft2gi4` (`weights025`, DT eval): step 44 `-0.939453`;
    latest step 49 `-0.949219`.
  - worker 12 / `j33thog7` (`policeval_w1`): step 44 `-0.957031`;
    latest step 47 `-0.964844`.
- Strong-loss sharper-target variants now have early evidence:
  - worker 4 / `0zc4tcr7` (`w4ps4`): best/latest step 3 `-0.964844`;
    target entropy `1.2185`, `q_loss_weight_mean=2.04`.
  - worker 14 / `x7aavdzs` (`w4ps8`): best/latest step 2 `-0.980469`;
    target entropy `1.7415`, `q_loss_weight_mean=2.10`.
  - worker 15 / `yf3bmdkz` (`w4ps4_b1024`): best/latest step 2
    `-0.984375`; target entropy `1.2046`, `q_loss_weight_mean=2.17`.
- Pane checks for the newer zero-row W&B slots:
  - worker 0 / `7vxaa74t` completed step 0 in the pane with DT eval
    `-0.9844`; W&B sync has not surfaced the row yet.
  - worker 1 / `94cwovky` completed step 0 in the pane with policy eval
    `-0.9902`; W&B sync has not surfaced the row yet.
  - worker 8 / `29q6oem5` is in initial evaluation.
- Decision: no kill/replacement this pass. The run states and panes indicate
  the TPUs are occupied, and the newer high-loss/low-entropy variants need a
  little more data before they are judged.

Replacement 2026-06-14 16:47 UTC:
- W&B status immediately before replacement still showed no hit
  (`avg_R >= 0` by step <= 100).
- Reclaimed worker 3:
  - Killed remote Python PID `526290`, W&B `xa5lwecf`, checkpoint
    `dtps_3_policeval_outcome`.
  - TERM did not exit within 5 seconds, so used `kill -9` on PID `526290`.
    The old `wandb-core` helper exited shortly afterward.
  - Rationale: this was an old outcome-only policy-eval control at step 45
    with best100 step 16 `-0.980469` and latest step 45 `-0.998047`.
    It was dominated by worker 7's outcome DT-eval control and by the newer
    strong-loss variants.
- Launched worker 3 / W&B `fc8bcgee` / remote Python PID `540916`:
  `dtps_3_p05w5clip300`, self-play DT, policy eval, `policy_samples=4`,
  `policy_weight=0.5`, Dirichlet loss scale `5/5`, outcome losses `0.25/0.1`,
  evidence-mass Q weighting, masked-mean Q Dirichlet KL, value mask, terminal
  targets, concentration clip `300`.
- Hypothesis: DT's early search targets are still noisy, so policy CE should
  not dominate the first updates. This variant follows the existing
  non-Go config's direction by shifting more gradient into value/Q posterior
  shaping while keeping the same single-worker search compute.

Status pass 2026-06-14 16:49 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Active best100 leaders remain:
  - worker 7 / `18od56jt` (`outcome`, DT eval): step 7 `-0.931641`,
    latest step 44 `-0.972656`.
  - worker 5 / `iqd5vvyh` (`weights1`, DT eval): step 47 `-0.939453`,
    latest step 50 `-0.960938`.
  - worker 6 / `kvft2gi4` (`weights025`, DT eval): step 44 `-0.939453`,
    latest step 50 `-0.960938`.
  - worker 12 / `j33thog7` (`policeval_w1`): step 44 `-0.957031`,
    latest step 49 `-0.960938`.
- Newer stronger-loss policy-eval rows:
  - worker 4 / `0zc4tcr7` (`w4ps4`): best step 3 `-0.964844`,
    latest step 4 `-0.980469`.
  - worker 15 / `yf3bmdkz` (`w4ps4_b1024`): best step 3 `-0.970703`,
    latest step 4 `-0.974609`.
  - worker 14 / `x7aavdzs` (`w4ps8`): best step 3 `-0.978516`,
    latest step 4 `-0.982422`.
  - worker 1 / `94cwovky` (`w4ps1`): W&B now has step 0/1, best
    `-0.990234`; target entropy is zero as expected for ps1.
- Pane/API lag checks:
  - worker 8 / `29q6oem5` (`w4ps4_lr3e3`) is healthy in the pane despite
    W&B API still returning no rows: pane shows step 0 `-0.9902` and step 1
    `-0.9844`, now evaluating step 2.
  - worker 3 / `fc8bcgee` (`p05w5clip300`) remains in iteration 0 eval.
- Decision: no further replacement this pass. Worker 8 is not idle, worker 3
  is fresh, and the strong-loss variants still need enough rows to separate
  signal from first-eval noise.

Status pass 2026-06-14 16:50 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Current active best100 leaders:
  - worker 7 / `18od56jt` (`outcome`, DT eval): step 7 `-0.931641`,
    latest step 45 `-0.960938`.
  - worker 5 / `iqd5vvyh` (`weights1`, DT eval): step 47 `-0.939453`,
    latest step 50 `-0.960938`.
  - worker 6 / `kvft2gi4` (`weights025`, DT eval): step 44 `-0.939453`,
    latest step 50 `-0.960938`.
  - worker 12 / `j33thog7` (`policeval_w1`): step 44 `-0.957031`,
    latest step 49 `-0.960938`.
- Newer strong-loss rows remain early:
  - worker 4 / `0zc4tcr7` (`w4ps4`): best step 3 `-0.964844`,
    latest step 5 `-0.978516`.
  - worker 15 / `yf3bmdkz` (`w4ps4_b1024`): best step 3 `-0.970703`,
    latest step 5 `-0.980469`.
  - worker 14 / `x7aavdzs` (`w4ps8`): best step 3 `-0.978516`,
    latest step 5 `-0.984375`.
  - worker 8 / `29q6oem5` (`w4ps4_lr3e3`): W&B now has step 0
    `-0.990234`; pane earlier showed it had reached later evals.
- Worker 3 / `fc8bcgee` (`p05w5clip300`) still has no W&B rows and the pane
  shows iteration 0 eval. This is within the initial compile/eval latency seen
  on other fresh launches, so it is not treated as stuck yet.
- Decision: no replacement this pass. All slots are occupied, and the only
  missing-row run is too fresh to judge.

Status pass 2026-06-14 16:52 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Worker 6 / `kvft2gi4` (`weights025`, DT eval) improved the active best100
  table to step 51 `-0.937500`; latest is also step 51 `-0.937500`.
  This is now the best active run after worker 7's early step-7 outlier
  `-0.931641`.
- Other DT-eval leaders:
  - worker 7 / `18od56jt` (`outcome`): best step 7 `-0.931641`, latest
    step 45 `-0.960938`.
  - worker 5 / `iqd5vvyh` (`weights1`): best step 47 `-0.939453`, latest
    step 51 `-0.957031`.
- Newer strong-loss policy-eval variants remain early:
  - worker 4 / `0zc4tcr7`: best step 3 `-0.964844`, latest step 6
    `-0.978516`.
  - worker 15 / `yf3bmdkz`: best step 3 `-0.970703`, latest step 5
    `-0.980469`.
  - worker 14 / `x7aavdzs`: best step 3 `-0.978516`, latest step 5
    `-0.984375`.
- Worker 3 / `fc8bcgee` (`p05w5clip300`) still has no W&B rows and the pane
  remains in iteration 0 eval. This is about six minutes after W&B launch and
  still within the first-eval latency observed on worker 0, so no action yet.
- Decision: no replacement this pass. The clearest new signal is that the
  lower Dirichlet/outcome weighting (`weights025`) is currently better than the
  heavier 4/4 or 5/5 variants, but the new variants are too young to terminate.

Replacement 2026-06-14 16:56 UTC:
- W&B still showed no hit (`avg_R >= 0` by step <= 100) before replacement.
- Reclaimed worker 10:
  - Killed remote Python PID `502922`, W&B `ft5sqx2u`, checkpoint
    `dtps_10_policeval_target`.
  - TERM did not exit within 5 seconds, so used `kill -9` on PID `502922`.
    The stale `wandb-core` helper exited shortly afterward.
  - Rationale: this was an old `policy_samples=0` control, step 24 with
    best100 step 6 `-0.980469` and latest step 24 `-0.992188`. It had no
    strong value/Q learning signal and was dominated by the DT-eval
    `weights025`, `weights1`, and outcome controls.
- Launched worker 10 / W&B `z14ixz57` / remote Python PID `512706`:
  `dtps_10_weights0125`, self-play DT, DT eval, `policy_samples=32`,
  Dirichlet loss scale `0.125/0.125`, outcome losses `0.05/0.025`,
  evidence-mass Q weighting, masked-mean Q Dirichlet KL, value mask, terminal
  targets, concentration clip `16`.
- Hypothesis: worker 6 `weights025` is currently the best non-outlier
  trajectory, and it uses a much lower Dirichlet/outcome scale than the recent
  4/4 and 5/5 launches. This tests the next lower neighbor while keeping
  search compute and DT evaluation unchanged.

Replacement 2026-06-14 16:59 UTC:
- W&B still showed no hit (`avg_R >= 0` by step <= 100).
- New best active trajectory before replacement:
  - worker 5 / `iqd5vvyh` (`weights1`, DT eval): step 52 `-0.925781`.
  - worker 6 / `kvft2gi4` (`weights025`, DT eval): best step 51 `-0.937500`,
    latest step 52 `-0.951172`.
  - worker 7 / `18od56jt` (`outcome`, DT eval): best step 7 `-0.931641`,
    latest step 46 `-0.976563`.
- Reclaimed worker 11:
  - Killed remote Python PID `504326`, W&B `guaasdn5`, checkpoint
    `dtps_11_ps4`.
  - TERM did not exit within 5 seconds, so used `kill -9` on PID `504326`.
  - Rationale: this was a plain `policy_samples=4` policy-eval control with
    no value/Q loss, best100 step 4 `-0.976563` and latest step 16
    `-0.988281`, dominated by the DT-eval weight-scale runs.
- Launched worker 11 / W&B `etroxcu6` / remote Python PID `512474`:
  `dtps_11_weights05`, self-play DT, DT eval, `policy_samples=32`,
  Dirichlet loss scale `0.5/0.5`, outcome losses `0.15/0.075`,
  evidence-mass Q weighting, masked-mean Q Dirichlet KL, value mask, terminal
  targets, concentration clip `64`.
- Hypothesis: after worker 5 (`weights1`) moved ahead of worker 6
  (`weights025`), the best local search over loss scale is around
  `0.25..1.0`. This tests the missing midpoint while preserving DT self-play
  and DT eval.

Status pass 2026-06-14 17:01 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Worker 5 / `iqd5vvyh` (`weights1`, DT eval) improved again:
  best/latest step 53 `-0.921875`. This is the strongest active trajectory so
  far and is now better than worker 7's early outlier.
- Other DT-eval leaders:
  - worker 7 / `18od56jt` (`outcome`): best step 7 `-0.931641`, latest
    step 46 `-0.976563`.
  - worker 6 / `kvft2gi4` (`weights025`): best step 51 `-0.937500`, latest
    step 52 `-0.951172`.
- The fresh DT-eval scale-neighbor runs are alive but still in initial eval:
  - worker 10 / `z14ixz57` (`weights0125`), launched at 16:55 UTC.
  - worker 11 / `etroxcu6` (`weights05`), launched at 16:59 UTC.
- Worker 3 / `fc8bcgee` now has rows: step 0/1 both `-0.990234`; this 5/5,
  clip-300, policy-eval setup does not look promising yet, but it is still too
  early to kill.
- Decision: no additional replacement this pass. The best current evidence
  points at the moderate DT-eval loss-scale family, and the two adjacent
  scale probes need first rows before further churn.

Status pass 2026-06-14 17:03 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Worker 5 / `iqd5vvyh` (`weights1`, DT eval) remains the best active run:
  best/latest step 53 `-0.921875`.
- Worker 6 / `kvft2gi4` (`weights025`, DT eval) improved again:
  best/latest step 53 `-0.933594`.
- Other useful comparators:
  - worker 7 / `18od56jt` (`outcome`, DT eval): best step 7 `-0.931641`,
    latest step 47 `-0.976563`.
  - worker 12 / `j33thog7` (`policeval_w1`): best step 44 `-0.957031`,
    latest step 53 `-0.958984`.
  - worker 2 / `4slwcxag` (`winner_action`): improved to best/latest
    step 13 `-0.968750`, still far behind the DT-eval weight-scale family.
- Worker 10 / `z14ixz57` (`weights0125`) and worker 11 / `etroxcu6`
  (`weights05`) are both alive in their tmux panes and still in iteration 0
  eval. No W&B rows yet; this is expected initial DT-eval latency.
- Decision: no replacement this pass. The current best hypothesis is still the
  moderate DT-eval value/Q loss scale; adjacent probes are running and need
  rows before changing more slots.

Status pass 2026-06-14 17:05 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Worker 5 / `iqd5vvyh` (`weights1`, DT eval) remains the best active run:
  best/latest step 53 `-0.921875`.
- Worker 6 / `kvft2gi4` (`weights025`, DT eval) remains second among active
  DT-eval scale runs: best/latest step 53 `-0.933594`.
- Worker 7 / `18od56jt` (`outcome`, DT eval) is now mostly an early outlier:
  best step 7 `-0.931641`, latest step 47 `-0.976563`.
- Newer/lower-confidence observations:
  - worker 2 / `4slwcxag` (`winner_action`, policy eval) improved to
    best/latest step 13 `-0.968750`, still behind the DT-eval scale runs.
  - worker 3 / `fc8bcgee` (`p05w5clip300`, policy eval) has early rows,
    best step 2 `-0.984375`.
  - worker 0 / `7vxaa74t` (`w4ps4_dteval`) reached best/latest step 3
    `-0.978516`.
- Worker 10 / `z14ixz57` (`weights0125`) and worker 11 / `etroxcu6`
  (`weights05`) are still alive in tmux and in iteration 0 eval. No W&B rows
  yet; this is expected DT-eval first-pass latency.
- Decision: no replacement this pass. All worker slots are occupied, and the
  adjacent DT-eval loss-scale probes need first rows before more changes.

Status pass 2026-06-14 17:06 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Best active trajectory remains worker 5 / `iqd5vvyh` (`weights1`, DT eval):
  best step 53 `-0.921875`, latest step 54 `-0.935547`.
- Worker 12 / `j33thog7` (`policeval_w1`, policy eval) jumped to
  best/latest step 54 `-0.931641`. This is useful confirmation that the
  `weights1` loss scale is helping, but it is policy eval rather than the
  stricter DT-eval track.
- Worker 6 / `kvft2gi4` (`weights025`, DT eval): best/latest step 53
  `-0.933594`.
- Fresh adjacent probes:
  - worker 10 / `z14ixz57` (`weights0125`, DT eval): pane shows step 0
    completed at `-0.9883`; W&B API has not surfaced the row yet.
  - worker 11 / `etroxcu6` (`weights05`, DT eval): alive in initial eval.
- Decision: no replacement this pass. The fleet is occupied, and the best
  hypothesis is still the moderate DT-eval value/Q loss-scale family. Worker
  10/11 need first synced rows before further changes.

Status pass 2026-06-14 17:08 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- DT-eval weight-scale leaders:
  - worker 5 / `iqd5vvyh` (`weights1`): best step 53 `-0.921875`,
    latest step 54 `-0.935547`.
  - worker 6 / `kvft2gi4` (`weights025`): best/latest step 54
    `-0.931641`.
  - worker 7 / `18od56jt` (`outcome`): best step 7 `-0.931641`,
    latest step 48 `-0.966797`.
- Policy-eval confirmation:
  - worker 12 / `j33thog7` (`policeval_w1`): best/latest step 54
    `-0.931641`, suggesting the `weights1` scale improves the raw policy too.
- Fresh adjacent probes:
  - worker 10 / `z14ixz57` (`weights0125`, DT eval): pane shows step 0
    `-0.9883` and step 1 eval in progress; W&B API still has no row.
  - worker 11 / `etroxcu6` (`weights05`, DT eval): alive in initial eval.
- Decision: no replacement this pass. The most promising family is already
  being bracketed by `weights0125`, `weights025`, `weights05`, and `weights1`;
  the two newest bracket points need rows before changing more workers.

Status pass 2026-06-14 17:12 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- All worker slots 0-15 have local tmux/launcher processes; no idle TPU slot
  was found.
- Best active trajectory remains worker 5 / `iqd5vvyh` (`weights1`, DT eval):
  best step 53 `-0.921875`, latest step 55 `-0.931641`.
- Worker 6 / `kvft2gi4` (`weights025`, DT eval): best step 54 `-0.931641`,
  latest step 55 `-0.960938`.
- Worker 7 / `18od56jt` (`outcome`, DT eval): best step 7 `-0.931641`,
  latest step 48 `-0.966797`.
- Worker 12 / `j33thog7` (`policeval_w1`, policy eval): best step 54
  `-0.931641`, latest step 56 `-0.939453`; this still supports the `weights1`
  loss scale but does not satisfy the DT-eval target.
- Fresh adjacent DT-eval probes:
  - worker 10 / `z14ixz57` (`weights0125`): pane shows step 0 `-0.9883`,
    step 1 eval in progress; W&B API still has no eval row.
  - worker 11 / `etroxcu6` (`weights05`): pane shows step 0 `-0.9883`,
    step 1 eval in progress; W&B API still has no eval row.
- Decision: no replacement this pass. The currently occupied fleet is still
  testing the most promising bracket (`0.125`, `0.25`, `0.5`, `1.0` value/Q
  Dirichlet weights under DT eval). Replacing before worker 10/11 sync first
  rows would add churn without evidence.

Status pass 2026-06-14 17:13 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Best active run remains worker 5 / `iqd5vvyh` (`weights1`, DT eval):
  best step 53 `-0.921875`, latest step 55 `-0.931641`.
- Worker 6 / `kvft2gi4` (`weights025`, DT eval): best step 54 `-0.931641`,
  latest step 55 `-0.960938`.
- Worker 10 / `z14ixz57` (`weights0125`, DT eval): W&B now has step 0
  `-0.988281`; local pane has step 1 eval running.
- Worker 11 / `etroxcu6` (`weights05`, DT eval): local pane has step 0
  `-0.9883` and step 1 eval running; W&B row is still sync-lagged.
- Decision: no replacement. The one unsynced run is alive, and the bracket
  needs actual step rows before another worker is killed or retargeted.

Status pass 2026-06-14 18:14 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- All tmux worker windows are occupied.
- Best valid DT-eval run is worker 5 / `iqd5vvyh` (`weights1`):
  best step 61 `-0.906250`, latest step 66 `-0.921875`.
- Worker 6 / `kvft2gi4` (`weights025`, DT eval): best step 54 `-0.931641`,
  latest step 67 `-0.957031`; weaker than `weights1`.
- Worker 10 / `z14ixz57` (`weights0125`, DT eval): best step 5
  `-0.980469`, latest step 11 `-0.982422`; weak early.
- Worker 11 / `etroxcu6` (`weights05`, DT eval): best step 9
  `-0.974609`, latest step 10 `-0.976563`; weak early.
- Worker 12 / `j33thog7` (`policeval_w1`, policy eval): best step 68
  `-0.878906`, latest step 73 `-0.931641`; this is policy eval only but
  confirms the `weights1` scale has some learning signal.
- Worker 9 / `xnh5t2vz` (`ps1_outcome`, policy eval): best step 34
  `-0.873047`, latest step 37 `-0.902344`, but latest train row has NaNs for
  policy/value/q losses and alpha concentrations. The run is alive but no
  longer trustworthy.
- Decision: reclaim worker 9 only. Exact old remote Python PID `502958`
  (`checkpoints/dtps_9_ps1_outcome`) was terminated because the run had
  entered NaN training. Replacement keeps the promising ps1/outcome shape but
  adds only `training.grad_clip_norm=1.0`.
- New worker 9 run: W&B `7a8uqr1o` (`polar-butterfly-461`), checkpoint dir
  `checkpoints/dtps_9_ps1_outcome_clip1`, remote Python PID `516207`.

Status pass 2026-06-14 19:18 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Best valid DT-eval trajectory is worker 5 / `iqd5vvyh` (`weights1`):
  best step 67 `-0.857422`, latest step 78 `-0.916016`. This is a real
  improvement over the prior pass, but still far from the zero target.
- Worker 12 / `j33thog7` (`policeval_w1`, policy eval): best step 83
  `-0.873047`, latest step 92 `-0.914063`. Same loss-scale family confirms
  some policy-side learning.
- Worker 6 / `kvft2gi4` (`weights025`, DT eval): best step 69 `-0.919922`,
  latest step 79 `-0.953125`; weaker than `weights1`.
- Worker 8 / `29q6oem5` (`w4ps4_lr3e3`, policy eval): best step 44
  `-0.925781`, latest step 45 `-0.933594`; lr 3e-3 is not destabilizing and
  may speed early learning.
- Worker 9 / `7a8uqr1o` (`ps1_outcome_clip1`, policy eval): best step 7
  `-0.982422`, latest step 15 `-0.986328`; no NaNs yet, but early signal is
  much weaker than the old unstable `ps1_outcome` run.
- Worker 13 / `wtj54ap3` (`policeval_ps8`, policy eval): best step 6
  `-0.970703`, latest step 65 `-0.986328`; dominated and no longer useful.
- Decision: reclaim only worker 13. Exact old remote Python PID `502227`
  (`checkpoints/dtps_13_policeval_ps8`) was terminated because it was an old,
  dominated comparator. Replacement keeps DT self-play and the strongest
  current loss-scale family (`weights1`) but adds `training.learning_rate=3e-3`
  for faster signal, using policy eval to keep the run cheap.
- New worker 13 run: W&B `ua6kysq4` (`mild-haze-462`), checkpoint dir
  `checkpoints/dtps_13_w1_lr3e3`, remote Python PID `521098`.

Status pass 2026-06-14 20:22 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Best valid DT-eval trajectory remains worker 5 / `iqd5vvyh` (`weights1`):
  best step 85 `-0.828125`, latest step 90 `-0.906250`. It improved again,
  but it is still far from the zero target and has only ten iterations left
  inside the speedrun horizon.
- Worker 6 / `kvft2gi4` (`weights025`, DT eval): best step 90 `-0.896484`,
  latest step 91 `-0.925781`; worse than `weights1` but still improving.
- Worker 12 / `j33thog7` (`policeval_w1`, policy eval): best within horizon
  step 83 `-0.873047`; now past step 100 without a hit.
- Worker 8 / `29q6oem5` (`w4ps4_lr3e3`, policy eval): best step 52
  `-0.890625`, latest step 64 `-0.931641`.
- Worker 13 / `ua6kysq4` (`w1_lr3e3`, policy eval): early, best step 13
  `-0.974609`, latest step 15 `-0.980469`; not enough evidence yet.
- Worker 9 / `7a8uqr1o` (`ps1_outcome_clip1`, policy eval): repeated the
  previous failure mode. It reached best/latest step 34 `-0.873047`, but the
  latest train row has NaNs for policy/value/q losses and alpha
  concentrations. `training.grad_clip_norm=1.0` was not sufficient to keep the
  outcome-only ps1 variant stable.
- Decision: reclaim only worker 9. Exact old remote Python PID `516207`
  (`checkpoints/dtps_9_ps1_outcome_clip1`) was terminated for repeated NaN
  training. Replacement keeps the ps1/outcome shape but adds small Dirichlet
  KL supervision (`value_dir_kl_weight=0.25`, `q_dir_kl_weight=0.25`) plus
  `q_loss_weight_mode=evidence_mass`, `q_dir_kl_reduction=masked_mean`,
  `grad_clip_norm=1.0`, and concentration clip `32` to constrain the
  Dirichlet heads instead of letting outcome-only training drift into NaNs.
- New worker 9 run: W&B `ynwnc48h` (`unique-galaxy-463`), checkpoint dir
  `checkpoints/dtps_9_ps1_outcome_dir025`, remote Python PID `528011`.

Follow-up on worker 9 / `7a8uqr1o` NaNs, 2026-06-14 20:48 UTC:
- The run's policy-eval result stayed relatively good after NaNs because eval
  used `eval.player_search.kind=policy`, so it mostly exercised the policy
  head while the Dirichlet heads were already numerically broken.
- W&B history shows `train/alpha_V_concentration` collapsing before the NaN:
  about `3.76` at step 0, `0.311` at step 15, `0.000256` at step 27, then
  policy/value/q train metrics became NaN at step 28. This points to
  outcome-only training driving the value Dirichlet concentration toward zero;
  grad clipping alone did not prevent that.
- This supports the worker 9 replacement choice (`ynwnc48h`): keep the
  promising ps1/outcome shape, but add small Dirichlet KL supervision so the
  Dirichlet heads stay constrained instead of relying only on outcome NLL.

Staged depth-fix rollout, 2026-06-14 21:51 UTC:
- User pointed out the staged fix: before this patch the DT search depth was
  effectively tied to `num_blocks`, so many active runs with `num_blocks=1`
  were doing very shallow search and likely struggling to get useful signal.
- Staged patch summary:
  - `DirichletThompsonSearchConfig` adds `max_depth`.
  - config aliases default DT `max_depth` to `num_simulations`.
  - `make_search_player` passes `max_depth` into `dirichlet_q_policy`.
- Verified the backend already accepts `max_depth`. A CPU-targeted
  `tests/test_config_validation.py` run got past TPU initialization, but two
  unrelated/staged config assertions currently fail for `hex6`/`hex8`
  checkpoint baseline search kind. Go experiment launch is not blocked by
  those failures.
- W&B still showed no hit before relaunch. Best old-code DT-eval trajectory
  was worker 5 / `iqd5vvyh` (`weights1`): best step 85 `-0.828125`, latest
  step 105 `-0.900391`. I kept it running as the old-code reference.
- Reclaimed four weak old-code slots:
  - worker 2 / `4slwcxag` (`winner_action`), old PID `544101`,
    best step 13 `-0.968750`, latest step 90 `-0.992188`.
  - worker 9 / `ynwnc48h` (`ps1_outcome_dir025`), old PID `528011`,
    best step 6 `-0.976563`, latest step 21 `-0.980469`.
  - worker 10 / `z14ixz57` (`weights0125`), old PID `512706`,
    best step 35 `-0.943359`, latest step 50 `-0.990234`.
  - worker 11 / `etroxcu6` (`weights05`), old PID `512474`,
    best step 35 `-0.960938`, latest step 51 `-0.978516`.
- First relaunch attempt used plain `selfplay.search.dirichlet_thompson.max_depth`
  overrides and failed before W&B because Hydra requires `+...max_depth=32`
  when inserting a key absent from the YAML. Retried with `+` overrides.
- Live fixed-depth probes:
  - worker 2 / W&B `07h9yjtl` (`serene-lake-465`), remote PID `572032`,
    checkpoint `checkpoints/dtps_2_fixdepth_w1_dteval`: `weights1` loss scale,
    DT self-play max_depth 32, DT eval max_depth 32.
  - worker 10 / W&B `ua5kn63n` (`glorious-plant-464`), remote PID `538895`,
    checkpoint `checkpoints/dtps_10_fixdepth_w1_policy`: `weights1` loss scale,
    DT self-play max_depth 32, policy eval for cheaper early readout.
  - worker 11 / W&B `e7trtu9g` (`trim-armadillo-467`), remote PID `536458`,
    checkpoint `checkpoints/dtps_11_fixdepth_w4ps4_dteval`: stronger 4/4
    Dirichlet losses, policy_samples 4, DT self-play/eval max_depth 32.
  - worker 9 / W&B `0tgil6cc` (`usual-wave-466`), remote PID `539130`,
    checkpoint `checkpoints/dtps_9_fixdepth_ps1_dir025`: stabilized ps1/outcome
    variant, DT self-play max_depth 32, policy eval.

Status pass 2026-06-14 22:52 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Best overall remains old-code worker 5 / `iqd5vvyh` (`weights1`, DT eval):
  best step 85 `-0.828125`, latest step 118 `-0.916016`. Kept running as the
  old-code reference.
- First fixed-depth rows are live but still weak:
  - worker 10 / `ua5kn63n` (`fix_w1_policy`): best step 4 `-0.972656`,
    latest step 15 `-0.988281`.
  - worker 2 / `07h9yjtl` (`fix_w1_dteval`): best step 2 `-0.974609`,
    latest step 8 `-0.974609`.
  - worker 11 / `e7trtu9g` (`fix_w4ps4_dteval`): best step 6 `-0.974609`,
    latest step 8 `-0.980469`.
  - worker 9 / `0tgil6cc` (`fix_ps1_dir025`): best step 6 `-0.976563`,
    latest step 15 `-0.984375`.
- Interpretation: the depth fix is not an immediate step-0/early-step jump, but
  these runs are much younger than the old reference. Need more rows before
  deciding whether deeper search helps later learning.
- Concrete failure: old-code worker 7 / `18od56jt` (`outcome`, DT eval) now has
  NaN train metrics and latest eval `-1` at step 104. Exact old remote Python
  PID `469113` was terminated.
- Replacement worker 7: W&B `56o561fb` (`comfy-firefly-468`), checkpoint
  `checkpoints/dtps_7_fixdepth_w1_lr3e3`, remote PID `514031`. It uses
  fixed-depth DT self-play (`max_depth=32`), `weights1` losses, lr `3e-3`, and
  policy eval for cheaper early readout.

Status pass 2026-06-14 23:56 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Best old-code reference is still worker 5 / `iqd5vvyh` (`weights1`, DT eval):
  best step 85 `-0.828125`, latest step 130 `-0.921875`.
- Fixed-depth probes are stable so far but not strong early:
  - worker 9 / `0tgil6cc` (`fix_ps1_dir025`, policy eval): best step 32
    `-0.945313`, latest step 35 `-0.980469`. This is the best fixed-depth
    score so far and has avoided the prior outcome-only value-concentration
    collapse.
  - worker 11 / `e7trtu9g` (`fix_w4ps4_dteval`): best step 9 `-0.960938`,
    latest step 21 `-0.972656`.
  - worker 2 / `07h9yjtl` (`fix_w1_dteval`): best step 13 `-0.964844`,
    latest step 19 `-0.974609`.
  - worker 10 / `ua5kn63n` (`fix_w1_policy`): best step 26 `-0.968750`,
    latest step 34 `-0.982422`.
  - worker 7 / `56o561fb` (`fix_w1_lr3e3`, policy eval): best step 13
    `-0.974609`, latest step 15 `-0.980469`; still fresh.
- Interpretation: the depth fix did not create an immediate early reward jump
  by steps 20-35. It may still improve later learning, but current best remains
  the old shallow `weights1` run. Need more rows before killing the fresh
  fixed-depth runs.
- Reclaimed one stale old-code duplicate: worker 13 / `ua6kysq4`
  (`w1_lr3e3`, policy eval), old remote PID `521098`, best step 65
  `-0.919922`, latest step 77 `-0.955078`. It was weak and duplicated by the
  fixed-depth worker 7 `w1_lr3e3` probe.
- Replacement worker 13: W&B `rtco03sy` (`trim-spaceship-469`), checkpoint
  `checkpoints/dtps_13_fixdepth_w4ps1`, remote PID `542404`. It tests
  fixed-depth `w4ps1` because the old `w4ps1` policy-eval run had one of the
  better late policy scores (`-0.875` after step 100), while still keeping a
  single-worker, similar-compute setup.

Status pass 2026-06-15 01:00 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Best overall remains the old-code worker 5 / `iqd5vvyh` (`weights1`, DT eval):
  best step 85 `-0.828125`, latest step 142 `-0.910156`.
- Fixed-depth runs are stable but still materially behind the old best:
  - worker 9 / `0tgil6cc` (`fix_ps1_dir025`, policy eval): best step 50
    `-0.941406`, latest step 53 `-0.960938`. This remains the best fixed-depth
    score and has not repeated the old ps1/outcome NaN collapse.
  - worker 11 / `e7trtu9g` (`fix_w4ps4_dteval`): best step 26 `-0.955078`,
    latest step 33 `-0.984375`.
  - worker 2 / `07h9yjtl` (`fix_w1_dteval`): best/latest step 30-31
    `-0.957031`.
  - worker 10 / `ua5kn63n` (`fix_w1_policy`): best step 44 `-0.957031`,
    latest step 52 `-0.972656`.
  - worker 7 / `56o561fb` (`fix_w1_lr3e3`, policy eval): best step 29
    `-0.962891`, latest step 33 `-0.972656`.
  - worker 13 / `rtco03sy` (`fix_w4ps1`, policy eval): still fresh, best step
    13 `-0.976563`, latest step 15 `-0.978516`.
- Decision: no replacement this pass. The fixed-depth batch is not good yet,
  but it is not dead or numerically unstable. Replacing based only on weak
  early rows would add churn before we know whether deeper DT search improves
  learning later in the first 100 steps.

Status pass 2026-06-15 02:07 UTC:
- Checked the staged depth-fix files again. The important path is present:
  `DirichletThompsonSearchConfig.max_depth` defaults from `num_simulations`,
  config aliases propagate it, and `make_search_player` passes it into
  `dirichlet_q_policy`. Fixed a small staged assertion-message bug in
  `play_search.py` where the failure branch referenced an undefined local.
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Best overall remains old-code worker 5 / `iqd5vvyh` (`weights1`, DT eval):
  best step 85 `-0.828125`, latest step 154 `-0.892578`.
- Best fixed-depth score so far is worker 10 / `ua5kn63n`
  (`fix_w1_policy`): best step 68 `-0.878906`, latest step 71
  `-0.935547`. This is now close to the weaker old-code references, but still
  far from a non-negative early eval.
- Fixed-depth worker 4 / `koojyw4u` was launched earlier from a reclaimed
  old-code slot: checkpoint `checkpoints/dtps_4_fixdepth_w025_policy`, remote
  PID `522247`, DT self-play `max_depth=32`, policy eval, and the `weights025`
  loss family.
- Reclaimed two additional old-depth slots after confirming exact PIDs:
  - worker 15 / W&B `yf3bmdkz` (`w4ps4_b1024`), remote PID `490392`, step
    171, best100 `-0.941406`, latest `-0.943359`. It was old-depth, past the
    target window, and dominated by preserved references.
  - worker 8 / W&B `29q6oem5` (`w4ps4_lr3e3`), remote PID `535810`, step 166,
    best100 `-0.890625`, latest `-0.945313`. It was old-depth, past the target
    window, and not improving.
- Replacements:
  - worker 8 / W&B `khlgjrwt` (`driven-spaceship-471`), remote PID `575691`,
    checkpoint `checkpoints/dtps_8_fixdepth_w4ps4_lr3e3`: mirrors the killed
    `w4ps4_lr3e3` recipe, adds DT self-play `max_depth=32`, keeps policy eval.
  - worker 15 / W&B `ati43om2` (`vital-universe-472`), remote PID `529679`,
    checkpoint `checkpoints/dtps_15_fixdepth_p05w5clip300`: mirrors the
    promising late-learning `p05w5clip300` family, adds DT self-play
    `max_depth=32`, keeps policy eval.

Status pass 2026-06-15 03:12 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Best overall remains old-code worker 5 / `iqd5vvyh` (`weights1`, DT eval):
  best step 85 `-0.828125`, latest step 166 `-0.910156`.
- Best fixed-depth run is now worker 10 / `ua5kn63n` (`fix_w1_policy`): best
  step 83 `-0.873047`, latest step 90 `-0.896484`. This has caught the weaker
  old policy-eval reference band, but it is still far from the objective.
- Other fixed-depth probes after the wait:
  - worker 11 / `e7trtu9g` (`fix_w4ps4_dteval`): best step 54 `-0.902344`,
    latest step 59 `-0.939453`.
  - worker 2 / `07h9yjtl` (`fix_w1_dteval`): best step 56 `-0.916016`,
    latest step 57 `-0.917969`.
  - worker 7 / `56o561fb` (`fix_w1_lr3e3`): best step 65 `-0.919922`,
    latest step 71 `-0.960938`.
  - worker 9 / `0tgil6cc` (`fix_ps1_dir025`): best step 57 `-0.933594`,
    latest step 93 `-0.976563`.
  - fresh replacements worker 15 / `ati43om2`, worker 4 / `koojyw4u`, and
    worker 8 / `khlgjrwt` are still too early to judge.
- Old shallow runs still provide late-learning references but no target hit:
  worker 1 / `94cwovky` improved late to `-0.794922` at step 185; worker 14 /
  `x7aavdzs` improved late to `-0.871094` at step 187; worker 3 / `fc8bcgee`
  remains best at `-0.837891` after step 100.
- No live run had NaN fields in the last train row.
- Worker 12 / `j33thog7` finished normally at step 219, so I reused that idle
  worker instead of killing another live job.
- Replacement worker 12: W&B `recu5muu` (`scarlet-monkey-473`), remote PID
  `561814`, checkpoint `checkpoints/dtps_12_fixdepth_w4ps8`. It mirrors the
  old `w4ps8` recipe, adds DT self-play `max_depth=32`, keeps
  `policy_samples=8`, and uses policy eval.

Status pass 2026-06-15 04:15 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Best overall remains old-code worker 5 / `iqd5vvyh` (`weights1`, DT eval):
  best step 85 `-0.828125`, latest step 178 `-0.884766`.
- Best fixed-depth result improved to worker 2 / `07h9yjtl` (`fix_w1_dteval`):
  best step 67 `-0.857422`, latest step 68 `-0.921875`. This is the strongest
  depth-fix evidence so far, though still far from non-negative.
- Other fixed-depth first-100 bests:
  - worker 10 / `ua5kn63n` (`fix_w1_policy`): best step 83 `-0.873047`,
    latest step 108 `-0.898438`.
  - worker 11 / `e7trtu9g` (`fix_w4ps4_dteval`): best step 66 `-0.888672`,
    latest step 72 `-0.916016`.
  - worker 7 / `56o561fb` (`fix_w1_lr3e3`): best step 65 `-0.919922`,
    latest step 89 `-0.947266`.
  - worker 9 / `0tgil6cc` (`fix_ps1_dir025`): best step 98 `-0.923828`,
    latest step 111 `-0.953125`.
  - worker 8 / `khlgjrwt`, worker 15 / `ati43om2`, worker 4 / `koojyw4u`,
    and worker 12 / `recu5muu` remain too fresh or too weak to judge.
- Old shallow references still show some late improvement but no target-window
  hit: worker 1 / `94cwovky` best late `-0.794922`, worker 3 / `fc8bcgee`
  best late `-0.814453`, worker 14 / `x7aavdzs` best late `-0.871094`.
- No NaN fields were present in the last train rows checked. No live process was
  killed on this pass.

Status pass 2026-06-15 05:17 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Best overall remains old-code worker 5 / `iqd5vvyh` (`weights1`, DT eval):
  best step 85 `-0.828125`, latest step 189 `-0.886719`.
- Best fixed-depth result remains worker 2 / `07h9yjtl` (`fix_w1_dteval`):
  best step 67 `-0.857422`, latest step 80 `-0.886719`.
- Fixed-depth runs now at or past the early window still missed:
  - worker 10 / `ua5kn63n` (`fix_w1_policy`): best step 83 `-0.873047`,
    latest step 125 `-0.917969`.
  - worker 11 / `e7trtu9g` (`fix_w4ps4_dteval`): best step 66 `-0.888672`,
    latest step 84 `-0.947266`.
  - worker 13 / `rtco03sy` (`fix_w4ps1`): best step 78 `-0.896484`, latest
    step 92 `-0.955078`.
  - worker 7 / `56o561fb` (`fix_w1_lr3e3`): best step 90 `-0.916016`, latest
    step 107 `-0.923828`.
  - worker 9 / `0tgil6cc` (`fix_ps1_dir025`): best step 98 `-0.923828`,
    latest step 129 `-0.960938`.
- Two old shallow slots finished normally and were idle:
  - worker 1 / `94cwovky`, final step 219 `-0.853516`, best late
    `-0.794922`, no target-window hit.
  - worker 14 / `x7aavdzs`, final step 219 `-0.912109`, best late
    `-0.869141`, no target-window hit.
- I reused those idle workers for a real search-strength test rather than
  killing live runs:
  - worker 1 / W&B `evq0t6g8` (`solar-monkey-475`), remote PID `571507`,
    checkpoint `checkpoints/dtps_1_fixdepth_w1_dteval_s64`: `weights1` losses,
    DT self-play `num_simulations=64`, `max_depth=64`, DT eval
    `num_simulations=64`, `max_depth=64`.
  - worker 14 / W&B `xyv2kc3f` (`major-valley-474`), remote PID `550708`,
    checkpoint `checkpoints/dtps_14_fixdepth_w1_policy_s64`: `weights1`
    losses, DT self-play `num_simulations=64`, `max_depth=64`, policy eval.

Idle-slot refill 2026-06-15 05:35 UTC:
- Worker 3 / `fc8bcgee` finished normally at step 219, final `-0.914063`,
  best100 `-0.935547`, best late `-0.814453`. It had no target-window hit.
- Replacement worker 3: W&B `b389datz` (`dainty-sun-476`), remote PID
  `593301`, checkpoint `checkpoints/dtps_3_fixdepth_w1_dteval_b2`. It keeps
  the `weights1` losses, uses DT self-play/eval with `num_simulations=32`,
  `num_blocks=2`, and `max_depth=32`. This pairs with worker 1's 64-simulation
  single-block run to compare two ways of spending roughly 64 DT simulations.

Status pass 2026-06-15 06:37 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Fixed-depth worker 2 / `07h9yjtl` (`fix_w1_dteval`) has now matched the old
  best early score: best step 85 `-0.828125`, latest step 94 `-0.896484`.
  This is the clearest evidence that the staged depth fix recovers the best
  early signal, but it is still far from non-negative.
- Old-code worker 5 / `iqd5vvyh` remains tied for best100 at step 85
  `-0.828125`, latest step 204 `-0.902344`.
- Other mature fixed-depth runs remain behind:
  - worker 10 / `ua5kn63n` (`fix_w1_policy`): best step 83 `-0.873047`,
    latest step 148 `-0.892578`.
  - worker 11 / `e7trtu9g` (`fix_w4ps4_dteval`): best step 66 `-0.888672`,
    latest step 101 `-0.947266`.
  - worker 8 / `khlgjrwt` (`fix_w4ps4_lr3e3`): best step 52 `-0.890625`,
    latest step 77 `-0.923828`.
  - worker 13 / `rtco03sy` (`fix_w4ps1`): best step 78 `-0.896484`, latest
    step 116 `-0.916016`.
- New search-strength probes are still too fresh:
  - worker 1 / `evq0t6g8` (`s64` DT eval): best step 3 `-0.984375`.
  - worker 14 / `xyv2kc3f` (`s64` policy eval): best step 7 `-0.982422`.
  - worker 3 / `b389datz` (`num_blocks=2` DT eval): best step 0 `-0.986328`.
- No NaN fields were present in the last train rows checked.

Status pass 2026-06-15 07:28 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Best fixed-depth run remains worker 2 / `07h9yjtl` (`fix_w1_dteval`): best
  step 85 `-0.828125`, latest step 104 `-0.900391`.
- Old-code worker 5 / `iqd5vvyh` remains tied for best100 at step 85
  `-0.828125`, latest step 213 `-0.892578`.
- Mature fixed-depth runs remain behind:
  - worker 10 / `ua5kn63n`: best100 `-0.873047`, latest step 162
    `-0.904297`.
  - worker 8 / `khlgjrwt`: best100 `-0.890625`, latest step 92 `-0.896484`.
  - worker 13 / `rtco03sy`: best100 `-0.896484`, late best step 126
    `-0.875000`, latest step 130 `-0.875000`.
  - worker 9 / `0tgil6cc`: best100 `-0.923828`, late best step 118
    `-0.914063`.
- The 64-simulation and 2-block probes are still early and currently weak:
  worker 1 / `evq0t6g8` best `-0.974609`, worker 14 / `xyv2kc3f` best
  `-0.978516`, worker 3 / `b389datz` best `-0.968750`.
- No NaN fields were present in the last train rows checked. Worker 5 and
  worker 6 are close to normal completion, so the next check should be shorter
  to refill them if they finish.

Status/refill 2026-06-15 08:09 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Worker 5 / `iqd5vvyh` finished normally: final step 219 `-0.888672`,
  best100 `-0.828125`, no hit.
- Best fixed-depth run remains worker 2 / `07h9yjtl`: best step 85
  `-0.828125`, latest step 112 `-0.921875`.
- Worker 6 / `kvft2gi4` is close to finishing but still running at step 217,
  latest `-0.906250`.
- Worker 5 replacement: W&B `og8pq0tm` (`pious-cosmos-477`), remote PID
  `558523`, checkpoint `checkpoints/dtps_5_fixdepth_w1_dteval_ps0`. It keeps
  `weights1` losses, fixed-depth DT self-play/eval, and sets
  `policy_samples=0` so the policy target uses search action weights rather
  than posterior-best sampled targets.

Idle-slot refill 2026-06-15 08:27 UTC:
- Worker 6 / `kvft2gi4` finished normally: final step 219 `-0.919922`,
  best100 `-0.896484`, best late `-0.882813`, no hit.
- Replacement worker 6: W&B `vj3bgkc6` (`copper-cherry-478`), remote PID
  `557894`, checkpoint `checkpoints/dtps_6_fixdepth_w1_dteval_ps64`. It keeps
  `weights1` losses and fixed-depth DT self-play/eval, but increases
  posterior-best policy target sampling to `policy_samples=64` with chunk size
  32.

Status pass 2026-06-15 09:29 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Best fixed-depth run remains worker 2 / `07h9yjtl`: best step 85
  `-0.828125`, latest step 127 `-0.919922`.
- Mature fixed-depth late movement:
  - worker 7 / `56o561fb` (`fix_w1_lr3e3`) improved late to best step 170
    `-0.851563`, latest step 179 `-0.876953`; still no target-window hit.
  - worker 13 / `rtco03sy` (`fix_w4ps1`) remains best late step 126
    `-0.875000`, latest step 166 `-0.892578`.
  - worker 10 / `ua5kn63n` is at step 196 with best100 `-0.873047`.
- New probes:
  - worker 5 / `og8pq0tm` (`policy_samples=0`) is early, best step 12
    `-0.964844`.
  - worker 6 / `vj3bgkc6` (`policy_samples=64`) is early, best step 3
    `-0.966797`.
  - worker 1 / `evq0t6g8` (`s64` DT eval) remains weak, best `-0.974609`.
  - worker 14 / `xyv2kc3f` (`s64` policy eval) remains weak, best `-0.974609`.
  - worker 3 / `b389datz` (`num_blocks=2`) improved to best step 20
    `-0.949219`, still weak.
- No NaN fields were present in the last train rows checked. Worker 0, 9, and
  10 are close to normal completion.

Status/refill 2026-06-15 10:30 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Worker 9 / `0tgil6cc` finished normally: final step 219 `-0.960938`,
  best100 `-0.923828`, best late `-0.914063`, no hit.
- Best fixed-depth run remains worker 2 / `07h9yjtl`: best step 85
  `-0.828125`, latest step 138 `-0.935547`.
- Mature late movement:
  - worker 13 / `rtco03sy` (`fix_w4ps1`) improved late to best step 181
    `-0.822266`, latest step 183 `-0.828125`; still no target-window hit.
  - worker 7 / `56o561fb` (`fix_w1_lr3e3`) remains best late step 170
    `-0.851563`, latest step 196 `-0.898438`.
  - worker 15 / `ati43om2` (`fix_p05w5clip300`) improved late to best step
    143 `-0.882813`, latest step 146 `-0.894531`.
- Fresh probes:
  - worker 5 / `og8pq0tm` (`policy_samples=0`) best step 22 `-0.933594`.
  - worker 6 / `vj3bgkc6` (`policy_samples=64`) best step 17 `-0.962891`.
  - worker 3 / `b389datz` (`num_blocks=2`) best step 26 `-0.937500`.
  - 64-simulation probes remain weak.
- Worker 9 replacement: W&B `11pzw4at` (`honest-rain-479`), remote PID
  `591644`, checkpoint `checkpoints/dtps_9_fixdepth_w1_dteval_searchmask`.
  It keeps `weights1` losses and fixed-depth DT self-play/eval, but switches
  `training.losses.loss_mask_mode=search` to test whether the broader value
  mask is hurting early policy signal.

Idle-slot refill 2026-06-15 10:58 UTC:
- Worker 0 / `7vxaa74t` finished normally: final step 219 `-0.892578`,
  best100 `-0.888672`, best late `-0.880859`, no hit.
- Worker 10 / `ua5kn63n` finished normally: final step 219 `-0.916016`,
  best100 `-0.873047`, no hit.
- Replacement worker 0: W&B `t3dn7v6z` (`scarlet-shape-480`), remote PID
  `1416346`, checkpoint `checkpoints/dtps_0_fixdepth_w4ps1_dteval`. This is
  the late-moving `w4ps1` recipe with fixed-depth DT self-play and DT eval.
- Replacement worker 10: W&B `bdwc1l7d` (`stellar-shadow-480`), remote PID
  `593062`, checkpoint `checkpoints/dtps_10_fixdepth_w1_lr3e3_dteval`. This is
  the late-moving `w1_lr3e3` recipe with fixed-depth DT self-play and DT eval.

Status/refill 2026-06-15 12:00 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- Best fixed-depth early result remains worker 2 / `07h9yjtl`: best step 85
  `-0.828125`, latest step 155 `-0.908203`.
- Worker 7 / `56o561fb` finished normally: final step 219 `-0.931641`,
  best100 `-0.916016`, best late step 170 `-0.851563`, no hit.
- Late-moving runs:
  - worker 13 / `rtco03sy` improved late to best step 185 `-0.794922`, latest
    step 208 `-0.871094`; still no target-window hit.
  - worker 15 / `ati43om2` improved late to best step 159 `-0.837891`, latest
    step 171 `-0.855469`; still no target-window hit.
  - worker 12 / `recu5muu` now has late best step 135 `-0.886719`.
- Fresh probes:
  - worker 5 / `og8pq0tm` (`policy_samples=0`) best step 37 `-0.888672`.
  - worker 6 / `vj3bgkc6` (`policy_samples=64`) best step 18 `-0.955078`.
  - worker 3 / `b389datz` (`num_blocks=2`) best step 26 `-0.937500`.
  - worker 9 / `11pzw4at` (`loss_mask_mode=search`) is still very fresh,
    best step 11 `-0.972656`.
- Worker 7 replacement: W&B `82pbt3q8` (`crisp-water-482`), remote PID
  `563250`, checkpoint `checkpoints/dtps_7_fixdepth_w4ps1_lr3e3`. This tests
  the late-moving `w4ps1` family with lr `3e-3` and policy eval.

Status/refill 2026-06-15 12:39 UTC:
- Worker 13 / `rtco03sy` (`trim-spaceship-469`) is finished and is the current
  best late fixed-depth DT signal: final step 219 `-0.853516`, best100 step 78
  `-0.896484`, best late step 185 `-0.794922`. It did not hit the target
  window, but its late trend is materially better than the other recipes.
- Worker 13 remote process check showed no live training process; the only match
  was the `pgrep` command itself. No running job was killed.
- Replacement worker 13: W&B `2pw00705` (`giddy-donkey-483`), remote Python PID
  `595396`, checkpoint `checkpoints/dtps_13_fixdepth_w4ps1_lr2e3`. This keeps
  the `rtco03sy` recipe fixed (`policy_samples=1`, fixed-depth DT self-play,
  policy eval, loss weights `1/4/4`, clip `100`) and only changes
  `training.learning_rate` from `1e-3` to `2e-3`, aiming to move the same late
  improvement earlier. The `3e-3` version is already running as worker 7.

Status pass 2026-06-15 12:40 UTC:
- W&B still shows no hit (`avg_R >= 0` by step <= 100).
- All 16 tracked workers are marked running, so no job was killed and no extra
  refill was needed.
- Best live early fixed-depth result remains worker 2 / `07h9yjtl`: best100
  step 85 `-0.828125`, latest step 162 `-0.912109`, best late step 107
  `-0.843750`.
- Strongest live late movers:
  - worker 15 / `ati43om2` (`fix_p05w5clip300`): latest step 182
    `-0.863281`, best late step 159 `-0.837891`.
  - worker 12 / `recu5muu` (`fix_w4ps8`): latest step 163 `-0.916016`, best
    late step 135 `-0.886719`.
  - worker 11 / `e7trtu9g` (`fix_w4ps4_dteval`): latest step 175
    `-0.910156`, best late step 173 `-0.902344`.
- Worker 13 / `2pw00705` is live but had not logged an eval row yet at this
  check. Next pass should be coarse, around an hour later unless a slot visibly
  finishes sooner.

Status pass 2026-06-15 16:32 UTC:
- Checked the latest three W&B runs by creation time:
  - `c7is0n65` (`giddy-galaxy-491`): DT self-play with only 2 simulations,
    `max_depth=64`, policy eval, no outcome-loss weights. It is running at
    step 53, latest `avg_R=-0.867188`, best step 24 `-0.830078`.
  - `3cuaezdb` (`ethereal-resonance-490`): low-budget Gumbel self-play
    control with 5 simulations, policy eval, outcome-loss weights on. It is
    running at step 111, latest `avg_R=-1.000000`, best step 3 `-0.951172`.
    This one is not part of the apparent improvement.
  - `xmg67m67` (`resilient-shadow-489`): DT self-play with 5 simulations,
    `max_depth=32`, policy eval, no outcome-loss weights. It is running at
    step 41, latest `avg_R=-0.896484`, best step 38 `-0.839844`.
- These two DT policy-eval runs are the best fresh signals from the autoresearch
  batch. `c7is0n65` nearly matches the old fixed-depth early best
  (`07h9yjtl`, step 85 `-0.828125`) by step 24, and `xmg67m67` reaches
  `-0.839844` by step 38. They are also much earlier than the late-moving
  `rtco03sy` recipe, whose best within the speedrun horizon was only
  `-0.896484`.
- The common factor in the better-looking pair is not deeper full-width search:
  both use policy eval, low DT self-play simulation counts, and no
  value/Q outcome-loss terms. The likely explanation is that the high-budget
  DT runs were overcommitting to noisy early tree targets, while these
  low-budget DT runs behave more like a narrow stochastic data generator. Their
  policy-target entropy is still diffuse (`~2.5`), but it is lower than the
  initial `~2.77-2.79` and does not collapse into the bad low-budget Gumbel
  trajectory.
- The depth/outcome-loss comparison supports that interpretation: the nearby
  `wamub2qr` run (`simulations=5`, `max_depth=64`, outcome losses on) is worse
  so far, with best step 30 `-0.876953` and latest step 41 `-0.935547`.
  `3cuaezdb` sharpens the policy target more (`policy_target_entropy ~1.82`)
  but reward collapses to `-1`, so simply making targets sharper is not enough.
- Caveat: these are policy-eval results, not the stricter DT-eval objective, and
  no latest run has hit `avg_R >= 0` by step 100. The next useful confirmation
  is to keep `c7is0n65` and `xmg67m67` running, then repeat the same low-budget,
  no-outcome DT recipes under DT eval or with a paired seed before treating the
  effect as real rather than favorable eval noise.
