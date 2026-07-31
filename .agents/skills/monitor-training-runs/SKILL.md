---
name: monitor-training-runs
description: Monitor long-running machine-learning training runs, experiments, sweeps, jobs, and checkpoints without wasteful rapid polling. Use when Codex is asked to watch, babysit, follow, monitor, wait for, or periodically inspect a training run in W&B, TensorBoard, a scheduler, logs, or a local/remote process, especially under Goal mode or when a run may last from minutes to days.
---

# Monitor Training Runs

Monitor with sparse scheduled wake-ups. Treat waiting as expected work, not as
loss of progress.

## Establish the monitor

1. Identify the run, the authoritative status source, terminal states, failure
   signals, and the user's definition of done. Reuse known context; ask only for
   information that cannot be discovered safely.
2. Inspect the run once. Record a compact snapshot containing status, step or
   epoch, last-update time, important metrics, and an estimated completion time
   when available.
3. Distinguish these conditions:
   - `running`: the process or service reports a live run.
   - `quiet`: no new metric arrived, but there is no positive evidence of
     failure. Quiet is normal during evaluation, checkpointing, compilation,
     data loading, or long steps.
   - `stalled`: a heartbeat or progress signal is older than a justified
     threshold and corroborating evidence suggests a problem.
   - `terminal`: completed, failed, cancelled, preempted, or otherwise ended.
4. Never infer failure or completion merely from unchanged metrics or several
   uneventful checks.

## Choose a wake-up interval

Estimate remaining duration from scheduler metadata, throughput, completed
steps, prior runs, or the user's estimate. Start with this table:

| Estimated time remaining | Next check |
| --- | --- |
| up to 15 minutes | 2 minutes |
| 15–60 minutes | 5 minutes |
| 1–4 hours | 15 minutes |
| 4–12 hours | 30 minutes |
| 12–36 hours | 60 minutes |
| more than 36 hours | 2 hours |

Adjust the interval as follows:

- Shorten it near expected completion or when a credible anomaly appears.
- Lengthen it by up to 2× after two healthy, uneventful checks, without
  exceeding 2 hours unless the user requests a slower cadence.
- Do not check more frequently than the run can produce meaningful new
  information.
- Honor an explicit user cadence over these defaults.

## Sleep by scheduling, not polling

1. Search for the `automation_update` capability when it is not already
   available.
2. When scheduled tasks are supported, create or update a scheduled task
   **inside the current chat** for the chosen next-check time. Keep it in the
   current chat so it retains the run context.
3. Put durable instructions in the scheduled prompt: invoke
   `$monitor-training-runs`, identify the exact run, name the authoritative
   status source, include the previous snapshot, state terminal conditions, and
   request one bounded status check followed by cadence adjustment.
4. After scheduling succeeds, yield control. Do not fill the interval with
   repeated reasoning turns, shell `sleep` loops, or 20-second checks.
5. Do not use Goal mode itself as a timer. When monitoring is the only remaining
   work and an active goal keeps auto-continuing, pause the goal through the
   available goal control; if that control is unavailable, tell the user once
   to run `/goal pause`. The in-chat scheduled task remains responsible for the
   next check.
6. If scheduled tasks are unavailable, do not pretend a long sleep is possible.
   Perform at most one bounded wait supported by the current environment, then
   explain that reliable low-token monitoring requires an in-chat scheduled
   task or an external scheduler. Do not fall back to rapid polling.

## Handle each wake-up

1. Make one read-only status query and compare it with the saved snapshot.
2. Report only a terminal event, a credible anomaly, a request for user action,
   or a concise material progress update. Silence or a one-line update is
   appropriate for an uneventful healthy check.
3. If still running, calculate the next interval and update the existing
   scheduled task rather than creating duplicates.
4. If terminal, stop or remove the monitor, report the outcome and relevant
   final metrics, and resume or complete the surrounding goal only when its
   actual completion criteria are satisfied.

Do not cancel, kill, restart, or modify a training run unless the user
explicitly authorized that action. Do not mark a goal blocked merely because a
healthy run has not changed since the last check.
