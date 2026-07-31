---
name: inspect-training-curves
description: Render training histories as temporary PNG plots and inspect them visually before analyzing learning dynamics. Use for Weights & Biases (W&B/WandB) runs, TensorBoard exports, CSV or JSONL histories, learning curves, loss curves, evaluation curves, instability diagnosis, convergence analysis, or comparisons between two or more training runs. Trigger especially when run histories are available as arrays or tables and the user asks which run is better, faster, stabler, converged, collapsed, plateaued, or changed after a configuration modification.
---

# Inspect training curves

Treat numeric history as data for a visualization, not as a substitute for one.

## Required workflow

1. Identify the decision and the metrics that can answer it. Include the primary outcome plus likely explanatory metrics.
2. Fetch full-resolution history when practical. Preserve the run name, run ID, x-axis, missing values, and actual last recorded step. Do not silently compare sparse API samples as if they were complete.
3. Render the relevant runs together with `scripts/plot_curves.py`, or an equivalent plotting command when the input format requires it. Write images to a temporary directory.
4. Open every generated PNG with the available local image-viewing tool. This visual inspection is mandatory; creating a plot without viewing it does not satisfy the skill.
5. If overlap, scale, noise, or truncation hides a conclusion, render a second diagnostic view such as log scale, a focused step range, separate panels, or raw plus smoothed curves, and inspect it too.
6. Cross-check visual impressions against a few computed values before reporting them. Useful checks include final value, best value, best step, rolling variability, area under the curve, slope over the final window, and values at matched steps.
7. Explain the observed dynamics, uncertainty, and comparison limits. Distinguish evidence from hypotheses about causes.

## Comparison rules

- Compare runs on the same x-axis semantics. Prefer optimizer/environment steps over row index or wall time unless time efficiency is the question.
- Use the shared step range for direct claims. Call out when one run is longer; do not treat extra training as an intrinsic quality advantage.
- Overlay the same metric across runs. Use one subplot per metric unless units and scales genuinely match.
- Show raw data faintly and an EMA trace prominently. Treat smoothing only as a visual aid and keep spikes, collapse, oscillation, and regime changes visible.
- Use identical y-limits for compared runs within a metric. Never autoscale separate panels in a way that visually exaggerates or hides differences.
- Inspect both outcome and mechanism when available: evaluation strength/reward, losses, learning rate, gradient/update norms, entropy, value/policy diagnostics, throughput, and numerical-health metrics.
- Prefer several legible plots over one overloaded dashboard.
- Do not infer dynamics from endpoints alone.

## W&B history

Use the authenticated W&B API already available in the environment. Request only needed keys, but obtain enough samples to preserve the curve shape. When `run.history()` sampling may omit structure, prefer `run.scan_history(keys=[...])` or explicitly state the sampling limitation.

Convert histories to CSV or JSONL, then plot them. Keep temporary artifacts out of the repository unless the user asks to save them.

## Plot helper

```bash
python scripts/plot_curves.py \
  --input baseline=/tmp/baseline.csv \
  --input candidate=/tmp/candidate.csv \
  --metrics eval/score,train/loss \
  --x _step \
  --output /tmp/training-curves.png
```

The helper accepts CSV and JSONL files. Use `--ema 0` to disable smoothing, `--log-y` for positive metrics spanning orders of magnitude, and `--x-min`/`--x-max` for focused views.

## Reporting

Base the answer on the inspected figures and numeric cross-checks. State:

- what happened over time;
- where runs diverged, plateaued, destabilized, recovered, or crossed;
- which run is preferable under the user's criterion;
- whether the conclusion holds at matched compute/steps;
- any caveat from noise, sparse logging, different lengths, seeds, or scales.

Include the plot in the response when it helps the user verify the conclusion. Remove temporary artifacts afterward only when they are no longer needed and removal is safe.
