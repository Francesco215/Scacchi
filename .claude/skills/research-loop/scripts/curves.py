"""Draw the training curves and evals of one or more runs on shared axes.

    .venv/bin/python .claude/skills/research-loop/scripts/curves.py \
        ratio=outputs/ratio_hex6/binned_ratio.jsonl \
        mixture=wandb:pal/scacchi-az/abc123 \
        --out outputs/ratio_hex6/curves.png

Each run is ``label=source``. A source is a local JSONL history written by
``+logging.jsonl_path=...`` or ``wandb:entity/project/run_id``. Raw values are
drawn faint and a running mean solid, as in the blog's figures. The script
prints nothing about final values on purpose: open the image and read shapes.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (panel title, regex over metric keys, log-scale y). A panel is drawn only if
# some run logged a matching key, so Beta and binned runs can share a figure.
PANELS = [
    ("eval: win rate vs baseline", r"^eval/.*/win_rate$", False),
    ("value NLL (search target)", r"^train/value_(dir_kl|binned)_loss$", True),
    ("Q NLL (search target)", r"^train/q_(dir_kl|binned)_loss$", True),
    ("value outcome NLL", r"^train/value_outcome_loss$", False),
    ("policy target entropy", r"^train/policy_target_entropy$", False),
    ("policy KL to search", r"^train/policy_kl_hat$", True),
    ("value belief entropy", r"^train/value_belief_entropy$", False),
    ("value belief variance", r"^train/value_belief_variance$", False),
    ("V concentration: predicted", r"V_C_pred_mean_dir$", True),
    ("V concentration: search target", r"V_C_target_mean_dir$", True),
    ("Q concentration: predicted", r"Q_C_pred_mean_dir$", True),
    ("Q concentration: search target", r"Q_C_target_mean_dir$", True),
    ("fraction of proven V targets", r"v_categorical_target_fraction$", False),
]


def load(source: str) -> list[dict]:
    if source.startswith("wandb:"):
        import wandb

        return list(wandb.Api().run(source.removeprefix("wandb:")).scan_history())
    return [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]


def series(rows: list[dict], pattern: str, x_key: str):
    regex = re.compile(pattern)
    xs, ys = [], []
    for index, row in enumerate(rows):
        values = [v for k, v in row.items() if regex.search(k) and isinstance(v, (int, float)) and math.isfinite(v)]
        if values:
            x = row.get(x_key)
            xs.append(x if isinstance(x, (int, float)) else row.get("step", row.get("_step", index)))
            ys.append(values[0])
    return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)


def running_mean(y: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(y) < 3:
        return y
    padded = np.pad(y, (window // 2, window - 1 - window // 2), mode="edge")
    return np.convolve(padded, np.ones(window) / window, mode="valid")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="label=path.jsonl or label=wandb:entity/project/run_id")
    parser.add_argument("--out", required=True)
    parser.add_argument("--x", default="train/frames", help="history key for the x axis")
    parser.add_argument("--logx", action="store_true")
    parser.add_argument("--smooth", type=int, default=7, help="running-mean window in logged points")
    parser.add_argument("--metric", action="append", default=[], help="extra regex to plot as its own panel")
    args = parser.parse_args()

    runs = {label: load(source) for label, source in (item.split("=", 1) for item in args.runs)}
    panels = PANELS + [(pattern, pattern, False) for pattern in args.metric]
    panels = [p for p in panels if any(len(series(rows, p[1], args.x)[0]) for rows in runs.values())]
    columns = 3
    figure, axes = plt.subplots(math.ceil(len(panels) / columns), columns, figsize=(5.2 * columns, 3.4 * math.ceil(len(panels) / columns)), squeeze=False)
    colors = plt.get_cmap("tab10")
    for axis, (title, pattern, log_y) in zip(axes.flat, panels):
        for index, (label, rows) in enumerate(runs.items()):
            x, y = series(rows, pattern, args.x)
            if not len(x):
                continue
            sparse = len(x) < 24  # evals: show every point, no smoothing
            axis.plot(x, y, color=colors(index), alpha=0.9 if sparse else 0.25, marker="o" if sparse else None, markersize=4, linewidth=1.4 if sparse else 0.8, label=label if sparse else None)
            if not sparse:
                axis.plot(x, running_mean(y, args.smooth), color=colors(index), linewidth=1.6, label=label)
        if "win rate" in title:
            axis.axhline(0.5, color="gray", linestyle=":", linewidth=1)
            axis.set_ylim(-0.02, 1.02)
        elif log_y:
            axis.set_yscale("log")
        if args.logx:
            axis.set_xscale("log")
        axis.set_title(title, fontsize=10)
        axis.set_xlabel(args.x, fontsize=8)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    for axis in list(axes.flat)[len(panels):]:
        axis.axis("off")
    figure.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=110)
    print(f"wrote {args.out}: {len(panels)} panels, {len(runs)} runs")


if __name__ == "__main__":
    main()
