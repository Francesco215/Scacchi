"""Figure for the curious self-play runs: training curves and the noise check.

    .venv/bin/python scripts/plot_curiosity.py \
        "no curiosity"=outputs/curiosity/h8/head_only.jsonl "T = 1"=outputs/curiosity/h8/curious_t1.jsonl \
        --noise "T = 1"=outputs/curiosity/h8/noise_t1.json --out reports/curiosity/hex8.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CURVES = [
    ("win rate against the solved network", "eval/vs_baseline/win_rate"),
    ("bits of a self-play position before it is trained on", "data/bits"),
    ("correlation of predicted and measured next-position bits", "curiosity/correlation"),
    ("games finished per lane and self-play call", "data/terminations_per_row"),
    ("share of value targets that are proven", "data/v_categorical_target_fraction"),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", help="label=history.jsonl")
    parser.add_argument("--noise", action="append", default=[], help="label=noise_check.json")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    runs = {label: [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()] for label, path in (item.rsplit("=", 1) for item in args.runs)}
    colors = dict(zip(runs, ["black", "tab:blue", "tab:red", "tab:green", "tab:orange"]))
    figure, axes = plt.subplots(2, 3, figsize=(17, 9.5))
    for axis, (title, key) in zip(axes.flat, CURVES):
        for label, rows in runs.items():
            if key.startswith("eval/"):  # the loop evaluates before it trains: row i is the network of row i - 1
                points = [(rows[i - 1]["train/frames"], r[key]) for i, r in enumerate(rows) if i and key in r]
            else:
                points = [(r["train/frames"], r[key]) for r in rows if key in r]
            if points:
                x, y = np.asarray(points).T
                axis.plot(x / 1e6, y, marker="o", markersize=3, color=colors[label], label=label)
        if key.startswith("eval/"):
            axis.axhline(0.5, color="gray", linestyle=":", linewidth=1)
        axis.set_title(title, fontsize=10)
        axis.set_xlabel("self-play frames (millions)")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.2)
    axis = axes.flat[5]
    for item in args.noise:
        label, path = item.rsplit("=", 1)
        report = json.loads(Path(path).read_text())
        for name, style in (("plain", "--"), ("curious", "-")):
            part = report[name]["by_predicted"]
            axis.plot(part["x"], part["signal_share"], marker="o", linestyle=style, color=colors.get(label, "tab:blue"), label=f"{label} network, {name} games")
    axis.set_ylim(0, 1)
    axis.set_title("share of a target's pull that a second search reproduces", fontsize=10)
    axis.set_xlabel("bits the head predicted for the position (octile means)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.out, dpi=110)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
