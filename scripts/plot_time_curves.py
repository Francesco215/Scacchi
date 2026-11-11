"""Training curves against wall-clock time, and seconds per iteration.

Runs are label=path.jsonl or label=wandb:entity/project/run_id, as in the
research-loop curves script. Time is the running sum of train/iter_seconds,
so it counts training iterations only (not start-up or compilation gaps).
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(spec):
    if spec.startswith("wandb:"):
        import wandb

        return [dict(row) for row in wandb.Api().run(spec[6:]).scan_history()]
    return [json.loads(line) for line in open(spec) if line.strip()]


def series(rows, key):
    """(elapsed seconds, value) where the key is logged; elapsed = running sum of train/iter_seconds."""
    elapsed, out = 0.0, []
    for row in rows:
        if row.get("train/iter_seconds") is not None:
            elapsed += float(row["train/iter_seconds"])
        if row.get(key) is not None:
            out.append((elapsed, float(row[key])))
    return np.array(out).reshape(-1, 2)


def find(rows, *needles):
    keys = {k for row in rows for k in row}
    for k in sorted(keys):
        if all(n in k.lower() for n in needles):
            return k
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    runs = {spec.split("=", 1)[0]: load(spec.split("=", 1)[1]) for spec in args.runs}
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
    for label, rows in runs.items():
        win = find(rows, "win_rate")
        nll = find(rows, "train/value_binned_loss") or find(rows, "train/value_dir_kl_loss")
        for ax, key in ((axes[0], win), (axes[1], nll)):
            if key:
                xy = series(rows, key)
                ax.plot(xy[:, 0] / 60, xy[:, 1], label=label)
        secs = np.array([float(r["train/iter_seconds"]) for r in rows if r.get("train/iter_seconds") is not None])
        axes[2].plot(np.arange(len(secs)), secs, label=f"{label} (median {np.median(secs):.2f} s)")
    axes[0].set(title="eval: win rate vs baseline", xlabel="training time (minutes)")
    axes[1].set(title="value NLL (search target)", xlabel="training time (minutes)", yscale="log")
    axes[2].set(title="seconds per iteration", xlabel="iteration", ylim=(0, None))
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
