"""Plot prediction entropy for the Hex 3--9 Dirichlet--Thompson runs."""

from __future__ import annotations

import argparse
import csv
import tempfile
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from plot_hex3_9_logbins import (
    AXIS_LABEL_COLOR,
    BIN_WIDTH,
    BOARD_COLORS,
    FLOPS_PER_ITERATION,
    RUNS,
    SPINE_COLOR,
    TICK_COLOR,
    log_bin,
)


METRIC = "train/policy_target_entropy"
HISTORY_KEYS = ("_step", METRIC)
DT_RUNS = {key: run_id for key, run_id in RUNS.items() if key.startswith("dt_")}


@dataclass(frozen=True)
class EntropySeries:
    key: str
    step: np.ndarray
    entropy: np.ndarray

    @property
    def flops(self) -> np.ndarray:
        return self.step * FLOPS_PER_ITERATION[self.key]


def fetch_histories(project: str, history_dir: Path) -> dict[str, Path]:
    """Fetch full-resolution prediction entropy with W&B scan_history."""
    import wandb

    history_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    paths: dict[str, Path] = {}
    for key, run_id in DT_RUNS.items():
        run = api.run(f"{project}/{run_id}")
        rows = list(run.scan_history(keys=list(HISTORY_KEYS), page_size=1000))
        path = history_dir / f"{key}-{run_id}-prediction-entropy.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_KEYS)
            writer.writeheader()
            writer.writerows(
                {field: row.get(field) for field in HISTORY_KEYS} for row in rows
            )
        paths[key] = path
        steps = [row.get("_step") for row in rows if row.get("_step") is not None]
        print(
            f"fetched {key:5s} {run_id}: {len(rows):4d} values, "
            f"last step {max(steps) if steps else None}"
        )
    return paths


def load_histories(paths: dict[str, Path]) -> dict[str, EntropySeries]:
    histories: dict[str, EntropySeries] = {}
    for key, path in paths.items():
        steps: list[float] = []
        entropies: list[float] = []
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                steps.append(float(row["_step"]) if row["_step"] else np.nan)
                entropies.append(float(row[METRIC]) if row[METRIC] else np.nan)
        step = np.asarray(steps, dtype=np.float64)
        entropy = np.asarray(entropies, dtype=np.float64)
        order = np.argsort(step, kind="stable")
        histories[key] = EntropySeries(
            key=key,
            step=step[order],
            entropy=entropy[order],
        )
    return histories


def style_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.tick_params(
        axis="both",
        which="major",
        colors=TICK_COLOR,
        labelsize=7.5,
        length=2.7,
        width=0.6,
    )
    ax.tick_params(
        axis="x",
        which="minor",
        colors=SPINE_COLOR,
        length=1.5,
        width=0.45,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("bottom", "left"):
        ax.spines[side].set_color(SPINE_COLOR)
        ax.spines[side].set_linewidth(0.65)


def plot(histories: dict[str, EntropySeries], output_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    for board_size in range(3, 10):
        series = histories[f"dt_{board_size}"]
        x = series.flops
        y = series.entropy
        raw_mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
        ax.scatter(
            x[raw_mask],
            y[raw_mask],
            s=3.5,
            alpha=0.14,
            color=BOARD_COLORS[board_size],
            edgecolors="none",
            rasterized=True,
            zorder=2,
        )
        binned_x, binned_y = log_bin(x, y, width=BIN_WIDTH)
        ax.plot(
            binned_x,
            binned_y,
            color=BOARD_COLORS[board_size],
            linewidth=1.45,
            zorder=3,
        )

        final_window = y[np.isfinite(y)][-min(10, np.isfinite(y).sum()):]
        print(
            f"summary dt_{board_size}: raw plotted={raw_mask.sum():4d}, "
            f"bins={len(binned_x):3d}, final={y[-1]:.4f}, "
            f"late mean={np.mean(final_window):.4f}, "
            f"late std={np.std(final_window):.4f}"
        )

    style_axis(ax)
    ax.set_ylabel("prediction entropy (nats)", color=AXIS_LABEL_COLOR, fontsize=8.2)
    ax.set_xlabel(
        "Training FLOPs",
        color=AXIS_LABEL_COLOR,
        fontsize=8.0,
        labelpad=4,
    )
    legend_handles = [
        Line2D([0], [0], color=BOARD_COLORS[size], linewidth=1.7, label=f"{size}")
        for size in range(3, 10)
    ]
    legend = ax.legend(
        handles=legend_handles,
        title="Board size",
        loc="upper left",
        bbox_to_anchor=(0.015, 0.985),
        ncol=2,
        frameon=False,
        fontsize=6.9,
        handlelength=1.6,
        handletextpad=0.4,
        labelcolor=AXIS_LABEL_COLOR,
        columnspacing=0.9,
        borderaxespad=0,
        alignment="left",
    )
    legend.get_title().set_color(AXIS_LABEL_COLOR)
    legend.get_title().set_fontsize(7.2)
    fig.subplots_adjust(left=0.14, right=0.99, top=0.97, bottom=0.16)

    stem = "hex3-9-dt-prediction-entropy-logbins"
    fig.savefig(output_dir / f"{stem}.png", dpi=300)
    fig.savefig(output_dir / f"{stem}.svg", dpi=300)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="pal/scacchi-az")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--history-dir", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpl.rcParams["svg.fonttype"] = "none"

    if args.history_dir is not None:
        paths = fetch_histories(args.project, args.history_dir)
        histories = load_histories(paths)
        plot(histories, args.output_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="dt-prediction-entropy-") as tmp:
            paths = fetch_histories(args.project, Path(tmp))
            histories = load_histories(paths)
            plot(histories, args.output_dir)


if __name__ == "__main__":
    main()
