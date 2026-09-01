"""Plot Hex 3--9 scaling curves from full-resolution W&B histories.

The evaluation logged at W&B step k precedes training iteration k, so its
training+self-play compute is ``k * FLOPS_PER_ITERATION[run]``. Evaluation and
non-neural-network FLOPs are intentionally excluded.
"""

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


METRIC = "eval/vs_baseline/win_rate"
HISTORY_KEYS = ("_step", METRIC)
BIN_WIDTH = 0.04

RUNS = {
    "dt_3": "qvi6kdax",
    "gumbel_3": "7xnbgnbi",
    "dt_4": "6qfhg2yu",
    "gumbel_4": "avk3qzwi",
    "dt_5": "hwv90a57",
    "gumbel_5": "j039hqlf",
    "dt_6": "cwu7nh51",
    "gumbel_6": "gfyib5ip",
    "dt_7": "2sqo0dyb",
    "gumbel_7": "hqzcf2eo",
    "dt_8": "gwtq38d3",
    "gumbel_8": "1gt9rc26",
    "dt_9": "xucms7r4",
    "gumbel_9": "k2jdarbu",
}

FLOPS_PER_ITERATION = {
    "dt_3": 25_830_343_507_968,
    "gumbel_3": 25_799_812_005_888,
    "dt_4": 93_570_514_550_784,
    "gumbel_4": 93_477_443_993_600,
    "dt_5": 328_879_893_708_800,
    "gumbel_5": 328_656_460_185_600,
    "dt_6": 538_389_969_371_136,
    "gumbel_6": 537_930_831_495_168,
    "dt_7": 1_033_137_176_379_392,
    "gumbel_7": 1_032_291_040_428_032,
    "dt_8": 1_808_902_001_786_880,
    "gumbel_8": 1_807_463_942_717_440,
    "dt_9": 2_956_043_127_029_760,
    "gumbel_9": 2_953_746_387_763_200,
}

BOARD_COLORS = {
    3: "#4477AA",
    4: "#66CCEE",
    5: "#228833",
    6: "#CCBB44",
    7: "#EE6677",
    8: "#AA3377",
    9: "#663399",
}

METHOD_COLORS = {
    "dt": "#ef6c23",
    "gumbel": "#2478b5",
}

METHOD_LABELS = {
    "dt": "Curiosity-Driven",
    "gumbel": "Gumbel-AlphaZero",
}

SPINE_COLOR = "#aeb9b5"
TICK_COLOR = "#75817d"
AXIS_LABEL_COLOR = "#3f4945"
TITLE_COLOR = "#31413d"


@dataclass(frozen=True)
class RunSeries:
    key: str
    run_id: str
    step: np.ndarray
    win_rate: np.ndarray

    @property
    def flops(self) -> np.ndarray:
        return self.step * FLOPS_PER_ITERATION[self.key]


def log_bin(
    x: np.ndarray, y: np.ndarray, width: float = BIN_WIDTH
) -> tuple[np.ndarray, np.ndarray]:
    """Average observations in fixed-width log10(x) bins."""
    mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[mask], y[mask]

    bin_ids = np.floor(np.log10(x) / width).astype(int)

    binned_x = []
    binned_y = []

    for bin_id in np.unique(bin_ids):
        selected = bin_ids == bin_id
        binned_x.append(10 ** np.mean(np.log10(x[selected])))
        binned_y.append(np.mean(y[selected]))

    return np.asarray(binned_x), np.asarray(binned_y)


def fetch_histories(project: str, history_dir: Path) -> dict[str, Path]:
    """Fetch requested W&B runs with scan_history and cache them as CSV."""
    import wandb

    history_dir.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    paths: dict[str, Path] = {}
    for key, run_id in RUNS.items():
        run = api.run(f"{project}/{run_id}")
        if run.id != run_id:
            raise RuntimeError(f"expected run {run_id}, got {run.id}")

        rows = list(run.scan_history(keys=list(HISTORY_KEYS), page_size=1000))
        path = history_dir / f"{key}-{run_id}.csv"
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=HISTORY_KEYS)
            writer.writeheader()
            writer.writerows(
                {field: row.get(field) for field in HISTORY_KEYS} for row in rows
            )
        paths[key] = path
        steps = [row.get("_step") for row in rows if row.get("_step") is not None]
        last_step = max(steps) if steps else None
        print(
            f"fetched {key:8s} {run_id}: {len(rows):4d} evaluations, "
            f"last evaluation step {last_step}"
        )
    return paths


def load_histories(paths: dict[str, Path]) -> dict[str, RunSeries]:
    histories: dict[str, RunSeries] = {}
    for key, path in paths.items():
        steps: list[float] = []
        win_rates: list[float] = []
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                steps.append(float(row["_step"]) if row["_step"] else np.nan)
                win_rates.append(float(row[METRIC]) if row[METRIC] else np.nan)

        step = np.asarray(steps, dtype=np.float64)
        win_rate = np.asarray(win_rates, dtype=np.float64)
        order = np.argsort(step, kind="stable")
        histories[key] = RunSeries(
            key=key,
            run_id=RUNS[key],
            step=step[order],
            win_rate=win_rate[order],
        )
    return histories


def _plot_series(
    ax: plt.Axes,
    series: RunSeries,
    *,
    color: str,
    point_size: float,
    line_width: float,
    zorder: float = 2,
) -> tuple[np.ndarray, np.ndarray]:
    x = series.flops
    y = series.win_rate
    raw_mask = np.isfinite(x) & np.isfinite(y) & (x > 0)
    ax.scatter(
        x[raw_mask],
        y[raw_mask],
        s=point_size,
        alpha=0.14,
        color=color,
        edgecolors="none",
        rasterized=True,
        zorder=zorder,
    )
    binned_x, binned_y = log_bin(x, y)
    ax.plot(
        binned_x,
        binned_y,
        color=color,
        linewidth=line_width,
        zorder=zorder + 1,
    )
    return binned_x, binned_y


def _style_data_axis(ax: plt.Axes) -> None:
    ax.set_xscale("log")
    ax.set_ylim(-0.015, 0.67)
    ax.axhline(
        0.5,
        color=TICK_COLOR,
        linestyle=":",
        linewidth=0.7,
        zorder=1,
    )
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


def plot_overview(histories: dict[str, RunSeries], output_dir: Path) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.2, 3.15),
        sharex=True,
        sharey=True,
    )
    for ax, method, title in zip(
        axes,
        ("dt", "gumbel"),
        ("Curiosity-Driven", "Gumbel-AlphaZero"),
        strict=True,
    ):
        for board_size in range(3, 10):
            _plot_series(
                ax,
                histories[f"{method}_{board_size}"],
                color=BOARD_COLORS[board_size],
                point_size=3.5,
                line_width=1.45,
        )
        _style_data_axis(ax)
        ax.set_title(title, color=TITLE_COLOR, fontsize=9.5, pad=5)

    axes[0].set_ylabel(
        "Win rate vs solved baseline",
        color=AXIS_LABEL_COLOR,
        fontsize=8.5,
    )
    fig.supxlabel(
        "Training FLOPs",
        color=AXIS_LABEL_COLOR,
        fontsize=8.5,
        y=0.175,
    )
    legend_handles = [
        Line2D([0], [0], color=BOARD_COLORS[size], linewidth=1.7, label=f"Hex {size}")
        for size in range(3, 10)
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.015),
        ncol=7,
        frameon=False,
        fontsize=7.4,
        handlelength=1.8,
        handletextpad=0.45,
        labelcolor=AXIS_LABEL_COLOR,
        columnspacing=1.15,
    )
    fig.subplots_adjust(left=0.085, right=0.99, top=0.88, bottom=0.275, wspace=0.08)
    fig.savefig(output_dir / "hex3-9-scaling-paper-logbins.png", dpi=300)
    fig.savefig(output_dir / "hex3-9-scaling-paper-logbins.pdf")
    fig.savefig(output_dir / "hex3-9-scaling-paper-logbins.svg", dpi=300)
    plt.close(fig)


def plot_paired_facets(histories: dict[str, RunSeries], output_dir: Path) -> None:
    fig, axes_grid = plt.subplots(
        2,
        4,
        figsize=(7.2, 5.5),
        sharey=True,
    )
    axes = axes_grid.flat
    for board_size, ax in zip(range(3, 10), axes[:7], strict=True):
        for method in ("gumbel", "dt"):
            _plot_series(
                ax,
                histories[f"{method}_{board_size}"],
                color=METHOD_COLORS[method],
                point_size=3.0,
                line_width=1.35,
                zorder=2 if method == "gumbel" else 4,
            )
        _style_data_axis(ax)
        ax.set_title(
            f"Hex {board_size}",
            color=TITLE_COLOR,
            fontsize=9.5,
            pad=5,
        )

    legend_ax = axes[7]
    legend_ax.axis("off")
    method_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLORS[method],
            linewidth=1.6,
            marker="o",
            markersize=3,
            label=METHOD_LABELS[method],
        )
        for method in ("dt", "gumbel")
    ]
    legend_ax.legend(
        handles=method_handles,
        loc="center",
        frameon=False,
        fontsize=8.5,
        handlelength=2.2,
        labelcolor=AXIS_LABEL_COLOR,
    )
    fig.supxlabel(
        "Training FLOPs",
        color=AXIS_LABEL_COLOR,
        fontsize=8.5,
        y=0.04,
    )
    fig.supylabel(
        "Win rate",
        color=AXIS_LABEL_COLOR,
        fontsize=8.5,
        x=0.025,
    )
    fig.subplots_adjust(
        left=0.08,
        right=0.99,
        top=0.94,
        bottom=0.105,
        wspace=0.12,
        hspace=0.30,
    )
    fig.savefig(output_dir / "hex3-9-paired-paper-logbins.png", dpi=300)
    fig.savefig(output_dir / "hex3-9-paired-paper-logbins.pdf")
    fig.savefig(output_dir / "hex3-9-paired-paper-logbins.svg", dpi=300)
    plt.close(fig)


def print_numeric_summary(histories: dict[str, RunSeries]) -> None:
    for key, series in histories.items():
        raw_mask = (
            np.isfinite(series.flops)
            & np.isfinite(series.win_rate)
            & (series.flops > 0)
        )
        binned_x, binned_y = log_bin(series.flops, series.win_rate)
        max_idx = int(np.nanargmax(series.win_rate))
        print(
            f"summary {key:8s}: raw plotted={raw_mask.sum():4d}, "
            f"bins={len(binned_x):4d}, final={series.win_rate[-1]:.4f}, "
            f"best={series.win_rate[max_idx]:.4f}@step{series.step[max_idx]:.0f}, "
            f"last_bin={binned_y[-1]:.4f}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="pal/scacchi-az")
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--history-dir",
        type=Path,
        default=None,
        help="optional persistent directory for raw scan_history CSV files",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["ps.fonttype"] = 42
    mpl.rcParams["svg.fonttype"] = "none"

    if args.history_dir is not None:
        paths = fetch_histories(args.project, args.history_dir)
        histories = load_histories(paths)
        plot_overview(histories, args.output_dir)
        plot_paired_facets(histories, args.output_dir)
        print_numeric_summary(histories)
    else:
        with tempfile.TemporaryDirectory(prefix="hex3-9-wandb-") as tmp:
            paths = fetch_histories(args.project, Path(tmp))
            histories = load_histories(paths)
            plot_overview(histories, args.output_dir)
            plot_paired_facets(histories, args.output_dir)
            print_numeric_summary(histories)

    for filename in (
        "hex3-9-scaling-paper-logbins.png",
        "hex3-9-scaling-paper-logbins.pdf",
        "hex3-9-scaling-paper-logbins.svg",
        "hex3-9-paired-paper-logbins.png",
        "hex3-9-paired-paper-logbins.pdf",
        "hex3-9-paired-paper-logbins.svg",
    ):
        print(f"wrote {args.output_dir / filename}")


if __name__ == "__main__":
    main()
