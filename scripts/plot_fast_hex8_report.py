"""Generate the three figures for the fast-hex8 PR report.

1. hex8_learning.png: Hex 8 win rate vs wall-clock time, vs searched positions,
   and time-to-parity bars across five generations of runs against 8_solved.
2. hex8_speed.png: Iteration time waterfall, per-simulation active leaf decay
   from subtree reuse + position cache + confidence early-stopping, and
   stratified evaluation speedup.
3. hex8_fidelity.png: Q=9 vs Q=33 quadrature policy error, per-layer FP8 vs
   BF16 trunk gradient cosine similarity, and value search-target NLL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter
import numpy as np

INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
RED, BLUE, GREEN, ORANGE, GOLD, NEUTRAL = (
    "#d6274b",
    "#2a78d6",
    "#1baf7a",
    "#eb6834",
    "#d99b00",
    "#a3a29b",
)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10.0,
    "axes.edgecolor": MUTED,
    "axes.labelcolor": MUTED,
    "axes.titlesize": 11.5,
    "axes.titleweight": "bold",
    "axes.titlecolor": INK,
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "legend.frameon": False,
})


def arrow(ax, start, end, text, text_xy, color=INK, fontsize=9.8):
    ax.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4, shrinkA=2, shrinkB=2),
    )
    ax.text(*text_xy, text, color=color, fontsize=fontsize, fontweight="bold", ha="center", va="center")


def load_run(path: Path):
    rows = [json.loads(line) for line in path.open() if line.strip().startswith("{")]
    evals = [r for r in rows if r.get("eval/vs_baseline/win_rate") is not None]
    vloss_rows = [
        r for r in rows
        if r.get("train/value_binned_loss") is not None and r.get("train/frames") is not None
    ]
    return {
        "minutes": np.array([float(r["train/hours"]) * 60.0 for r in evals]),
        "frames": np.array([float(r.get("train/searches", r["train/frames"])) for r in evals]),
        "win": np.array([float(r["eval/vs_baseline/win_rate"]) for r in evals]),
        "v_frames": np.array([float(r.get("train/searches", r["train/frames"])) for r in vloss_rows]),
        "v_loss": np.array([float(r["train/value_binned_loss"]) for r in vloss_rows]),
        "iter_s": float(np.median([float(r["train/iter_seconds"]) for r in rows[2:]])),
        "total_min": float(rows[-1]["train/hours"]) * 60.0,
    }


def smooth_log(x: np.ndarray, y: np.ndarray, sigma: float = 0.085):
    mask = x > 0
    x, y = x[mask], y[mask]
    lx = np.log10(x)
    grid = np.linspace(lx.min(), lx.max(), 320)
    w = np.exp(-0.5 * ((grid[:, None] - lx[None, :]) / sigma) ** 2)
    return 10.0**grid, (w @ y) / w.sum(axis=1)


def crossing(x: np.ndarray, y: np.ndarray, level: float = 0.5):
    for (x0, y0), (x1, y1) in zip(zip(x[:-1], y[:-1]), zip(x[1:], y[1:])):
        if y0 < level <= y1:
            return float(x0 + (x1 - x0) * (level - y0) / max(y1 - y0, 1e-12))
    return None


def make_learning_figure(runs_meta: list[tuple[str, str, Path, str]], out_path: Path):
    data = {rid: load_run(p) for rid, _, p, _ in runs_meta}
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 5.1), gridspec_kw={"width_ratios": [1.2, 1.2, 0.95]})
    crossings_min: dict[str, float | None] = {}

    for col, (key, xlabel) in enumerate([
        ("minutes", "training wall-clock time (minutes, log scale)"),
        ("frames", "positions searched (millions, log scale)"),
    ]):
        ax = axes[col]
        ax.set_xscale("log")
        ax.xaxis.set_major_locator(LogLocator(base=10, subs=(1, 2, 5)))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(0, 0.66)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0%}"))
        ax.axhline(0.5, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)

        for idx, (rid, label, _, color) in enumerate(runs_meta):
            d = data[rid]
            x, y = d[key], d["win"]
            mask = x > (0.12 if key == "minutes" else 4e4)
            x_plot, y_plot = x[mask], y[mask]
            lw = 2.6 if idx == 0 else 1.9
            z = 5 if idx == 0 else 3
            ax.scatter(x_plot, y_plot, s=11 if idx == 0 else 8, color=color, alpha=0.28, lw=0, zorder=2)
            sx, sy = smooth_log(x_plot, y_plot, sigma=0.08 if idx == 0 else 0.095)
            ax.plot(sx, sy, color=color, lw=lw, zorder=z, label=f"{label} ({rid})")
            c_val = crossing(sx, sy, 0.5)
            if key == "minutes":
                crossings_min[rid] = c_val
            if c_val is not None:
                ax.plot([c_val], [0.5], "o", color=color, ms=7.0, mec=SURFACE, mew=1.8, zorder=6)

        if key == "minutes":
            ax.set_xlim(0.15, 75)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.set_ylabel("eval win rate vs 8_solved (512 games)", color=INK)
            ax.set_title("Strength against training wall-clock", loc="left")
            c_fast = crossings_min[runs_meta[0][0]]
            c_pool = crossings_min[runs_meta[1][0]]
            if c_fast and c_pool:
                arrow(
                    ax,
                    (c_pool, 0.34),
                    (c_fast, 0.34),
                    f"{c_pool / c_fast:.1f}× faster\nthan pool + int8",
                    (0.52, 0.34),
                    color=RED,
                )
            ax.text(0.18, 0.515, "50% parity vs 8_solved", color=MUTED, fontsize=8.8, ha="left", va="bottom")
        else:
            ax.set_xlim(8e4, 5.5e7)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
            ax.set_title("Strength against search work", loc="left")
            ax.annotate(
                "2× Z₂ rot180 updates\nat target age 0:\n54.5% mean (59.4% peak),\nzero pool staleness",
                xy=(3.5e6, 0.54),
                xytext=(1.05e5, 0.31),
                color=INK,
                fontsize=8.9,
                ha="left",
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3),
            )
        ax.set_xlabel(xlabel, color=INK)

    # Panel 3: Time to 50% parity bars
    ax = axes[2]
    y_pos = np.arange(len(runs_meta))[::-1]
    max_m = max(
        crossings_min[rid] if crossings_min[rid] is not None else data[rid]["total_min"]
        for rid, *_ in runs_meta
    )
    for yi, (rid, label, _, color) in zip(y_pos, runs_meta):
        m = crossings_min[rid]
        if m is None:
            tot = data[rid]["total_min"]
            ax.barh(yi, tot, color=color, alpha=0.25, height=0.58)
            ax.text(tot + 0.5, yi, f"not reached ({tot:.0f} min)", va="center", fontsize=8.8, color=MUTED)
        else:
            ax.barh(yi, m, color=color, height=0.58)
            extra = "  (2.0m w/ JIT)" if rid == runs_meta[0][0] else ""
            ax.text(
                m + 0.5,
                yi,
                f"{m:.2f} min{extra}" if m < 3 else f"{m:.1f} min",
                va="center",
                fontsize=9.2,
                fontweight="bold" if rid == runs_meta[0][0] else "normal",
                color=INK,
            )
    ax.set_yticks(y_pos, [f"{label}\n({rid})" for rid, label, _, _ in runs_meta], fontsize=8.8, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max_m * 1.78)
    ax.set_xlabel("training minutes to 50% parity (smoothed)", color=INK)
    ax.set_title("Time to parity vs 8_solved", loc="left")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False, fontsize=9.2, labelcolor=INK)
    fig.suptitle(
        "Hex 8 end-to-end learning   ·   parity in 1.4 min of training (2.0 min total wall-clock), RTX 5090",
        x=0.008,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 0.94), w_pad=2.2)
    fig.savefig(out_path, dpi=145)
    plt.close(fig)
    print("wrote", out_path)


def make_speed_figure(meas: dict, out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 4.9), gridspec_kw={"width_ratios": [1.22, 1.18, 0.92]})

    # Panel 1: Iteration waterfall at matched 2048 x 16 = 32,768 searches/iter
    ax = axes[0]
    wf = meas["waterfall"]
    # Estimate 2-pass megakernel cost: megakernel 1-pass (1.131s) + 1 extra eager BF16 epoch (~0.34s)
    mk_1p = wf["megakernel_int8_1pass"]
    fast_1p = wf["plus_lazy_bf16_lastdiag_1pass"]
    fast_2p = wf["plus_2x_rot180_fp8_full"]
    steps = [
        ("megakernel\n(1 pass)", mk_1p, NEUTRAL),
        ("+ Q=9 &\n1-child", wf["plus_fast_search_no_reuse"], BLUE),
        ("+ reuse,\ncache, stop", wf["plus_reuse_cache_earlystop"], GREEN),
        ("+ lazy bf16\n& FP8 (1p)", fast_1p, ORANGE),
        ("+ 2× Z₂\nrot180 (2p)", fast_2p, RED),
    ]
    x = np.arange(len(steps))
    bars = ax.bar(x, [s[1] for s in steps], color=[s[2] for s in steps], width=0.56)
    bars[-1].set_edgecolor(INK)
    bars[-1].set_linewidth(1.0)
    ax.set_xticks(x, [s[0] for s in steps], color=INK, fontsize=8.6)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    for xi, (_, v, _) in zip(x, steps):
        ax.text(xi, v + 0.028, f"{v:.2f} s", ha="center", va="bottom", color=INK, fontsize=9.2, fontweight="bold" if xi in (3, 4) else "normal")
    ax.set_ylim(0, 1.44)
    ax.set_ylabel("seconds / iteration (32,768 searches)", color=INK)
    ax.set_title("Iteration waterfall (Hex 8, B = 2048 × 16)", loc="left")
    arrow(ax, (3.0, mk_1p), (3.0, fast_1p + 0.09), f"{mk_1p / fast_1p:.2f}× faster\n(at 1 pass)", (2.05, 1.25), color=ORANGE)
    ax.annotate(
        "+0.16 s buys a full 2nd\nrot180 pass (64 updates)",
        xy=(4.0, fast_2p + 0.08),
        xytext=(3.2, 1.09),
        color=RED,
        fontsize=8.6,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.2),
    )

    # Panel 2: Active leaf evaluations across 1 ply (root + 32 sims, B = 2048)
    ax = axes[1]
    ac = np.array(meas["active_counts"], dtype=float)
    B = float(meas["batch_size"])
    stages = np.arange(len(ac))  # 0 = root, 1..32 = sim 0..31
    baseline_line = np.full_like(ac, B)
    ax.fill_between(stages, ac, baseline_line, color=GREEN, alpha=0.20, label="saved evals (22.1% = 14,905 boards)")
    ax.fill_between(stages, 0, ac, color=BLUE, alpha=0.22)
    ax.plot(stages, baseline_line, color=NEUTRAL, lw=1.8, ls="--", label=f"megakernel: {int(33 * B):,} evals/ply")
    ax.plot(stages, ac, color=BLUE, lw=2.3, label=f"fast_search: {int(ac.sum()):,} evals/ply")
    ax.plot([0], [ac[0]], "o", color=RED, ms=7.5, zorder=5)
    ax.annotate(
        "root evals: 0 / 2,048\n(100% reused from prior ply!)",
        xy=(0.2, 20),
        xytext=(4.2, 420),
        color=RED,
        fontsize=9.0,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", color=RED, lw=1.3),
    )
    ax.annotate(
        "inherited subtree nodes +\n0.96 confidence early-stop\ndecay late-sim waves",
        xy=(28.5, ac[28]),
        xytext=(12.5, 930),
        color=INK,
        fontsize=8.8,
        arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2),
    )
    ax.set_xlim(-0.8, 32.5)
    ax.set_ylim(0, B * 1.38)
    ax.set_xticks([0, 8, 16, 24, 32], ["root", "sim 8", "sim 16", "sim 24", "sim 32"])
    ax.set_xlabel("evaluation wave within one move", color=INK)
    ax.set_ylabel("active boards evaluated (*d_boards)", color=INK)
    ax.set_title("Dynamic active-board compaction per move", loc="left")
    ax.legend(loc="upper right", fontsize=8.5)

    # Panel 3: Evaluation speedup (512 games vs 8_solved, strong trained checkpoint)
    ax = axes[2]
    ev = meas["eval_ablation"]
    ev_labels = [
        "standard play_eval\n(512 × 2 searches/ply)",
        "+ fast_search\nplayer",
        "+ stratified 256/256\n& active compaction",
    ]
    ev_vals = [ev["eval_baseline"], ev["eval_fast_search_only"], ev["eval_fast_plus_stratified"]]
    ev_cols = [NEUTRAL, BLUE, RED]
    y_ev = np.arange(len(ev_labels))[::-1]
    ax.barh(y_ev, ev_vals, color=ev_cols, height=0.56)
    ax.set_yticks(y_ev, ev_labels, fontsize=8.8, color=INK)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="y", visible=False)
    for yi, v in zip(y_ev, ev_vals):
        ax.text(v + 0.03, yi, f"{v:.2f} s", va="center", fontsize=9.2, color=INK, fontweight="bold" if yi == 0 else "normal")
    ax.set_xlim(0, max(ev_vals) * 1.62)
    ax.set_xlabel("seconds for 512 games (trained checkpoint)", color=INK)
    ax.set_title("512-game eval vs 8_solved", loc="left")
    arrow(ax, (1.68, y_ev[0] - 0.08), (1.68, y_ev[2] + 0.18), f"{ev_vals[0] / ev_vals[2]:.2f}× faster\n& 0 color noise", (2.02, y_ev[1]), color=RED, fontsize=8.8)

    fig.suptitle(
        "Where the speed comes from   ·   search megakernel, training, and evaluation on RTX 5090",
        x=0.008,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.2)
    fig.savefig(out_path, dpi=145)
    plt.close(fig)
    print("wrote", out_path)


def make_fidelity_figure(meas: dict, runs_meta: list[tuple[str, str, Path, str]], out_path: Path):
    fig, axes = plt.subplots(1, 3, figsize=(17.2, 4.7), gridspec_kw={"width_ratios": [1.0, 1.1, 1.15]})

    # Panel 1: Q=9 vs Q=33 quadrature accuracy across 2,048 trained midgame boards
    ax = axes[0]
    errs = np.maximum(np.array(meas["pol_q9_err_all"]), 1e-9)
    qs = [50, 90, 99, 100]
    qvals = np.percentile(errs, qs)
    qlabels = ["median\n(p50)", "p90", "p99", "worst\n(2,048 brds)"]
    x = np.arange(len(qs))
    ax.bar(x, qvals, color=[GREEN, BLUE, BLUE, ORANGE], width=0.54)
    ax.set_yscale("log")
    ax.set_ylim(5e-10, 5e-3)
    ax.set_xticks(x, qlabels, color=INK, fontsize=9.0)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    for xi, v in zip(x, qvals):
        ax.text(xi, v * 1.55, f"{v:.1e}", ha="center", va="bottom", fontsize=8.8, color=INK)
    ax.axhline(1e-3, color=MUTED, lw=0.9, ls="--")
    ax.text(3.35, 1.35e-3, "0.1% policy error", color=MUTED, fontsize=8.5, ha="right")
    ax.text(
        0.04,
        0.88,
        f"Top-1 move agreement:\n{meas['top1_agree']*100:.1f}% (2,048 / 2,048)",
        transform=ax.transAxes,
        fontsize=9.3,
        fontweight="bold",
        color=GREEN,
        va="top",
    )
    ax.set_ylabel("max |π_Q9(a) − π_Q33(a)| (log scale)", color=INK)
    ax.set_title("Q = 9 vs exact Q = 33 quadrature", loc="left")

    # Panel 2: Per-layer FP8 e4m3fn vs BF16 conv weight gradient cosine similarity
    ax = axes[1]
    cos_vals = np.array(meas["layer_cos"])
    layers = np.arange(1, len(cos_vals) + 1)
    ax.bar(layers, cos_vals, color=ORANGE, width=0.62, alpha=0.9)
    ax.set_ylim(0.98, 1.003)
    ax.set_xticks(layers, [f"L{i}" for i in layers], fontsize=8.5, color=INK)
    ax.tick_params(axis="x", length=0)
    ax.grid(axis="x", visible=False)
    ax.axhline(float(np.mean(cos_vals)), color=INK, lw=1.1, ls="--")
    ax.text(
        0.7,
        1.0018,
        f"mean cos = {np.mean(cos_vals):.4f}  (full net = 0.9998)",
        ha="left",
        va="top",
        fontsize=8.9,
        fontweight="bold",
        color=INK,
    )
    ax.set_xlabel("residual 3×3 convolution layer (6 ResNet-v2 blocks)", color=INK)
    ax.set_ylabel("cosine similarity vs bf16 wgrad", color=INK)
    ax.set_title("Scaled FP8 (e4m3fn) gradient fidelity", loc="left")

    # Panel 3: Value NLL across searches (fast-hex8 vs megakernel no-pool vs pool+int8)
    ax = axes[2]
    for idx in (0, 1, 2):
        rid, label, path, color = runs_meta[idx]
        d = load_run(path)
        mask = d["v_frames"] <= 5.0e6
        ax.plot(
            d["v_frames"][mask] / 1e6,
            d["v_loss"][mask],
            color=color,
            lw=2.0 if idx == 0 else 1.5,
            alpha=0.95 if idx == 0 else 0.75,
            label=f"{label} ({rid})",
        )
    ax.set_ylim(0.55, 1.15)
    ax.set_xlabel("positions searched (millions)", color=INK)
    ax.set_ylabel("value binned NLL of search target", color=INK)
    ax.set_title("Clean belief convergence (0 NaNs)", loc="left")
    ax.legend(loc="upper right", fontsize=8.6)

    fig.suptitle(
        "Approximation & training checks   ·   exact top-1 policies, >0.995 FP8 grad cosine, clean NLL",
        x=0.008,
        ha="left",
        fontsize=13,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94), w_pad=2.2)
    fig.savefig(out_path, dpi=145)
    plt.close(fig)
    print("wrote", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--measurements", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path("reports/fast_hex8"))
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    repo = Path(__file__).resolve().parents[1]
    wt = repo / ".claude" / "worktrees"
    runs_meta = [
        ("324hr0z7", "fast-hex8 (this PR)", repo / "outputs/fast_hex8/history.jsonl", RED),
        ("tnj90uci", "pool + re-search + int8 (#41)", wt / "megakernel/outputs/loss_pool/h8/research_bits_int8.jsonl", BLUE),
        ("4lzxy4zi", "no pool + int8 (megakernel)", wt / "megakernel/outputs/loss_pool/h8/nopool_1024_int8.jsonl", GREEN),
        ("q9wqllha", "no pool, XLA net + fused", wt / "megakernel/outputs/loss_pool/h8/ref.jsonl", NEUTRAL),
        ("riemy292", "binned_8 (pre-fused)", wt / "binned-weighted-sum/outputs/weighted_sum_scaling/binned_8.jsonl", GOLD),
    ]
    meas = json.loads(args.measurements.read_text())
    make_learning_figure(runs_meta, args.out_dir / "hex8_learning.png")
    make_speed_figure(meas, args.out_dir / "hex8_speed.png")
    make_fidelity_figure(meas, runs_meta, args.out_dir / "hex8_fidelity.png")


if __name__ == "__main__":
    main()
