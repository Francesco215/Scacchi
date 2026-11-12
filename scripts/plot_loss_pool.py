"""Curves of loss-pool runs against a run without a pool, on shared axes.

    .venv/bin/python scripts/plot_loss_pool.py \
        baseline=wandb:pal/scacchi-az/qn746218 \
        tau1=outputs/loss_pool/hex6_pool_tau1.jsonl \
        --out outputs/loss_pool/hex6

Writes ``<out>_frames.png``, ``<out>_searches.png`` (self-play frames plus
re-searched pool rows), ``<out>_updates.png`` and ``<out>_minutes.png``
(training time without evaluation, the first iteration's compilation replaced
by a typical iteration). A pool run's train/*
losses average a batch that is mostly replayed rows, so its fresh/* losses
(this iteration's self-play, before any update on it) are drawn in their place:
that is what train/* measures in a run without a pool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

CURVES = Path(__file__).resolve().parents[1] / ".claude/skills/research-loop/scripts/curves.py"
POOL_PANELS = [
    r"^fresh/bits$",
    r"^replay/bits$",
    r"^pool/bits$",
    r"^replay/bits_change_since_last_visit$",
    r"^replay/bits_rising_fraction$",
    r"^replay/age_updates$",
    r"^replay/visits$",
    r"^pool/evicted_visits$",
    r"^pool/evicted_bits$",
    r"^replay/bits_by_age/0_200$",
    r"^replay/bits_by_age/200_800$",
    r"^replay/bits_by_age/800_3200$",
    r"^replay/bits_by_age/3200_up$",
    r"^pool/fresh_kept_fraction$",
    r"^research/agreement$",
    r"^research/wrong_way$",
    r"^research/bits_before$",
    r"^research/bits_after$",
]


def load(source: str) -> list[dict]:
    if source.startswith("wandb:"):
        import wandb

        return list(wandb.Api().run(source.removeprefix("wandb:")).scan_history())
    return [json.loads(line) for line in Path(source).read_text().splitlines() if line.strip()]


def comparable(rows: list[dict], training_batch_size: int) -> list[dict]:
    out = []
    seconds = [row.get("train/iter_seconds") for row in rows]
    # A run with a fill phase compiles twice: once per phase. The first
    # iteration of each phase counts as a typical iteration of that phase.
    phases = [row.get("pool/filling", 0.0) for row in rows]
    typical = {}
    for phase in set(phases):
        inside = sorted(s for i, s in enumerate(seconds) if s is not None and phases[i] == phase and i > 0 and phases[i - 1] == phase)
        typical[phase] = inside[len(inside) // 2] if inside else None
    elapsed = 0.0
    for index, row in enumerate(rows):
        row = dict(row)
        if seconds[index] is not None:
            first_of_phase = index == 0 or phases[index] != phases[index - 1]
            usual = typical[phases[index]]
            elapsed += usual if first_of_phase and usual is not None else seconds[index]
            row["train/minutes"] = elapsed / 60
        for key in [k for k in row if k.startswith("fresh/")]:
            row["train/" + key.removeprefix("fresh/")] = row[key]
        if "train/frames" in row:  # without re-search every search is a self-play frame
            row.setdefault("train/searches", row["train/frames"])
        if "train/updates" not in row and "train/frames" in row:
            row["train/updates"] = row["train/frames"] / training_batch_size
        # The loop evaluates before it trains: row i's eval is of the network
        # that row i-1's frames produced.
        evals = {key: row.pop(key) for key in [k for k in row if k.startswith("eval/")]}
        if out:
            out[-1].update(evals)
        out.append(row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--training-batch-size", type=int, default=1024)
    args = parser.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    specs = []
    for item in args.runs:
        label, source = item.split("=", 1)
        path = out.parent / f".{out.name}_{label}.jsonl"
        path.write_text("\n".join(json.dumps(row) for row in comparable(load(source), args.training_batch_size)))
        specs.append(f"{label}={path}")
    extra = [arg for pattern in POOL_PANELS for arg in ("--metric", pattern)]
    for axis in ("frames", "searches", "updates", "minutes"):
        subprocess.run(
            [sys.executable, str(CURVES), *specs, "--out", f"{out}_{axis}.png", "--x", f"train/{axis}", "--logx", *extra],
            check=True,
        )


if __name__ == "__main__":
    main()
