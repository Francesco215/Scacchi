#!/usr/bin/env python3
"""Upload the finalized local E8+ training runs and evaluation evidence to W&B.

This is a retrospective metrics import.  It uploads resolved configurations,
JSONL metrics, and selected evaluation JSON artifacts.  It never uploads
checkpoint weights or a source-code snapshot.
"""

from __future__ import annotations

import hashlib
import json
import math
import argparse
from pathlib import Path
from typing import Any

import numpy as np
import wandb


ROOT = Path(__file__).resolve().parent.parent
PROJECT = "scacchi-az"

E8_JSONL = ROOT / "experiments/e8/hex6_root_q21_target_s0/metrics.jsonl"
E9_JSONL = ROOT / "experiments/e9/hex6_q21_target_action_s0/metrics.jsonl"
E10_JSONL = (
    ROOT / "experiments/e10/hex6_q21_posterior_sample_s0/metrics.jsonl"
)
E11_JSONL = (
    ROOT / "experiments/e11/hex6_q21_posterior_plurality32_s0/metrics.jsonl"
)
E8_Q21 = (
    ROOT
    / "experiments/e8/hex6_root_q21_target_s0/kappa_sweep/"
    "confirm8192_q21_kappa_3.json"
)
E8_M32 = (
    ROOT
    / "experiments/e8/hex6_root_q21_target_s0/kappa_sweep/"
    "confirm8192_m32_kappa_3.json"
)
E8_M32_VS_Q21 = (
    ROOT
    / "experiments/e8/hex6_root_q21_target_s0/kappa_sweep/"
    "paired_confirm8192_m32_vs_q21_kappa_3.json"
)
E9_Q21 = (
    ROOT
    / "experiments/e9/hex6_q21_target_action_s0/"
    "balanced_q21_step75_8192_audit_replay.json"
)
E8_VS_E9 = (
    ROOT
    / "experiments/e9/hex6_q21_target_action_s0/"
    "paired_e8_vs_e9_true_q21_step75_8192.json"
)
E10_EVALS = tuple(
    ROOT
    / "experiments/e10/hex6_q21_posterior_sample_s0"
    / f"balanced_q21_step{step}_4096.json"
    for step in (75, 100)
)
E11_EVALS = tuple(
    ROOT
    / "experiments/e11/hex6_q21_posterior_plurality32_s0"
    / f"balanced_q21_step{step}_8192.json"
    for step in (50, 75, 100, 125, 150, 175, 200)
)
KAPPA_PROBE = ROOT / "experiments/kappa_fixed_probe/e8_step75.json"
LEAGUE_SCREEN_MANIFEST = (
    ROOT / "experiments/league/hex6_early_checkpoint_screen_manifest.json"
)
LEAGUE_CONFIRM_MANIFEST = (
    ROOT
    / "experiments/league/hex6_early_checkpoint_confirmation_manifest.json"
)
LEAGUE_SCREEN_RUN = (
    ROOT / "experiments/league/hex6_early_checkpoint_screen_run.json"
)
LEAGUE_CONFIRM_RUN = (
    ROOT / "experiments/league/hex6_early_checkpoint_confirmation_run.json"
)
LEAGUE_ANALYSIS = (
    ROOT / "experiments/league/hex6_early_checkpoint_combined_analysis.json"
)
LEDGER = ROOT / "experiments/hex6_speedrun.md"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _wandb_value(value: Any) -> Any:
    if (
        isinstance(value, dict)
        and value.get("_type") == "histogram"
        and "counts" in value
        and "bin_edges" in value
    ):
        return wandb.Histogram(
            np_histogram=(
                np.asarray(value["counts"]),
                np.asarray(value["bin_edges"]),
            )
        )
    return value


def _seat_summary(payload: dict[str, Any]) -> dict[str, float | int]:
    seats = {
        item["candidate_seat"]: item
        for item in payload["marginals"]["candidate_seat"]
    }
    overall = payload["overall"]
    first = float(seats["first"]["win_rate"])
    second = float(seats["second"]["win_rate"])
    return {
        "games": int(overall["games"]),
        "win_rate": float(overall["win_rate"]),
        "first_seat_win_rate": first,
        "second_seat_win_rate": second,
        "seat_optimal_error": ((1.0 - first) + second) / 2.0,
    }


def _read_training(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not records or records[0].get("record_type") != "run_start":
        raise RuntimeError(f"{path} must begin with a run_start record")
    metrics = [
        record for record in records if record.get("record_type") == "metrics"
    ]
    steps = [int(record["step"]) for record in metrics]
    expected_count = int(records[0]["config"]["run"]["max_num_iters"])
    if steps != list(range(expected_count)):
        raise RuntimeError(
            f"{path} expected metric steps 0..{expected_count - 1}"
        )
    return records[0], metrics


def _log_training_run(
    *,
    experiment_id: str,
    run_id: str,
    name: str,
    jsonl: Path,
    evaluation_summary: dict[str, Any],
    evidence_paths: tuple[Path, ...],
    tags: list[str],
) -> str:
    start, metrics = _read_training(jsonl)
    final_step = int(metrics[-1]["step"])
    source_hash = _sha256(jsonl)
    config = dict(start["config"])
    config["retrospective_import"] = {
        "enabled": True,
        "source": str(jsonl.relative_to(ROOT)),
        "source_sha256": source_hash,
        "original_run_name": start["run_name"],
        "reason": "Finalized local run imported with explicit user request.",
    }
    run = wandb.init(
        project=PROJECT,
        id=run_id,
        resume="allow",
        name=name,
        group="hex6-speedrun",
        job_type="training-retrospective-import",
        tags=["hex6", experiment_id, "retrospective-import", "seed-0", *tags],
        notes=(
            f"Exact retrospective import of {experiment_id} through step "
            f"{final_step}. "
            "Uploads metrics/config/evaluation JSON only; no checkpoints or "
            "source snapshot."
        ),
        config=config,
        save_code=False,
        reinit="finish_previous",
    )
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    for record in metrics:
        run.log(
            {
                key: _wandb_value(value)
                for key, value in record["metrics"].items()
            },
            step=int(record["step"]),
        )
    run.summary.update(
        {
            "experiment/id": experiment_id,
            "experiment/retrospective_import": True,
            "experiment/source_jsonl_sha256": source_hash,
            "experiment/imported_metric_steps": len(metrics),
            "experiment/final_checkpoint_step": final_step,
            **evaluation_summary,
        }
    )
    artifact = wandb.Artifact(
        name=f"{experiment_id.lower()}-hex6-finalized-evidence",
        type="training-evidence",
        metadata={
            "retrospective_import": True,
            "source_jsonl_sha256": source_hash,
            "includes_checkpoints": False,
            "includes_source_snapshot": False,
        },
    )
    artifact.add_file(str(jsonl), name="metrics.jsonl")
    for path in evidence_paths:
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    url = str(run.url)
    wandb.finish()
    return url


def _log_evaluation_study() -> str:
    kappa_files = sorted(
        (
            E8_Q21.parent.glob("confirm8192_q21_kappa_*.json")
        ),
        key=lambda path: float(_load(path)["candidate_kappa"]["effective"]),
    )
    kappa_rows: list[list[Any]] = []
    probe = _load(KAPPA_PROBE)["summary"]["overall"]["kappas"]
    for path in kappa_files:
        payload = _load(path)
        kappa = float(payload["candidate_kappa"]["effective"])
        summary = _seat_summary(payload)
        elo_proxy = 400.0 * math.log10(
            summary["win_rate"] / (1.0 - summary["win_rate"])
        )
        fixed = probe[f"{kappa:g}"]
        kappa_rows.append(
            [
                kappa,
                summary["games"],
                summary["win_rate"],
                summary["first_seat_win_rate"],
                summary["second_seat_win_rate"],
                summary["seat_optimal_error"],
                elo_proxy,
                fixed["root_policy_vs_reference_kappa"]["mean_l1"],
                fixed["root_policy_vs_reference_kappa"][
                    "top_action_flip_fraction"
                ],
                fixed["search_structure"][
                    "implied_local_e_fold_length"
                ]["mean"],
            ]
        )

    league = _load(LEAGUE_ANALYSIS)
    robust = league["robust_first_seat_conversion"]["players"]
    ratings = {
        item["id"]: item
        for item in league["seat_adjusted_regularized_bradley_terry"][
            "ratings"
        ]
    }
    league_rows = [
        [
            competitor,
            robust[competitor]["minimum_first_seat_win_rate"],
            robust[competitor]["minimum_bonferroni_simultaneous_lower_95"],
            robust[competitor]["worst_opponent_by_point_estimate"],
            ratings[competitor]["rank"],
            ratings[competitor]["elo_like"],
        ]
        for competitor in league["competitor_ids"]
    ]

    run = wandb.init(
        project=PROJECT,
        id="hex6kappaleague1",
        resume="allow",
        name="hex6-kappa-and-early-league-final",
        group="hex6-speedrun",
        job_type="evaluation-study",
        tags=["hex6", "kappa", "league", "q21", "paired-evaluation"],
        notes=(
            "Finalized test-time kappa response, fixed-root mechanism probe, "
            "and seat-conditioned early-checkpoint league. No checkpoints."
        ),
        config={
            "kappa_checkpoint": "E8 step 75",
            "kappa_games_per_confirmation": 8192,
            "kappa_reference": 3.0,
            "root_action_estimator": "prefix_cdf",
            "prefix_cdf_half_width": 10,
            "league_screen_games_per_pair": 1024,
            "league_confirmation_games_per_pair": 4096,
            "external_logging_scope": "metrics and finalized JSON evidence",
        },
        save_code=False,
        reinit="finish_previous",
    )
    if run is None:
        raise RuntimeError("wandb.init returned no run")
    run.log(
        {
            "kappa/confirmation_table": wandb.Table(
                columns=[
                    "kappa",
                    "games",
                    "pooled_win_rate",
                    "first_seat_win_rate",
                    "second_seat_win_rate",
                    "seat_optimal_error",
                    "pooled_elo_proxy",
                    "fixed_root_mean_l1_vs_kappa3",
                    "fixed_root_top_flip_fraction_vs_kappa3",
                    "fixed_root_mean_local_e_fold_length",
                ],
                data=kappa_rows,
            ),
            "league/robustness_table": wandb.Table(
                columns=[
                    "competitor",
                    "worst_first_seat_win_rate",
                    "simultaneous_lower_95",
                    "worst_opponent",
                    "descriptive_bt_rank",
                    "descriptive_bt_elo",
                ],
                data=league_rows,
            ),
        }
    )
    for index, row in enumerate(kappa_rows):
        run.log(
            {
                "kappa/value": row[0],
                "kappa/pooled_win_rate": row[2],
                "kappa/first_seat_win_rate": row[3],
                "kappa/second_seat_win_rate": row[4],
                "kappa/seat_optimal_error": row[5],
                "kappa/pooled_elo_proxy": row[6],
                "kappa/fixed_root_mean_l1_vs_3": row[7],
                "kappa/fixed_root_top_flip_fraction_vs_3": row[8],
                "kappa/fixed_root_mean_local_e_fold_length": row[9],
            },
            step=index,
        )
    cycles = league["nontransitivity"]["three_cycles"]
    run.summary.update(
        {
            "study/kappa_plateau_conclusion": (
                "kappa 3-8 differs by about one pooled Elo; retain kappa=3"
            ),
            "study/league_competitors": len(league["competitor_ids"]),
            "study/league_observed_pairs": league["coverage"][
                "observed_unordered_pairs"
            ],
            "study/league_supported_three_cycles": sum(
                bool(item["all_edges_bootstrap_lower_above_threshold"])
                for item in cycles
            ),
            "study/league_best_checkpoint": ratings[
                min(ratings, key=lambda key: ratings[key]["rank"])
            ]["id"],
            "study/kappa_probe_sha256": _sha256(KAPPA_PROBE),
            "study/league_analysis_sha256": _sha256(LEAGUE_ANALYSIS),
        }
    )
    artifact = wandb.Artifact(
        name="hex6-kappa-league-finalized-evidence",
        type="evaluation-evidence",
        metadata={
            "includes_checkpoints": False,
            "includes_source_snapshot": False,
            "kappa_probe_sha256": _sha256(KAPPA_PROBE),
            "league_analysis_sha256": _sha256(LEAGUE_ANALYSIS),
        },
    )
    for path in (
        KAPPA_PROBE,
        LEAGUE_SCREEN_MANIFEST,
        LEAGUE_CONFIRM_MANIFEST,
        LEAGUE_SCREEN_RUN,
        LEAGUE_CONFIRM_RUN,
        LEAGUE_ANALYSIS,
        LEDGER,
    ):
        artifact.add_file(str(path), name=path.name)
    run.log_artifact(artifact)
    url = str(run.url)
    wandb.finish()
    return url


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-only",
        action="store_true",
        help="Upload E8+ training runs without the separate evaluation study.",
    )
    args = parser.parse_args()

    e8_q21 = _seat_summary(_load(E8_Q21))
    e8_m32 = _seat_summary(_load(E8_M32))
    readout_pair = _load(E8_M32_VS_Q21)["paired"]["overall"]
    e8_url = _log_training_run(
        experiment_id="E8",
        run_id="e8q21targetretro1",
        name="e8-hex6-root-q21-target-s0-retro",
        jsonl=E8_JSONL,
        evaluation_summary={
            "balanced/step75/q21_win_rate": e8_q21["win_rate"],
            "balanced/step75/q21_first_seat_win_rate": e8_q21[
                "first_seat_win_rate"
            ],
            "balanced/step75/m32_win_rate": e8_m32["win_rate"],
            "balanced/step75/q21_minus_m32_paired": readout_pair[
                "delta_win_rate"
            ],
            "balanced/step75/q21_minus_m32_ci95_low": readout_pair[
                "delta_win_rate_ci95"
            ]["interval"][0],
            "balanced/step75/q21_minus_m32_ci95_high": readout_pair[
                "delta_win_rate_ci95"
            ]["interval"][1],
        },
        evidence_paths=(E8_Q21, E8_M32, E8_M32_VS_Q21),
        tags=["E8", "q21-target", "m32-selfplay"],
    )

    e9_q21 = _seat_summary(_load(E9_Q21))
    trajectory_pair = _load(E8_VS_E9)["paired"]["overall"]
    e9_url = _log_training_run(
        experiment_id="E9",
        run_id="e9q21actionretro1",
        name="e9-hex6-q21-target-action-s0-retro",
        jsonl=E9_JSONL,
        evaluation_summary={
            "balanced/step75/q21_win_rate": e9_q21["win_rate"],
            "balanced/step75/q21_first_seat_win_rate": e9_q21[
                "first_seat_win_rate"
            ],
            "balanced/step75/e9_minus_e8_paired": trajectory_pair[
                "delta_win_rate"
            ],
            "balanced/step75/e9_minus_e8_ci95_low": trajectory_pair[
                "delta_win_rate_ci95"
            ]["interval"][0],
            "balanced/step75/e9_minus_e8_ci95_high": trajectory_pair[
                "delta_win_rate_ci95"
            ]["interval"][1],
        },
        evidence_paths=(E9_Q21, E8_VS_E9),
        tags=["E9", "q21-target", "q21-selfplay"],
    )

    e10_summaries = {
        step: _seat_summary(_load(path))
        for step, path in zip((75, 100), E10_EVALS)
    }
    e10_url = _log_training_run(
        experiment_id="E10",
        run_id="e10q21sampleretro1",
        name="e10-hex6-q21-posterior-sample-s0-retro",
        jsonl=E10_JSONL,
        evaluation_summary={
            f"balanced/step{step}/q21_win_rate": summary["win_rate"]
            for step, summary in e10_summaries.items()
        },
        evidence_paths=E10_EVALS,
        tags=["E10", "q21-target", "q21-posterior-sample"],
    )

    e11_summaries = {
        step: _seat_summary(_load(path))
        for step, path in zip((50, 75, 100, 125, 150, 175, 200), E11_EVALS)
    }
    e11_url = _log_training_run(
        experiment_id="E11",
        run_id="e11q21plurality32retro1",
        name="e11-hex6-q21-posterior-plurality32-s0-retro",
        jsonl=E11_JSONL,
        evaluation_summary={
            f"balanced/step{step}/q21_win_rate": summary["win_rate"]
            for step, summary in e11_summaries.items()
        },
        evidence_paths=E11_EVALS,
        tags=["E11", "q21-target", "q21-posterior-plurality32"],
    )

    study_url = None if args.training_only else _log_evaluation_study()
    print(
        json.dumps(
            {
                "e8_run_url": e8_url,
                "e9_run_url": e9_url,
                "e10_run_url": e10_url,
                "e11_run_url": e11_url,
                "evaluation_study_url": study_url,
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
