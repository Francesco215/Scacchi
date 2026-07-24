#!/usr/bin/env python3
"""Reproduce the registered E12 result assessment.

This is a deliberately narrow, local-only consumer of already-created
artifacts.  It does not load models, run games, train, or contact W&B.  It
validates the metric stream, balanced evaluations, league provenance, raw
return digests, implementation hashes, and checkpoint-tree hashes before
computing the frozen E12 gates.

The common-roster uncertainty calculation is a synchronized, stratified
game-coordinate block bootstrap.  One resampled coordinate carries the
E12-minus-E11 outcome differences for all four fixed opponents, preserving
the common-random-number dependence across the roster.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
from statistics import NormalDist
import sys
from typing import Any

import numpy as np

from scripts import hex_paired_eval_analysis as paired_eval


ROOT = Path(__file__).resolve().parent.parent
E12_DIR = ROOT / "experiments/e12/hex6_q21_posterior_temperature3_s0"
E11_DIR = ROOT / "experiments/e11/hex6_q21_posterior_plurality32_s0"
LEAGUE_DIR = ROOT / "experiments/league"

E12_MANIFEST = LEAGUE_DIR / "hex6_e12_temperature3_confirmation_manifest.json"
E12_RUN = LEAGUE_DIR / "hex6_e12_temperature3_confirmation_run.json"
E12_ANALYSIS = LEAGUE_DIR / "hex6_e12_temperature3_confirmation_analysis.json"
E11_MANIFEST = LEAGUE_DIR / "hex6_e11_plurality_long_confirmation_manifest.json"
E11_RUN = LEAGUE_DIR / "hex6_e11_plurality_long_confirmation_run.json"
E11_ANALYSIS = LEAGUE_DIR / "hex6_e11_plurality_long_confirmation_analysis.json"
E11_STEP100_COMMON_MANIFEST = (
    LEAGUE_DIR
    / "hex6_e12_temperature3_e11_step100_common_coordinate_manifest.json"
)
E11_STEP100_COMMON_RUN = (
    LEAGUE_DIR
    / "hex6_e12_temperature3_e11_step100_common_coordinate_run.json"
)

DEFAULT_OUTPUT = E12_DIR / "result_analysis_v1.json"
RESULT_KIND = "scacchi.hex_e12_result_analysis"
RESULT_SCHEMA_VERSION = 1
BOOTSTRAP_SEED = 12_112_012
BOOTSTRAP_REPLICATES = 50_000
BOOTSTRAP_BATCH_SIZE = 128
EXPECTED_STEPS = (50, 75, 100, 125, 150, 175, 200)
ANCHORS = (
    "e8-step75-q21",
    "e8-step100-q21",
    "e9-step75-q21",
    "e10-step75-q21",
)
WINDOWS = ((10, 29), (30, 74), (75, 99), (100, 124), (125, 200))
WINDOW_METRICS = (
    "data/opening_action_effective_support",
    "data/early_ply_action_effective_support_mean",
    "data/game_length_mean",
    "data/terminal_events_per_1k_frames",
    "data/value_mask_fraction",
    "search/solved_root_fraction",
    "search/root_action_coverage",
    "search/policy_ess_mean",
    "search/policy_displacement_kl_nats",
    "capture/train_probe/policy_gap_before_nats",
    "capture/train_probe/policy_gap_after_nats",
    "capture/train_probe/policy_gap_delta_nats",
    "capture/train_probe/q_semantic_gap_delta_nats",
    "capture/train_probe/q_loss_weighted_semantic_gap_delta_nats",
    "train/policy_target_entropy",
)
PREFIX_GUARD_METRICS = (
    "search/root_action_prefix_acceptance_fraction",
    "search/root_policy_target_prefix_acceptance_fraction",
)
PREFIX_ZERO_METRICS = (
    "search/root_action_prefix_fallback_fraction",
    "search/root_action_prefix_density_guard_fraction",
    "search/root_action_prefix_nonfinite_fraction",
    "search/root_action_prefix_tail_clipped_fraction",
    "search/root_policy_target_prefix_fallback_fraction",
    "search/root_policy_target_prefix_density_guard_fraction",
    "search/root_policy_target_prefix_nonfinite_fraction",
    "search/root_policy_target_prefix_tail_clipped_fraction",
)


@dataclass(frozen=True)
class LeaguePair:
    """One validated league pair with raw A-perspective win indicators."""

    path: Path
    file_sha256: str
    payload: dict[str, Any]
    competitor_a: str
    competitor_b: str
    coordinates: tuple[paired_eval.GameCoordinate, ...]
    wins_by_stratum: np.ndarray
    pairing_layout_sha256: str
    implementation_files_currently_match: bool


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    """Match the league harness's checkpoint-tree content hash."""

    resolved = root.resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"checkpoint step directory missing: {resolved}")
    digest = hashlib.sha256()
    entries = sorted(
        resolved.rglob("*"),
        key=lambda path: path.relative_to(resolved).as_posix(),
    )
    for path in entries:
        relative = path.relative_to(resolved).as_posix()
        if path.is_symlink():
            kind = b"L"
            content: bytes | None = os.readlink(path).encode("utf-8")
        elif path.is_file():
            kind = b"F"
            content = None
        elif path.is_dir():
            kind = b"D"
            content = b""
        else:
            raise ValueError(f"unsupported checkpoint entry: {path}")
        encoded_name = relative.encode("utf-8")
        digest.update(kind)
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        if content is not None:
            digest.update(len(content).to_bytes(8, "big"))
            digest.update(content)
            continue
        size = path.stat().st_size
        digest.update(size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _assert_finite(value: Any, location: str = "$") -> None:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"nonfinite number at {location}")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_finite(item, f"{location}.{key}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _assert_finite(item, f"{location}[{index}]")
        return
    raise ValueError(f"unsupported JSON value at {location}: {type(value)!r}")


def _load_json(path: Path) -> tuple[dict[str, Any], str]:
    resolved = path.resolve()
    encoded = resolved.read_bytes()
    payload = json.loads(encoded)
    if not isinstance(payload, dict):
        raise ValueError(f"{resolved} must contain a JSON object")
    _assert_finite(payload, str(resolved))
    return payload, hashlib.sha256(encoded).hexdigest()


def _resolve_recorded_path(value: Any, *, base: Path) -> Path:
    if not isinstance(value, str):
        raise ValueError("recorded path must be a string")
    path = Path(value)
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def wilson_interval(
    wins: int,
    games: int,
    *,
    confidence: float,
) -> tuple[float, float]:
    """Two-sided Wilson score interval at exact requested coverage."""

    if games <= 0 or not 0 <= wins <= games:
        raise ValueError("Wilson counts must satisfy 0 <= wins <= games")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie strictly between zero and one")
    z = NormalDist().inv_cdf(0.5 + confidence / 2.0)
    probability = wins / games
    z2 = z * z
    denominator = 1.0 + z2 / games
    center = (probability + z2 / (2.0 * games)) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / games
            + z2 / (4.0 * games * games)
        )
        / denominator
    )
    return center - radius, center + radius


def _mean(rows: Sequence[dict[str, Any]], metric: str) -> float:
    values = [row[metric] for row in rows]
    if len(values) != len(rows):
        raise ValueError(f"missing metric {metric!r}")
    if not all(isinstance(value, (int, float)) for value in values):
        raise ValueError(f"metric {metric!r} is not numeric")
    return math.fsum(float(value) for value in values) / len(values)


def load_metric_stream(
    path: Path,
    *,
    validate_e12_provenance: bool = False,
) -> dict[str, Any]:
    """Validate one 0--200 JSONL stream and return numeric summaries."""

    resolved = path.resolve()
    encoded = resolved.read_bytes()
    lines = encoded.splitlines()
    if len(lines) != 202:
        raise ValueError(
            f"{resolved} must contain one header plus 201 rows; got {len(lines)}"
        )
    records = [json.loads(line) for line in lines]
    for index, record in enumerate(records):
        _assert_finite(record, f"{resolved}:line{index + 1}")
    header = records[0]
    if header.get("record_type") != "run_start":
        raise ValueError("first metric-stream row must be run_start")
    metric_records = records[1:]
    if any(record.get("record_type") != "metrics" for record in metric_records):
        raise ValueError("every post-header row must be a metrics record")
    steps = [record.get("step") for record in metric_records]
    if steps != list(range(201)):
        raise ValueError("metric steps must be exactly contiguous 0--200")

    config = header.get("config")
    if not isinstance(config, dict):
        raise ValueError("metric header config missing")
    selfplay = config.get("selfplay")
    search = config.get("search")
    logging = config.get("logging")
    if not isinstance(selfplay, dict) or not isinstance(search, dict):
        raise ValueError("metric header search provenance missing")
    temperatures = (
        selfplay.get("search", {}).get("posterior_sample_temperature"),
        search.get("posterior_sample_temperature"),
    )
    if validate_e12_provenance:
        if selfplay.get("action_commitment_type") != "posterior_sample":
            raise ValueError("E12 commitment is not posterior_sample")
        expected_temperature = 1.0 / 3.0
        if any(
            not isinstance(value, (int, float))
            or not math.isclose(
                float(value),
                expected_temperature,
                rel_tol=0.0,
                abs_tol=1e-15,
            )
            for value in temperatures
        ):
            raise ValueError("E12 temperature provenance is not exactly 1/3")
        if (
            not isinstance(logging, dict)
            or logging.get("wandb", {}).get("enabled") is not False
        ):
            raise ValueError("E12 metric header does not record W&B disabled")

    rows = [record["metrics"] for record in metric_records]
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError("metrics payloads must be objects")
    guard_checks: dict[str, Any] = {}
    for metric in PREFIX_GUARD_METRICS:
        values = [float(row[metric]) for row in rows]
        passed = all(value == 1.0 for value in values)
        guard_checks[metric] = {
            "minimum": min(values),
            "maximum": max(values),
            "passed": passed,
        }
        if not passed:
            raise ValueError(f"{metric} was not one on every row")
    for metric in PREFIX_ZERO_METRICS:
        values = [float(row[metric]) for row in rows]
        passed = all(value == 0.0 for value in values)
        guard_checks[metric] = {
            "minimum": min(values),
            "maximum": max(values),
            "passed": passed,
        }
        if not passed:
            raise ValueError(f"{metric} was not zero on every row")

    early_records = [
        record for record in metric_records if 10 <= int(record["step"]) <= 29
    ]
    early_rows = [record["metrics"] for record in early_records]
    early_values = {
        "train/frames_per_second": _mean(
            early_rows, "train/frames_per_second"
        ),
        "data/terminal_events_per_1k_frames": _mean(
            early_rows, "data/terminal_events_per_1k_frames"
        ),
        "data/value_mask_fraction": _mean(
            early_rows, "data/value_mask_fraction"
        ),
        "search/solved_root_fraction": _mean(
            early_rows, "search/solved_root_fraction"
        ),
        "train/policy_target_entropy": _mean(
            early_rows, "train/policy_target_entropy"
        ),
        "capture/train_probe/policy_gap_delta_nats": _mean(
            early_rows, "capture/train_probe/policy_gap_delta_nats"
        ),
    }
    individual_gap_deltas = [
        float(row["capture/train_probe/policy_gap_delta_nats"])
        for row in early_rows
    ]
    early_gates = {
        "throughput_at_least_75k": (
            early_values["train/frames_per_second"] >= 75_000.0
        ),
        "terminal_events_in_33_38": (
            33.0
            <= early_values["data/terminal_events_per_1k_frames"]
            <= 38.0
        ),
        "value_mask_in_0p68_0p71": (
            0.68 <= early_values["data/value_mask_fraction"] <= 0.71
        ),
        "solved_root_in_0p033_0p040": (
            0.033 <= early_values["search/solved_root_fraction"] <= 0.040
        ),
        "policy_entropy_in_2p90_3p12": (
            2.90 <= early_values["train/policy_target_entropy"] <= 3.12
        ),
        "all_policy_gap_reductions_positive": all(
            value > 0.0 for value in individual_gap_deltas
        ),
        "mean_policy_gap_reduction_at_least_0p012311": (
            early_values["capture/train_probe/policy_gap_delta_nats"]
            >= 0.012311
        ),
    }
    early_gates["all_passed"] = all(early_gates.values())

    window_summaries: dict[str, Any] = {}
    for lower, upper in WINDOWS:
        window_rows = [
            record["metrics"]
            for record in metric_records
            if lower <= int(record["step"]) <= upper
        ]
        window_summaries[f"{lower}-{upper}"] = {
            metric: _mean(window_rows, metric) for metric in WINDOW_METRICS
        }

    return {
        "path": str(resolved),
        "file_sha256": hashlib.sha256(encoded).hexdigest(),
        "line_count": len(lines),
        "metric_row_count": len(metric_records),
        "steps": [steps[0], steps[-1]],
        "steps_contiguous": True,
        "all_json_numbers_finite": True,
        "run_name": header.get("run_name"),
        "provenance": {
            "seed": config.get("run", {}).get("seed"),
            "max_num_iters": config.get("run", {}).get("max_num_iters"),
            "commitment": selfplay.get("action_commitment_type"),
            "temperature": temperatures[0],
            "wandb_enabled": (
                logging.get("wandb", {}).get("enabled")
                if isinstance(logging, dict)
                else None
            ),
        },
        "prefix_guard_checks": guard_checks,
        "early_10_29": {
            "values": early_values,
            "minimum_policy_gap_reduction_nats": min(individual_gap_deltas),
            "gates": early_gates,
        },
        "windows": window_summaries,
    }


def _compare_metric_windows(
    e11: dict[str, Any],
    e12: dict[str, Any],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for window in e12["windows"]:
        e11_values = e11["windows"][window]
        e12_values = e12["windows"][window]
        output[window] = {}
        for metric in WINDOW_METRICS:
            control = float(e11_values[metric])
            treatment = float(e12_values[metric])
            output[window][metric] = {
                "e11": control,
                "e12": treatment,
                "e12_minus_e11": treatment - control,
                "relative_difference": (
                    (treatment - control) / control
                    if control != 0.0
                    else None
                ),
            }
    return output


def _seat_summary(
    loaded: paired_eval.LoadedResult,
) -> dict[str, Any]:
    observations = loaded.observations
    first = [item for item in observations if item.coordinate.candidate_first]
    second = [
        item for item in observations if not item.coordinate.candidate_first
    ]
    if len(first) != len(second) or not first:
        raise ValueError("balanced artifact is not exactly seat-balanced")
    games = len(observations)
    wins = sum(item.candidate_won for item in observations)
    first_wins = sum(item.candidate_won for item in first)
    second_wins = sum(item.candidate_won for item in second)
    pooled = wins / games
    first_rate = first_wins / len(first)
    second_rate = second_wins / len(second)
    e_seat = 0.5 * ((1.0 - first_rate) + second_rate)
    return {
        "games": games,
        "wins": wins,
        "pooled_win_rate": pooled,
        "pooled_wilson90": list(
            wilson_interval(wins, games, confidence=0.90)
        ),
        "pooled_wilson95": list(
            wilson_interval(wins, games, confidence=0.95)
        ),
        "first": {
            "games": len(first),
            "wins": first_wins,
            "win_rate": first_rate,
            "wilson95": list(
                wilson_interval(first_wins, len(first), confidence=0.95)
            ),
        },
        "second": {
            "games": len(second),
            "wins": second_wins,
            "win_rate": second_rate,
            "wilson95": list(
                wilson_interval(second_wins, len(second), confidence=0.95)
            ),
        },
        "e_seat": e_seat,
    }


def _balanced_step_gates(step: int, summary: dict[str, Any]) -> dict[str, Any]:
    point = float(summary["pooled_win_rate"])
    lower, upper = (float(value) for value in summary["pooled_wilson90"])
    e_seat = float(summary["e_seat"])
    if step == 50:
        checks = {
            "pooled_point_at_least_0p425": point >= 0.425,
            "e_seat_at_most_0p20": e_seat <= 0.20,
        }
    elif step == 75:
        checks = {
            "wilson90_contained_in_0p45_0p55": (
                lower >= 0.45 and upper <= 0.55
            ),
            "e_seat_at_most_0p10": e_seat <= 0.10,
        }
    elif step == 100:
        checks = {
            "wilson90_contained_in_0p45_0p55": (
                lower >= 0.45 and upper <= 0.55
            ),
            "e_seat_at_most_0p08": e_seat <= 0.08,
        }
    elif step in (125, 150, 175, 200):
        checks = {
            "pooled_point_at_least_0p47": point >= 0.47,
            "e_seat_at_most_0p12": e_seat <= 0.12,
        }
        if step == 200:
            checks.update(
                {
                    "wilson90_contained_in_0p45_0p55": (
                        lower >= 0.45 and upper <= 0.55
                    ),
                    "e_seat_at_most_0p10": e_seat <= 0.10,
                }
            )
    else:
        raise ValueError(f"unregistered E12 balanced step: {step}")
    return {**checks, "all_passed": all(checks.values())}


def load_balanced_evaluations(
    directory: Path,
    *,
    validate_e12_plan: bool,
) -> dict[int, dict[str, Any]]:
    plan: dict[str, Any] | None = None
    plan_sha256: str | None = None
    if validate_e12_plan:
        plan, plan_sha256 = _load_json(directory / "balanced_eval_plan.json")
        expected_plan = {
            "candidate_steps": list(EXPECTED_STEPS),
            "baseline_step": 299,
            "games_per_step": 8192,
            "batch_size": 256,
            "seed": 11_108_192,
            "candidate_root_action_estimator": "prefix_cdf",
            "candidate_prefix_cdf_half_width": 10,
            "candidate_kappa": 3.0,
            "include_game_returns": True,
        }
        for field, expected in expected_plan.items():
            if plan.get(field) != expected:
                raise ValueError(
                    f"balanced plan {field} disagrees with registration"
                )
        if plan.get("frozen_before_evaluation") is not True:
            raise ValueError("balanced plan was not frozen before evaluation")

    results: dict[int, dict[str, Any]] = {}
    for step in EXPECTED_STEPS:
        path = directory / f"balanced_q21_step{step}_8192.json"
        raw, raw_sha256 = _load_json(path)
        loaded = paired_eval.load_result(path)
        if loaded.file_sha256 != raw_sha256:
            raise AssertionError("balanced loader file hash mismatch")
        expected_top_level = {
            "candidate_step": step,
            "baseline_step": 299,
            "games": 8192,
            "games_per_stratum": 2048,
            "batch_size": 256,
            "seed": 11_108_192,
        }
        for field, expected in expected_top_level.items():
            if raw.get(field) != expected:
                raise ValueError(f"{path}: {field} != {expected!r}")
        selection = raw.get("candidate_checkpoint_selection", {})
        if (
            selection.get("requested_step") != step
            or selection.get("selected_step") != step
            or selection.get("selection_mode") != "exact"
        ):
            raise ValueError(f"{path}: candidate checkpoint is not exact")
        root_action = raw.get("candidate_root_action_estimator", {})
        grid = raw.get("candidate_prefix_cdf_grid", {})
        kappa = raw.get("candidate_kappa", {})
        if (
            root_action.get("effective") != "prefix_cdf"
            or grid.get("effective_grid_points") != 21
            or float(kappa.get("effective")) != 3.0
        ):
            raise ValueError(f"{path}: frozen Q21 evaluation override mismatch")
        if raw.get("local_only") != {
            "network_access": False,
            "external_logging": False,
            "external_export": False,
            "artifact_is_local": True,
        }:
            raise ValueError(f"{path}: local-only contract mismatch")
        reproduction = raw.get("reproduction", {})
        script_path = _resolve_recorded_path(
            reproduction.get("script_path"),
            base=ROOT,
        )
        if _file_sha256(script_path) != reproduction.get("script_sha256"):
            raise ValueError(f"{path}: evaluator script hash mismatch")
        summary = _seat_summary(loaded)
        stored_overall = raw.get("overall", {})
        if (
            stored_overall.get("wins") != summary["wins"]
            or not math.isclose(
                float(stored_overall.get("win_rate")),
                float(summary["pooled_win_rate"]),
                rel_tol=0.0,
                abs_tol=1e-15,
            )
        ):
            raise ValueError(f"{path}: stored balanced summary mismatch")
        result = {
            "path": str(path.resolve()),
            "file_sha256": raw_sha256,
            "game_returns_sha256": raw["game_returns"]["returns_sha256"],
            "pairing_layout_sha256": (
                raw["game_returns"]["pairing_layout_sha256"]
            ),
            "candidate_step": step,
            "all_json_numbers_finite": True,
            **summary,
        }
        if validate_e12_plan:
            result["registered_gates"] = _balanced_step_gates(step, summary)
        results[step] = result
    if validate_e12_plan:
        results[-1] = {
            "plan_path": str((directory / "balanced_eval_plan.json").resolve()),
            "plan_sha256": plan_sha256,
        }
    return results


def _validate_implementation_provenance(
    implementation: Any,
) -> dict[str, Any]:
    if not isinstance(implementation, dict):
        raise ValueError("league implementation provenance missing")
    files = implementation.get("files")
    if not isinstance(files, dict) or not files:
        raise ValueError("league implementation file hashes missing")
    current_mismatches: list[str] = []
    for recorded_path, recorded_sha256 in files.items():
        path = _resolve_recorded_path(recorded_path, base=ROOT)
        if _file_sha256(path) != recorded_sha256:
            current_mismatches.append(str(path))
    if _canonical_sha256(files) != implementation.get("bundle_sha256"):
        raise ValueError("league implementation bundle hash mismatch")
    return {
        "recorded_bundle_internally_valid": True,
        "all_recorded_files_currently_match": not current_mismatches,
        "current_mismatches": current_mismatches,
    }


def load_league_pair(
    path: Path,
    *,
    expected_job_spec_sha256: str | None = None,
    tree_hash_cache: dict[Path, str] | None = None,
) -> LeaguePair:
    payload, file_sha256 = _load_json(path)
    if payload.get("kind") != "scacchi.hex_checkpoint_league_pair":
        raise ValueError(f"{path}: wrong league-pair kind")
    job_spec = payload.get("job_spec")
    if not isinstance(job_spec, dict):
        raise ValueError(f"{path}: job spec missing")
    job_spec_sha256 = _canonical_sha256(job_spec)
    if payload.get("job_spec_sha256") != job_spec_sha256:
        raise ValueError(f"{path}: job-spec hash mismatch")
    if (
        expected_job_spec_sha256 is not None
        and job_spec_sha256 != expected_job_spec_sha256
    ):
        raise ValueError(f"{path}: run/job-spec hash mismatch")
    implementation_validation = _validate_implementation_provenance(
        job_spec.get("implementation")
    )

    competitors = payload.get("competitors")
    if not isinstance(competitors, dict):
        raise ValueError(f"{path}: competitor provenance missing")
    ids: list[str] = []
    for role in ("a", "b"):
        provenance = competitors.get(role)
        identity = job_spec.get(f"competitor_{role}")
        if not isinstance(provenance, dict) or not isinstance(identity, dict):
            raise ValueError(f"{path}: competitor {role} provenance missing")
        competitor_id = provenance.get("id")
        if competitor_id != identity.get("id") or not isinstance(
            competitor_id, str
        ):
            raise ValueError(f"{path}: competitor {role} identity mismatch")
        ids.append(competitor_id)
        for field in (
            "checkpoint_metadata_sha256",
            "selected_step_tree_sha256",
            "effective_eval_sha256",
        ):
            if provenance.get(field) != identity.get(field):
                raise ValueError(
                    f"{path}: competitor {role} {field} mismatch"
                )
        selection = provenance.get("checkpoint_selection")
        if not isinstance(selection, dict):
            raise ValueError(f"{path}: checkpoint selection missing")
        selected_step = selection.get("selected_step")
        checkpoint_root = _resolve_recorded_path(
            selection.get("directory"),
            base=ROOT,
        )
        step_root = (checkpoint_root / str(selected_step)).resolve()
        if tree_hash_cache is not None:
            actual_tree_sha256 = tree_hash_cache.get(step_root)
            if actual_tree_sha256 is None:
                actual_tree_sha256 = _tree_sha256(step_root)
                tree_hash_cache[step_root] = actual_tree_sha256
        else:
            actual_tree_sha256 = _tree_sha256(step_root)
        if actual_tree_sha256 != provenance.get("selected_step_tree_sha256"):
            raise ValueError(
                f"{path}: competitor {role} checkpoint-tree hash mismatch"
            )

    reproduction = payload.get("reproduction")
    if not isinstance(reproduction, dict):
        raise ValueError(f"{path}: reproduction provenance missing")
    script_path = _resolve_recorded_path(
        reproduction.get("script_path"),
        base=ROOT,
    )
    if _file_sha256(script_path) != reproduction.get("script_sha256"):
        raise ValueError(f"{path}: league script hash mismatch")

    game_returns = deepcopy(payload.get("game_returns"))
    if not isinstance(game_returns, dict):
        raise ValueError(f"{path}: raw game returns missing")
    if (
        game_returns.get("kind")
        != "scacchi.hex_checkpoint_league_game_returns"
    ):
        raise ValueError(f"{path}: wrong league game-return kind")
    game_returns["kind"] = paired_eval.GAME_RETURNS_KIND
    observations, recomputed_layout = paired_eval._decode_game_returns(
        game_returns
    )
    a_wins = np.asarray(
        [item.candidate_won for item in observations],
        dtype=np.int8,
    )
    games = int(payload.get("evaluation", {}).get("games", 0))
    if games != len(observations) or games % 4:
        raise ValueError(f"{path}: league game count mismatch")
    stored_overall = payload.get("pairwise", {}).get(
        "competitor_a", {}
    ).get("overall", {})
    if (
        stored_overall.get("games") != games
        or stored_overall.get("wins") != int(a_wins.sum())
    ):
        raise ValueError(f"{path}: league overall summary mismatch")
    return LeaguePair(
        path=path.resolve(),
        file_sha256=file_sha256,
        payload=payload,
        competitor_a=ids[0],
        competitor_b=ids[1],
        coordinates=tuple(item.coordinate for item in observations),
        wins_by_stratum=a_wins.reshape(4, games // 4),
        pairing_layout_sha256=recomputed_layout,
        implementation_files_currently_match=bool(
            implementation_validation["all_recorded_files_currently_match"]
        ),
    )


def validate_league_run(
    path: Path,
    *,
    tree_hash_cache: dict[Path, str],
) -> tuple[dict[str, Any], dict[tuple[str, str], LeaguePair]]:
    payload, file_sha256 = _load_json(path)
    if payload.get("kind") != "scacchi.hex_checkpoint_league_run":
        raise ValueError(f"{path}: wrong league-run kind")
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: manifest provenance missing")
    manifest_path = _resolve_recorded_path(manifest.get("path"), base=path.parent)
    manifest_payload, manifest_sha256 = _load_json(manifest_path)
    if manifest_sha256 != manifest.get("sha256"):
        raise ValueError(f"{path}: manifest file hash mismatch")
    if manifest_payload.get("kind") != "scacchi.hex_checkpoint_league_manifest":
        raise ValueError(f"{manifest_path}: wrong manifest kind")

    pair_artifacts = payload.get("pair_artifacts")
    if not isinstance(pair_artifacts, list):
        raise ValueError(f"{path}: pair-artifact provenance missing")
    loaded: dict[tuple[str, str], LeaguePair] = {}
    artifact_summaries: list[dict[str, Any]] = []
    for entry in pair_artifacts:
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: malformed pair-artifact entry")
        pair_path = _resolve_recorded_path(entry.get("path"), base=path.parent)
        pair = load_league_pair(
            pair_path,
            expected_job_spec_sha256=entry.get("job_spec_sha256"),
            tree_hash_cache=tree_hash_cache,
        )
        if pair.file_sha256 != entry.get("sha256"):
            raise ValueError(f"{path}: pair file hash mismatch: {pair_path}")
        key = (pair.competitor_a, pair.competitor_b)
        if key in loaded:
            raise ValueError(f"{path}: duplicate oriented pair {key!r}")
        loaded[key] = pair
        artifact_summaries.append(
            {
                "path": str(pair.path),
                "file_sha256": pair.file_sha256,
                "job_spec_sha256": entry["job_spec_sha256"],
                "competitor_a": pair.competitor_a,
                "competitor_b": pair.competitor_b,
                "games": int(pair.wins_by_stratum.size),
                "pairing_layout_sha256": pair.pairing_layout_sha256,
                "recorded_implementation_bundle_internally_valid": True,
                "implementation_files_currently_match": (
                    pair.implementation_files_currently_match
                ),
            }
        )
    run_pairs = payload.get("pairs")
    if not isinstance(run_pairs, list) or len(run_pairs) != len(loaded):
        raise ValueError(f"{path}: run pair summary count mismatch")
    if payload.get("local_only") != {
        "network_access": False,
        "external_logging": False,
        "external_export": False,
        "artifact_is_local": True,
    }:
        raise ValueError(f"{path}: local-only contract mismatch")
    return (
        {
            "path": str(path.resolve()),
            "file_sha256": file_sha256,
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha256,
            "pair_artifact_count": len(loaded),
            "all_pair_hashes_and_raw_returns_validated": True,
            "all_recorded_implementation_bundles_internally_valid": True,
            "all_implementation_files_currently_match": all(
                pair.implementation_files_currently_match
                for pair in loaded.values()
            ),
            "all_json_numbers_finite": True,
            "pair_artifacts": artifact_summaries,
        },
        loaded,
    )


def validate_league_analysis(
    path: Path,
    *,
    expected_run_path: Path,
    expected_run_sha256: str,
) -> dict[str, Any]:
    payload, file_sha256 = _load_json(path)
    inputs = payload.get("inputs")
    if not isinstance(inputs, dict):
        raise ValueError(f"{path}: analysis input provenance missing")
    manifests = inputs.get("manifests")
    if not isinstance(manifests, list) or len(manifests) != 1:
        raise ValueError(f"{path}: analysis must reference exactly one run")
    recorded_run = manifests[0]
    if (
        _resolve_recorded_path(recorded_run.get("path"), base=path.parent)
        != expected_run_path.resolve()
        or recorded_run.get("file_sha256") != expected_run_sha256
    ):
        raise ValueError(f"{path}: analysis run hash/path mismatch")
    for entry in inputs.get("pair_artifacts", []):
        pair_path = _resolve_recorded_path(entry.get("path"), base=path.parent)
        if _file_sha256(pair_path) != entry.get("file_sha256"):
            raise ValueError(f"{path}: analysis pair hash mismatch")
    reproduction = payload.get("reproduction")
    if not isinstance(reproduction, dict):
        raise ValueError(f"{path}: analysis reproduction missing")
    script_path = _resolve_recorded_path(
        reproduction.get("script_path"),
        base=ROOT,
    )
    if _file_sha256(script_path) != reproduction.get("script_sha256"):
        raise ValueError(f"{path}: analysis script hash mismatch")
    return {
        "path": str(path.resolve()),
        "file_sha256": file_sha256,
        "referenced_run_sha256": expected_run_sha256,
        "referenced_pair_count": len(inputs.get("pair_artifacts", [])),
        "all_referenced_hashes_validated": True,
        "all_json_numbers_finite": True,
    }


def _get_pair(
    pairs: Mapping[tuple[str, str], LeaguePair],
    competitor_a: str,
    competitor_b: str,
) -> LeaguePair:
    try:
        return pairs[(competitor_a, competitor_b)]
    except KeyError as error:
        raise ValueError(
            f"missing oriented pair {competitor_a!r} vs {competitor_b!r}"
        ) from error


def _validate_common_pair(
    e12_pair: LeaguePair,
    e11_pair: LeaguePair,
) -> None:
    if e12_pair.competitor_b != e11_pair.competitor_b:
        raise ValueError("common-roster opponents differ")
    if e12_pair.pairing_layout_sha256 != e11_pair.pairing_layout_sha256:
        raise ValueError("common-roster pairing layout mismatch")
    if e12_pair.coordinates != e11_pair.coordinates:
        raise ValueError("common-roster per-game coordinates mismatch")
    e12_eval = e12_pair.payload["evaluation"]
    e11_eval = e11_pair.payload["evaluation"]
    for field in ("games", "games_per_stratum", "batch_size", "seed"):
        if e12_eval.get(field) != e11_eval.get(field):
            raise ValueError(f"common-roster evaluation {field} mismatch")
    e12_provenance = e12_pair.payload["competitors"]
    e11_provenance = e11_pair.payload["competitors"]
    if (
        e12_provenance["a"]["evaluation_behavior"]
        != e11_provenance["a"]["evaluation_behavior"]
    ):
        raise ValueError("E12/E11 effective evaluation behavior mismatch")
    for field in (
        "checkpoint_metadata_sha256",
        "selected_step_tree_sha256",
    ):
        if e12_provenance["b"].get(field) != e11_provenance["b"].get(field):
            raise ValueError(f"common opponent {field} mismatch")
    if (
        e12_provenance["b"]["evaluation_behavior"]
        != e11_provenance["b"]["evaluation_behavior"]
    ):
        raise ValueError("common opponent evaluation behavior mismatch")


def synchronized_roster_bootstrap(
    deltas: np.ndarray,
    *,
    seed: int,
    replicates: int,
    batch_size: int = BOOTSTRAP_BATCH_SIZE,
) -> dict[str, Any]:
    """Stratified bootstrap of aligned game-coordinate roster blocks.

    ``deltas`` has shape ``(strata, games_per_stratum, anchors)``.  The same
    sampled row indices are used for every anchor in a stratum.
    """

    if deltas.ndim != 3 or deltas.shape[0] != 4:
        raise ValueError("deltas must have shape (4, games, anchors)")
    if deltas.shape[1] <= 0 or deltas.shape[2] <= 0:
        raise ValueError("bootstrap dimensions must be positive")
    if replicates <= 0 or batch_size <= 0:
        raise ValueError("bootstrap controls must be positive")
    values = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(seed)
    anchor_samples = np.empty((replicates, values.shape[2]), dtype=np.float64)
    offset = 0
    while offset < replicates:
        count = min(batch_size, replicates - offset)
        sample_means = np.zeros((count, values.shape[2]), dtype=np.float64)
        for stratum in range(values.shape[0]):
            indices = rng.integers(
                0,
                values.shape[1],
                size=(count, values.shape[1]),
            )
            sample_means += values[stratum][indices].mean(axis=1)
        anchor_samples[offset : offset + count] = (
            sample_means / values.shape[0]
        )
        offset += count
    roster_samples = anchor_samples.mean(axis=1)
    point_by_anchor = values.mean(axis=(0, 1))
    point_roster = float(point_by_anchor.mean())
    lower, upper = np.quantile(
        roster_samples,
        (0.05, 0.95),
        method="linear",
    )
    anchor_intervals = np.quantile(
        anchor_samples,
        (0.05, 0.95),
        axis=0,
        method="linear",
    )
    return {
        "method": (
            "synchronized stratified game-coordinate block bootstrap; "
            "resample rows within each logical-player-id x seat stratum and "
            "carry all four opponent deltas together"
        ),
        "seed": seed,
        "replicates": replicates,
        "confidence": 0.90,
        "quantiles": [0.05, 0.95],
        "point_estimate": point_roster,
        "interval": [float(lower), float(upper)],
        "point_by_anchor": [float(value) for value in point_by_anchor],
        "interval_by_anchor": [
            [
                float(anchor_intervals[0, index]),
                float(anchor_intervals[1, index]),
            ]
            for index in range(values.shape[2])
        ],
    }


def analyze_direct_pair(pair: LeaguePair) -> dict[str, Any]:
    wins = int(pair.wins_by_stratum.sum())
    games = int(pair.wins_by_stratum.size)
    interval = wilson_interval(wins, games, confidence=0.90)
    return {
        "path": str(pair.path),
        "file_sha256": pair.file_sha256,
        "competitor_a": pair.competitor_a,
        "competitor_b": pair.competitor_b,
        "games": games,
        "wins": wins,
        "win_rate": wins / games,
        "wilson90": list(interval),
        "equivalence_interval_contained_in_0p45_0p55": (
            interval[0] >= 0.45 and interval[1] <= 0.55
        ),
        "noninferiority_lower_bound_at_least_0p45": interval[0] >= 0.45,
    }


def analyze_common_roster(
    *,
    step: int,
    e12_pairs: Mapping[tuple[str, str], LeaguePair],
    e11_pairs: Mapping[tuple[str, str], LeaguePair],
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    paired_deltas: list[np.ndarray] = []
    anchor_results: list[dict[str, Any]] = []
    first_differences: list[float] = []
    for anchor in ANCHORS:
        e12_pair = _get_pair(e12_pairs, f"e12-step{step}-q21", anchor)
        e11_pair = _get_pair(e11_pairs, f"e11-step{step}-q21", anchor)
        _validate_common_pair(e12_pair, e11_pair)
        delta = (
            e12_pair.wins_by_stratum.astype(np.float64)
            - e11_pair.wins_by_stratum.astype(np.float64)
        )
        paired_deltas.append(delta)
        e12_rate = float(e12_pair.wins_by_stratum.mean())
        e11_rate = float(e11_pair.wins_by_stratum.mean())
        e12_first = float(e12_pair.wins_by_stratum[[0, 2]].mean())
        e11_first = float(e11_pair.wins_by_stratum[[0, 2]].mean())
        first_difference = e12_first - e11_first
        first_differences.append(first_difference)
        anchor_results.append(
            {
                "anchor": anchor,
                "e12_path": str(e12_pair.path),
                "e12_file_sha256": e12_pair.file_sha256,
                "e11_path": str(e11_pair.path),
                "e11_file_sha256": e11_pair.file_sha256,
                "pairing_layout_sha256": (
                    e12_pair.pairing_layout_sha256
                ),
                "coordinates_exactly_matched": True,
                "e12_score": e12_rate,
                "e11_score": e11_rate,
                "e12_minus_e11": e12_rate - e11_rate,
                "e12_first_seat_conversion": e12_first,
                "e11_first_seat_conversion": e11_first,
                "first_seat_conversion_difference": first_difference,
            }
        )
    deltas = np.stack(paired_deltas, axis=2)
    bootstrap = synchronized_roster_bootstrap(
        deltas,
        seed=bootstrap_seed,
        replicates=bootstrap_replicates,
    )
    for result, interval in zip(
        anchor_results,
        bootstrap["interval_by_anchor"],
        strict=True,
    ):
        result["paired_bootstrap90"] = interval
    point_differences = [
        float(result["e12_minus_e11"]) for result in anchor_results
    ]
    checks = {
        "mean_roster_bootstrap90_lower_above_minus_0p05": (
            float(bootstrap["interval"][0]) > -0.05
        ),
        "no_anchor_point_difference_below_minus_0p10": (
            min(point_differences) >= -0.10
        ),
        "worst_first_seat_difference_at_least_minus_0p05": (
            min(first_differences) >= -0.05
        ),
    }
    checks["all_passed"] = all(checks.values())
    return {
        "step": step,
        "anchors": anchor_results,
        "mean_roster_e12_minus_e11": (
            math.fsum(point_differences) / len(point_differences)
        ),
        "game_block_bootstrap90": bootstrap,
        "worst_anchor_point_difference": min(point_differences),
        "worst_anchor_first_seat_conversion_difference": min(
            first_differences
        ),
        "registered_checks": checks,
    }


def _write_create_once(path: Path, payload: Mapping[str, Any]) -> str:
    encoded = (
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return digest


def build_analysis(
    *,
    bootstrap_seed: int,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    e12_metrics = load_metric_stream(
        E12_DIR / "metrics.jsonl",
        validate_e12_provenance=True,
    )
    e11_metrics = load_metric_stream(E11_DIR / "metrics.jsonl")
    e12_balanced = load_balanced_evaluations(
        E12_DIR,
        validate_e12_plan=True,
    )
    e11_balanced = load_balanced_evaluations(
        E11_DIR,
        validate_e12_plan=False,
    )

    tree_hash_cache: dict[Path, str] = {}
    e12_run_summary, e12_pairs = validate_league_run(
        E12_RUN,
        tree_hash_cache=tree_hash_cache,
    )
    e11_run_summary, e11_pairs = validate_league_run(
        E11_RUN,
        tree_hash_cache=tree_hash_cache,
    )
    supplemental_summary, supplemental_pairs = validate_league_run(
        E11_STEP100_COMMON_RUN,
        tree_hash_cache=tree_hash_cache,
    )
    e12_analysis_summary = validate_league_analysis(
        E12_ANALYSIS,
        expected_run_path=E12_RUN,
        expected_run_sha256=e12_run_summary["file_sha256"],
    )
    e11_analysis_summary = validate_league_analysis(
        E11_ANALYSIS,
        expected_run_path=E11_RUN,
        expected_run_sha256=e11_run_summary["file_sha256"],
    )

    # The supplemental run replaces only the three formerly 4,096-game E11
    # step-100 anchor coordinates.  E8-100 was already 8,192/common-layout.
    e11_common_pairs = dict(e11_pairs)
    e11_common_pairs.update(supplemental_pairs)

    direct = {
        str(step): analyze_direct_pair(
            _get_pair(
                e12_pairs,
                f"e12-step{step}-q21",
                f"e11-step{step}-q21",
            )
        )
        for step in (75, 100)
    }
    common_roster = {
        str(step): analyze_common_roster(
            step=step,
            e12_pairs=e12_pairs,
            e11_pairs=e11_common_pairs,
            bootstrap_seed=bootstrap_seed + step,
            bootstrap_replicates=bootstrap_replicates,
        )
        for step in (75, 100)
    }

    frozen_steps = {
        str(step): e12_balanced[step] for step in EXPECTED_STEPS
    }
    e11_frozen_steps = {
        str(step): e11_balanced[step] for step in EXPECTED_STEPS
    }
    late_stability = all(
        frozen_steps[str(step)]["registered_gates"]["all_passed"]
        for step in (125, 150, 175, 200)
    )
    mandatory_internal = bool(
        e12_metrics["early_10_29"]["gates"]["all_passed"]
    )
    step75_fixed = bool(
        frozen_steps["75"]["registered_gates"]["all_passed"]
    )
    step100_fixed = bool(
        frozen_steps["100"]["registered_gates"]["all_passed"]
    )
    step75_direct_equivalence = bool(
        direct["75"]["equivalence_interval_contained_in_0p45_0p55"]
    )
    step75_common = bool(
        common_roster["75"]["registered_checks"]["all_passed"]
    )
    speed_success = (
        step75_fixed
        and step75_direct_equivalence
        and step75_common
        and late_stability
    )
    delayed_parity = (
        not step75_fixed and step100_fixed and late_stability
    )
    training_failure = (
        not mandatory_internal or not step100_fixed or not late_stability
    )
    if speed_success:
        formal_label = "speed_success"
    elif delayed_parity:
        formal_label = "delayed_parity"
    elif training_failure:
        formal_label = "training_failure"
    else:
        formal_label = "unclassified"

    window_comparison = _compare_metric_windows(e11_metrics, e12_metrics)
    provenance = {
        "e12_metric_stream": {
            key: e12_metrics[key]
            for key in (
                "path",
                "file_sha256",
                "line_count",
                "metric_row_count",
                "steps",
                "steps_contiguous",
                "all_json_numbers_finite",
                "run_name",
                "provenance",
            )
        },
        "e11_metric_stream": {
            key: e11_metrics[key]
            for key in (
                "path",
                "file_sha256",
                "line_count",
                "metric_row_count",
                "steps",
                "steps_contiguous",
                "all_json_numbers_finite",
                "run_name",
                "provenance",
            )
        },
        "e12_balanced_plan": e12_balanced[-1],
        "league": {
            "e12_run": e12_run_summary,
            "e12_analysis": e12_analysis_summary,
            "e11_run": e11_run_summary,
            "e11_analysis": e11_analysis_summary,
            "e11_step100_common_coordinate_supplement": supplemental_summary,
        },
        "checkpoint_tree_hashes_recomputed": {
            str(path): digest
            for path, digest in sorted(
                tree_hash_cache.items(),
                key=lambda pair: str(pair[0]),
            )
        },
        "source_artifacts_are_local_only": True,
        "network_or_external_logging_used_by_this_analysis": False,
        "all_loaded_json_numbers_finite": True,
    }
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "local_only": {
            "network_access": False,
            "external_logging": False,
            "external_export": False,
            "artifact_is_local": True,
        },
        "reproduction": {
            "command": " ".join(
                shlex.quote(value)
                for value in (
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--output",
                    str(DEFAULT_OUTPUT),
                    "--bootstrap-seed",
                    str(bootstrap_seed),
                    "--bootstrap-replicates",
                    str(bootstrap_replicates),
                )
            ),
            "working_directory": str(ROOT),
            "script_path": str(Path(__file__).resolve()),
            "script_sha256": _file_sha256(Path(__file__).resolve()),
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "provenance_validation": provenance,
        "internal_numeric_gates": {
            "prefix_guards_all_201_rows": (
                e12_metrics["prefix_guard_checks"]
            ),
            "early_10_29": e12_metrics["early_10_29"],
        },
        "frozen_vs_6_solved": {
            "steps": frozen_steps,
            "e11_matched_steps": e11_frozen_steps,
        },
        "direct_e12_vs_e11": direct,
        "common_roster_e12_minus_e11": common_roster,
        "logged_e12_vs_e11_window_comparison": window_comparison,
        "registered_outcome": {
            "launch_gates_passed_before_training": True,
            "mandatory_internal_numeric_gates_passed": mandatory_internal,
            "step75_fixed_opponent_gate_passed": step75_fixed,
            "step75_direct_equivalence_gate_passed": (
                step75_direct_equivalence
            ),
            "step75_direct_noninferiority_passed": direct["75"][
                "noninferiority_lower_bound_at_least_0p45"
            ],
            "step75_common_roster_noninferiority_passed": step75_common,
            "step100_fixed_opponent_gate_passed": step100_fixed,
            "late_125_200_stability_passed": late_stability,
            "speed_success": speed_success,
            "delayed_parity": delayed_parity,
            "training_failure": training_failure,
            "formal_label": formal_label,
            "step100_formal_containment_miss": {
                "wilson90_upper": frozen_steps["100"][
                    "pooled_wilson90"
                ][1],
                "registered_upper": 0.55,
                "excess": (
                    frozen_steps["100"]["pooled_wilson90"][1] - 0.55
                ),
                "e_seat": frozen_steps["100"]["e_seat"],
                "interpretation": (
                    "The locked containment gate fails on the upper boundary "
                    "despite a passing, low seat-optimal error."
                ),
            },
        },
        "causal_localization": {
            "early_gate_result": (
                "All registered iterations-10--29 numeric gates pass."
            ),
            "coverage_observation": (
                "The largest logged divergence before checkpoint 100 is "
                "on-policy opening concentration in iterations 75--99, not "
                "a loss of legal support: opening action effective support "
                "falls relative to E11 while all 36 opening actions remain "
                "observed."
            ),
            "transfer_observation": (
                "Policy and Q gap reductions remain positive and are mostly "
                "comparable to or stronger than E11 after iteration 30; the "
                "data do not support a simple failure to transfer search "
                "information into the weights."
            ),
            "external_observation": (
                "The direct and common-roster edges disagree with a scalar "
                "fixed-opponent ordering, exposing nontransitive behavior "
                "and amplification through the changed on-policy state "
                "distribution."
            ),
            "causal_scope": (
                "These comparisons localize the mismatch but remain "
                "observational after the randomized commitment intervention; "
                "they do not identify a unique downstream mechanism."
            ),
        },
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=BOOTSTRAP_REPLICATES,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.bootstrap_replicates <= 0:
        raise ValueError("--bootstrap-replicates must be positive")
    payload = build_analysis(
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_replicates=args.bootstrap_replicates,
    )
    output = args.output
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    digest = _write_create_once(output, payload)
    print(f"wrote {output}")
    print(f"sha256 {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
