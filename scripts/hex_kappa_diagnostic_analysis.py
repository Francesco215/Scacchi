#!/usr/bin/env python3
"""Test whether frozen-root kappa diagnostics predict deployed movement.

The input is a v2 artifact from ``hex_kappa_fixed_probe.py`` containing a
reference kappa, symmetric ``log(kappa) +/- h`` points, and the requested
nonlocal comparison points.  The analysis is deliberately root-level:

* the central finite-difference policy response is
  ``||pi(+h) - pi(-h)||_1 / (2 h)``;
* its rank association with the reference normalized-cache derivative is
  reported overall and by stage both end to end and conditional on the
  reference root remaining unresolved, with a game-cluster bootstrap;
* the exact raw-cache derivative bound is audited;
* response is summarized by reference-derivative decile; and
* top-action flips are stratified by the reference top-two policy margin,
  including the necessary L1 and Linf movement bounds.

The script reads the immutable corpus ``roots.npz`` to recover the original
game clusters.  It performs no search, training, network access, or external
logging.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import shlex
import sys
import tempfile
from typing import Any, Mapping, Sequence, cast

import numpy as np


FORMAT = "scacchi.hex_kappa_diagnostic_informativeness.v2"
PROBE_FORMAT = "scacchi.hex_kappa_fixed_probe.v2"
STAGE_NAMES = ("early", "mid", "late")
DEPLOYMENT_SCOPE = "deployment_all_numeric_repair_roots"
CHANNEL_ACTIVE_SCOPE = "reference_unresolved_numeric_channel"
DEFAULT_LOCAL_LOG_STEP = 0.05
DEFAULT_COMPARISON_KAPPAS = (8.0, 64.0)
DEFAULT_BOOTSTRAP_REPLICATES = 50_000
DEFAULT_BOOTSTRAP_SEED = 2_026_072_5
BOUND_ABSOLUTE_TOLERANCE = 1e-7
BOUND_RELATIVE_TOLERANCE = 1e-5
MARGIN_BOUND_TOLERANCE = 1e-10


@dataclass(frozen=True)
class ProbeArrays:
    """Validated root-aligned arrays extracted from a v2 probe."""

    reference_kappa: float
    local_minus_kappa: float
    local_plus_kappa: float
    local_log_step: float
    comparison_kappas: tuple[float, ...]
    root_ids: np.ndarray
    stage_ids: np.ndarray
    game_cluster_ids: np.ndarray
    root_weights: np.ndarray
    policies: Mapping[float, np.ndarray]
    numeric_repair_counts: Mapping[float, np.ndarray]
    raw_innovation_l2: Mapping[float, np.ndarray]
    raw_dcache_dlogkappa_l2: Mapping[float, np.ndarray]
    mean_dcache_dlogkappa_l2: Mapping[float, np.ndarray]
    reference_solved_roots: np.ndarray
    reference_margins: np.ndarray
    reference_margin_scales: np.ndarray


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _kappa_key(kappa: float) -> str:
    return format(float(kappa), ".17g")


def _parse_kappas(value: str) -> tuple[float, ...]:
    try:
        kappas = tuple(
            float(part.strip())
            for part in value.split(",")
            if part.strip()
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "kappas must be comma-separated floating-point values"
        ) from error
    if not kappas:
        raise argparse.ArgumentTypeError("at least one kappa is required")
    if any(not math.isfinite(kappa) or kappa <= 0.0 for kappa in kappas):
        raise argparse.ArgumentTypeError(
            "every kappa must be finite and positive"
        )
    if len(set(kappas)) != len(kappas):
        raise argparse.ArgumentTypeError("kappas must be unique")
    return kappas


def _finite_summary(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return {
        "count": int(len(finite)),
        "mean": float(np.mean(finite)) if finite.size else None,
        "median": float(np.median(finite)) if finite.size else None,
        "p05": (
            float(np.quantile(finite, 0.05)) if finite.size else None
        ),
        "p25": (
            float(np.quantile(finite, 0.25)) if finite.size else None
        ),
        "p75": (
            float(np.quantile(finite, 0.75)) if finite.size else None
        ),
        "p95": (
            float(np.quantile(finite, 0.95)) if finite.size else None
        ),
        "max": float(np.max(finite)) if finite.size else None,
    }


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float | None:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = (
        np.isfinite(values)
        & np.isfinite(weights)
        & (weights >= 0.0)
    )
    total = float(np.sum(weights[valid]))
    if total <= 0.0:
        return None
    return float(np.sum(values[valid] * weights[valid]) / total)


def _average_ranks(values: np.ndarray) -> np.ndarray:
    """Return one-based average ranks with exact ties."""

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("rank input must be a finite one-dimensional array")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while (
            stop < len(values)
            and values[order[stop]] == values[order[start]]
        ):
            stop += 1
        average = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray) -> float | None:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(left) & np.isfinite(right)
    left = left[valid]
    right = right[valid]
    if len(left) < 2:
        return None
    left_rank = _average_ranks(left)
    right_rank = _average_ranks(right)
    left_centered = left_rank - np.mean(left_rank)
    right_centered = right_rank - np.mean(right_rank)
    denominator = math.sqrt(
        float(np.sum(np.square(left_centered)))
        * float(np.sum(np.square(right_centered)))
    )
    if denominator <= 0.0:
        return None
    return float(
        np.sum(left_centered * right_centered) / denominator
    )


def _percentile_interval(values: np.ndarray) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if not values.size:
        return None
    return [
        float(np.quantile(values, 0.025)),
        float(np.quantile(values, 0.975)),
    ]


def _cluster_bootstrap(
    *,
    derivative: np.ndarray,
    response: np.ndarray,
    cluster_ids: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    """Resample complete source-game clusters and recompute estimands."""

    derivative = np.asarray(derivative, dtype=np.float64)
    response = np.asarray(response, dtype=np.float64)
    cluster_ids = np.asarray(cluster_ids)
    valid = (
        np.isfinite(derivative)
        & np.isfinite(response)
    )
    derivative = derivative[valid]
    response = response[valid]
    cluster_ids = cluster_ids[valid]
    unique_clusters = np.unique(cluster_ids)
    groups = [
        np.flatnonzero(cluster_ids == cluster)
        for cluster in unique_clusters
    ]
    if not groups:
        return {
            "game_clusters": 0,
            "replicates_requested": int(replicates),
            "valid_spearman_replicates": 0,
            "spearman_95": None,
            "mean_response_95": None,
        }
    rho_samples = np.full(replicates, np.nan, dtype=np.float64)
    mean_samples = np.full(replicates, np.nan, dtype=np.float64)
    for replicate in range(replicates):
        sampled_groups = rng.integers(
            0,
            len(groups),
            size=len(groups),
        )
        indices = np.concatenate(
            [groups[group] for group in sampled_groups]
        )
        rho = _spearman(derivative[indices], response[indices])
        if rho is not None:
            rho_samples[replicate] = rho
        mean_samples[replicate] = float(np.mean(response[indices]))
    return {
        "game_clusters": int(len(unique_clusters)),
        "replicates_requested": int(replicates),
        "valid_spearman_replicates": int(
            np.sum(np.isfinite(rho_samples))
        ),
        "spearman_95": _percentile_interval(rho_samples),
        "mean_response_95": _percentile_interval(mean_samples),
    }


def _find_kappa(
    available: Sequence[float],
    target: float,
    *,
    label: str,
    log_tolerance: float = 1e-10,
) -> float:
    candidates = [
        float(kappa)
        for kappa in available
        if abs(math.log(float(kappa) / float(target))) <= log_tolerance
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"{label}={target:.17g} has {len(candidates)} matches in "
            f"available kappas {list(available)}"
        )
    return candidates[0]


def _optional_float(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{location} must be finite")
    return result


def _optional_channel_value(
    record: Mapping[str, Any],
    *path: str,
) -> float:
    value: Any = record
    for component in path:
        if not isinstance(value, Mapping) or component not in value:
            return math.nan
        value = value[component]
    if value is None:
        return math.nan
    return _optional_float(value, ".".join(path))


def _extract_probe_arrays(
    *,
    payload: Mapping[str, Any],
    roots: Mapping[str, np.ndarray],
    local_log_step: float,
    comparison_kappas: Sequence[float],
) -> ProbeArrays:
    if payload.get("format") != PROBE_FORMAT:
        raise ValueError(
            f"probe format must be {PROBE_FORMAT!r}; "
            f"got {payload.get('format')!r}"
        )
    reference_kappa = _optional_float(
        payload["protocol"]["reference_kappa"],
        "protocol.reference_kappa",
    )
    available = tuple(
        _optional_float(value, "execution.kappas")
        for value in payload["execution"]["kappas"]
    )
    local_minus = _find_kappa(
        available,
        reference_kappa * math.exp(-local_log_step),
        label="local minus kappa",
    )
    local_plus = _find_kappa(
        available,
        reference_kappa * math.exp(local_log_step),
        label="local plus kappa",
    )
    resolved_comparisons = tuple(
        _find_kappa(
            available,
            float(kappa),
            label="comparison kappa",
        )
        for kappa in comparison_kappas
    )
    required_kappas = (
        reference_kappa,
        local_minus,
        local_plus,
        *resolved_comparisons,
    )
    if len(set(required_kappas)) != len(required_kappas):
        raise ValueError(
            "reference, local, and comparison kappas must be distinct"
        )

    positions = payload.get("positions")
    if not isinstance(positions, list) or not positions:
        raise ValueError("probe positions must be a nonempty list")
    corpus_root_ids = np.asarray(roots["root_id"], dtype=np.int64)
    if len(np.unique(corpus_root_ids)) != len(corpus_root_ids):
        raise ValueError("roots.npz root_id values must be unique")
    corpus_index = {
        int(root_id): index
        for index, root_id in enumerate(corpus_root_ids)
    }
    root_ids: list[int] = []
    stage_ids: list[int] = []
    cluster_ids: list[int] = []
    root_weights: list[float] = []
    policy_rows: dict[float, list[np.ndarray]] = {
        kappa: [] for kappa in required_kappas
    }
    numeric_counts: dict[float, list[float]] = {
        kappa: [] for kappa in required_kappas
    }
    raw_innovation: dict[float, list[float]] = {
        kappa: [] for kappa in required_kappas
    }
    raw_derivative: dict[float, list[float]] = {
        kappa: [] for kappa in required_kappas
    }
    mean_derivative: dict[float, list[float]] = {
        kappa: [] for kappa in required_kappas
    }
    reference_margins: list[float] = []
    reference_margin_scales: list[float] = []
    reference_solved_roots: list[bool] = []
    action_count: int | None = None
    for position_index, position_value in enumerate(positions):
        if not isinstance(position_value, Mapping):
            raise ValueError(f"positions[{position_index}] must be an object")
        position = cast(Mapping[str, Any], position_value)
        root_id = int(position["root_id"])
        if root_id not in corpus_index:
            raise ValueError(f"probe root_id {root_id} is absent from roots.npz")
        corpus_row = corpus_index[root_id]
        stage_id = int(roots["stage_id"][corpus_row])
        if position.get("stage_id") != stage_id:
            raise ValueError(f"root {root_id} stage disagrees with roots.npz")
        if "state_sha256" in roots:
            corpus_state_sha256 = roots["state_sha256"][corpus_row]
            if isinstance(corpus_state_sha256, bytes):
                corpus_state_sha256 = corpus_state_sha256.decode("ascii")
            if position.get("state_sha256") != str(corpus_state_sha256):
                raise ValueError(
                    f"root {root_id} state digest disagrees with roots.npz"
                )
        artifact_weight = _optional_float(
            position["corpus_root_weight"],
            f"root {root_id} corpus_root_weight",
        )
        corpus_weight = float(roots["root_weight"][corpus_row])
        if not math.isclose(
            artifact_weight,
            corpus_weight,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(f"root {root_id} weight disagrees with roots.npz")
        root_ids.append(root_id)
        stage_ids.append(stage_id)
        cluster_ids.append(int(roots["game_cluster_id"][corpus_row]))
        root_weights.append(corpus_weight)

        by_kappa_value = position.get("by_kappa")
        if not isinstance(by_kappa_value, Mapping):
            raise ValueError(f"root {root_id} by_kappa must be an object")
        by_kappa = cast(Mapping[str, Any], by_kappa_value)
        keyed_records = {
            float(key): value
            for key, value in by_kappa.items()
        }
        per_kappa_records: dict[float, Mapping[str, Any]] = {}
        for kappa in required_kappas:
            matched = _find_kappa(
                tuple(keyed_records),
                kappa,
                label=f"root {root_id} kappa",
            )
            record = keyed_records[matched]
            if not isinstance(record, Mapping):
                raise ValueError(
                    f"root {root_id} kappa {kappa} record must be an object"
                )
            per_kappa_records[kappa] = cast(Mapping[str, Any], record)
            policy = np.asarray(record["root_policy"], dtype=np.float64)
            if (
                policy.ndim != 1
                or not np.all(np.isfinite(policy))
                or np.any(policy < -1e-10)
                or not math.isclose(
                    float(np.sum(policy)),
                    1.0,
                    rel_tol=0.0,
                    abs_tol=1e-8,
                )
            ):
                raise ValueError(
                    f"root {root_id} kappa {kappa} has invalid policy"
                )
            if action_count is None:
                action_count = len(policy)
            elif len(policy) != action_count:
                raise ValueError("root policy action dimensions disagree")
            policy_rows[kappa].append(policy)
            numeric_counts[kappa].append(
                _optional_channel_value(
                    record,
                    "kappa_channel",
                    "numeric_repairs",
                    "count",
                )
            )
            raw_innovation[kappa].append(
                _optional_channel_value(
                    record,
                    "kappa_channel",
                    "numeric_repairs",
                    "raw_innovation_l2_mean",
                )
            )
            raw_derivative[kappa].append(
                _optional_channel_value(
                    record,
                    "kappa_channel",
                    "numeric_repairs",
                    "raw_dcache_dlogkappa_l2_mean",
                )
            )
            mean_derivative[kappa].append(
                _optional_channel_value(
                    record,
                    "kappa_channel",
                    "numeric_repairs",
                    "mean_dcache_dlogkappa_l2_mean",
                )
            )
        reference_record = per_kappa_records[reference_kappa]
        reference_solved = reference_record.get("solved_root")
        if not isinstance(reference_solved, bool):
            raise ValueError(
                f"root {root_id} reference solved_root must be Boolean"
            )
        reference_solved_roots.append(reference_solved)
        reference_margins.append(
            _optional_channel_value(
                reference_record,
                "kappa_channel",
                "commitment_policy_top2",
                "margin",
            )
        )
        reference_margin_scales.append(
            _optional_channel_value(
                reference_record,
                "kappa_channel",
                "commitment_policy_top2",
                "reference_scale",
            )
        )
    if len(set(root_ids)) != len(root_ids):
        raise ValueError("probe positions contain duplicate root_id values")

    return ProbeArrays(
        reference_kappa=reference_kappa,
        local_minus_kappa=local_minus,
        local_plus_kappa=local_plus,
        local_log_step=local_log_step,
        comparison_kappas=resolved_comparisons,
        root_ids=np.asarray(root_ids, dtype=np.int64),
        stage_ids=np.asarray(stage_ids, dtype=np.int8),
        game_cluster_ids=np.asarray(cluster_ids, dtype=np.int64),
        root_weights=np.asarray(root_weights, dtype=np.float64),
        policies={
            kappa: np.stack(rows, axis=0)
            for kappa, rows in policy_rows.items()
        },
        numeric_repair_counts={
            kappa: np.asarray(values, dtype=np.float64)
            for kappa, values in numeric_counts.items()
        },
        raw_innovation_l2={
            kappa: np.asarray(values, dtype=np.float64)
            for kappa, values in raw_innovation.items()
        },
        raw_dcache_dlogkappa_l2={
            kappa: np.asarray(values, dtype=np.float64)
            for kappa, values in raw_derivative.items()
        },
        mean_dcache_dlogkappa_l2={
            kappa: np.asarray(values, dtype=np.float64)
            for kappa, values in mean_derivative.items()
        },
        reference_solved_roots=np.asarray(
            reference_solved_roots,
            dtype=bool,
        ),
        reference_margins=np.asarray(
            reference_margins,
            dtype=np.float64,
        ),
        reference_margin_scales=np.asarray(
            reference_margin_scales,
            dtype=np.float64,
        ),
    )


def _central_response(arrays: ProbeArrays) -> tuple[np.ndarray, np.ndarray]:
    span = np.sum(
        np.abs(
            arrays.policies[arrays.local_plus_kappa]
            - arrays.policies[arrays.local_minus_kappa]
        ),
        axis=-1,
    )
    return span, span / (2.0 * arrays.local_log_step)


def _central_response_summary(
    *,
    arrays: ProbeArrays,
    mask: np.ndarray,
    replicates: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    span, response = _central_response(arrays)
    derivative = arrays.mean_dcache_dlogkappa_l2[
        arrays.reference_kappa
    ]
    eligible = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(derivative)
        & (arrays.numeric_repair_counts[arrays.reference_kappa] > 0.0)
    )
    point = _spearman(derivative[eligible], response[eligible])
    bootstrap = _cluster_bootstrap(
        derivative=derivative[eligible],
        response=response[eligible],
        cluster_ids=arrays.game_cluster_ids[eligible],
        replicates=replicates,
        rng=rng,
    )
    return {
        "roots_in_slice": int(np.sum(mask)),
        "numeric_derivative_eligible_roots": int(np.sum(eligible)),
        "central_log_step": arrays.local_log_step,
        "local_minus_kappa": arrays.local_minus_kappa,
        "local_plus_kappa": arrays.local_plus_kappa,
        "policy_l1_span": _finite_summary(span[mask]),
        "policy_l1_response_per_log_kappa": _finite_summary(
            response[mask]
        ),
        "occupancy_weighted_mean_policy_l1_response_per_log_kappa": (
            _weighted_mean(response[mask], arrays.root_weights[mask])
        ),
        "reference_normalized_cache_derivative_l2": _finite_summary(
            derivative[eligible]
        ),
        "spearman_derivative_vs_policy_l1_response": point,
        "game_cluster_bootstrap": bootstrap,
    }


def _derivative_bound_summary(
    *,
    arrays: ProbeArrays,
    kappa: float,
    mask: np.ndarray,
) -> dict[str, Any]:
    count = arrays.numeric_repair_counts[kappa]
    innovation = arrays.raw_innovation_l2[kappa]
    derivative = arrays.raw_dcache_dlogkappa_l2[kappa]
    eligible = (
        np.asarray(mask, dtype=bool)
        & (count > 0.0)
        & np.isfinite(innovation)
        & np.isfinite(derivative)
    )
    bound = 0.25 * innovation[eligible]
    observed = derivative[eligible]
    tolerance = (
        BOUND_ABSOLUTE_TOLERANCE
        + BOUND_RELATIVE_TOLERANCE * bound
    )
    excess = observed - bound
    violated = excess > tolerance
    eligible_root_ids = arrays.root_ids[eligible]
    positive_bound = bound > 0.0
    ratios = np.divide(
        observed,
        bound,
        out=np.full_like(observed, np.nan),
        where=positive_bound,
    )
    return {
        "eligible_roots": int(np.sum(eligible)),
        "bound": (
            "mean raw ||dC/dlog(kappa)||_2 <= "
            "0.25 * mean raw ||Abar-V||_2"
        ),
        "absolute_tolerance": BOUND_ABSOLUTE_TOLERANCE,
        "relative_tolerance": BOUND_RELATIVE_TOLERANCE,
        "violation_count": int(np.sum(violated)),
        "violation_fraction": (
            float(np.mean(violated)) if violated.size else None
        ),
        "violation_root_ids": eligible_root_ids[violated].tolist(),
        "maximum_excess_before_tolerance": (
            float(np.max(excess)) if excess.size else None
        ),
        "maximum_observed_to_bound_ratio": (
            float(np.nanmax(ratios))
            if np.any(np.isfinite(ratios))
            else None
        ),
    }


def _derivative_deciles(
    *,
    arrays: ProbeArrays,
    mask: np.ndarray,
) -> list[dict[str, Any]]:
    _, response = _central_response(arrays)
    derivative = arrays.mean_dcache_dlogkappa_l2[
        arrays.reference_kappa
    ]
    eligible_indices = np.flatnonzero(
        np.asarray(mask, dtype=bool)
        & np.isfinite(derivative)
        & (arrays.numeric_repair_counts[arrays.reference_kappa] > 0.0)
    )
    order = np.lexsort(
        (arrays.root_ids[eligible_indices], derivative[eligible_indices])
    )
    ordered = eligible_indices[order]
    if not len(ordered):
        return []
    groups = np.array_split(ordered, min(10, len(ordered)))
    output: list[dict[str, Any]] = []
    for decile, indices in enumerate(groups, start=1):
        if not len(indices):
            continue
        output.append(
            {
                "decile": decile,
                "roots": int(len(indices)),
                "derivative_min": float(np.min(derivative[indices])),
                "derivative_max": float(np.max(derivative[indices])),
                "derivative_mean": float(np.mean(derivative[indices])),
                "policy_l1_response": _finite_summary(response[indices]),
                "occupancy_weighted_mean_policy_l1_response": (
                    _weighted_mean(
                        response[indices],
                        arrays.root_weights[indices],
                    )
                ),
            }
        )
    return output


def _margin_stratum_masks(
    margin: np.ndarray,
    scale: np.ndarray,
    eligible: np.ndarray,
) -> tuple[tuple[str, np.ndarray], ...]:
    tie = eligible & (margin == 0.0)
    positive = eligible & (margin > 0.0)
    return (
        ("tie", tie),
        (
            "positive_at_most_one_reference_scale",
            positive & (margin <= scale),
        ),
        (
            "one_to_two_reference_scales",
            positive & (margin > scale) & (margin <= 2.0 * scale),
        ),
        (
            "above_two_reference_scales",
            positive & (margin > 2.0 * scale),
        ),
    )


def _margin_flip_summary(
    *,
    arrays: ProbeArrays,
    candidate_kappa: float,
) -> dict[str, Any]:
    reference = arrays.policies[arrays.reference_kappa]
    candidate = arrays.policies[candidate_kappa]
    movement = np.abs(candidate - reference)
    l1 = np.sum(movement, axis=-1)
    linf = np.max(movement, axis=-1)
    flips = np.argmax(candidate, axis=-1) != np.argmax(reference, axis=-1)
    margin = arrays.reference_margins
    scale = arrays.reference_margin_scales
    eligible = (
        np.isfinite(margin)
        & np.isfinite(scale)
        & (scale > 0.0)
    )
    considered_flips = eligible & flips
    l1_violations = considered_flips & (
        l1 + MARGIN_BOUND_TOLERANCE < margin
    )
    linf_violations = considered_flips & (
        linf + MARGIN_BOUND_TOLERANCE < 0.5 * margin
    )
    eligible_weight = float(np.sum(arrays.root_weights[eligible]))
    flip_weight = float(np.sum(arrays.root_weights[considered_flips]))
    total_flips = int(np.sum(considered_flips))
    strata: dict[str, Any] = {}
    for name, stratum in _margin_stratum_masks(margin, scale, eligible):
        roots = int(np.sum(stratum))
        stratum_flips = stratum & flips
        flip_count = int(np.sum(stratum_flips))
        weight = float(np.sum(arrays.root_weights[stratum]))
        stratum_flip_weight = float(
            np.sum(arrays.root_weights[stratum_flips])
        )
        strata[name] = {
            "roots": roots,
            "root_share": (
                float(roots / np.sum(eligible))
                if np.any(eligible)
                else None
            ),
            "occupancy_weight_share": (
                weight / eligible_weight
                if eligible_weight > 0.0
                else None
            ),
            "flip_count": flip_count,
            "flip_rate_within_stratum": (
                float(flip_count / roots) if roots else None
            ),
            "share_of_all_flips": (
                float(flip_count / total_flips)
                if total_flips
                else None
            ),
            "occupancy_weighted_flip_rate_within_stratum": (
                stratum_flip_weight / weight if weight > 0.0 else None
            ),
            "occupancy_weighted_share_of_all_flips": (
                stratum_flip_weight / flip_weight
                if flip_weight > 0.0
                else None
            ),
            "policy_l1_movement": _finite_summary(l1[stratum]),
            "policy_linf_movement": _finite_summary(linf[stratum]),
        }
    return {
        "candidate_kappa": candidate_kappa,
        "delta_log_kappa": math.log(
            candidate_kappa / arrays.reference_kappa
        ),
        "reference_margin_eligible_roots": int(np.sum(eligible)),
        "top_action_flip_count": total_flips,
        "top_action_flip_rate": (
            float(np.mean(flips[eligible])) if np.any(eligible) else None
        ),
        "occupancy_weighted_top_action_flip_rate": (
            flip_weight / eligible_weight
            if eligible_weight > 0.0
            else None
        ),
        "necessary_movement_bounds": {
            "l1_bound": "top-action flip implies ||delta policy||_1 >= margin",
            "linf_bound": (
                "top-action flip implies "
                "||delta policy||_inf >= margin/2"
            ),
            "tolerance": MARGIN_BOUND_TOLERANCE,
            "l1_violation_count": int(np.sum(l1_violations)),
            "l1_violation_root_ids": arrays.root_ids[
                l1_violations
            ].tolist(),
            "linf_violation_count": int(np.sum(linf_violations)),
            "linf_violation_root_ids": arrays.root_ids[
                linf_violations
            ].tolist(),
        },
        "by_reference_margin": strata,
    }


def analyze(
    *,
    payload: Mapping[str, Any],
    roots: Mapping[str, np.ndarray],
    local_log_step: float = DEFAULT_LOCAL_LOG_STEP,
    comparison_kappas: Sequence[float] = DEFAULT_COMPARISON_KAPPAS,
    bootstrap_replicates: int = DEFAULT_BOOTSTRAP_REPLICATES,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    if not math.isfinite(local_log_step) or local_log_step <= 0.0:
        raise ValueError("local_log_step must be finite and positive")
    comparison_kappas = tuple(float(kappa) for kappa in comparison_kappas)
    if (
        not comparison_kappas
        or len(set(comparison_kappas)) != len(comparison_kappas)
        or any(
            not math.isfinite(kappa) or kappa <= 0.0
            for kappa in comparison_kappas
        )
    ):
        raise ValueError(
            "comparison_kappas must be unique finite positive values"
        )
    if bootstrap_replicates < 200:
        raise ValueError("bootstrap_replicates must be at least 200")
    arrays = _extract_probe_arrays(
        payload=payload,
        roots=roots,
        local_log_step=local_log_step,
        comparison_kappas=comparison_kappas,
    )
    stage_masks = {
        "overall": np.ones(len(arrays.root_ids), dtype=bool),
        **{
            name: arrays.stage_ids == stage_id
            for stage_id, name in enumerate(STAGE_NAMES)
        },
    }
    scope_masks = {
        DEPLOYMENT_SCOPE: np.ones(len(arrays.root_ids), dtype=bool),
        CHANNEL_ACTIVE_SCOPE: ~arrays.reference_solved_roots,
    }
    scope_seeds = {
        DEPLOYMENT_SCOPE: bootstrap_seed,
        CHANNEL_ACTIVE_SCOPE: bootstrap_seed + 1,
    }
    scoped_masks = {
        scope: {
            name: stage_mask & scope_mask
            for name, stage_mask in stage_masks.items()
        }
        for scope, scope_mask in scope_masks.items()
    }
    central: dict[str, dict[str, Any]] = {}
    for scope, masks in scoped_masks.items():
        rng = np.random.default_rng(scope_seeds[scope])
        central[scope] = {
            name: _central_response_summary(
                arrays=arrays,
                mask=mask,
                replicates=bootstrap_replicates,
                rng=rng,
            )
            for name, mask in masks.items()
        }
    derivative_bounds = {
        _kappa_key(kappa): {
            name: _derivative_bound_summary(
                arrays=arrays,
                kappa=kappa,
                mask=mask,
            )
            for name, mask in stage_masks.items()
        }
        for kappa in arrays.policies
    }
    derivative_deciles = {
        scope: {
            name: _derivative_deciles(arrays=arrays, mask=mask)
            for name, mask in masks.items()
        }
        for scope, masks in scoped_masks.items()
    }
    flip_kappas = (
        arrays.local_minus_kappa,
        arrays.local_plus_kappa,
        *arrays.comparison_kappas,
    )
    margin_flips = {
        _kappa_key(kappa): _margin_flip_summary(
            arrays=arrays,
            candidate_kappa=kappa,
        )
        for kappa in flip_kappas
    }
    return {
        "format": FORMAT,
        "protocol": {
            "reference_kappa": arrays.reference_kappa,
            "local_log_step": arrays.local_log_step,
            "local_minus_kappa": arrays.local_minus_kappa,
            "local_plus_kappa": arrays.local_plus_kappa,
            "comparison_kappas": list(arrays.comparison_kappas),
            "central_response": (
                "||pi(kappa*exp(h))-pi(kappa*exp(-h))||_1/(2h)"
            ),
            "rank_association": (
                "unweighted root-level Spearman correlation between the "
                "central root-policy L1 response and the reference root's "
                "mean normalized-cache derivative norm across numeric "
                "repair events"
            ),
            "rank_association_scopes": {
                DEPLOYMENT_SCOPE: (
                    "all roots with a finite reference derivative and at "
                    "least one reference numeric repair; final solved-root "
                    "bypasses remain in this end-to-end deployment estimand"
                ),
                CHANNEL_ACTIVE_SCOPE: (
                    "the same eligibility restricted to roots unresolved at "
                    "reference kappa; this conditions on the numeric "
                    "commitment channel remaining active at the reference"
                ),
            },
            "bootstrap": {
                "unit": "source self-play game_cluster_id from roots.npz",
                "replicates": bootstrap_replicates,
                "scope_seeds": scope_seeds,
                "interval": "percentile 95%",
            },
            "derivative_deciles": (
                "stable equal-count rank bins; root_id breaks derivative ties"
            ),
            "margin_strata": (
                "reference-policy tie, (0, one vote scale], "
                "(one, two vote scales], and above two vote scales"
            ),
        },
        "root_count": int(len(arrays.root_ids)),
        "reference_root_status": {
            "solved_bypass_roots": int(
                np.sum(arrays.reference_solved_roots)
            ),
            "unresolved_numeric_channel_roots": int(
                np.sum(~arrays.reference_solved_roots)
            ),
            "by_stage": {
                name: {
                    "roots": int(np.sum(mask)),
                    "solved_bypass_roots": int(
                        np.sum(mask & arrays.reference_solved_roots)
                    ),
                    "unresolved_numeric_channel_roots": int(
                        np.sum(mask & ~arrays.reference_solved_roots)
                    ),
                }
                for name, mask in stage_masks.items()
                if name != "overall"
            },
        },
        "central_local_response": central,
        "raw_derivative_bound_audit": derivative_bounds,
        "reference_normalized_derivative_deciles": derivative_deciles,
        "margin_stratified_policy_flips": margin_flips,
    }


def _write_immutable(path: Path, payload: Mapping[str, Any]) -> str:
    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise FileExistsError(
            f"refusing to overwrite analysis artifact: {resolved}"
        )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=resolved.parent,
        prefix=f".{resolved.name}.",
        suffix=".tmp",
    )
    try:
        with open(descriptor, "w", encoding="utf-8", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
        Path(temporary_name).replace(resolved)
    except BaseException:
        Path(temporary_name).unlink(missing_ok=True)
        raise
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--roots", type=Path, required=True)
    parser.add_argument(
        "--local-log-step",
        type=float,
        default=DEFAULT_LOCAL_LOG_STEP,
    )
    parser.add_argument(
        "--comparison-kappas",
        type=_parse_kappas,
        default=DEFAULT_COMPARISON_KAPPAS,
    )
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPLICATES,
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    payload = json.loads(args.probe.read_text(encoding="utf-8"))
    with np.load(args.roots, allow_pickle=False) as archive:
        roots = {name: np.asarray(archive[name]) for name in archive.files}
    result = analyze(
        payload=payload,
        roots=roots,
        local_log_step=args.local_log_step,
        comparison_kappas=args.comparison_kappas,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    result["sources"] = {
        "probe": {
            "path": str(args.probe.resolve()),
            "sha256": _sha256(args.probe),
        },
        "roots": {
            "path": str(args.roots.resolve()),
            "sha256": _sha256(args.roots),
        },
    }
    result["execution"] = {
        "command": shlex.join(sys.argv),
        "python": sys.version,
        "platform": platform.platform(),
        "external_logging": False,
    }
    digest = _write_immutable(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": digest,
                "root_count": result["root_count"],
                "central_local_response": result[
                    "central_local_response"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
