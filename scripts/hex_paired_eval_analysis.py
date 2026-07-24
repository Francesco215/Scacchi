#!/usr/bin/env python3
"""Compare two matched outputs from ``hex_balanced_eval.py``.

The balanced evaluator can preserve every game return together with the RNG
and row coordinates that generated it.  This script verifies that two result
artifacts really use the same coordinates before treating their outcomes as
paired observations.

Only the Python standard library is required.  The primary estimand is the
treatment-minus-control change in candidate win probability.  Confidence
intervals use the matched differences and preserve the evaluator's fixed
logical-player-id x seat strata.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from decimal import Decimal, localcontext
import hashlib
import json
import math
from pathlib import Path
import platform
import shlex
import sys
from collections.abc import Mapping
from typing import Any, Callable, Sequence


ANALYSIS_SCHEMA_VERSION = 1
GAME_RETURNS_KIND = "scacchi.hex_balanced_eval_game_returns"
RESULT_KIND = "scacchi.hex_balanced_eval_paired_comparison"
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class GameCoordinate:
    """Everything needed to identify one matched evaluator row."""

    global_game_index: int
    stratum_index: int
    candidate_player_id: int
    candidate_first: bool
    candidate_seat: str
    chunk_index: int
    run_key_index: int
    rng_key_data: tuple[int, ...]
    row_index: int


@dataclass(frozen=True)
class GameObservation:
    coordinate: GameCoordinate
    candidate_return: int

    @property
    def candidate_won(self) -> int:
        return int(self.candidate_return > 0)


@dataclass(frozen=True)
class LoadedResult:
    path: Path
    file_sha256: str
    result: dict[str, Any]
    game_returns: dict[str, Any]
    observations: tuple[GameObservation, ...]
    recomputed_layout_sha256: str


def _canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    return value


def _require_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    return value


def _require_str(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a string")
    return value


def _validate_summary(
    summary: Any,
    observations: Sequence[GameObservation],
    location: str,
) -> None:
    summary = _require_mapping(summary, location)
    games = len(observations)
    wins = sum(item.candidate_won for item in observations)
    draws = sum(item.candidate_return == 0 for item in observations)
    losses = games - wins - draws
    expected_counts = {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
    }
    for field, expected in expected_counts.items():
        actual = _require_int(summary.get(field), f"{location}.{field}")
        if actual != expected:
            raise ValueError(
                f"{location}.{field}={actual} disagrees with raw "
                f"returns ({expected})"
            )
    expected_win_rate = wins / games
    actual_win_rate = summary.get("win_rate")
    if not isinstance(actual_win_rate, (int, float)) or not math.isclose(
        float(actual_win_rate),
        expected_win_rate,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            f"{location}.win_rate disagrees with raw returns"
        )


def _decode_game_returns(
    payload: dict[str, Any],
) -> tuple[tuple[GameObservation, ...], str]:
    if payload.get("kind") != GAME_RETURNS_KIND:
        raise ValueError(
            "game_returns.kind must be "
            f"{GAME_RETURNS_KIND!r}"
        )
    if _require_int(
        payload.get("schema_version"),
        "game_returns.schema_version",
    ) != 1:
        raise ValueError("unsupported game_returns schema_version")

    seed = _require_int(payload.get("seed"), "game_returns.seed")
    games = _require_int(payload.get("games"), "game_returns.games")
    games_per_stratum = _require_int(
        payload.get("games_per_stratum"),
        "game_returns.games_per_stratum",
    )
    batch_size = _require_int(
        payload.get("batch_size"),
        "game_returns.batch_size",
    )
    num_chunks = _require_int(
        payload.get("num_chunks_per_stratum"),
        "game_returns.num_chunks_per_stratum",
    )
    if min(games, games_per_stratum, batch_size, num_chunks) <= 0:
        raise ValueError("game-return dimensions must be positive")
    if games != 4 * games_per_stratum:
        raise ValueError(
            "game_returns must contain four equally sized strata"
        )
    if games_per_stratum != batch_size * num_chunks:
        raise ValueError(
            "games_per_stratum must equal batch_size * "
            "num_chunks_per_stratum"
        )

    strata_value = payload.get("strata")
    if not isinstance(strata_value, list) or len(strata_value) != 4:
        raise ValueError(
            "game_returns.strata must contain the four canonical strata"
        )
    expected_strata = (
        (0, True, "first"),
        (0, False, "second"),
        (1, True, "first"),
        (1, False, "second"),
    )
    observations: list[GameObservation] = []
    layout_chunks: list[dict[str, Any]] = []
    stratum_order: list[dict[str, Any]] = []
    expected_global_index = 0

    for stratum_index, (stratum_value, expected_stratum) in enumerate(
        zip(strata_value, expected_strata, strict=True)
    ):
        location = f"game_returns.strata[{stratum_index}]"
        stratum = _require_mapping(stratum_value, location)
        actual_index = _require_int(
            stratum.get("stratum_index"),
            f"{location}.stratum_index",
        )
        candidate_player_id = _require_int(
            stratum.get("candidate_player_id"),
            f"{location}.candidate_player_id",
        )
        candidate_first = stratum.get("candidate_first")
        if not isinstance(candidate_first, bool):
            raise ValueError(f"{location}.candidate_first must be boolean")
        candidate_seat = _require_str(
            stratum.get("candidate_seat"),
            f"{location}.candidate_seat",
        )
        actual_stratum = (
            candidate_player_id,
            candidate_first,
            candidate_seat,
        )
        if actual_index != stratum_index or actual_stratum != expected_stratum:
            raise ValueError(
                f"{location} is not the expected canonical stratum "
                f"{(stratum_index, *expected_stratum)!r}"
            )
        stratum_order.append(
            {
                "candidate_player_id": candidate_player_id,
                "candidate_first": candidate_first,
                "candidate_seat": candidate_seat,
            }
        )

        chunks_value = stratum.get("chunks")
        if not isinstance(chunks_value, list) or len(chunks_value) != num_chunks:
            raise ValueError(
                f"{location}.chunks must contain {num_chunks} chunks"
            )
        stratum_observations: list[GameObservation] = []
        for chunk_index, chunk_value in enumerate(chunks_value):
            chunk_location = f"{location}.chunks[{chunk_index}]"
            chunk = _require_mapping(chunk_value, chunk_location)
            actual_chunk_index = _require_int(
                chunk.get("chunk_index"),
                f"{chunk_location}.chunk_index",
            )
            run_key_index = _require_int(
                chunk.get("run_key_index"),
                f"{chunk_location}.run_key_index",
            )
            global_start = _require_int(
                chunk.get("global_game_index_start"),
                f"{chunk_location}.global_game_index_start",
            )
            game_count = _require_int(
                chunk.get("game_count"),
                f"{chunk_location}.game_count",
            )
            rng_value = chunk.get("rng_key_data")
            if not isinstance(rng_value, list) or not rng_value:
                raise ValueError(
                    f"{chunk_location}.rng_key_data must be a nonempty list"
                )
            rng_key_data = tuple(
                _require_int(value, f"{chunk_location}.rng_key_data")
                for value in rng_value
            )
            expected_run_key_index = (
                stratum_index * num_chunks + chunk_index
            )
            if actual_chunk_index != chunk_index:
                raise ValueError(
                    f"{chunk_location}.chunk_index is not canonical"
                )
            if run_key_index != expected_run_key_index:
                raise ValueError(
                    f"{chunk_location}.run_key_index is not canonical"
                )
            if global_start != expected_global_index:
                raise ValueError(
                    f"{chunk_location}.global_game_index_start is not "
                    "contiguous"
                )
            if game_count != batch_size:
                raise ValueError(
                    f"{chunk_location}.game_count must equal batch_size"
                )
            returns = chunk.get("returns")
            if not isinstance(returns, list) or len(returns) != game_count:
                raise ValueError(
                    f"{chunk_location}.returns length must equal game_count"
                )
            layout_chunks.append(
                {
                    "stratum_index": stratum_index,
                    "chunk_index": chunk_index,
                    "run_key_index": run_key_index,
                    "rng_key_data": list(rng_key_data),
                    "global_game_index_start": global_start,
                    "game_count": game_count,
                }
            )
            for row_index, return_value in enumerate(returns):
                candidate_return = _require_int(
                    return_value,
                    f"{chunk_location}.returns[{row_index}]",
                )
                if candidate_return not in (-1, 0, 1):
                    raise ValueError(
                        "candidate returns must be encoded as -1, 0, or 1"
                    )
                observation = GameObservation(
                    coordinate=GameCoordinate(
                        global_game_index=global_start + row_index,
                        stratum_index=stratum_index,
                        candidate_player_id=candidate_player_id,
                        candidate_first=candidate_first,
                        candidate_seat=candidate_seat,
                        chunk_index=chunk_index,
                        run_key_index=run_key_index,
                        rng_key_data=rng_key_data,
                        row_index=row_index,
                    ),
                    candidate_return=candidate_return,
                )
                observations.append(observation)
                stratum_observations.append(observation)
                expected_global_index += 1
        _validate_summary(
            stratum.get("summary"),
            stratum_observations,
            f"{location}.summary",
        )

    if expected_global_index != games:
        raise ValueError("raw return count disagrees with game_returns.games")

    layout_material = {
        "seed": seed,
        "games": games,
        "games_per_stratum": games_per_stratum,
        "batch_size": batch_size,
        "stratum_order": stratum_order,
        "chunks": layout_chunks,
    }
    recomputed_layout_sha256 = _canonical_sha256(layout_material)
    declared_layout_sha256 = _require_str(
        payload.get("pairing_layout_sha256"),
        "game_returns.pairing_layout_sha256",
    )
    if declared_layout_sha256 != recomputed_layout_sha256:
        raise ValueError(
            "declared pairing_layout_sha256 does not match the canonical "
            "layout reconstructed from coordinates"
        )

    declared_returns_sha256 = _require_str(
        payload.get("returns_sha256"),
        "game_returns.returns_sha256",
    )
    flat_returns = [item.candidate_return for item in observations]
    recomputed_returns_sha256 = hashlib.sha256(
        json.dumps(flat_returns, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if declared_returns_sha256 != recomputed_returns_sha256:
        raise ValueError(
            "declared returns_sha256 does not match serialized returns"
        )
    _validate_summary(
        payload.get("overall"),
        observations,
        "game_returns.overall",
    )
    return tuple(observations), recomputed_layout_sha256


def load_result(path: Path) -> LoadedResult:
    """Load and validate one complete balanced-evaluation result."""

    resolved = path.resolve()
    encoded = resolved.read_bytes()
    try:
        value = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise ValueError(f"{resolved} is not valid JSON: {error}") from error
    result = _require_mapping(value, str(resolved))
    game_returns = _require_mapping(
        result.get("game_returns"),
        f"{resolved}.game_returns",
    )
    observations, layout_sha256 = _decode_game_returns(game_returns)
    for field in ("games", "games_per_stratum", "batch_size", "seed"):
        if field in result and result[field] != game_returns[field]:
            raise ValueError(
                f"{resolved}: top-level {field} disagrees with game_returns"
            )
    return LoadedResult(
        path=resolved,
        file_sha256=hashlib.sha256(encoded).hexdigest(),
        result=result,
        game_returns=game_returns,
        observations=observations,
        recomputed_layout_sha256=layout_sha256,
    )


VARYING_FACTOR_FIELDS = {
    "root_action_estimator": (
        "root_action_estimator",
        "candidate_root_action_estimator",
    ),
    "kappa": ("kappa", "candidate_kappa"),
    "checkpoint": (None, "candidate_checkpoint_selection"),
}


def _normalized_candidate_search(
    result: dict[str, Any],
    varying_factor: str,
) -> Any:
    monitor = result.get("monitor")
    if not isinstance(monitor, dict):
        return None
    candidate_search = monitor.get("candidate_search")
    if not isinstance(candidate_search, dict):
        return candidate_search
    normalized = dict(candidate_search)
    try:
        search_field, _ = VARYING_FACTOR_FIELDS[varying_factor]
    except KeyError as error:
        raise ValueError(
            f"unsupported varying factor {varying_factor!r}"
        ) from error
    if search_field is not None:
        normalized.pop(search_field, None)
    return normalized


def _shared_context_checks(
    control: LoadedResult,
    treatment: LoadedResult,
    varying_factor: str,
) -> list[dict[str, Any]]:
    """Validate invariant result-level fields when the harness recorded them."""

    comparisons: list[tuple[str, Any, Any]] = [
        (
            "baseline_checkpoint_selection",
            control.result.get("baseline_checkpoint_selection"),
            treatment.result.get("baseline_checkpoint_selection"),
        ),
        (
            "baseline_checkpoint_metadata",
            control.result.get("baseline_checkpoint_metadata"),
            treatment.result.get("baseline_checkpoint_metadata"),
        ),
        (
            "environment",
            control.result.get("environment"),
            treatment.result.get("environment"),
        ),
        (
            "role_and_rng_construction",
            control.result.get("role_and_rng_construction"),
            treatment.result.get("role_and_rng_construction"),
        ),
        (
            f"candidate_search_except_{varying_factor}",
            _normalized_candidate_search(
                control.result,
                varying_factor,
            ),
            _normalized_candidate_search(
                treatment.result,
                varying_factor,
            ),
        ),
    ]
    if varying_factor != "checkpoint":
        comparisons[:0] = [
            (
                "candidate_checkpoint_selection",
                control.result.get("candidate_checkpoint_selection"),
                treatment.result.get("candidate_checkpoint_selection"),
            ),
            (
                "candidate_checkpoint_metadata",
                control.result.get("candidate_checkpoint_metadata"),
                treatment.result.get("candidate_checkpoint_metadata"),
            ),
        ]
    for monitor_field in (
        "candidate_action_commitment",
        "opponent_search",
        "opponent_action_commitment",
    ):
        control_monitor = control.result.get("monitor")
        treatment_monitor = treatment.result.get("monitor")
        control_value = (
            control_monitor.get(monitor_field)
            if isinstance(control_monitor, dict)
            else None
        )
        treatment_value = (
            treatment_monitor.get(monitor_field)
            if isinstance(treatment_monitor, dict)
            else None
        )
        comparisons.append(
            (f"monitor.{monitor_field}", control_value, treatment_value)
        )

    checks: list[dict[str, Any]] = []
    for field, control_value, treatment_value in comparisons:
        available = control_value is not None or treatment_value is not None
        matched = control_value == treatment_value
        checks.append(
            {
                "field": field,
                "available": available,
                "matched": matched if available else None,
            }
        )
        if available and not matched:
            raise ValueError(
                f"paired artifacts differ in invariant field {field}"
            )
    return checks


def validate_pairing(
    control: LoadedResult,
    treatment: LoadedResult,
    *,
    varying_factor: str = "root_action_estimator",
) -> dict[str, Any]:
    """Require both the canonical layout digest and every row coordinate."""

    declared_control = control.game_returns["pairing_layout_sha256"]
    declared_treatment = treatment.game_returns[
        "pairing_layout_sha256"
    ]
    if declared_control != declared_treatment:
        raise ValueError(
            "pairing_layout_sha256 mismatch: the artifacts are not paired"
        )
    control_coordinates = tuple(
        item.coordinate for item in control.observations
    )
    treatment_coordinates = tuple(
        item.coordinate for item in treatment.observations
    )
    if control_coordinates != treatment_coordinates:
        mismatch = next(
            (
                index
                for index, (left, right) in enumerate(
                    zip(
                        control_coordinates,
                        treatment_coordinates,
                        strict=False,
                    )
                )
                if left != right
            ),
            min(len(control_coordinates), len(treatment_coordinates)),
        )
        raise ValueError(
            "per-game pairing coordinates differ at observation "
            f"{mismatch}"
        )
    try:
        _, result_field = VARYING_FACTOR_FIELDS[varying_factor]
    except KeyError as error:
        raise ValueError(
            f"unsupported varying factor {varying_factor!r}"
        ) from error
    context_checks = _shared_context_checks(
        control,
        treatment,
        varying_factor,
    )
    return {
        "valid": True,
        "pairing_layout_sha256": declared_control,
        "layout_digest_recomputed_for_both": True,
        "per_game_coordinates_compared": len(control_coordinates),
        "per_game_coordinates_identical": True,
        "shared_context_checks": context_checks,
        "deliberately_varying_factor": {
            "field": result_field,
            "control": control.result.get(result_field),
            "treatment": treatment.result.get(result_field),
        },
    }


def exact_mcnemar_two_sided(
    control_only_wins: int,
    treatment_only_wins: int,
) -> float:
    """Exact conditional McNemar p-value under Binomial(n, 1/2)."""

    if control_only_wins < 0 or treatment_only_wins < 0:
        raise ValueError("discordant counts must be nonnegative")
    discordant = control_only_wins + treatment_only_wins
    numerator, denominator = _exact_mcnemar_fraction(
        control_only_wins,
        treatment_only_wins,
    )
    return numerator / denominator


def _exact_mcnemar_fraction(
    control_only_wins: int,
    treatment_only_wins: int,
) -> tuple[int, int]:
    """Return the exact p-value as a (possibly unreduced) integer ratio."""

    if control_only_wins < 0 or treatment_only_wins < 0:
        raise ValueError("discordant counts must be nonnegative")
    discordant = control_only_wins + treatment_only_wins
    if discordant == 0:
        return (1, 1)
    smaller = min(control_only_wins, treatment_only_wins)
    lower_tail_numerator = sum(
        math.comb(discordant, value)
        for value in range(smaller + 1)
    )
    denominator = 1 << discordant
    numerator = min(denominator, 2 * lower_tail_numerator)
    return (numerator, denominator)


def _exact_mcnemar_result(
    control_only_wins: int,
    treatment_only_wins: int,
) -> dict[str, Any]:
    numerator, denominator = _exact_mcnemar_fraction(
        control_only_wins,
        treatment_only_wins,
    )
    numeric = numerator / denominator
    with localcontext() as context:
        context.prec = 18
        decimal_probability = Decimal(numerator) / Decimal(denominator)
    discordant = control_only_wins + treatment_only_wins
    return {
        "p_value_two_sided": numeric,
        "p_value_two_sided_decimal": format(
            decimal_probability,
            ".17E",
        ),
        "floating_point_underflow": numeric == 0.0 and numerator != 0,
        "integer_tail_evaluation": True,
        "discordant_trials": discordant,
        "smaller_direction_count": min(
            control_only_wins,
            treatment_only_wins,
        ),
        "null": (
            "conditional on discordance, each direction has "
            "probability 0.5"
        ),
        "success_definition": "candidate return > 0",
    }


def _stratified_paired_ci(
    differences_by_stratum: Mapping[int, Sequence[int | float]],
    *,
    bounds: tuple[float, float] = (-1.0, 1.0),
) -> dict[str, Any]:
    """Normal CI for an equally weighted mean of matched stratum means."""

    if not differences_by_stratum:
        raise ValueError("at least one stratum is required")
    stratum_means: list[float] = []
    variance_terms: list[float] = []
    sizes: list[int] = []
    complete_variance = True
    for stratum_index in sorted(differences_by_stratum):
        values = list(differences_by_stratum[stratum_index])
        if not values:
            raise ValueError(f"stratum {stratum_index} contains zero pairs")
        size = len(values)
        mean = sum(values) / size
        sizes.append(size)
        stratum_means.append(mean)
        if size < 2:
            complete_variance = False
            continue
        sample_variance = sum(
            (value - mean) ** 2 for value in values
        ) / (size - 1)
        variance_terms.append(sample_variance / size)
    estimate = sum(stratum_means) / len(stratum_means)
    result: dict[str, Any] = {
        "estimate": estimate,
        "method": (
            "paired stratified normal interval over fixed, equally "
            "weighted strata"
        ),
        "nominal_coverage": 0.95,
        "sidedness": "two-sided",
        "z": Z_95,
        "stratum_indices": sorted(differences_by_stratum),
        "pairs_per_stratum": sizes,
    }
    if not complete_variance:
        result.update(
            {
                "standard_error": None,
                "interval": None,
                "note": (
                    "interval unavailable because at least one fixed "
                    "stratum has fewer than two pairs"
                ),
            }
        )
        return result
    standard_error = math.sqrt(
        sum(variance_terms) / (len(stratum_means) ** 2)
    )
    result.update(
        {
            "standard_error": standard_error,
            "interval": [
                max(bounds[0], estimate - Z_95 * standard_error),
                min(bounds[1], estimate + Z_95 * standard_error),
            ],
        }
    )
    return result


def _method_summary(
    observations: Sequence[GameObservation],
) -> dict[str, Any]:
    games = len(observations)
    wins = sum(item.candidate_won for item in observations)
    draws = sum(item.candidate_return == 0 for item in observations)
    losses = games - wins - draws
    return {
        "games": games,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "win_rate": wins / games,
        "avg_return": (
            sum(item.candidate_return for item in observations) / games
        ),
    }


def _select(
    observations: Sequence[GameObservation],
    predicate: Callable[[GameCoordinate], bool],
) -> tuple[GameObservation, ...]:
    return tuple(
        item for item in observations if predicate(item.coordinate)
    )


def _grouped_differences(
    control: Sequence[GameObservation],
    treatment: Sequence[GameObservation],
    *,
    transform: Callable[[GameCoordinate, float], float] | None = None,
) -> dict[int, list[float]]:
    grouped: dict[int, list[float]] = {}
    for control_item, treatment_item in zip(
        control,
        treatment,
        strict=True,
    ):
        difference = float(
            treatment_item.candidate_won - control_item.candidate_won
        )
        if transform is not None:
            difference = transform(control_item.coordinate, difference)
        grouped.setdefault(
            control_item.coordinate.stratum_index,
            [],
        ).append(difference)
    return grouped


def _paired_summary(
    control: Sequence[GameObservation],
    treatment: Sequence[GameObservation],
) -> dict[str, Any]:
    if len(control) != len(treatment) or not control:
        raise ValueError("paired summaries require equal nonzero samples")
    control_wins = [item.candidate_won for item in control]
    treatment_wins = [item.candidate_won for item in treatment]
    both_win = sum(
        left == 1 and right == 1
        for left, right in zip(control_wins, treatment_wins, strict=True)
    )
    control_only = sum(
        left == 1 and right == 0
        for left, right in zip(control_wins, treatment_wins, strict=True)
    )
    treatment_only = sum(
        left == 0 and right == 1
        for left, right in zip(control_wins, treatment_wins, strict=True)
    )
    neither_win = len(control) - both_win - control_only - treatment_only
    control_rate = sum(control_wins) / len(control_wins)
    treatment_rate = sum(treatment_wins) / len(treatment_wins)
    confidence = _stratified_paired_ci(
        _grouped_differences(control, treatment)
    )
    delta = treatment_rate - control_rate
    if not math.isclose(
        confidence["estimate"],
        delta,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise ValueError(
            "fixed strata are not equally sized; paired estimand would "
            "not equal the reported pooled delta"
        )
    return {
        "games": len(control),
        "control_win_rate": control_rate,
        "treatment_win_rate": treatment_rate,
        "delta_win_rate": delta,
        "discordance_table": {
            "both_win": both_win,
            "control_win_treatment_not_win": control_only,
            "control_not_win_treatment_win": treatment_only,
            "neither_win": neither_win,
            "discordant_total": control_only + treatment_only,
        },
        "exact_mcnemar": _exact_mcnemar_result(
            control_only,
            treatment_only,
        ),
        "delta_win_rate_ci95": confidence,
    }


def _seat_rates(
    observations: Sequence[GameObservation],
) -> dict[str, dict[str, Any]]:
    return {
        seat: _method_summary(
            _select(
                observations,
                lambda coordinate, seat=seat: (
                    coordinate.candidate_seat == seat
                ),
            )
        )
        for seat in ("first", "second")
    }


def _seat_optimal_error(
    seat_rates: dict[str, dict[str, Any]],
) -> float:
    first_win_rate = float(seat_rates["first"]["win_rate"])
    second_win_rate = float(seat_rates["second"]["win_rate"])
    return ((1.0 - first_win_rate) + second_win_rate) / 2.0


def analyze_pair(
    control: LoadedResult,
    treatment: LoadedResult,
    *,
    control_label: str,
    treatment_label: str,
    varying_factor: str = "root_action_estimator",
) -> dict[str, Any]:
    pairing = validate_pairing(
        control,
        treatment,
        varying_factor=varying_factor,
    )
    control_observations = control.observations
    treatment_observations = treatment.observations
    control_seats = _seat_rates(control_observations)
    treatment_seats = _seat_rates(treatment_observations)
    control_error = _seat_optimal_error(control_seats)
    treatment_error = _seat_optimal_error(treatment_seats)

    paired_by_seat: dict[str, Any] = {}
    for seat in ("first", "second"):
        predicate = lambda coordinate, seat=seat: (
            coordinate.candidate_seat == seat
        )
        paired_by_seat[seat] = _paired_summary(
            _select(control_observations, predicate),
            _select(treatment_observations, predicate),
        )

    paired_by_stratum: list[dict[str, Any]] = []
    for stratum_index in range(4):
        predicate = lambda coordinate, index=stratum_index: (
            coordinate.stratum_index == index
        )
        summary = _paired_summary(
            _select(control_observations, predicate),
            _select(treatment_observations, predicate),
        )
        coordinate = next(
            item.coordinate
            for item in control_observations
            if item.coordinate.stratum_index == stratum_index
        )
        summary.update(
            {
                "stratum_index": stratum_index,
                "candidate_player_id": coordinate.candidate_player_id,
                "candidate_seat": coordinate.candidate_seat,
            }
        )
        paired_by_stratum.append(summary)

    error_differences = _grouped_differences(
        control_observations,
        treatment_observations,
        transform=lambda coordinate, difference: (
            -difference
            if coordinate.candidate_seat == "first"
            else difference
        ),
    )
    error_confidence = _stratified_paired_ci(error_differences)
    error_delta = treatment_error - control_error
    if not math.isclose(
        error_confidence["estimate"],
        error_delta,
        rel_tol=0.0,
        abs_tol=1e-15,
    ):
        raise AssertionError(
            "seat-optimal error contrast disagrees with paired transform"
        )

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "local_only": {
            "network_access": False,
            "external_logging": False,
            "external_export": False,
            "artifact_is_local": True,
        },
        "inputs": {
            "control": {
                "label": control_label,
                "path": str(control.path),
                "file_sha256": control.file_sha256,
                "returns_sha256": control.game_returns["returns_sha256"],
                "candidate_root_action_estimator": control.result.get(
                    "candidate_root_action_estimator"
                ),
                "candidate_kappa": control.result.get("candidate_kappa"),
                "candidate_prefix_cdf_grid": control.result.get(
                    "candidate_prefix_cdf_grid"
                ),
                "candidate_checkpoint_selection": control.result.get(
                    "candidate_checkpoint_selection"
                ),
            },
            "treatment": {
                "label": treatment_label,
                "path": str(treatment.path),
                "file_sha256": treatment.file_sha256,
                "returns_sha256": treatment.game_returns["returns_sha256"],
                "candidate_root_action_estimator": treatment.result.get(
                    "candidate_root_action_estimator"
                ),
                "candidate_kappa": treatment.result.get("candidate_kappa"),
                "candidate_prefix_cdf_grid": treatment.result.get(
                    "candidate_prefix_cdf_grid"
                ),
                "candidate_checkpoint_selection": treatment.result.get(
                    "candidate_checkpoint_selection"
                ),
            },
        },
        "pairing_validation": pairing,
        "methods": {
            "control": {
                "label": control_label,
                "overall": _method_summary(control_observations),
                "by_seat": control_seats,
                "seat_optimal_error": control_error,
            },
            "treatment": {
                "label": treatment_label,
                "overall": _method_summary(treatment_observations),
                "by_seat": treatment_seats,
                "seat_optimal_error": treatment_error,
            },
        },
        "paired": {
            "contrast": f"{treatment_label} - {control_label}",
            "overall": _paired_summary(
                control_observations,
                treatment_observations,
            ),
            "by_seat": paired_by_seat,
            "by_stratum": paired_by_stratum,
            "seat_optimal_error": {
                "definition": (
                    "((1 - first-seat win rate) + "
                    "second-seat win rate) / 2"
                ),
                "interpretation": (
                    "Hex has no draws and optimal play is a first-seat win; "
                    "zero is seat-optimal and lower is better"
                ),
                "control": control_error,
                "treatment": treatment_error,
                "delta": error_delta,
                "delta_ci95": error_confidence,
            },
        },
        "statistics_contract": {
            "paired_unit": (
                "same stratum, chunk RNG key, and row index"
            ),
            "win_definition": "candidate return > 0",
            "delta_direction": "treatment minus control",
            "confidence_interval": (
                "matched-difference normal interval, with fixed canonical "
                "strata equally weighted; no independence assumption is "
                "made between paired methods"
            ),
            "mcnemar": (
                "exact two-sided conditional Binomial(discordant, 0.5) test"
            ),
        },
    }


def _write_json(path: Path, payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and compare two game-for-game paired outputs from "
            "scripts/hex_balanced_eval.py."
        )
    )
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--treatment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--control-label", default="control")
    parser.add_argument("--treatment-label", default="treatment")
    parser.add_argument(
        "--varying-factor",
        choices=tuple(VARYING_FACTOR_FIELDS),
        default="root_action_estimator",
        help=(
            "the sole candidate search field allowed to differ between "
            "paired artifacts"
        ),
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="write the full artifact but print only its path and digest",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    control = load_result(args.control)
    treatment = load_result(args.treatment)
    result = analyze_pair(
        control,
        treatment,
        control_label=args.control_label,
        treatment_label=args.treatment_label,
        varying_factor=args.varying_factor,
    )
    result["reproduction"] = {
        "command": " ".join(
            shlex.quote(argument)
            for argument in (
                sys.argv
                if argv is None
                else [sys.argv[0], *argv]
            )
        ),
        "working_directory": str(Path.cwd().resolve()),
        "python": platform.python_version(),
        "script_path": str(Path(__file__).resolve()),
        "script_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
    }
    output_sha256 = _write_json(args.output, result)
    if not args.quiet:
        print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    print(
        f"Wrote {args.output.resolve()} sha256={output_sha256}",
        flush=True,
    )


if __name__ == "__main__":
    main()
