#!/usr/bin/env python3
"""Analyze a seat-balanced league of pairwise Hex checkpoint evaluations.

The primary output is the directed, seat-conditioned matrix

    P(row checkpoint wins | row has first/second seat, column opponent).

Balanced pooled scores, three-cycles, and a scalar rating are deliberately
secondary summaries.  A scalar cannot faithfully represent a non-transitive
league, and an unconditioned score can hide failures to convert Hex's
first-seat advantage.

Inputs may be pair artifacts produced by ``hex_checkpoint_league.py`` or a
manifest/run-summary containing references to such artifacts.  Multiple
artifacts for the same unordered pair are treated as independent replicate
batches and their counts are aggregated.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import platform
import shlex
import statistics
import sys
from typing import Any

import numpy as np


ANALYSIS_SCHEMA_VERSION = 1
PAIR_KIND = "scacchi.hex_checkpoint_league_pair"
MANIFEST_KIND = "scacchi.hex_checkpoint_league_manifest"
RUN_KIND = "scacchi.hex_checkpoint_league_run"
RESULT_KIND = "scacchi.hex_checkpoint_league_analysis"
Z_90 = 1.6448536269514722
Z_95 = 1.959963984540054
DEFAULT_EQUIVALENCE_MARGIN = 0.05


@dataclass(frozen=True)
class OutcomeSummary:
    """Sufficient statistics from one fixed directed seat cell."""

    games: int
    wins: int
    draws: int
    losses: int

    @classmethod
    def from_json(cls, value: Any, location: str) -> OutcomeSummary:
        mapping = _require_mapping(value, location)
        summary = cls(
            games=_require_int(mapping.get("games"), f"{location}.games"),
            wins=_require_int(mapping.get("wins"), f"{location}.wins"),
            draws=_require_int(mapping.get("draws"), f"{location}.draws"),
            losses=_require_int(mapping.get("losses"), f"{location}.losses"),
        )
        if min(
            summary.games,
            summary.wins,
            summary.draws,
            summary.losses,
        ) < 0:
            raise ValueError(f"{location} contains a negative count")
        if summary.games <= 0:
            raise ValueError(f"{location}.games must be positive")
        if (
            summary.wins + summary.draws + summary.losses
            != summary.games
        ):
            raise ValueError(
                f"{location} wins+draws+losses disagrees with games"
            )
        _validate_reported_rate(mapping, "win_rate", summary.win_rate, location)
        return summary

    @property
    def win_rate(self) -> float:
        return self.wins / self.games

    @property
    def score_rate(self) -> float:
        return (self.wins + 0.5 * self.draws) / self.games

    def reverse(self) -> OutcomeSummary:
        """Return the opponent's summary from the same games."""

        return OutcomeSummary(
            games=self.games,
            wins=self.losses,
            draws=self.draws,
            losses=self.wins,
        )

    def __add__(self, other: OutcomeSummary) -> OutcomeSummary:
        return OutcomeSummary(
            games=self.games + other.games,
            wins=self.wins + other.wins,
            draws=self.draws + other.draws,
            losses=self.losses + other.losses,
        )

    def counts_json(self) -> dict[str, int]:
        return {
            "games": self.games,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
        }


@dataclass(frozen=True)
class PairEvidence:
    competitor_a: str
    competitor_b: str
    a_first: OutcomeSummary
    a_second: OutcomeSummary
    path: Path
    file_sha256: str


@dataclass(frozen=True)
class ArtifactReference:
    path: Path
    expected_sha256: str | None
    source_manifest: Path | None


@dataclass(frozen=True)
class LoadedInputs:
    evidence: tuple[PairEvidence, ...]
    roster_ids: tuple[str, ...]
    manifests: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class RatingObservation:
    first_player: int
    second_player: int
    games: int
    first_score: float


def _require_mapping(value: Any, location: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    return value


def _require_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    return value


def _require_str(value: Any, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a nonempty string")
    return value


def _validate_reported_rate(
    mapping: Mapping[str, Any],
    field: str,
    expected: float,
    location: str,
) -> None:
    if field not in mapping:
        return
    actual = mapping[field]
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(float(actual))
        or not math.isclose(
            float(actual),
            expected,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
    ):
        raise ValueError(f"{location}.{field} disagrees with counts")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def wilson_interval(
    wins: int,
    games: int,
    *,
    z: float = Z_95,
) -> tuple[float, float]:
    """Two-sided Wilson score interval for a Bernoulli win probability."""

    if games <= 0:
        raise ValueError("Wilson interval requires positive games")
    if wins < 0 or wins > games:
        raise ValueError("Wilson wins must lie in [0, games]")
    probability = wins / games
    z_squared = z * z
    denominator = 1.0 + z_squared / games
    centre = (
        probability + z_squared / (2.0 * games)
    ) / denominator
    radius = (
        z
        * math.sqrt(
            probability * (1.0 - probability) / games
            + z_squared / (4.0 * games * games)
        )
        / denominator
    )
    return (max(0.0, centre - radius), min(1.0, centre + radius))


def wilson_lower_bound(
    wins: int,
    games: int,
    *,
    z: float,
) -> float:
    return wilson_interval(wins, games, z=z)[0]


def _summary_equal(left: OutcomeSummary, right: OutcomeSummary) -> bool:
    return left == right


def _competitor_id(
    competitors: Mapping[str, Any],
    side: str,
    path: Path,
) -> str:
    value = competitors.get(side)
    mapping = _require_mapping(value, f"{path}.competitors.{side}")
    return _require_str(
        mapping.get("id"),
        f"{path}.competitors.{side}.id",
    )


def _looks_like_perspective(value: Any) -> bool:
    return (
        isinstance(value, dict)
        and isinstance(value.get("by_seat"), dict)
        and "first" in value["by_seat"]
        and "second" in value["by_seat"]
    )


def _find_perspective(
    pairwise: Mapping[str, Any],
    *,
    side: str,
    competitor_id: str,
) -> Mapping[str, Any] | None:
    candidates: list[Any] = [
        pairwise.get(f"competitor_{side}"),
        pairwise.get(side),
        pairwise.get(competitor_id),
    ]
    by_competitor = pairwise.get("by_competitor")
    if isinstance(by_competitor, dict):
        candidates.extend(
            [
                by_competitor.get(f"competitor_{side}"),
                by_competitor.get(side),
                by_competitor.get(competitor_id),
            ]
        )
    for candidate in candidates:
        if _looks_like_perspective(candidate):
            return candidate
    return None


def _summary_from_nested(
    value: Any,
    location: str,
) -> OutcomeSummary:
    mapping = _require_mapping(value, location)
    if "summary" in mapping:
        return OutcomeSummary.from_json(
            mapping["summary"],
            f"{location}.summary",
        )
    return OutcomeSummary.from_json(mapping, location)


def _aggregate_strata_fallback(
    artifact: Mapping[str, Any],
    *,
    side: str,
    path: Path,
) -> dict[str, OutcomeSummary]:
    strata = artifact.get("strata")
    if not isinstance(strata, list) or not strata:
        raise ValueError(
            f"{path} has neither usable pairwise perspectives nor strata"
        )
    grouped: dict[str, OutcomeSummary] = {}
    for index, value in enumerate(strata):
        location = f"{path}.strata[{index}]"
        stratum = _require_mapping(value, location)
        seat_value = (
            stratum.get(f"competitor_{side}_seat")
            if f"competitor_{side}_seat" in stratum
            else stratum.get("candidate_seat")
        )
        seat = _require_str(seat_value, f"{location}.seat")
        if seat not in ("first", "second"):
            raise ValueError(f"{location}.seat must be first or second")
        summary_value: Any = None
        for key in (
            f"competitor_{side}",
            f"competitor_{side}_summary",
            side,
        ):
            if key in stratum:
                summary_value = stratum[key]
                break
        if summary_value is None and side == "a" and "summary" in stratum:
            summary_value = stratum["summary"]
        if summary_value is None:
            raise ValueError(
                f"{location} lacks a competitor_{side} summary"
            )
        summary = _summary_from_nested(
            summary_value,
            f"{location}.competitor_{side}",
        )
        grouped[seat] = grouped.get(seat, OutcomeSummary(0, 0, 0, 0)) + summary
    if set(grouped) != {"first", "second"}:
        raise ValueError(f"{path} does not cover both seats")
    return grouped


def _extract_perspective_summaries(
    artifact: Mapping[str, Any],
    *,
    side: str,
    competitor_id: str,
    path: Path,
) -> dict[str, OutcomeSummary] | None:
    pairwise_value = artifact.get("pairwise")
    if isinstance(pairwise_value, dict):
        perspective = _find_perspective(
            pairwise_value,
            side=side,
            competitor_id=competitor_id,
        )
        if perspective is not None:
            by_seat = _require_mapping(
                perspective.get("by_seat"),
                f"{path}.pairwise.competitor_{side}.by_seat",
            )
            return {
                seat: _summary_from_nested(
                    by_seat.get(seat),
                    (
                        f"{path}.pairwise.competitor_{side}"
                        f".by_seat.{seat}"
                    ),
                )
                for seat in ("first", "second")
            }
    try:
        return _aggregate_strata_fallback(
            artifact,
            side=side,
            path=path,
        )
    except ValueError:
        if side == "b":
            return None
        raise


def load_pair_artifact(
    path: Path,
    *,
    expected_sha256: str | None = None,
) -> PairEvidence:
    resolved = path.resolve()
    digest = file_sha256(resolved)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(
            f"{resolved} sha256={digest} does not match manifest "
            f"sha256={expected_sha256}"
        )
    artifact = _require_mapping(
        json.loads(resolved.read_text(encoding="utf-8")),
        str(resolved),
    )
    if artifact.get("kind") != PAIR_KIND:
        raise ValueError(
            f"{resolved}.kind must be {PAIR_KIND!r}"
        )
    schema_version = _require_int(
        artifact.get("schema_version"),
        f"{resolved}.schema_version",
    )
    if schema_version != 1:
        raise ValueError(
            f"{resolved} has unsupported pair schema {schema_version}"
        )
    competitors = _require_mapping(
        artifact.get("competitors"),
        f"{resolved}.competitors",
    )
    competitor_a = _competitor_id(competitors, "a", resolved)
    competitor_b = _competitor_id(competitors, "b", resolved)

    a_summaries = _extract_perspective_summaries(
        artifact,
        side="a",
        competitor_id=competitor_a,
        path=resolved,
    )
    if a_summaries is None:
        raise ValueError(f"{resolved} lacks competitor A summaries")
    b_summaries = _extract_perspective_summaries(
        artifact,
        side="b",
        competitor_id=competitor_b,
        path=resolved,
    )
    expected_b = {
        "first": a_summaries["second"].reverse(),
        "second": a_summaries["first"].reverse(),
    }
    if b_summaries is not None:
        for seat in ("first", "second"):
            if not _summary_equal(b_summaries[seat], expected_b[seat]):
                raise ValueError(
                    f"{resolved} competitor B {seat}-seat summary is not "
                    "the exact reverse of competitor A's opposite-seat games"
                )
    return PairEvidence(
        competitor_a=competitor_a,
        competitor_b=competitor_b,
        a_first=a_summaries["first"],
        a_second=a_summaries["second"],
        path=resolved,
        file_sha256=digest,
    )


def _possible_reference(
    value: Any,
    *,
    manifest_path: Path,
) -> ArtifactReference | None:
    if isinstance(value, str):
        raw_path = value
        expected_sha256 = None
    elif isinstance(value, dict):
        raw_path = None
        for key in (
            "artifact_path",
            "result_path",
            "output_path",
            "artifact",
            "path",
            "output",
        ):
            candidate = value.get(key)
            if isinstance(candidate, str):
                raw_path = candidate
                break
        if raw_path is None:
            return None
        expected_sha256 = None
        for key in (
            "sha256",
            "file_sha256",
            "artifact_sha256",
            "result_sha256",
        ):
            candidate = value.get(key)
            if isinstance(candidate, str):
                expected_sha256 = candidate
                break
    else:
        return None
    candidate_path = Path(raw_path)
    if not candidate_path.is_absolute():
        candidate_path = manifest_path.parent / candidate_path
    return ArtifactReference(
        path=candidate_path.resolve(),
        expected_sha256=expected_sha256,
        source_manifest=manifest_path,
    )


def _references_from_container(
    value: Any,
    *,
    manifest_path: Path,
) -> list[ArtifactReference]:
    references: list[ArtifactReference] = []
    if isinstance(value, list):
        items = value
    elif isinstance(value, dict):
        items = list(value.values())
    else:
        return references
    for item in items:
        reference = _possible_reference(
            item,
            manifest_path=manifest_path,
        )
        if reference is not None:
            references.append(reference)
            continue
        if isinstance(item, dict):
            item_mapping = _require_mapping(item, "manifest reference")
            for key in (
                "artifact",
                "result",
                "execution",
            ):
                nested_value = item_mapping.get(key)
                if nested_value is not None:
                    nested = _possible_reference(
                        nested_value,
                        manifest_path=manifest_path,
                    )
                    if nested is not None:
                        references.append(nested)
    return references


def _manifest_references(
    manifest: Mapping[str, Any],
    path: Path,
) -> list[ArtifactReference]:
    references: list[ArtifactReference] = []
    for key in (
        "pair_artifacts",
        "completed_pairs",
        "artifacts",
        "results",
        "outputs",
    ):
        if key in manifest:
            references.extend(
                _references_from_container(
                    manifest[key],
                    manifest_path=path,
                )
            )
    for parent_key in ("run_summary", "execution", "league_run"):
        parent = manifest.get(parent_key)
        if not isinstance(parent, dict):
            continue
        for key in (
            "pair_artifacts",
            "completed_pairs",
            "artifacts",
            "results",
            "outputs",
        ):
            if key in parent:
                references.extend(
                    _references_from_container(
                        parent[key],
                        manifest_path=path,
                    )
                )
    if not references:
        output_directory_value = manifest.get("output_directory")
        if isinstance(output_directory_value, str):
            output_directory = Path(output_directory_value)
            if not output_directory.is_absolute():
                output_directory = path.parent / output_directory
            if output_directory.is_dir():
                for child in sorted(output_directory.glob("*.json")):
                    try:
                        payload = json.loads(
                            child.read_text(encoding="utf-8")
                        )
                    except (OSError, json.JSONDecodeError):
                        continue
                    if (
                        isinstance(payload, dict)
                        and payload.get("kind") == PAIR_KIND
                    ):
                        references.append(
                            ArtifactReference(
                                child.resolve(),
                                None,
                                path,
                            )
                        )
    unique: dict[Path, ArtifactReference] = {}
    for reference in references:
        previous = unique.get(reference.path)
        if (
            previous is not None
            and previous.expected_sha256 is not None
            and reference.expected_sha256 is not None
            and previous.expected_sha256 != reference.expected_sha256
        ):
            raise ValueError(
                f"{path} gives conflicting hashes for {reference.path}"
            )
        if previous is None or previous.expected_sha256 is None:
            unique[reference.path] = reference
    return list(unique.values())


def _roster_ids(manifest: Mapping[str, Any], path: Path) -> list[str]:
    roster = manifest.get("roster", manifest.get("competitors"))
    if roster is None:
        return []
    if (
        isinstance(roster, dict)
        and isinstance(roster.get("competitors"), (list, dict))
    ):
        roster = roster["competitors"]
    ids: list[str] = []
    if isinstance(roster, list):
        for index, value in enumerate(roster):
            if isinstance(value, str):
                ids.append(_require_str(value, f"{path}.roster[{index}]"))
            else:
                mapping = _require_mapping(
                    value,
                    f"{path}.roster[{index}]",
                )
                ids.append(
                    _require_str(
                        mapping.get("id"),
                        f"{path}.roster[{index}].id",
                    )
                )
    elif isinstance(roster, dict):
        for key, value in roster.items():
            if isinstance(value, dict) and "id" in value:
                ids.append(
                    _require_str(
                        value["id"],
                        f"{path}.roster.{key}.id",
                    )
                )
            else:
                ids.append(_require_str(key, f"{path}.roster id"))
    else:
        raise ValueError(f"{path}.roster must be an array or object")
    if len(ids) != len(set(ids)):
        raise ValueError(f"{path}.roster contains duplicate ids")
    return ids


def load_inputs(paths: Sequence[Path]) -> LoadedInputs:
    pending: list[ArtifactReference] = []
    manifests: list[dict[str, Any]] = []
    roster_ids: list[str] = []
    manifest_provenance: list[dict[str, Any]] = []

    for input_path in paths:
        resolved = input_path.resolve()
        if resolved.is_dir():
            found = 0
            for child in sorted(resolved.glob("*.json")):
                try:
                    payload = json.loads(child.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if isinstance(payload, dict) and payload.get("kind") == PAIR_KIND:
                    pending.append(
                        ArtifactReference(child.resolve(), None, None)
                    )
                    found += 1
            if found == 0:
                raise ValueError(
                    f"{resolved} contains no {PAIR_KIND} artifacts"
                )
            continue

        payload = _require_mapping(
            json.loads(resolved.read_text(encoding="utf-8")),
            str(resolved),
        )
        if payload.get("kind") == PAIR_KIND:
            pending.append(ArtifactReference(resolved, None, None))
            continue
        if payload.get("kind") not in (MANIFEST_KIND, RUN_KIND):
            raise ValueError(
                f"{resolved}.kind is neither {PAIR_KIND!r} nor "
                f"{MANIFEST_KIND!r}/{RUN_KIND!r}"
            )
        roster_ids.extend(_roster_ids(payload, resolved))
        references = _manifest_references(payload, resolved)
        if not references:
            raise ValueError(
                f"{resolved} contains no completed pair artifact references"
            )
        pending.extend(references)
        provenance = {
            "path": str(resolved),
            "file_sha256": file_sha256(resolved),
            "pair_artifacts_referenced": len(references),
        }
        manifest_provenance.append(provenance)
        manifests.append(provenance)

    unique: dict[Path, ArtifactReference] = {}
    for reference in pending:
        previous = unique.get(reference.path)
        if (
            previous is not None
            and previous.expected_sha256 is not None
            and reference.expected_sha256 is not None
            and previous.expected_sha256 != reference.expected_sha256
        ):
            raise ValueError(
                f"conflicting expected hashes for {reference.path}"
            )
        if previous is None or previous.expected_sha256 is None:
            unique[reference.path] = reference
    evidence = tuple(
        load_pair_artifact(
            reference.path,
            expected_sha256=reference.expected_sha256,
        )
        for reference in sorted(
            unique.values(),
            key=lambda item: str(item.path),
        )
    )
    if not evidence:
        raise ValueError("no pair evidence was loaded")
    all_ids = set(roster_ids)
    for pair in evidence:
        all_ids.update((pair.competitor_a, pair.competitor_b))
    return LoadedInputs(
        evidence=evidence,
        roster_ids=tuple(sorted(all_ids)),
        manifests=tuple(manifest_provenance),
    )


def _aggregate_directed(
    evidence: Sequence[PairEvidence],
) -> tuple[
    dict[tuple[str, str, str], OutcomeSummary],
    dict[tuple[str, str], int],
]:
    directed: dict[tuple[str, str, str], OutcomeSummary] = {}
    replicate_counts: dict[tuple[str, str], int] = {}

    def add(
        row: str,
        column: str,
        seat: str,
        summary: OutcomeSummary,
    ) -> None:
        key = (row, column, seat)
        if key in directed:
            directed[key] = directed[key] + summary
        else:
            directed[key] = summary

    for pair in evidence:
        a = pair.competitor_a
        b = pair.competitor_b
        add(a, b, "first", pair.a_first)
        add(a, b, "second", pair.a_second)
        add(b, a, "first", pair.a_second.reverse())
        add(b, a, "second", pair.a_first.reverse())
        unordered = (a, b) if a < b else (b, a)
        replicate_counts[unordered] = replicate_counts.get(unordered, 0) + 1
    return directed, replicate_counts


def _bootstrap_summary(
    summary: OutcomeSummary,
    rng: np.random.Generator,
    replicates: int,
) -> tuple[np.ndarray, np.ndarray]:
    probabilities = np.asarray(
        [summary.wins, summary.draws, summary.losses],
        dtype=np.float64,
    ) / summary.games
    counts = rng.multinomial(
        summary.games,
        probabilities,
        size=replicates,
    )
    win_rate = counts[:, 0] / summary.games
    score_rate = (counts[:, 0] + 0.5 * counts[:, 1]) / summary.games
    return win_rate, score_rate


def _percentile_interval(values: np.ndarray) -> list[float]:
    quantiles = np.quantile(values, [0.025, 0.975])
    return [float(quantiles[0]), float(quantiles[1])]


def _seat_cell(
    summary: OutcomeSummary,
    win_bootstrap: np.ndarray,
    score_bootstrap: np.ndarray,
) -> dict[str, Any]:
    wilson_low, wilson_high = wilson_interval(
        summary.wins,
        summary.games,
    )
    return {
        **summary.counts_json(),
        "win_rate": summary.win_rate,
        "score_rate": summary.score_rate,
        "win_rate_wilson_95": [wilson_low, wilson_high],
        "win_rate_bootstrap_95": _percentile_interval(win_bootstrap),
        "score_rate_bootstrap_95": _percentile_interval(score_bootstrap),
    }


def _pooled_cell(
    first: OutcomeSummary,
    second: OutcomeSummary,
    first_win_bootstrap: np.ndarray,
    second_win_bootstrap: np.ndarray,
    first_score_bootstrap: np.ndarray,
    second_score_bootstrap: np.ndarray,
    *,
    equivalence_margin: float,
) -> dict[str, Any]:
    balanced_win_bootstrap = (
        first_win_bootstrap + second_win_bootstrap
    ) / 2.0
    balanced_score_bootstrap = (
        first_score_bootstrap + second_score_bootstrap
    ) / 2.0
    combined = first + second
    wilson_90_low, wilson_90_high = wilson_interval(
        combined.wins,
        combined.games,
        z=Z_90,
    )
    equivalence_bounds = [
        0.5 - equivalence_margin,
        0.5 + equivalence_margin,
    ]
    equal_seat_sample_sizes = first.games == second.games
    draw_free = combined.draws == 0
    equivalence_eligible = equal_seat_sample_sizes and draw_free
    interval_contained = (
        wilson_90_low >= equivalence_bounds[0]
        and wilson_90_high <= equivalence_bounds[1]
        if equivalence_eligible
        else None
    )
    return {
        "first_games": first.games,
        "second_games": second.games,
        "seat_balanced_win_rate": (
            first.win_rate + second.win_rate
        )
        / 2.0,
        "seat_balanced_score_rate": (
            first.score_rate + second.score_rate
        )
        / 2.0,
        "seat_balanced_win_rate_bootstrap_95": (
            _percentile_interval(balanced_win_bootstrap)
        ),
        "seat_balanced_score_rate_bootstrap_95": (
            _percentile_interval(balanced_score_bootstrap)
        ),
        "raw_pooled_win_rate": combined.win_rate,
        "raw_pooled_score_rate": combined.score_rate,
        "raw_pooled_counts": combined.counts_json(),
        "raw_pooled_win_rate_wilson_90": [
            wilson_90_low,
            wilson_90_high,
        ],
        "seat_balanced_score_rate_wilson_90": (
            [wilson_90_low, wilson_90_high]
            if equivalence_eligible
            else None
        ),
        "seat_balanced_score_rate_wilson_90_valid": (
            equivalence_eligible
        ),
        "equivalence_90": {
            "eligible": equivalence_eligible,
            "equal_seat_sample_sizes": equal_seat_sample_sizes,
            "draw_free": draw_free,
            "acceptance_bounds": equivalence_bounds,
            "interval_contained": interval_contained,
        },
        "equal_seat_sample_sizes": equal_seat_sample_sizes,
    }


def _matrix(
    ids: Sequence[str],
    values: Mapping[tuple[str, str], Any],
) -> list[list[Any]]:
    return [
        [
            values.get((row, column))
            for column in ids
        ]
        for row in ids
    ]


def _three_cycles(
    ids: Sequence[str],
    pooled_cells: Mapping[tuple[str, str], Mapping[str, Any]],
    *,
    margin: float,
) -> list[dict[str, Any]]:
    def edge(left: str, right: str) -> bool:
        cell = pooled_cells.get((left, right))
        return (
            cell is not None
            and float(cell["seat_balanced_score_rate"])
            > 0.5 + margin
        )

    cycles: list[dict[str, Any]] = []
    for a_index in range(len(ids)):
        for b_index in range(a_index + 1, len(ids)):
            for c_index in range(b_index + 1, len(ids)):
                a, b, c = ids[a_index], ids[b_index], ids[c_index]
                if edge(a, b) and edge(b, c) and edge(c, a):
                    order = (a, b, c)
                elif edge(a, c) and edge(c, b) and edge(b, a):
                    order = (a, c, b)
                else:
                    continue
                edges: list[dict[str, Any]] = []
                confidence_supported = True
                for left, right in zip(
                    order,
                    (*order[1:], order[0]),
                    strict=True,
                ):
                    cell = pooled_cells[(left, right)]
                    interval = cell[
                        "seat_balanced_score_rate_bootstrap_95"
                    ]
                    confidence_supported &= (
                        float(interval[0]) > 0.5 + margin
                    )
                    edges.append(
                        {
                            "winner": left,
                            "loser": right,
                            "seat_balanced_score_rate": cell[
                                "seat_balanced_score_rate"
                            ],
                            "bootstrap_95": interval,
                        }
                    )
                cycles.append(
                    {
                        "cycle": [*order, order[0]],
                        "edges": edges,
                        "all_edges_bootstrap_lower_above_threshold": (
                            confidence_supported
                        ),
                    }
                )
    return cycles


def _rating_objective(
    theta: np.ndarray,
    design: np.ndarray,
    games: np.ndarray,
    scores: np.ndarray,
    regularization: float,
) -> float:
    eta = design @ theta
    likelihood = np.sum(games * np.logaddexp(0.0, eta) - scores * eta)
    penalty = 0.5 * regularization * float(theta @ theta)
    return float(likelihood + penalty)


def _fit_regularized_bradley_terry(
    ids: Sequence[str],
    directed: Mapping[tuple[str, str, str], OutcomeSummary],
    *,
    regularization: float,
    max_iterations: int = 100,
    tolerance: float = 1e-10,
) -> dict[str, Any]:
    if regularization <= 0.0:
        raise ValueError("rating regularization must be positive")
    index = {competitor_id: offset for offset, competitor_id in enumerate(ids)}
    observations: list[RatingObservation] = []
    for left_index, left in enumerate(ids):
        for right in ids[left_index + 1 :]:
            left_first = directed.get((left, right, "first"))
            right_first = directed.get((right, left, "first"))
            if left_first is None or right_first is None:
                continue
            observations.extend(
                [
                    RatingObservation(
                        first_player=index[left],
                        second_player=index[right],
                        games=left_first.games,
                        first_score=(
                            left_first.wins + 0.5 * left_first.draws
                        ),
                    ),
                    RatingObservation(
                        first_player=index[right],
                        second_player=index[left],
                        games=right_first.games,
                        first_score=(
                            right_first.wins + 0.5 * right_first.draws
                        ),
                    ),
                ]
            )
    for competitor_id in ids:
        self_first = directed.get(
            (competitor_id, competitor_id, "first")
        )
        if self_first is not None:
            observations.append(
                RatingObservation(
                    first_player=index[competitor_id],
                    second_player=index[competitor_id],
                    games=self_first.games,
                    first_score=(
                        self_first.wins + 0.5 * self_first.draws
                    ),
                )
            )
    if not observations:
        raise ValueError("rating requires at least one completed pair")

    parameter_count = len(ids) + 1
    design = np.zeros(
        (len(observations), parameter_count),
        dtype=np.float64,
    )
    games = np.empty(len(observations), dtype=np.float64)
    scores = np.empty(len(observations), dtype=np.float64)
    for row, observation in enumerate(observations):
        design[row, observation.first_player] += 1.0
        design[row, observation.second_player] -= 1.0
        design[row, -1] = 1.0
        games[row] = observation.games
        scores[row] = observation.first_score

    theta = np.zeros(parameter_count, dtype=np.float64)
    converged = False
    accepted_iterations = 0
    gradient_max = math.inf
    for _ in range(max_iterations):
        eta = design @ theta
        probabilities = 1.0 / (1.0 + np.exp(-np.clip(eta, -40.0, 40.0)))
        gradient = (
            design.T @ (games * probabilities - scores)
            + regularization * theta
        )
        curvature = games * probabilities * (1.0 - probabilities)
        hessian = (
            design.T @ (curvature[:, None] * design)
            + regularization * np.eye(parameter_count)
        )
        gradient_max = float(np.max(np.abs(gradient)))
        if gradient_max < tolerance:
            converged = True
            break
        step = np.linalg.solve(hessian, gradient)
        old_objective = _rating_objective(
            theta,
            design,
            games,
            scores,
            regularization,
        )
        factor = 1.0
        accepted = False
        while factor >= 2.0**-20:
            candidate = theta - factor * step
            candidate[:-1] -= np.mean(candidate[:-1])
            candidate_objective = _rating_objective(
                candidate,
                design,
                games,
                scores,
                regularization,
            )
            if candidate_objective <= old_objective:
                theta = candidate
                accepted = True
                accepted_iterations += 1
                if float(np.max(np.abs(factor * step))) < tolerance:
                    converged = True
                break
            factor *= 0.5
        if not accepted or converged:
            break

    eta = design @ theta
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(eta, -40.0, 40.0)))
    final_gradient = (
        design.T @ (games * probabilities - scores)
        + regularization * theta
    )
    gradient_max = float(np.max(np.abs(final_gradient)))
    converged |= gradient_max < tolerance
    curvature = games * probabilities * (1.0 - probabilities)
    final_hessian = (
        design.T @ (curvature[:, None] * design)
        + regularization * np.eye(parameter_count)
    )
    curvature_covariance = np.linalg.inv(final_hessian)
    elo_scale = 400.0 / math.log(10.0)
    entries: list[dict[str, Any]] = []
    for competitor_id, ability, variance in zip(
        ids,
        theta[:-1],
        np.diag(curvature_covariance)[:-1],
        strict=True,
    ):
        standard_error = math.sqrt(max(0.0, float(variance)))
        entries.append(
            {
                "id": competitor_id,
                "ability_log_odds": float(ability),
                "elo_like": float(ability * elo_scale),
                "penalized_curvature_standard_error_log_odds": standard_error,
                "penalized_curvature_interval_95_elo_like": [
                    float((ability - Z_95 * standard_error) * elo_scale),
                    float((ability + Z_95 * standard_error) * elo_scale),
                ],
            }
        )
    entries.sort(key=lambda item: (-item["elo_like"], item["id"]))
    for rank, entry in enumerate(entries, start=1):
        entry["rank"] = rank

    residuals: list[dict[str, Any]] = []
    for observation, predicted in zip(
        observations,
        probabilities,
        strict=True,
    ):
        observed = observation.first_score / observation.games
        residuals.append(
            {
                "first_player": ids[observation.first_player],
                "second_player": ids[observation.second_player],
                "games": observation.games,
                "observed_first_score_rate": observed,
                "predicted_first_score_rate": float(predicted),
                "residual_observed_minus_predicted": (
                    observed - float(predicted)
                ),
            }
        )
    residuals.sort(
        key=lambda item: (
            -abs(item["residual_observed_minus_predicted"]),
            item["first_player"],
            item["second_player"],
        )
    )
    seat_log_odds = float(theta[-1])
    return {
        "model": (
            "P(first player i scores vs second player j) = "
            "logistic(ability_i - ability_j + first_seat_log_odds)"
        ),
        "draw_treatment": "a draw contributes one half-success",
        "l2_regularization": regularization,
        "identifiability": "abilities are centered to mean zero",
        "converged": converged,
        "accepted_newton_iterations": accepted_iterations,
        "final_max_absolute_penalized_gradient": gradient_max,
        "first_seat_log_odds": seat_log_odds,
        "first_seat_elo_like": seat_log_odds * elo_scale,
        "equal_ability_first_score_probability": (
            1.0 / (1.0 + math.exp(-seat_log_odds))
        ),
        "ratings": entries,
        "largest_absolute_residuals": residuals[: min(20, len(residuals))],
        "caveats": [
            (
                "This is a descriptive scalar projection of the sampled "
                "league, not a minimax value or proof of optimal play."
            ),
            (
                "The model has one constant seat intercept; player-specific "
                "seat sensitivity and game-stage effects are omitted."
            ),
            (
                "Bradley-Terry assumes a transitive one-dimensional ability. "
                "Three-cycles and other non-transitive interactions, together "
                "with large residuals, are evidence against that assumption, "
                "so the matrix takes precedence over the ranking."
            ),
            (
                "L2 regularization prevents infinite estimates under the "
                "near-separation expected in solved Hex. Ratings and the "
                "penalized-curvature intervals depend on its chosen scale."
            ),
            (
                "The curvature intervals are local diagnostics conditional "
                "on this fitted regularized model, not calibrated league-wide "
                "frequentist confidence intervals."
            ),
        ],
    }


def analyze_league(
    loaded: LoadedInputs,
    *,
    bootstrap_replicates: int = 10_000,
    bootstrap_seed: int = 600_613,
    cycle_margin: float = 0.0,
    rating_regularization: float = 1.0,
    equivalence_margin: float = DEFAULT_EQUIVALENCE_MARGIN,
) -> dict[str, Any]:
    if bootstrap_replicates < 200:
        raise ValueError("bootstrap_replicates must be at least 200")
    if cycle_margin < 0.0 or cycle_margin >= 0.5:
        raise ValueError("cycle_margin must lie in [0, 0.5)")
    if (
        not math.isfinite(equivalence_margin)
        or equivalence_margin <= 0.0
        or equivalence_margin >= 0.5
    ):
        raise ValueError("equivalence_margin must lie in (0, 0.5)")
    ids = loaded.roster_ids
    if not ids:
        raise ValueError("league analysis requires at least one competitor")
    directed, replicate_counts = _aggregate_directed(loaded.evidence)
    if len(ids) < 2 and not any(
        left == right for left, right in replicate_counts
    ):
        raise ValueError(
            "a one-competitor league requires a self-play artifact"
        )
    rng = np.random.default_rng(bootstrap_seed)

    seat_cells: dict[str, dict[tuple[str, str], dict[str, Any]]] = {
        "first": {},
        "second": {},
    }
    win_bootstraps: dict[tuple[str, str, str], np.ndarray] = {}
    score_bootstraps: dict[tuple[str, str, str], np.ndarray] = {}
    for row in ids:
        for column in ids:
            for seat in ("first", "second"):
                summary = directed.get((row, column, seat))
                if summary is None:
                    continue
                win_sample, score_sample = _bootstrap_summary(
                    summary,
                    rng,
                    bootstrap_replicates,
                )
                win_bootstraps[(row, column, seat)] = win_sample
                score_bootstraps[(row, column, seat)] = score_sample
                seat_cells[seat][(row, column)] = _seat_cell(
                    summary,
                    win_sample,
                    score_sample,
                )

    pooled_cells: dict[tuple[str, str], dict[str, Any]] = {}
    for row in ids:
        for column in ids:
            first = directed.get((row, column, "first"))
            second = directed.get((row, column, "second"))
            if first is None or second is None:
                continue
            pooled = _pooled_cell(
                first,
                second,
                win_bootstraps[(row, column, "first")],
                win_bootstraps[(row, column, "second")],
                score_bootstraps[(row, column, "first")],
                score_bootstraps[(row, column, "second")],
                equivalence_margin=equivalence_margin,
            )
            pooled_cells[(row, column)] = pooled

    robust_conversion: dict[str, Any] = {}
    for row in ids:
        opponents = [
            column
            for column in ids
            if column != row and (row, column, "first") in directed
        ]
        if not opponents:
            robust_conversion[row] = None
            continue
        point_rates = {
            opponent: directed[(row, opponent, "first")].win_rate
            for opponent in opponents
        }
        worst_point = min(opponents, key=lambda item: (point_rates[item], item))
        marginal_lowers = {
            opponent: wilson_lower_bound(
                directed[(row, opponent, "first")].wins,
                directed[(row, opponent, "first")].games,
                z=Z_95,
            )
            for opponent in opponents
        }
        one_sided_family_z = statistics.NormalDist().inv_cdf(
            1.0 - 0.05 / len(opponents)
        )
        simultaneous_lowers = {
            opponent: wilson_lower_bound(
                directed[(row, opponent, "first")].wins,
                directed[(row, opponent, "first")].games,
                z=one_sided_family_z,
            )
            for opponent in opponents
        }
        worst_conservative = min(
            opponents,
            key=lambda item: (simultaneous_lowers[item], item),
        )
        minimum_bootstrap = np.minimum.reduce(
            [
                win_bootstraps[(row, opponent, "first")]
                for opponent in opponents
            ]
        )
        robust_conversion[row] = {
            "opponents_covered": len(opponents),
            "opponent_ids": opponents,
            "minimum_first_seat_win_rate": point_rates[worst_point],
            "worst_opponent_by_point_estimate": worst_point,
            "minimum_first_seat_win_rate_bootstrap_95": (
                _percentile_interval(minimum_bootstrap)
            ),
            "minimum_marginal_wilson_lower_95": min(
                marginal_lowers.values()
            ),
            "minimum_bonferroni_simultaneous_lower_95": (
                simultaneous_lowers[worst_conservative]
            ),
            "worst_opponent_by_simultaneous_lower_bound": (
                worst_conservative
            ),
            "bonferroni_one_sided_z": one_sided_family_z,
        }

    observed_unordered = {
        pair for pair in replicate_counts if pair[0] != pair[1]
    }
    observed_self_play = {
        pair for pair in replicate_counts if pair[0] == pair[1]
    }
    expected_unordered = {
        (ids[left], ids[right])
        for left in range(len(ids))
        for right in range(left + 1, len(ids))
    }
    missing_pairs = sorted(expected_unordered - observed_unordered)
    draw_games = sum(
        pair.a_first.draws + pair.a_second.draws
        for pair in loaded.evidence
    )
    cycles = _three_cycles(
        ids,
        pooled_cells,
        margin=cycle_margin,
    )
    result = {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "local_only": {
            "network_access": False,
            "external_logging": False,
            "external_export": False,
            "artifact_is_local": True,
        },
        "inputs": {
            "manifests": list(loaded.manifests),
            "pair_artifacts": [
                {
                    "path": str(pair.path),
                    "file_sha256": pair.file_sha256,
                    "competitor_a": pair.competitor_a,
                    "competitor_b": pair.competitor_b,
                    "a_first_games": pair.a_first.games,
                    "a_second_games": pair.a_second.games,
                }
                for pair in loaded.evidence
            ],
        },
        "competitor_ids": list(ids),
        "coverage": {
            "competitors": len(ids),
            "expected_unordered_pairs": len(expected_unordered),
            "observed_unordered_pairs": len(observed_unordered),
            "observed_self_play_pairs": len(observed_self_play),
            "complete": not missing_pairs,
            "missing_unordered_pairs": [
                list(pair) for pair in missing_pairs
            ],
            "replicate_artifact_counts": {
                f"{left} vs {right}": count
                for (left, right), count in sorted(replicate_counts.items())
            },
        },
        "primary_seat_conditioned": {
            "row_semantics": "competitor whose win probability is reported",
            "column_semantics": "opponent",
            "win_definition": "row competitor wins; draws are not wins",
            "first": {
                "estimand": "P(row wins | row has the first seat)",
                "matrix": _matrix(ids, seat_cells["first"]),
                "win_rate_matrix": _matrix(
                    ids,
                    {
                        key: cell["win_rate"]
                        for key, cell in seat_cells["first"].items()
                    },
                ),
            },
            "second": {
                "estimand": "P(row wins | row has the second seat)",
                "matrix": _matrix(ids, seat_cells["second"]),
                "win_rate_matrix": _matrix(
                    ids,
                    {
                        key: cell["win_rate"]
                        for key, cell in seat_cells["second"].items()
                    },
                ),
            },
        },
        "secondary_balanced_pooled": {
            "status": "secondary; never substitutes for the seat matrices",
            "estimand": (
                "equal-weight mean of the row competitor's first-seat and "
                "second-seat score probabilities; draws count one half"
            ),
            "equivalence_contract": {
                "confidence_level": 0.90,
                "target_score_rate": 0.5,
                "margin": equivalence_margin,
                "acceptance_bounds": [
                    0.5 - equivalence_margin,
                    0.5 + equivalence_margin,
                ],
                "interval_field": (
                    "seat_balanced_score_rate_wilson_90"
                ),
                "decision_field": "equivalence_90.interval_contained",
                "eligibility": (
                    "The two-sided Wilson interval is an equivalence test for "
                    "the seat-balanced score only when seat sample sizes are "
                    "equal and there are no draws, as in the Hex league. "
                    "Otherwise interval_contained is null."
                ),
                "decision_rule": (
                    "Equivalence is supported exactly when the eligible 90% "
                    "Wilson interval is contained in the acceptance bounds."
                ),
            },
            "matrix": _matrix(ids, pooled_cells),
            "score_rate_matrix": _matrix(
                ids,
                {
                    key: cell["seat_balanced_score_rate"]
                    for key, cell in pooled_cells.items()
                },
            ),
        },
        "robust_first_seat_conversion": {
            "definition": (
                "minimum over sampled opponents of P(row wins | row first)"
            ),
            "players": robust_conversion,
            "simultaneous_bound_note": (
                "For each row, one-sided Wilson lower bounds use alpha/K "
                "Bonferroni correction over its K covered opponents. Their "
                "minimum is a conservative family-wise 95% lower bound for "
                "the worst sampled-opponent conversion probability."
            ),
            "scope_note": (
                "Robustness is only with respect to the listed opponent "
                "league. It is not a lower bound against arbitrary policies."
            ),
        },
        "nontransitivity": {
            "edge_rule": (
                "A -> B when A's seat-balanced score rate exceeds "
                f"{0.5 + cycle_margin:g}"
            ),
            "cycle_margin": cycle_margin,
            "three_cycles": cycles,
            "three_cycle_count": len(cycles),
            "confidence_flag_note": (
                "Cycle discovery uses point estimates. Each cycle separately "
                "reports whether all three parametric-bootstrap lower bounds "
                "also exceed the edge threshold; this is not multiplicity "
                "adjusted."
            ),
        },
        "seat_adjusted_regularized_bradley_terry": (
            _fit_regularized_bradley_terry(
                ids,
                directed,
                regularization=rating_regularization,
            )
        ),
        "uncertainty_contract": {
            "wilson": (
                "Two-sided 95% Wilson score intervals are primary for each "
                "fixed-seat Bernoulli win probability."
            ),
            "equivalence_wilson": (
                "Eligible balanced pooled cells additionally report a "
                "two-sided 90% Wilson score interval for the draw-free raw "
                "pooled win probability. With equal seat sample sizes this "
                "equals the seat-balanced score estimand and implements the "
                "registered equivalence-interval decision."
            ),
            "bootstrap": {
                "kind": (
                    "stratified parametric bootstrap of the observed "
                    "win/draw/loss multinomial counts"
                ),
                "replicates": bootstrap_replicates,
                "seed": bootstrap_seed,
                "percentile_interval": [0.025, 0.975],
                "independence_scope": (
                    "Pair artifacts and seat strata are treated as independent "
                    "batches. The bootstrap does not model shared checkpoint "
                    "training noise or common random numbers across artifacts."
                ),
            },
        },
        "interpretation_contract": {
            "hex_optimal_play": (
                "For finite Hex without pie/swap, there are no draws and "
                "perfect play converts the first seat and loses from the "
                "second seat. A balanced score near 0.5 alone is therefore "
                "not evidence of optimal play."
            ),
            "primary_order": [
                "seat-conditioned matrix",
                "worst-opponent first-seat conversion",
                "coverage and uncertainty",
                "cycles/residuals",
                "balanced scalar summaries and rating",
            ],
            "opaque_ids": (
                "Competitor IDs are treated as opaque labels. Checkpoints "
                "that differ only by test-time kappa/readout may be rostered "
                "as separate competitors without changing the analysis."
            ),
            "warnings": [
                *(
                    [
                        f"League coverage is incomplete: {len(missing_pairs)} "
                        "unordered pairs are missing."
                    ]
                    if missing_pairs
                    else []
                ),
                *(
                    [
                        f"Observed {draw_games} draws even though standard Hex "
                        "has no draws; inspect environment/evaluation integrity."
                    ]
                    if draw_games
                    else []
                ),
            ],
        },
    }
    return result


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
            "Analyze pair artifacts or completed manifests from the Hex "
            "checkpoint league."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help=(
            "pair artifact, completed manifest/run-summary, or directory of "
            "pair artifacts"
        ),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--bootstrap-replicates",
        type=int,
        default=10_000,
    )
    parser.add_argument("--bootstrap-seed", type=int, default=600_613)
    parser.add_argument(
        "--cycle-margin",
        type=float,
        default=0.0,
        help="minimum score margin above 0.5 required for a directed edge",
    )
    parser.add_argument(
        "--rating-regularization",
        type=float,
        default=1.0,
        help="positive L2 penalty on abilities and the seat intercept",
    )
    parser.add_argument(
        "--equivalence-margin",
        type=float,
        default=DEFAULT_EQUIVALENCE_MARGIN,
        help=(
            "half-width around 0.5 for the 90% Wilson equivalence decision "
            "(default: 0.05, giving [0.45, 0.55])"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    loaded = load_inputs(args.inputs)
    result = analyze_league(
        loaded,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
        cycle_margin=args.cycle_margin,
        rating_regularization=args.rating_regularization,
        equivalence_margin=args.equivalence_margin,
    )
    script_path = Path(__file__).resolve()
    result["reproduction"] = {
        "command": " ".join(
            shlex.quote(argument)
            for argument in (
                sys.executable,
                str(script_path),
                *(sys.argv[1:] if argv is None else argv),
            )
        ),
        "working_directory": str(Path.cwd().resolve()),
        "script_path": str(script_path),
        "script_sha256": file_sha256(script_path),
        "python": platform.python_version(),
        "numpy": np.__version__,
    }
    output_sha256 = _write_json(args.output, result)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "sha256": output_sha256,
                "competitors": len(result["competitor_ids"]),
                "observed_pairs": result["coverage"][
                    "observed_unordered_pairs"
                ],
                "coverage_complete": result["coverage"]["complete"],
                "three_cycles": result["nontransitivity"][
                    "three_cycle_count"
                ],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
