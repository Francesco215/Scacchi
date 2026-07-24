#!/usr/bin/env python3
"""Attach exact Hex-oracle decision labels to a balanced-eval trace.

The input is the local trace artifact written by
``scripts/hex_balanced_eval.py``.  Every recorded pre-action position is
validated against the trace's content-derived identifiers, reconstructed as a
fixed-colour :class:`scacchi.hex_oracle.HexPosition`, and solved exactly.

The output is a local, immutable analysis artifact.  It preserves the complete
source record and trace provenance, records the source file digest, and refuses
to overwrite an existing output path.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shlex
import sys
import tempfile
from typing import Any

from scacchi.hex_oracle import HexPosition, solve_hex


TRACE_KIND = "scacchi.hex_balanced_eval_trace"
TRACE_SCHEMA_VERSION = 1
RESULT_KIND = "scacchi.hex_trace_oracle_analysis"
ANALYSIS_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ValidatedTraceRecord:
    """One source record together with its validated exact position."""

    raw: dict[str, Any]
    position: HexPosition
    action: int
    actor_agent: str


@dataclass(frozen=True)
class LoadedTrace:
    """Validated trace input and byte-level provenance."""

    path: Path
    file_sha256: str
    canonical_json_sha256: str
    payload: dict[str, Any]
    board_size: int
    records: tuple[ValidatedTraceRecord, ...]


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


def _require_list(value: Any, location: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{location} must be a JSON array")
    return value


def _require_int(value: Any, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{location} must be an integer")
    return value


def _require_bool(value: Any, location: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{location} must be a boolean")
    return value


def _require_str(value: Any, location: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{location} must be a string")
    return value


def _require_sha256(value: Any, location: str) -> str:
    digest = _require_str(value, location)
    if len(digest) != 64:
        raise ValueError(f"{location} must be a 64-character SHA-256")
    try:
        int(digest, 16)
    except ValueError as error:
        raise ValueError(f"{location} must be hexadecimal") from error
    if digest != digest.lower():
        raise ValueError(f"{location} must use lowercase hexadecimal")
    return digest


def _trace_position_id(
    cells: Sequence[int],
    current_color: int,
) -> str:
    encoded = json.dumps(
        {
            "cells": list(cells),
            "current_color": current_color,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_trace_record(
    value: Any,
    *,
    index: int,
    sampling_seed: int,
) -> ValidatedTraceRecord:
    location = f"positions[{index}]"
    record = _require_mapping(value, location)
    cells_values = _require_list(record.get("cells"), f"{location}.cells")
    cells = tuple(
        _require_int(cell, f"{location}.cells[{cell_index}]")
        for cell_index, cell in enumerate(cells_values)
    )
    size = math.isqrt(len(cells))
    if size < 1 or size * size != len(cells):
        raise ValueError(
            f"{location}.cells must contain a positive square board"
        )

    current_color = _require_int(
        record.get("current_color"),
        f"{location}.current_color",
    )
    position = HexPosition(
        size=size,
        cells=cells,
        current_color=current_color,
    )
    empty_count = _require_int(
        record.get("empty_count"),
        f"{location}.empty_count",
    )
    if empty_count != position.empty_count:
        raise ValueError(
            f"{location}.empty_count={empty_count} disagrees with cells "
            f"({position.empty_count})"
        )
    ply_index = _require_int(
        record.get("ply_index"),
        f"{location}.ply_index",
    )
    expected_ply = size * size - empty_count
    if ply_index != expected_ply:
        raise ValueError(
            f"{location}.ply_index={ply_index} disagrees with cells "
            f"({expected_ply})"
        )

    action = _require_int(record.get("action"), f"{location}.action")
    if not 0 <= action < len(cells):
        raise ValueError(
            f"{location}.action={action} is outside the size-{size} board"
        )
    if cells[action] != 0:
        raise ValueError(
            f"{location}.action={action} is not legal in the recorded cells"
        )

    actor_player_id = _require_int(
        record.get("actor_player_id"),
        f"{location}.actor_player_id",
    )
    candidate_player_id = _require_int(
        record.get("candidate_player_id"),
        f"{location}.candidate_player_id",
    )
    if actor_player_id not in (0, 1) or candidate_player_id not in (0, 1):
        raise ValueError(
            f"{location} player ids must each be zero or one"
        )
    actor_agent = _require_str(
        record.get("actor_agent"),
        f"{location}.actor_agent",
    )
    if actor_agent not in ("candidate", "baseline"):
        raise ValueError(
            f"{location}.actor_agent must be 'candidate' or 'baseline'"
        )
    expected_actor = (
        "candidate"
        if actor_player_id == candidate_player_id
        else "baseline"
    )
    if actor_agent != expected_actor:
        raise ValueError(
            f"{location}.actor_agent={actor_agent!r} disagrees with player ids"
        )
    candidate_seat = _require_str(
        record.get("candidate_seat"),
        f"{location}.candidate_seat",
    )
    if candidate_seat not in ("first", "second"):
        raise ValueError(
            f"{location}.candidate_seat must be 'first' or 'second'"
        )

    final_return = _require_int(
        record.get("final_candidate_return"),
        f"{location}.final_candidate_return",
    )
    if final_return not in (-1, 0, 1):
        raise ValueError(
            f"{location}.final_candidate_return must be -1, 0, or 1"
        )
    candidate_won = _require_bool(
        record.get("candidate_won"),
        f"{location}.candidate_won",
    )
    if candidate_won != (final_return > 0):
        raise ValueError(
            f"{location}.candidate_won disagrees with final_candidate_return"
        )

    stratum_index = _require_int(
        record.get("stratum_index"),
        f"{location}.stratum_index",
    )
    chunk_index = _require_int(
        record.get("chunk_index"),
        f"{location}.chunk_index",
    )
    row_index = _require_int(
        record.get("row_index"),
        f"{location}.row_index",
    )
    if stratum_index not in range(4):
        raise ValueError(f"{location}.stratum_index must be in [0, 3]")
    if chunk_index < 0 or row_index < 0:
        raise ValueError(
            f"{location} chunk_index and row_index must be non-negative"
        )

    expected_position_id = _trace_position_id(cells, current_color)
    position_id = _require_sha256(
        record.get("position_id"),
        f"{location}.position_id",
    )
    if position_id != expected_position_id:
        raise ValueError(
            f"{location}.position_id disagrees with cells/current_color"
        )
    expected_priority = hashlib.sha256(
        (
            f"{sampling_seed}:{empty_count}:{stratum_index}:"
            f"{chunk_index}:{row_index}:{position_id}"
        ).encode("utf-8")
    ).hexdigest()
    priority = _require_sha256(
        record.get("sample_priority"),
        f"{location}.sample_priority",
    )
    if priority != expected_priority:
        raise ValueError(
            f"{location}.sample_priority disagrees with sampling provenance"
        )
    expected_trace_id = hashlib.sha256(
        f"{priority}:{action}:{actor_player_id}".encode("utf-8")
    ).hexdigest()
    trace_id = _require_sha256(
        record.get("trace_id"),
        f"{location}.trace_id",
    )
    if trace_id != expected_trace_id:
        raise ValueError(
            f"{location}.trace_id disagrees with action/actor provenance"
        )
    return ValidatedTraceRecord(
        raw=record,
        position=position,
        action=action,
        actor_agent=actor_agent,
    )


def load_trace(path: Path) -> LoadedTrace:
    """Load and validate one balanced-evaluation trace artifact."""

    resolved = path.resolve()
    encoded = resolved.read_bytes()
    file_sha256 = hashlib.sha256(encoded).hexdigest()
    try:
        payload_value = json.loads(encoded)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid trace JSON at {resolved}: {error}") from error
    payload = _require_mapping(payload_value, "trace")
    if payload.get("kind") != TRACE_KIND:
        raise ValueError(f"trace.kind must be {TRACE_KIND!r}")
    if _require_int(
        payload.get("schema_version"),
        "trace.schema_version",
    ) != TRACE_SCHEMA_VERSION:
        raise ValueError("unsupported trace.schema_version")

    _require_mapping(payload.get("source"), "trace.source")
    _require_mapping(payload.get("board_encoding"), "trace.board_encoding")
    sampling = _require_mapping(payload.get("sampling"), "trace.sampling")
    sampling_seed = _require_int(
        sampling.get("seed"),
        "trace.sampling.seed",
    )
    requested_counts = {
        _require_int(value, f"trace.sampling.requested_empty_counts[{index}]")
        for index, value in enumerate(
            _require_list(
                sampling.get("requested_empty_counts"),
                "trace.sampling.requested_empty_counts",
            )
        )
    }

    positions = _require_list(payload.get("positions"), "trace.positions")
    if not positions:
        raise ValueError("trace.positions must contain at least one record")
    records = tuple(
        _validate_trace_record(
            value,
            index=index,
            sampling_seed=sampling_seed,
        )
        for index, value in enumerate(positions)
    )
    board_sizes = {record.position.size for record in records}
    if len(board_sizes) != 1:
        raise ValueError("trace positions disagree on board size")
    board_size = next(iter(board_sizes))

    actual_counts = Counter(
        record.position.empty_count for record in records
    )
    if not set(actual_counts).issubset(requested_counts):
        raise ValueError(
            "trace positions contain an unrequested empty count"
        )
    declared_counts = _require_mapping(
        sampling.get("counts"),
        "trace.sampling.counts",
    )
    for empty_count, actual in actual_counts.items():
        entry = _require_mapping(
            declared_counts.get(str(empty_count)),
            f"trace.sampling.counts[{empty_count!r}]",
        )
        selected = _require_int(
            entry.get("selected"),
            f"trace.sampling.counts[{empty_count!r}].selected",
        )
        if selected != actual:
            raise ValueError(
                f"trace sampling selected count for {empty_count} empties "
                f"is {selected}, but positions contain {actual}"
            )

    position_ids = [
        _require_str(
            record.raw["position_id"],
            f"positions[{index}].position_id",
        )
        for index, record in enumerate(records)
    ]
    trace_ids = [
        _require_str(
            record.raw["trace_id"],
            f"positions[{index}].trace_id",
        )
        for index, record in enumerate(records)
    ]
    if len(position_ids) != len(set(position_ids)):
        raise ValueError("trace positions contain duplicate position_id values")
    if len(trace_ids) != len(set(trace_ids)):
        raise ValueError("trace positions contain duplicate trace_id values")

    return LoadedTrace(
        path=resolved,
        file_sha256=file_sha256,
        canonical_json_sha256=_canonical_sha256(payload),
        payload=payload,
        board_size=board_size,
        records=records,
    )


def _summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    decisions = len(records)
    optimal = sum(bool(record["action_optimal"]) for record in records)
    winning_states = sum(
        int(record["oracle_outcome"]) > 0 for record in records
    )
    losing_states = sum(
        int(record["oracle_outcome"]) < 0 for record in records
    )
    drawn_states = decisions - winning_states - losing_states
    losing_moves = sum(
        bool(record["losing_move_on_winning_state"])
        for record in records
    )
    regret_sum = sum(float(record["exact_regret"]) for record in records)
    return {
        "decisions": decisions,
        "optimal_action_count": optimal,
        "optimal_action_fraction": (
            optimal / decisions if decisions else None
        ),
        "winning_state_decisions": winning_states,
        "losing_state_decisions": losing_states,
        "drawn_state_decisions": drawn_states,
        "losing_move_on_winning_state_count": losing_moves,
        "losing_move_on_winning_state_rate": (
            losing_moves / winning_states if winning_states else None
        ),
        "exact_regret_sum": regret_sum,
        "mean_exact_regret": (
            regret_sum / decisions if decisions else None
        ),
    }


def analyze_trace(trace: LoadedTrace) -> dict[str, Any]:
    """Solve every validated trace record and build exact summaries."""

    analyzed: list[dict[str, Any]] = []
    for index, record in enumerate(trace.records):
        result = solve_hex(record.position)
        try:
            action_value = result.action_value(record.action)
        except KeyError as error:
            raise ValueError(
                f"positions[{index}] records an action in a terminal or "
                "otherwise oracle-illegal position"
            ) from error
        exact_regret = (result.outcome - action_value) / 2.0
        if not 0.0 <= exact_regret <= 1.0:
            raise AssertionError(
                f"positions[{index}] exact regret is outside [0, 1]"
            )
        action_optimal = record.action in result.optimal_actions
        if action_optimal != (action_value == result.outcome):
            raise AssertionError(
                f"positions[{index}] oracle action labels are inconsistent"
            )
        analyzed.append(
            {
                "trace_id": record.raw["trace_id"],
                "position_id": record.raw["position_id"],
                "empty_count": record.position.empty_count,
                "actor_agent": record.actor_agent,
                "action": record.action,
                "trace_record": record.raw,
                "oracle_position_id": record.position.position_id,
                "oracle_outcome": result.outcome,
                "oracle_optimal_actions": list(result.optimal_actions),
                "oracle_action_values": [
                    [action, value]
                    for action, value in result.action_values
                ],
                "action_value": action_value,
                "action_optimal": action_optimal,
                "exact_regret": exact_regret,
                "losing_move_on_winning_state": (
                    result.outcome == 1 and action_value == -1
                ),
            }
        )

    actor_agents = sorted(
        {str(record["actor_agent"]) for record in analyzed}
    )
    empty_counts = sorted(
        {int(record["empty_count"]) for record in analyzed}
    )
    by_actor_agent = {
        actor_agent: _summary(
            [
                record
                for record in analyzed
                if record["actor_agent"] == actor_agent
            ]
        )
        for actor_agent in actor_agents
    }
    by_empty_count = {
        str(empty_count): _summary(
            [
                record
                for record in analyzed
                if record["empty_count"] == empty_count
            ]
        )
        for empty_count in empty_counts
    }
    candidate_records = [
        record
        for record in analyzed
        if record["actor_agent"] == "candidate"
    ]
    candidate_summary = _summary(candidate_records)
    candidate_by_empty_count = {
        str(empty_count): _summary(
            [
                record
                for record in candidate_records
                if record["empty_count"] == empty_count
            ]
        )
        for empty_count in sorted(
            {
                int(record["empty_count"])
                for record in candidate_records
            }
        )
    }

    return {
        "schema_version": ANALYSIS_SCHEMA_VERSION,
        "kind": RESULT_KIND,
        "local_only": {
            "network_access": False,
            "external_logging": False,
            "external_export": False,
            "artifact_is_local": True,
            "immutable_output": True,
        },
        "input_trace": {
            "path": str(trace.path),
            "file_sha256": trace.file_sha256,
            "canonical_json_sha256": trace.canonical_json_sha256,
            "kind": trace.payload["kind"],
            "schema_version": trace.payload["schema_version"],
            "board_size": trace.board_size,
            "position_count": len(trace.records),
            "source": trace.payload["source"],
            "sampling": trace.payload["sampling"],
            "board_encoding": trace.payload["board_encoding"],
            "local_only": trace.payload.get("local_only"),
        },
        "oracle_contract": {
            "solver": "scacchi.hex_oracle.solve_hex",
            "perspective": (
                "oracle_outcome and action_value are from current_color's "
                "perspective"
            ),
            "action_optimal": (
                "recorded action belongs to oracle_optimal_actions"
            ),
            "exact_regret": "(oracle_outcome - action_value) / 2",
            "exact_regret_range": [0.0, 1.0],
            "losing_move_on_winning_state": (
                "oracle_outcome == 1 and action_value == -1"
            ),
            "losing_move_rate_denominator": (
                "decisions whose exact oracle_outcome is +1"
            ),
        },
        "headline": {
            "candidate_decisions": candidate_summary["decisions"],
            "candidate_decision_optimal_fraction": (
                candidate_summary["optimal_action_fraction"]
            ),
            "candidate_losing_move_on_winning_state_rate": (
                candidate_summary[
                    "losing_move_on_winning_state_rate"
                ]
            ),
            "candidate_mean_exact_regret": (
                candidate_summary["mean_exact_regret"]
            ),
        },
        "summaries": {
            "overall": _summary(analyzed),
            "candidate_decisions": candidate_summary,
            "by_actor_agent": by_actor_agent,
            "by_empty_count": by_empty_count,
            "candidate_by_empty_count": candidate_by_empty_count,
        },
        "records": analyzed,
    }


def _write_immutable_json(path: Path, payload: dict[str, Any]) -> str:
    """Atomically create ``path`` without ever replacing existing content."""

    resolved = path.resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if resolved.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable artifact: {resolved}"
        )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    descriptor, temporary_name = tempfile.mkstemp(
        dir=resolved.parent,
        prefix=f".{resolved.name}.",
        suffix=".tmp",
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, resolved)
        except FileExistsError as error:
            raise FileExistsError(
                f"refusing to overwrite immutable artifact: {resolved}"
            ) from error
    finally:
        temporary.unlink(missing_ok=True)
    return digest


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate and exactly solve every recorded action in a local "
            "scripts/hex_balanced_eval.py trace."
        )
    )
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    trace_path = args.trace.resolve()
    output_path = args.output.resolve()
    if trace_path == output_path:
        raise ValueError("--trace and --output must be different paths")
    if output_path.exists():
        raise FileExistsError(
            f"refusing to overwrite immutable artifact: {output_path}"
        )

    trace = load_trace(trace_path)
    result = analyze_trace(trace)
    script_path = Path(__file__).resolve()
    oracle_path = (
        script_path.parents[1] / "scacchi" / "hex_oracle.py"
    ).resolve()
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
        "script_path": str(script_path),
        "script_sha256": hashlib.sha256(
            script_path.read_bytes()
        ).hexdigest(),
        "oracle_path": str(oracle_path),
        "oracle_sha256": hashlib.sha256(
            oracle_path.read_bytes()
        ).hexdigest(),
    }
    output_sha256 = _write_immutable_json(output_path, result)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "sha256": output_sha256,
                "positions": len(result["records"]),
                "headline": result["headline"],
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
