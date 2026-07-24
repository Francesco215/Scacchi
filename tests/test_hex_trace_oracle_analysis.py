from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import hex_balanced_eval as balanced
from scripts import hex_trace_oracle_analysis as analysis


def _record(
    *,
    cells: tuple[int, ...],
    current_color: int,
    action: int,
    actor_agent: str,
    source: int,
) -> balanced.TraceRecord:
    candidate_player_id = 0
    actor_player_id = 0 if actor_agent == "candidate" else 1
    return balanced.TraceRecord(
        empty_count=cells.count(0),
        cells=cells,
        current_color=current_color,
        action=action,
        actor_player_id=actor_player_id,
        actor_agent=actor_agent,
        candidate_player_id=candidate_player_id,
        candidate_seat="first",
        final_candidate_return=1 if source % 2 == 0 else -1,
        stratum_index=source % 4,
        chunk_index=source // 4,
        row_index=source,
    )


def _trace_payload() -> dict:
    records = [
        # Hex2's empty board is winning, but corner 0 loses exactly.
        _record(
            cells=(0, 0, 0, 0),
            current_color=0,
            action=0,
            actor_agent="candidate",
            source=0,
        ),
        # This winning colour-1 state has unique optimal action 1.
        _record(
            cells=(0, 0, 0, 1),
            current_color=1,
            action=1,
            actor_agent="candidate",
            source=1,
        ),
        # This colour-1 state is lost; every legal action is optimal.
        _record(
            cells=(0, 0, 1, 0),
            current_color=1,
            action=0,
            actor_agent="baseline",
            source=2,
        ),
    ]
    return balanced.build_trace_payload(
        records,
        empty_counts=(3, 4),
        per_empty_limit=8,
        seed=91,
        board_size=2,
        source={
            "candidate_checkpoint": {
                "directory": "/tmp/candidate",
                "selected_step": 200,
                "selection_mode": "exact",
            },
            "baseline_checkpoint": {
                "directory": "/tmp/baseline",
                "selected_step": 300,
                "selection_mode": "latest",
            },
        },
    )


def _write_trace(path: Path, payload: dict | None = None) -> Path:
    path.write_text(
        json.dumps(
            _trace_payload() if payload is None else payload,
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def test_exact_trace_analysis_labels_actions_and_candidate_failures(
    tmp_path: Path,
):
    trace_path = _write_trace(tmp_path / "trace.json")
    loaded = analysis.load_trace(trace_path)
    result = analysis.analyze_trace(loaded)

    assert loaded.board_size == 2
    assert result["input_trace"]["file_sha256"] == hashlib.sha256(
        trace_path.read_bytes()
    ).hexdigest()
    assert result["input_trace"]["source"] == _trace_payload()["source"]

    by_cells = {
        tuple(record["trace_record"]["cells"]): record
        for record in result["records"]
    }
    losing_move = by_cells[(0, 0, 0, 0)]
    assert losing_move["oracle_outcome"] == 1
    assert losing_move["action_value"] == -1
    assert losing_move["action_optimal"] is False
    assert losing_move["exact_regret"] == 1.0
    assert losing_move["losing_move_on_winning_state"] is True

    winning_move = by_cells[(0, 0, 0, 1)]
    assert winning_move["oracle_outcome"] == 1
    assert winning_move["action_value"] == 1
    assert winning_move["action_optimal"] is True
    assert winning_move["exact_regret"] == 0.0

    forced_loss = by_cells[(0, 0, 1, 0)]
    assert forced_loss["oracle_outcome"] == -1
    assert forced_loss["action_value"] == -1
    assert forced_loss["action_optimal"] is True
    assert forced_loss["exact_regret"] == 0.0

    headline = result["headline"]
    assert headline == {
        "candidate_decisions": 2,
        "candidate_decision_optimal_fraction": 0.5,
        "candidate_losing_move_on_winning_state_rate": 0.5,
        "candidate_mean_exact_regret": 0.5,
    }
    summaries = result["summaries"]
    assert summaries["overall"]["optimal_action_fraction"] == pytest.approx(
        2 / 3
    )
    assert summaries["overall"]["mean_exact_regret"] == pytest.approx(1 / 3)
    assert summaries["by_actor_agent"]["candidate"] == (
        summaries["candidate_decisions"]
    )
    assert summaries["by_actor_agent"]["baseline"][
        "optimal_action_fraction"
    ] == 1.0
    assert set(summaries["by_empty_count"]) == {"3", "4"}
    assert summaries["candidate_by_empty_count"]["4"][
        "losing_move_on_winning_state_rate"
    ] == 1.0
    assert all(
        0.0 <= record["exact_regret"] <= 1.0
        for record in result["records"]
    )


def test_loader_rejects_an_illegal_recorded_action(tmp_path: Path):
    payload = _trace_payload()
    record = payload["positions"][0]
    record["action"] = record["cells"].index(1)
    trace_path = _write_trace(tmp_path / "illegal.json", payload)

    with pytest.raises(ValueError, match="action=.*not legal"):
        analysis.load_trace(trace_path)


def test_loader_verifies_content_derived_trace_provenance(tmp_path: Path):
    payload = _trace_payload()
    payload["positions"][0]["position_id"] = "0" * 64
    trace_path = _write_trace(tmp_path / "corrupt.json", payload)

    with pytest.raises(
        ValueError,
        match="position_id disagrees with cells/current_color",
    ):
        analysis.load_trace(trace_path)


def test_cli_writes_local_artifact_once_and_preserves_source(
    tmp_path: Path,
):
    trace_path = _write_trace(tmp_path / "trace.json")
    output = tmp_path / "oracle-analysis.json"
    argv = [
        "--trace",
        str(trace_path),
        "--output",
        str(output),
    ]

    analysis.main(argv)
    artifact = json.loads(output.read_text(encoding="utf-8"))

    assert artifact["kind"] == analysis.RESULT_KIND
    assert artifact["local_only"]["external_export"] is False
    assert artifact["local_only"]["immutable_output"] is True
    assert artifact["input_trace"]["path"] == str(trace_path.resolve())
    assert artifact["input_trace"]["position_count"] == 3
    assert artifact["input_trace"]["source"] == _trace_payload()["source"]
    assert len(artifact["reproduction"]["script_sha256"]) == 64
    assert len(artifact["reproduction"]["oracle_sha256"]) == 64

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        analysis.main(argv)
