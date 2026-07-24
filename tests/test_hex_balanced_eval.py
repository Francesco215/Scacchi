from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from types import SimpleNamespace

from flax import nnx
import jax
import jax.numpy as jnp
import pytest

from scripts import hex_balanced_eval as harness
from scacchi.envs import make_env
from scacchi.play_search import PlayerOutput


def _record(
    *,
    empty_count: int,
    source: int,
    cells: tuple[int, ...] | None = None,
) -> harness.TraceRecord:
    if cells is None:
        cells = (1, 2, 0, 0)
    return harness.TraceRecord(
        empty_count=empty_count,
        cells=cells,
        current_color=0,
        action=cells.index(0),
        actor_player_id=source % 2,
        actor_agent="candidate" if source % 2 == 0 else "baseline",
        candidate_player_id=0,
        candidate_seat="first",
        final_candidate_return=1 if source % 2 == 0 else -1,
        stratum_index=source % 4,
        chunk_index=source // 4,
        row_index=source,
    )


def test_exact_checkpoint_selection_does_not_substitute_latest(
    tmp_path: Path,
):
    selection = harness.select_checkpoint_step(
        tmp_path / "run",
        (0, 25, 50, 100),
        requested_step=25,
    )

    assert selection.selected_step == 25
    assert selection.requested_step == 25
    assert selection.selection_mode == "exact"
    assert selection.provenance() == {
        "directory": str((tmp_path / "run").resolve()),
        "requested_step": 25,
        "selected_step": 25,
        "selection_mode": "exact",
        "available_steps": [0, 25, 50, 100],
    }


def test_checkpoint_selection_records_latest_and_rejects_missing(
    tmp_path: Path,
):
    latest = harness.select_checkpoint_step(
        tmp_path / "baseline",
        (7, 2, 7),
        requested_step=None,
    )
    assert latest.selected_step == 7
    assert latest.selection_mode == "latest"
    assert latest.provenance()["requested_step"] is None

    with pytest.raises(FileNotFoundError, match="available steps.*2, 7"):
        harness.select_checkpoint_step(
            tmp_path / "baseline",
            (2, 7),
            requested_step=5,
        )


def test_model_restore_uses_selected_step_not_manager_latest(monkeypatch):
    class EmptyModel(nnx.Module):
        pass

    restored_steps: list[int] = []

    class FakeManager:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

        def all_steps(self):
            return (25, 100)

        def restore(self, step, *, args):
            del args
            restored_steps.append(step)
            return {"model": object()}

    monkeypatch.setattr(harness, "build_model", lambda *args, **kwargs: EmptyModel())
    monkeypatch.setattr(
        harness.checkpoint_io,
        "_checkpoint_manager_options",
        lambda **kwargs: None,
    )
    monkeypatch.setattr(
        harness.checkpoint_io.ocp,
        "CheckpointManager",
        FakeManager,
    )
    monkeypatch.setattr(harness.nnx, "update", lambda *args: None)
    selection = harness.CheckpointSelection(
        directory=Path("/tmp/checkpoints/run"),
        requested_step=25,
        selected_step=25,
        available_steps=(25, 100),
    )
    loaded = harness.LoadedCheckpointMetadata(
        selection=selection,
        metadata={"step": 25},
        config=object(),
    )
    env = SimpleNamespace(num_actions=4, observation_shape=(2, 2, 1))

    model = harness.load_model_at_step(loaded, env)

    assert isinstance(model, EmptyModel)
    assert restored_steps == [25]


def test_four_strata_cover_each_player_id_and_seat_once():
    assert [
        (
            stratum.candidate_player_id,
            stratum.candidate_seat,
            harness.stratum_player_order(stratum),
        )
        for stratum in harness.STRATUM_SPEC
    ] == [
        (0, "first", (0, 1)),
        (0, "second", (1, 0)),
        (1, "first", (1, 0)),
        (1, "second", (0, 1)),
    ]
    assert harness.validate_evaluation_shape(4096, 256) == 1024


def test_game_return_pairing_order_is_stable_across_readouts():
    seed = 73
    run_keys = jax.random.split(jax.random.PRNGKey(seed), 8)
    m32_returns = [
        [[1, -1], [1, 1]],
        [[-1, -1], [1, -1]],
        [[1, 1], [-1, 1]],
        [[-1, 1], [-1, -1]],
    ]
    q21_returns = [
        [[1, 1], [1, 1]],
        [[-1, -1], [1, -1]],
        [[1, -1], [-1, 1]],
        [[-1, 1], [1, -1]],
    ]

    m32 = harness.build_game_returns_payload(
        m32_returns,
        run_keys=run_keys,
        seed=seed,
        games=16,
        games_per_stratum=4,
        batch_size=2,
    )
    q21 = harness.build_game_returns_payload(
        q21_returns,
        run_keys=run_keys,
        seed=seed,
        games=16,
        games_per_stratum=4,
        batch_size=2,
    )

    assert m32["pairing_layout_sha256"] == q21["pairing_layout_sha256"]
    assert m32["pairing_contract"] == q21["pairing_contract"]
    for m32_stratum, q21_stratum in zip(
        m32["strata"],
        q21["strata"],
        strict=True,
    ):
        assert m32_stratum["stratum_index"] == q21_stratum["stratum_index"]
        for m32_chunk, q21_chunk in zip(
            m32_stratum["chunks"],
            q21_stratum["chunks"],
            strict=True,
        ):
            for field in (
                "chunk_index",
                "run_key_index",
                "rng_key_data",
                "global_game_index_start",
                "game_count",
            ):
                assert m32_chunk[field] == q21_chunk[field]


def test_game_return_summary_matches_serialized_raw_returns():
    raw = [
        [[1, -1]],
        [[1, 1]],
        [[-1, -1]],
        [[1, -1]],
    ]
    payload = harness.build_game_returns_payload(
        raw,
        run_keys=jax.random.split(jax.random.PRNGKey(5), 4),
        seed=5,
        games=8,
        games_per_stratum=2,
        batch_size=2,
    )
    flat = harness.flatten_game_returns(payload)

    assert flat == [1, -1, 1, 1, -1, -1, 1, -1]
    assert payload["overall"] == harness.summarize_returns(flat)
    assert sum(
        stratum["summary"]["games"]
        for stratum in payload["strata"]
    ) == payload["games"]
    assert payload["overall"]["wins"] == 4
    assert payload["overall"]["losses"] == 4
    assert len(payload["returns_sha256"]) == 64


def test_game_returns_are_opt_in_in_cli():
    required = [
        "--candidate",
        "checkpoints/run",
        "--step",
        "25",
        "--output",
        "result.json",
    ]
    assert harness._parser().parse_args(required).include_game_returns is False
    assert (
        harness._parser()
        .parse_args([*required, "--include-game-returns"])
        .include_game_returns
        is True
    )


def test_root_action_override_is_opt_in_and_scope_limited():
    @dataclass
    class Active:
        root_action_estimator: str = "winner_mc"
        root_policy_target_estimator: str = "winner_mc"

    @dataclass
    class Search:
        dirichlet_thompson: Active
        kind: str = "dirichlet_thompson"

    @dataclass
    class Eval:
        player_search: Search

    @dataclass
    class Config:
        eval: Eval

    original = Config(eval=Eval(player_search=Search(Active())))
    untouched, exact = harness.override_candidate_root_action_estimator(
        original,
        None,
    )
    overridden, changed = (
        harness.override_candidate_root_action_estimator(
            original,
            "prefix_cdf",
        )
    )

    assert untouched is original
    assert exact["checkpoint_exact"] is True
    assert (
        overridden.eval.player_search.dirichlet_thompson.root_action_estimator
        == "prefix_cdf"
    )
    assert (
        overridden.eval.player_search.dirichlet_thompson
        .root_policy_target_estimator
        == "winner_mc"
    )
    assert (
        original.eval.player_search.dirichlet_thompson.root_action_estimator
        == "winner_mc"
    )
    assert changed["stored"] == "winner_mc"
    assert changed["effective"] == "prefix_cdf"
    assert changed["checkpoint_exact"] is False


def test_root_action_override_never_substitutes_policy_target_field():
    @dataclass
    class Active:
        root_policy_target_estimator: str = "winner_mc"

    @dataclass
    class Search:
        kind: str
        dirichlet_thompson: Active

    @dataclass
    class Eval:
        player_search: Search

    @dataclass
    class Config:
        eval: Eval

    config = Config(
        eval=Eval(
            player_search=Search(
                kind="dirichlet_thompson",
                dirichlet_thompson=Active(),
            )
        )
    )
    with pytest.raises(
        ValueError,
        match="root_policy_target_estimator is intentionally not substituted",
    ):
        harness.override_candidate_root_action_estimator(
            config,
            "prefix_cdf",
        )


def test_kappa_override_is_opt_in_and_scope_limited():
    @dataclass
    class Active:
        kappa: float = 3.0
        policy_samples: int = 32

    @dataclass
    class Search:
        dirichlet_thompson: Active
        kind: str = "dirichlet_thompson"

    @dataclass
    class Eval:
        player_search: Search

    @dataclass
    class Config:
        eval: Eval

    original = Config(eval=Eval(player_search=Search(Active())))
    untouched, exact = harness.override_candidate_kappa(original, None)
    overridden, changed = harness.override_candidate_kappa(original, 0.5)

    assert untouched is original
    assert exact == {
        "requested_override": None,
        "field_present": True,
        "stored": 3.0,
        "effective": 3.0,
        "checkpoint_exact": True,
    }
    assert overridden.eval.player_search.dirichlet_thompson.kappa == 0.5
    assert (
        overridden.eval.player_search.dirichlet_thompson.policy_samples
        == 32
    )
    assert original.eval.player_search.dirichlet_thompson.kappa == 3.0
    assert changed["stored"] == 3.0
    assert changed["effective"] == 0.5
    assert changed["checkpoint_exact"] is False


@pytest.mark.parametrize("kappa", ["0", "-1", "nan", "inf", "-inf"])
def test_candidate_kappa_cli_requires_positive_finite_value(kappa: str):
    required = [
        "--candidate",
        "checkpoints/run",
        "--step",
        "25",
        "--output",
        "result.json",
    ]
    with pytest.raises(SystemExit):
        harness._parser().parse_args(
            [*required, "--candidate-kappa", kappa]
        )


def test_prefix_grid_override_records_q21_and_preserves_other_fields():
    @dataclass
    class Active:
        prefix_cdf_half_width: int = 20
        kappa: float = 3.0

    @dataclass
    class Search:
        dirichlet_thompson: Active
        kind: str = "dirichlet_thompson"

    @dataclass
    class Eval:
        player_search: Search

    @dataclass
    class Config:
        eval: Eval

    original = Config(eval=Eval(player_search=Search(Active())))
    overridden, provenance = (
        harness.override_candidate_prefix_cdf_half_width(original, 10)
    )

    active = overridden.eval.player_search.dirichlet_thompson
    assert active.prefix_cdf_half_width == 10
    assert active.kappa == 3.0
    assert original.eval.player_search.dirichlet_thompson.prefix_cdf_half_width == 20
    assert provenance["stored"] == 20
    assert provenance["effective"] == 10
    assert provenance["effective_grid_points"] == 21
    assert provenance["checkpoint_exact"] is False


@pytest.mark.parametrize(
    ("games", "batch_size", "message"),
    [
        (0, 1, "positive"),
        (10, 1, "divisible by four"),
        (12, 2, "games per stratum"),
    ],
)
def test_balanced_shape_rejects_partial_strata(
    games: int,
    batch_size: int,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        harness.validate_evaluation_shape(games, batch_size)


def test_trace_selection_is_deterministic_order_independent_and_deduplicated():
    duplicate = _record(empty_count=2, source=8)
    records = [
        _record(empty_count=2, source=0),
        _record(
            empty_count=2,
            source=1,
            cells=(1, 0, 2, 0),
        ),
        duplicate,
    ]
    selected_forward, counts_forward = harness.select_trace_records(
        records,
        empty_counts=(2,),
        per_empty_limit=8,
        seed=123,
    )
    selected_reverse, counts_reverse = harness.select_trace_records(
        list(reversed(records)),
        empty_counts=(2,),
        per_empty_limit=8,
        seed=123,
    )

    assert selected_forward == selected_reverse
    assert counts_forward == counts_reverse == {
        "2": {
            "eligible": 3,
            "unique_positions": 2,
            "selected": 2,
        }
    }
    assert len({record.state_key for record in selected_forward}) == 2


def test_trace_payload_round_trips_board_action_and_provenance(
    tmp_path: Path,
):
    record = _record(
        empty_count=2,
        source=0,
        cells=(1, 2, 0, 0),
    )
    payload = harness.build_trace_payload(
        [record],
        empty_counts=(2,),
        per_empty_limit=4,
        seed=99,
        board_size=2,
        source={
            "candidate_checkpoint": {
                "selected_step": 25,
                "selection_mode": "exact",
            }
        },
    )
    output = tmp_path / "trace.json"
    digest = harness._write_json(output, payload)
    decoded = json.loads(output.read_text(encoding="utf-8"))
    position = decoded["positions"][0]

    assert len(digest) == 64
    assert decoded["schema_version"] == harness.TRACE_SCHEMA_VERSION
    assert decoded["source"]["candidate_checkpoint"]["selected_step"] == 25
    assert position["cells"] == [1, 2, 0, 0]
    assert position["empty_count"] == 2
    assert position["action"] == 2
    assert position["current_color"] == 0
    assert position["ply_index"] == 2
    assert position["actor_agent"] == "candidate"
    assert position["candidate_won"] is True
    assert len(position["position_id"]) == 64
    assert len(position["trace_id"]) == 64


def test_jitted_pgx_trace_captures_legal_pre_action_board(monkeypatch):
    class EmptyModel(nnx.Module):
        pass

    def fake_make_search_player(*args, **kwargs):
        del args, kwargs

        def player(state, key):
            return PlayerOutput(
                action=jax.random.categorical(
                    key,
                    jnp.where(
                        state.legal_action_mask,
                        0.0,
                        -1.0e9,
                    ),
                ).astype(jnp.int32),
                posterior=None,
            )

        return player

    monkeypatch.setattr(
        harness,
        "make_search_player",
        fake_make_search_player,
    )
    monkeypatch.setattr(
        harness,
        "baseline_search_config",
        lambda config: config.eval.player_search,
    )
    config = SimpleNamespace(
        env=SimpleNamespace(board_size=3),
        eval=SimpleNamespace(
            player_search=object(),
            player_action_commitment_type=object(),
            baseline_action_commitment_type=object(),
        ),
        training=SimpleNamespace(
            losses=SimpleNamespace(q_loss_weight_mode="uniform")
        ),
    )
    evaluator = harness.make_balanced_evaluator(
        make_env("hex", 3),
        config,
        EmptyModel(),
        batch_size=2,
        trace_empty_counts=(8,),
    )
    evaluator_without_trace = harness.make_balanced_evaluator(
        make_env("hex", 3),
        config,
        EmptyModel(),
        batch_size=2,
    )

    key = jax.random.PRNGKey(17)
    output = evaluator(
        key,
        EmptyModel(),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(False),
    )
    output_without_trace = evaluator_without_trace(
        key,
        EmptyModel(),
        jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(False),
    )
    host_returns = jax.device_get(output.returns).astype(int).tolist()
    records = harness.trace_records_from_chunk(
        output.trace,
        host_returns,
        trace_empty_counts=(8,),
        stratum=harness.Stratum(1, False, "second"),
        stratum_index=3,
        chunk_index=0,
    )

    assert len(records) == 2
    assert jnp.array_equal(output.returns, output_without_trace.returns)
    assert all(record.cells.count(0) == 8 for record in records)
    assert all(record.cells[record.action] == 0 for record in records)
    assert all(record.current_color == 1 for record in records)
    assert all(record.actor_player_id == 1 for record in records)
    assert all(record.actor_agent == "candidate" for record in records)


def test_trace_cli_rejects_positions_too_early_for_exact_oracle():
    assert harness._parse_empty_counts("15,10,15") == (10, 15)
    with pytest.raises(
        argparse.ArgumentTypeError,
        match="between 1 and 15",
    ):
        harness._parse_empty_counts("16")
