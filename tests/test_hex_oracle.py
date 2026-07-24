from __future__ import annotations

from dataclasses import replace
from functools import lru_cache
import math
from pathlib import Path
import random
import runpy
from types import SimpleNamespace

import jax
import jax.numpy as jnp

from scacchi import checkpoint as checkpoint_io
from scacchi.envs import make_env
from scacchi.hex_oracle import (
    COLOR_0,
    COLOR_1,
    EMPTY,
    HexOracleResult,
    HexPosition,
    assess_policy_against_oracle,
    assess_policy_readout_noise,
    canonical_action_sequence,
    clear_hex_oracle_cache,
    compare_binary_outcome_probabilities,
    compare_policies_against_oracle,
    hex_has_connection,
    hex_oracle_cache_info,
    hex_oracle_value_cache_info,
    load_frozen_hex_corpus,
    position_from_pgx_state,
    sample_late_game_hex_corpus,
    solve_hex,
    solve_pgx_hex_state,
    write_frozen_hex_corpus,
)
from scacchi.types import Config, EnvConfig, config_to_dict


def test_hex_connectivity_matches_pgx_edge_orientation():
    size = 3
    horizontal = (
        COLOR_1,
        COLOR_1,
        COLOR_1,
        EMPTY,
        EMPTY,
        EMPTY,
        EMPTY,
        EMPTY,
        EMPTY,
    )
    vertical = (
        COLOR_0,
        EMPTY,
        EMPTY,
        COLOR_0,
        EMPTY,
        EMPTY,
        COLOR_0,
        EMPTY,
        EMPTY,
    )
    assert hex_has_connection(horizontal, size=size, color=1)
    assert not hex_has_connection(horizontal, size=size, color=0)
    assert hex_has_connection(vertical, size=size, color=0)
    assert not hex_has_connection(vertical, size=size, color=1)


def test_one_cell_hex_and_terminal_pgx_state_have_opposite_perspectives():
    initial = HexPosition(size=1, cells=(EMPTY,), current_color=0)
    result = solve_hex(initial)
    assert result.outcome == 1
    assert result.optimal_actions == (0,)
    assert result.action_values == ((0, 1),)

    env = make_env("hex", board_size=1)
    state = env.init(jax.random.PRNGKey(0))
    terminal = env.step(state, jnp.int32(0))
    assert bool(terminal.terminated)
    converted = position_from_pgx_state(terminal)
    assert converted.cells == (COLOR_0,)
    assert converted.current_color == 1
    assert solve_pgx_hex_state(terminal).outcome == -1
    assert solve_pgx_hex_state(terminal).optimal_actions == ()


def _brute_force_pgx(state, env):
    """Independent exhaustive reference using only PGX step/reward semantics."""

    actor = int(state.current_player)
    if bool(state.terminated):
        reward = float(state.rewards[actor])
        return int(math.copysign(1, reward)), ()

    legal = [
        action
        for action in range(env.size * env.size)
        if bool(state.legal_action_mask[action])
    ]
    values: list[tuple[int, int]] = []
    for action in legal:
        child = env.step(state, jnp.int32(action))
        if bool(child.terminated):
            value = int(math.copysign(1, float(child.rewards[actor])))
        else:
            child_outcome, _ = _brute_force_pgx(child, env)
            value = -child_outcome
        values.append((action, value))
    best = max(value for _, value in values)
    return best, tuple(action for action, value in values if value == best)


def test_solver_matches_brute_force_pgx_on_every_reachable_size_two_state():
    env = make_env("hex", board_size=2)
    initial = env.init(jax.random.PRNGKey(7))
    stack = [initial]
    visited: set[tuple[int, ...]] = set()
    while stack:
        state = stack.pop()
        position = position_from_pgx_state(state)
        key = (position.current_color, *position.cells)
        if key in visited:
            continue
        visited.add(key)
        expected_outcome, expected_actions = _brute_force_pgx(state, env)
        actual = solve_hex(position)
        assert actual.outcome == expected_outcome
        assert actual.optimal_actions == expected_actions
        if not bool(state.terminated):
            stack.extend(
                env.step(state, jnp.int32(action))
                for action in range(env.size * env.size)
                if bool(state.legal_action_mask[action])
            )


def _reference_solve_hex(position: HexPosition) -> HexOracleResult:
    """Former exhaustive recurrence, kept here as an independent reference."""

    @lru_cache(maxsize=None)
    def solve(
        cells: tuple[int, ...],
        current_color: int,
    ) -> HexOracleResult:
        color_0_wins = hex_has_connection(
            cells,
            size=position.size,
            color=0,
        )
        color_1_wins = hex_has_connection(
            cells,
            size=position.size,
            color=1,
        )
        if color_0_wins and color_1_wins:
            raise ValueError("invalid Hex position")
        if color_0_wins:
            outcome = 1 if current_color == 0 else -1
            return HexOracleResult(outcome, (), ())
        if color_1_wins:
            outcome = 1 if current_color == 1 else -1
            return HexOracleResult(outcome, (), ())

        legal_actions = tuple(
            action
            for action, cell in enumerate(cells)
            if cell == EMPTY
        )
        if not legal_actions:
            return HexOracleResult(0, (), ())

        values = []
        for action in legal_actions:
            child = list(cells)
            child[action] = current_color + 1
            child_cells = tuple(child)
            if hex_has_connection(
                child_cells,
                size=position.size,
                color=current_color,
            ):
                value = 1
            else:
                value = -solve(child_cells, 1 - current_color).outcome
            values.append((action, value))
        best = max(value for _, value in values)
        return HexOracleResult(
            outcome=best,
            optimal_actions=tuple(
                action for action, value in values if value == best
            ),
            action_values=tuple(values),
        )

    return solve(position.cells, position.current_color)


def test_bitmask_value_solver_matches_exhaustive_reference_action_vectors():
    rng = random.Random(819245)
    positions: list[HexPosition] = []
    for size, empty_count, count in (
        (3, 9, 1),
        (3, 6, 3),
        (3, 3, 3),
        (4, 8, 3),
        (4, 6, 3),
        (4, 4, 3),
    ):
        occupied_count = size * size - empty_count
        color_0_count = (occupied_count + 1) // 2
        while sum(
            position.size == size
            and position.empty_count == empty_count
            for position in positions
        ) < count:
            actions = list(range(size * size))
            rng.shuffle(actions)
            cells = [EMPTY] * (size * size)
            for action in actions[:color_0_count]:
                cells[action] = COLOR_0
            for action in actions[color_0_count:occupied_count]:
                cells[action] = COLOR_1
            cell_tuple = tuple(cells)
            if hex_has_connection(cell_tuple, size=size, color=0):
                continue
            if hex_has_connection(cell_tuple, size=size, color=1):
                continue
            positions.append(
                HexPosition(
                    size=size,
                    cells=cell_tuple,
                    current_color=occupied_count % 2,
                )
            )

    clear_hex_oracle_cache()
    for position in positions:
        assert solve_hex(position) == _reference_solve_hex(position)


def test_solver_cache_is_used_for_repeated_positions():
    clear_hex_oracle_cache()
    assert hex_oracle_cache_info().currsize == 0
    assert hex_oracle_value_cache_info().currsize == 0
    position = HexPosition(
        size=3,
        cells=(EMPTY,) * 9,
        current_color=0,
    )
    first = solve_hex(position)
    after_first = hex_oracle_cache_info()
    value_after_first = hex_oracle_value_cache_info()
    assert value_after_first.currsize > 0
    second = solve_hex(position)
    after_second = hex_oracle_cache_info()
    value_after_second = hex_oracle_value_cache_info()
    assert first == second
    assert after_second.hits == after_first.hits + 1
    assert value_after_second == value_after_first

    clear_hex_oracle_cache()
    assert hex_oracle_cache_info().currsize == 0
    assert hex_oracle_cache_info().hits == 0
    assert hex_oracle_value_cache_info().currsize == 0
    assert hex_oracle_value_cache_info().hits == 0


def test_policy_regret_and_proper_scores_reward_oracle_improvement():
    result = HexOracleResult(
        outcome=1,
        optimal_actions=(0,),
        action_values=((0, 1), (1, -1)),
    )
    prior = assess_policy_against_oracle((0.25, 0.75), result)
    search = assess_policy_against_oracle((0.75, 0.25), result)
    assert prior.expected_outcome == -0.5
    assert prior.regret == 1.5
    assert search.expected_outcome == 0.5
    assert search.regret == 0.5
    assert not prior.top_action_is_optimal
    assert search.top_action_is_optimal

    comparison = compare_policies_against_oracle(
        prior_policy=(0.25, 0.75),
        search_policy=(0.75, 0.25),
        result=result,
    )
    assert comparison.regret_reduction == 1.0
    assert comparison.proper_scores.log_score_gain > 0.0
    assert comparison.proper_scores.brier_score_gain > 0.0

    outcome_scores = compare_binary_outcome_probabilities(
        oracle_outcome=1,
        prior_win_probability=0.25,
        search_win_probability=0.75,
    )
    assert outcome_scores.target_distribution == (0.0, 1.0)
    assert outcome_scores.log_score_gain > 0.0
    assert outcome_scores.brier_score_gain > 0.0


def test_canonical_action_sequence_reconstructs_fixed_colour_position():
    position = HexPosition(
        size=3,
        cells=(
            COLOR_0,
            COLOR_1,
            COLOR_0,
            EMPTY,
            EMPTY,
            EMPTY,
            EMPTY,
            EMPTY,
            EMPTY,
        ),
        current_color=1,
    )
    actions = canonical_action_sequence(position)
    assert actions == (0, 1, 2)

    env = make_env("hex", board_size=3)
    state = env.init(jax.random.PRNGKey(3))
    for action in actions:
        state = env.step(state, jnp.asarray(action, dtype=jnp.int32))
    assert not bool(state.terminated)
    assert position_from_pgx_state(state) == position


def test_canonical_action_sequence_rejects_inconsistent_side_to_move():
    position = HexPosition(
        size=2,
        cells=(COLOR_0, COLOR_1, EMPTY, EMPTY),
        current_color=1,
    )
    try:
        canonical_action_sequence(position)
    except ValueError as error:
        assert "current_color disagrees" in str(error)
    else:
        raise AssertionError("inconsistent current_color was not rejected")


def test_fixed_tree_policy_readouts_separate_displacement_and_noise():
    result = HexOracleResult(
        outcome=1,
        optimal_actions=(0,),
        action_values=((0, 1), (1, -1)),
    )
    noiseless = assess_policy_readout_noise(
        prior_policy=(0.5, 0.5),
        search_policy_readouts=((0.75, 0.25), (0.75, 0.25)),
        result=result,
    )
    assert noiseless.readout_noise_squared_l2 == 0.0
    assert math.isclose(
        noiseless.noise_corrected_displacement_squared_l2,
        0.125,
    )
    assert noiseless.prior_to_mean_search_js_nats > 0.0

    noisy = assess_policy_readout_noise(
        prior_policy=(0.5, 0.5),
        search_policy_readouts=((0.8, 0.2), (0.6, 0.4)),
        result=result,
    )
    assert math.isclose(noisy.readout_noise_squared_l2, 0.04)
    assert math.isclose(
        noisy.noise_corrected_displacement_squared_l2,
        0.06,
    )
    assert noisy.mean_pairwise_search_readout_js_nats > 0.0


def test_policy_readout_noise_requires_two_fixed_tree_readouts():
    result = HexOracleResult(
        outcome=1,
        optimal_actions=(0,),
        action_values=((0, 1), (1, -1)),
    )
    try:
        assess_policy_readout_noise(
            prior_policy=(0.5, 0.5),
            search_policy_readouts=((0.75, 0.25),),
            result=result,
        )
    except ValueError as error:
        assert "at least two" in str(error)
    else:
        raise AssertionError("one readout was not rejected")


def test_checkpoint_oracle_cli_accepts_and_documents_exact_step():
    harness = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "hex_oracle_harness.py"
        )
    )
    parser = harness["_parser"]()
    args = parser.parse_args(
        [
            "checkpoint",
            "--corpus",
            "corpus.json",
            "--checkpoint",
            "checkpoints/run",
            "--checkpoint-step",
            "25",
        ]
    )
    assert args.checkpoint_step == 25

    subparser_action = next(
        action
        for action in parser._actions
        if "checkpoint" in (getattr(action, "choices", None) or {})
    )
    checkpoint_parser = subparser_action.choices["checkpoint"]
    assert "--checkpoint-step" in checkpoint_parser.format_help()


def test_checkpoint_oracle_accepts_up_to_fifteen_empty_cells():
    harness = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "hex_oracle_harness.py"
        )
    )
    assert harness["MAX_CHECKPOINT_ORACLE_EMPTY_CELLS"] == 15
    predict = harness["_predict_checkpoint"]
    args = SimpleNamespace(
        policy_readouts=2,
        batch_size=1,
        checkpoint=Path("checkpoints/run"),
        checkpoint_step=0,
    )

    too_early = SimpleNamespace(
        max_empty=16,
        positions=(
            SimpleNamespace(position=SimpleNamespace(empty_count=16)),
        ),
    )
    try:
        predict(args, too_early)
    except ValueError as error:
        assert "at most 15 empty cells" in str(error)
    else:
        raise AssertionError("16-empty checkpoint corpus was not rejected")

    class ReachedCheckpointLoad(Exception):
        pass

    def stop_at_checkpoint_load(*args, **kwargs):
        del args, kwargs
        raise ReachedCheckpointLoad

    predict.__globals__["_load_checkpoint_config_and_step"] = (
        stop_at_checkpoint_load
    )
    boundary = SimpleNamespace(
        max_empty=15,
        positions=(
            SimpleNamespace(position=SimpleNamespace(empty_count=15)),
        ),
    )
    try:
        predict(args, boundary)
    except ReachedCheckpointLoad:
        pass
    else:
        raise AssertionError("15-empty checkpoint corpus did not pass guard")


def test_checkpoint_oracle_validates_step_and_reads_its_compute_counters(
    monkeypatch,
):
    harness = runpy.run_path(
        str(
            Path(__file__).resolve().parents[1]
            / "scripts"
            / "hex_oracle_harness.py"
        )
    )
    raw_config = config_to_dict(
        Config(env=EnvConfig(id="hex", board_size=2, num_outcomes=2))
    )
    restored_steps: list[int] = []

    class FakeCheckpointManager:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            del args

        def all_steps(self):
            return [0, 25]

        def latest_step(self):
            return 25

        def restore(self, step, *, args):
            del args
            restored_steps.append(step)
            return {
                "meta": {
                    "config": raw_config,
                    "step": step,
                    "hours": 1.25,
                    "frames": 1234,
                    "optimizer_updates": 56,
                }
            }

    monkeypatch.setattr(
        checkpoint_io.ocp,
        "CheckpointManager",
        FakeCheckpointManager,
    )
    load = harness["_load_checkpoint_config_and_step"]

    config, step, progress = load(Path("checkpoints/run"), 0)
    assert config.env.id == "hex"
    assert step == 0
    assert restored_steps == [0]
    assert progress == {
        "checkpoint_hours": 1.25,
        "checkpoint_frames": 1234,
        "checkpoint_optimizer_updates": 56,
        "checkpoint_completed_iterations": 1,
    }

    try:
        load(Path("checkpoints/run"), 10)
    except FileNotFoundError as error:
        assert "available steps: [0, 25]" in str(error)
    else:
        raise AssertionError("missing checkpoint step was not rejected")


def test_frozen_hex_corpus_round_trip_and_label_verification(tmp_path):
    corpus = sample_late_game_hex_corpus(
        count=4,
        size=3,
        min_empty=2,
        max_empty=3,
        seed=11,
        balanced_outcomes=False,
    )
    path = tmp_path / "oracle.json"
    write_frozen_hex_corpus(path, corpus)
    assert load_frozen_hex_corpus(path) == corpus

    first = corpus.positions[0]
    corrupted = replace(
        first,
        oracle_outcome=-first.oracle_outcome,
    )
    corrupted_corpus = replace(
        corpus,
        positions=(corrupted, *corpus.positions[1:]),
    )
    write_frozen_hex_corpus(path, corrupted_corpus)
    try:
        load_frozen_hex_corpus(path)
    except ValueError as error:
        assert "oracle labels disagree" in str(error)
    else:
        raise AssertionError("corrupted oracle labels were not rejected")
