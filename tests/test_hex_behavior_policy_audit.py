from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from scacchi.envs import make_env
from scacchi.types import (
    ActionCommitmentType,
    DirichletThompsonSearchConfig,
    PosteriorPolicyEstimator,
    SearchConfig,
    SearchKind,
)
from scripts import hex_behavior_policy_audit as audit


def _host_data() -> audit.HostModeData:
    # Four complete two-ply games on a 2x2 toy action/state space.  The
    # summarizer is environment-agnostic; the executable audit itself checks
    # that the restored environment is Hex.
    valid = np.ones((4, 2), dtype=bool)
    cells = np.asarray(
        [
            [[0, 0, 0, 0], [1, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 1, 0, 0]],
            [[0, 0, 0, 0], [1, 0, 0, 0]],
            [[0, 0, 0, 0], [0, 0, 1, 0]],
        ],
        dtype=np.int8,
    )
    current_color = np.asarray(
        [[0, 1], [0, 1], [0, 1], [0, 1]],
        dtype=np.int8,
    )
    action = np.asarray(
        [[0, 1], [1, 0], [0, 2], [2, 0]],
        dtype=np.int16,
    )
    legal_action_mask = np.asarray(
        [
            [[True, True, True, False], [False, True, True, False]],
            [[True, True, True, False], [True, False, True, False]],
            [[True, True, True, False], [False, True, True, False]],
            [[True, True, True, False], [True, False, True, False]],
        ],
        dtype=bool,
    )
    q21_policy = np.asarray(
        [
            [[0.5, 0.3, 0.2, 0.0], [0.0, 0.5, 0.5, 0.0]],
            [[0.2, 0.5, 0.3, 0.0], [0.5, 0.0, 0.5, 0.0]],
            [[0.5, 0.2, 0.3, 0.0], [0.0, 0.4, 0.6, 0.0]],
            [[0.2, 0.3, 0.5, 0.0], [0.6, 0.0, 0.4, 0.0]],
        ],
        dtype=np.float32,
    )
    all_true = np.ones((4, 2), dtype=bool)
    all_false = np.zeros((4, 2), dtype=bool)
    return audit.HostModeData(
        valid=valid,
        cells=cells,
        current_color=current_color,
        action=action,
        legal_action_mask=legal_action_mask,
        q21_policy=q21_policy,
        root_solved=all_false,
        target_prefix_eligible=all_true,
        target_prefix_accepted=all_true,
        target_prefix_fallback=all_false,
        action_prefix_eligible=all_true,
        action_prefix_accepted=all_true,
        action_prefix_fallback=all_false,
        prefix_tail_clipped=all_false,
        prefix_density_guard=all_false,
        prefix_nonfinite=all_false,
        first_player_return=np.asarray([1, -1, 1, -1], dtype=np.float32),
        game_length=np.asarray([2, 2, 2, 2], dtype=np.int16),
        completed=np.ones((4,), dtype=bool),
    )


def test_mode_searches_share_dt32_q21_and_change_only_commitment():
    stored = SearchConfig(
        kind=SearchKind.dirichlet_thompson,
        dirichlet_thompson=DirichletThompsonSearchConfig(
            num_simulations=7,
            max_depth=7,
            kappa=9.0,
            policy_samples=11,
            posterior_policy_estimator=PosteriorPolicyEstimator.winner_mc,
            root_policy_target_estimator=PosteriorPolicyEstimator.winner_mc,
            root_action_estimator=PosteriorPolicyEstimator.winner_mc,
            prefix_cdf_half_width=20,
        ),
    )

    effective = {
        spec.mode_id: audit.build_mode_search(
            stored,
            spec,
            posterior_sample_temperature=0.4,
        )
        for spec in audit.MODE_SPECS
    }

    for spec in audit.MODE_SPECS:
        search = effective[spec.mode_id]
        active = search.dirichlet_thompson
        assert search.posterior_plurality_samples == 32
        assert search.posterior_sample_temperature == (
            0.4 if spec.uses_configured_sample_temperature else 1.0
        )
        assert active.num_simulations == 32
        assert active.max_depth == 32
        assert active.policy_samples == 32
        assert active.kappa == 3.0
        assert active.prefix_cdf_half_width == 10
        assert (
            active.posterior_policy_estimator
            == PosteriorPolicyEstimator.prefix_cdf
        )
        assert (
            active.root_policy_target_estimator
            == PosteriorPolicyEstimator.prefix_cdf
        )
        assert active.root_action_estimator == spec.root_action_estimator

    assert {
        spec.mode_id: spec.action_commitment_type
        for spec in audit.MODE_SPECS
    } == {
        "q21_posterior_sample": ActionCommitmentType.posterior_sample,
        "q21_posterior_temperature": (
            ActionCommitmentType.posterior_sample
        ),
        "q21_posterior_plurality32": (
            ActionCommitmentType.posterior_plurality
        ),
        "q21_posterior_argmax": ActionCommitmentType.posterior_argmax,
        "m32_posterior_argmax": ActionCommitmentType.posterior_argmax,
    }


def test_mode_search_rejects_non_dt_and_invalid_numeric_overrides():
    with pytest.raises(ValueError, match="requires selfplay.search.kind"):
        audit.build_mode_search(SearchConfig(), audit.MODE_SPECS[0])

    stored = SearchConfig(kind=SearchKind.dirichlet_thompson)
    with pytest.raises(ValueError, match="kappa"):
        audit.build_mode_search(stored, audit.MODE_SPECS[0], kappa=0.0)
    with pytest.raises(ValueError, match="policy_samples"):
        audit.build_mode_search(
            stored,
            audit.MODE_SPECS[0],
            policy_samples=0,
        )
    with pytest.raises(ValueError, match="posterior_plurality_samples"):
        audit.build_mode_search(
            stored,
            audit.MODE_SPECS[0],
            posterior_plurality_samples=0,
        )
    with pytest.raises(ValueError, match="posterior_sample_temperature"):
        audit.build_mode_search(
            stored,
            audit.MODE_SPECS[0],
            posterior_sample_temperature=0.0,
        )


def test_pgx_hex_swap_action_is_separate_from_board_cell_storage():
    env = make_env("hex", 2)

    assert env.size == 2
    assert env.num_actions == 5
    # Exercise the public assumptions used by the compiled audit geometry:
    # four cells are recorded per state, while five action ids/max plies are
    # needed to include the pie-rule action.
    assert env.size * env.size == 4
    assert env.num_actions == env.size * env.size + 1


def test_summary_reports_state_prefix_game_and_q_calibration_metrics():
    result = audit.summarize_mode(
        _host_data(),
        num_actions=4,
        sampling_calibration_applicable=True,
    )

    assert result["games"]["games"] == 4
    assert result["games"]["first_player_win_rate"] == 0.5
    assert result["games"]["second_player_win_rate"] == 0.5
    assert result["games"]["length_mean"] == 2.0
    assert result["games"]["terminal_events_per_1k_frames"] == 500.0

    ply_zero = result["by_ply"][0]
    assert ply_zero["states"]["sample_count"] == 4
    assert ply_zero["states"]["unique_count"] == 1
    assert ply_zero["ordered_prefixes"]["unique_count"] == 3
    assert ply_zero["actions"]["counts"] == [2, 1, 1, 0]

    ply_one = result["by_ply"][1]
    assert ply_one["states"]["unique_count"] == 3
    assert ply_one["ordered_prefixes"]["unique_count"] == 4

    q = result["commitment_q21"]
    assert q["q21_accepted_unresolved_count"] == 8
    assert q["sampling_calibration_applicable"] is True
    assert sum(q["observed_action_counts"]) == 8
    assert sum(q["expected_action_counts_under_q21_sampling"]) == pytest.approx(
        8.0
    )
    assert np.isfinite(q["multinomial_mahalanobis_statistic"])
    assert result["numeric_guards"]["target"]["acceptance_fraction"] == 1.0
    assert result["numeric_guards"]["target"]["fallback_count"] == 0


def test_solved_roots_are_excluded_from_commitment_q_population():
    data = _host_data()
    solved = data.root_solved.copy()
    solved[:, 1] = True
    result = audit.summarize_mode(
        audit.HostModeData(
            **{
                field: (
                    solved
                    if field == "root_solved"
                    else getattr(data, field)
                )
                for field in data.__dataclass_fields__
            }
        ),
        num_actions=4,
        sampling_calibration_applicable=True,
    )

    assert result["commitment_q21"]["q21_accepted_unresolved_count"] == 4


def test_common_coordinates_are_stable_and_validate_shape():
    first_keys, first = audit.common_coordinate_layout(
        seed=17,
        games=8,
        batch_size=4,
    )
    second_keys, second = audit.common_coordinate_layout(
        seed=17,
        games=8,
        batch_size=4,
    )

    np.testing.assert_array_equal(first_keys, second_keys)
    assert first == second
    assert len(first["chunks"]) == 2
    assert first["chunks"][1]["global_game_index_start"] == 4

    with pytest.raises(ValueError, match="divisible"):
        audit.common_coordinate_layout(seed=0, games=7, batch_size=4)


def test_host_mode_data_concatenates_chunks():
    data = _host_data()

    def chunk(rows: slice) -> audit.AuditChunk:
        return audit.AuditChunk(
            **{
                field: jnp.asarray(getattr(data, field)[rows])
                for field in data.__dataclass_fields__
            }
        )

    joined = audit.host_mode_data((chunk(slice(0, 2)), chunk(slice(2, 4))))

    np.testing.assert_array_equal(joined.action, data.action)
    np.testing.assert_array_equal(
        joined.first_player_return,
        data.first_player_return,
    )


def test_mode_comparison_includes_plurality_against_native_m32():
    def summarized(scale: float) -> dict:
        return {
            "by_ply": [
                {
                    "actions": {
                        "effective_support": 20.0 * scale,
                    }
                }
            ],
            "early_plies_0_to_10": {
                "states": {
                    "effective_support": 100.0 * scale,
                    "unique_count": int(100 * scale),
                },
                "ordered_prefixes": {
                    "effective_support": 80.0 * scale,
                    "unique_count": int(80 * scale),
                },
                "actions": {
                    "effective_support": 20.0 * scale,
                },
            },
            "games": {
                "terminal_events_per_1k_frames": 50.0 * scale,
                "length_mean": 20.0 + 0.1 * (scale - 1.0),
                "first_player_win_rate": 0.5 + 0.01 * (scale - 1.0),
            },
        }

    result = audit.compare_modes(
        {
            "q21_posterior_sample": summarized(1.2),
            "q21_posterior_temperature": summarized(1.005),
            "q21_posterior_plurality32": summarized(1.01),
            "q21_posterior_argmax": summarized(0.01),
            "m32_posterior_argmax": summarized(1.0),
        }
    )

    plurality = result["contrasts"][
        "q21_posterior_plurality32_vs_m32_posterior_argmax"
    ]
    assert plurality["state_effective_support_ratio"] == pytest.approx(1.01)
    assert plurality["ordered_prefix_effective_support_ratio"] == (
        pytest.approx(1.01)
    )
    assert plurality["action_effective_support_ratio"] == pytest.approx(1.01)
    gate = result["plurality32_native_m32_equivalence"]
    assert gate["metrics"][
        "state_ess_error_fraction_of_nearer_endpoint_separation"
    ] == pytest.approx(0.05)
    assert gate["pass"]["all_pass"] is True
    temperature = result["contrasts"][
        "q21_posterior_temperature_vs_m32_posterior_argmax"
    ]
    assert temperature["state_effective_support_ratio"] == pytest.approx(
        1.005
    )
    assert result["temperature_native_m32_equivalence"]["pass"][
        "all_pass"
    ] is True


def test_temperature_calibration_uses_power_transformed_policy() -> None:
    result = audit.summarize_mode(
        _host_data(),
        num_actions=4,
        sampling_calibration_applicable=True,
        sampling_temperature=0.5,
    )

    summary = result["commitment_q21"]
    assert summary["sampling_temperature"] == 0.5
    assert (
        summary["sampling_policy_effective_support_mean"]
        < summary["policy_effective_support_mean"]
    )
    assert (
        summary["expected_argmax_fraction_under_commitment_sampling"]
        > 0.5
    )


def test_parser_defaults_temperature_to_one_third() -> None:
    args = audit._parser().parse_args(
        [
            "--checkpoint",
            "/tmp/checkpoint",
            "--step",
            "0",
            "--output",
            "/tmp/output.json",
        ]
    )

    assert args.posterior_sample_temperature == pytest.approx(1.0 / 3.0)


def test_create_once_output_is_immutable(tmp_path: Path):
    output = tmp_path / "audit.json"
    digest = audit._write_json_create_once(output, {"version": 1})

    assert len(digest) == 64
    assert output.is_file()
    with pytest.raises(FileExistsError):
        audit._write_json_create_once(output, {"version": 2})
