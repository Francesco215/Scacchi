from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from scacchi.types import PosteriorPolicyEstimator
from scripts.hex_kappa_fixed_probe import (
    Q21_HALF_WIDTH,
    _SEARCH_DIAGNOSTIC_EXPORTS,
    _effective_search_config,
    _effective_support,
    _entropy,
    _js,
    _parse_kappas,
    _position_rows,
    _summarize,
)


def test_kappa_grid_parser_requires_unique_positive_finite_values():
    assert _parse_kappas("0.25,3,4,8,16,64") == (
        0.25,
        3.0,
        4.0,
        8.0,
        16.0,
        64.0,
    )
    for invalid in ("", "0", "-1,3", "nan,3", "inf,3", "3,3", "wat"):
        with pytest.raises(Exception):
            _parse_kappas(invalid)


def test_entropy_js_and_inverse_simpson_ess_have_known_values():
    deterministic = np.asarray([[1.0, 0.0]])
    uniform = np.asarray([[0.5, 0.5]])

    np.testing.assert_allclose(_entropy(deterministic), [0.0])
    np.testing.assert_allclose(_entropy(uniform), [np.log(2.0)])
    np.testing.assert_allclose(_effective_support(deterministic), [1.0])
    np.testing.assert_allclose(_effective_support(uniform), [2.0])
    np.testing.assert_allclose(_js(uniform, uniform), [0.0])
    np.testing.assert_allclose(
        _js(deterministic, np.asarray([[0.0, 1.0]])),
        [np.log(2.0)],
    )


@dataclass(frozen=True)
class FakeSearchConfig:
    kappa: float = 3.0
    root_action_estimator: PosteriorPolicyEstimator = (
        PosteriorPolicyEstimator.winner_mc
    )
    root_policy_target_estimator: PosteriorPolicyEstimator = (
        PosteriorPolicyEstimator.winner_mc
    )
    posterior_policy_estimator: PosteriorPolicyEstimator = (
        PosteriorPolicyEstimator.winner_mc
    )
    prefix_cdf_half_width: int = 20
    policy_samples: int = 32


def test_effective_search_config_changes_only_kappa_and_q21_action_readout():
    stored = FakeSearchConfig()
    effective = _effective_search_config(stored, kappa=8.0)

    assert effective.kappa == 8.0
    assert (
        effective.root_action_estimator
        == PosteriorPolicyEstimator.prefix_cdf
    )
    assert effective.prefix_cdf_half_width == Q21_HALF_WIDTH == 10
    assert (
        effective.posterior_policy_estimator
        == stored.posterior_policy_estimator
    )
    assert (
        effective.root_policy_target_estimator
        == stored.root_policy_target_estimator
    )
    assert effective.policy_samples == stored.policy_samples
    assert stored.kappa == 3.0
    assert (
        stored.root_action_estimator
        == PosteriorPolicyEstimator.winner_mc
    )


def _diagnostics(solved: np.ndarray) -> dict[str, np.ndarray]:
    size = len(solved)
    unresolved = 1.0 - solved
    return {
        "solved": solved,
        "structural_support": np.full(size, 32.0),
        "repaired_actions": np.full(size, 2.0),
        "categorical_actions": solved.copy(),
        "legal_actions": np.full(size, 4.0),
        "prefix_eligible": 1.0 - solved,
        "prefix_accepted": 1.0 - solved,
        "prefix_fallback": np.zeros(size),
        "prefix_tail_clipped": np.zeros(size),
        "prefix_nonfinite": np.zeros(size),
        "kappa_numeric_repair_count": 2.0 * unresolved,
        "kappa_raw_innovation_l2_sum": 0.8 * unresolved,
        "kappa_semantic_innovation_l2_sum": 0.4 * unresolved,
        "kappa_concentration_innovation_abs_sum": 0.2 * unresolved,
        "kappa_raw_dcache_dlogkappa_l2_sum": 0.6 * unresolved,
        "kappa_mean_dcache_dlogkappa_l2_sum": 0.3 * unresolved,
        "kappa_log_concentration_dcache_dlogkappa_abs_sum": (
            0.1 * unresolved
        ),
        "kappa_numeric_path_count": 2.0 * unresolved,
        "kappa_path_gamma_product_sum": 1.2 * unresolved,
        "kappa_path_gamma_log_attenuation_sum": 1.0 * unresolved,
        "kappa_categorical_publication_path_count": solved.copy(),
        "active_simulation_rows": np.full(size, 32.0),
        "root_policy_top2_margin_sum": 0.5 * unresolved,
        "root_policy_top2_margin_count": unresolved,
        "root_policy_top2_margin_tie_count": np.zeros(size),
        "root_policy_top2_margin_below_reference_count": np.zeros(size),
        "root_policy_top2_margin_reference_scale_sum": (
            unresolved / 32.0
        ),
    }


def test_probe_export_table_is_covered_by_search_diagnostics_interface():
    from scacchi.search_diagnostics import SearchDiagnostics

    exported_fields = {
        field for _, field in _SEARCH_DIAGNOSTIC_EXPORTS
    }
    assert exported_fields <= set(SearchDiagnostics._fields)


def test_position_rows_export_per_root_kappa_channel_and_paired_response():
    roots = {
        "root_id": np.asarray([17]),
        "state_sha256": np.asarray([b"a" * 64]),
        "checkpoint_step": np.asarray([75]),
        "stage_id": np.asarray([1]),
        "action_count": np.asarray([12]),
        "root_weight": np.asarray([3.0]),
    }
    policies = {
        3.0: np.asarray([[0.75, 0.25]]),
        8.0: np.asarray([[0.25, 0.75]]),
    }
    diagnostics = {
        3.0: _diagnostics(np.zeros(1)),
        8.0: _diagnostics(np.zeros(1)),
    }
    diagnostics[3.0]["kappa_mean_dcache_dlogkappa_l2_sum"] = (
        np.asarray([0.6])
    )
    diagnostics[8.0]["root_policy_top2_margin_sum"] = np.asarray([0.5])

    rows = _position_rows(
        roots=roots,
        prior=np.asarray([[0.5, 0.5]]),
        policies=policies,
        diagnostics=diagnostics,
        reference_kappa=3.0,
        oracle_records=[None],
    )

    assert len(rows) == 1
    channel = rows[0]["by_kappa"]["3"]["kappa_channel"]
    assert channel["numeric_repairs"]["count"] == 2
    assert channel["numeric_repairs"][
        "mean_dcache_dlogkappa_l2_mean"
    ] == pytest.approx(0.3)
    assert channel["numeric_paths"]["gamma_product_mean"] == (
        pytest.approx(0.6)
    )
    assert channel["commitment_policy_top2"]["margin"] == 0.5

    paired = rows[0]["by_kappa"]["8"][
        "paired_response_vs_reference"
    ]
    assert paired["delta_log_kappa"] == pytest.approx(np.log(8.0 / 3.0))
    assert paired["root_policy_l1"] == 1.0
    assert paired["root_policy_l1_per_abs_delta_log_kappa"] == (
        pytest.approx(1.0 / np.log(8.0 / 3.0))
    )
    assert paired["top_action_flipped"] is True
    assert paired[
        "root_policy_l1_meets_flip_margin_necessary_condition"
    ] is True
    assert paired[
        "reference_numeric_repair_mean_dmean_dlogkappa_l2"
    ] == pytest.approx(0.3)
    assert paired[
        "reference_first_order_cache_movement_scale_l2"
    ] == pytest.approx(0.3 * np.log(8.0 / 3.0))


def test_summary_is_stage_stratified_and_compares_to_kappa3():
    prior = np.asarray(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
        ]
    )
    policies = {
        3.0: np.asarray(
            [
                [0.75, 0.25],
                [0.25, 0.75],
                [1.0, 0.0],
            ]
        ),
        8.0: np.asarray(
            [
                [0.50, 0.50],
                [0.25, 0.75],
                [0.0, 1.0],
            ]
        ),
    }
    diagnostics = {
        3.0: _diagnostics(np.asarray([0.0, 0.0, 1.0])),
        8.0: _diagnostics(np.asarray([0.0, 0.0, 1.0])),
    }
    summary = _summarize(
        stage_ids=np.asarray([0, 1, 2]),
        prior=prior,
        policies=policies,
        diagnostics=diagnostics,
        reference_kappa=3.0,
        oracle=None,
    )

    assert summary["overall"]["roots"] == 3
    assert summary["by_stage"]["early"]["roots"] == 1
    assert summary["by_stage"]["mid"]["roots"] == 1
    assert summary["by_stage"]["late"]["roots"] == 1
    control = summary["overall"]["kappas"]["3"]
    assert control["root_policy_vs_reference_kappa"]["mean_l1"] == 0.0
    assert control["root_policy_vs_reference_kappa"]["mean_js_nats"] == 0.0
    treatment = summary["overall"]["kappas"]["8"]
    assert treatment["root_policy_vs_reference_kappa"][
        "mean_l1"
    ] == pytest.approx((0.5 + 0.0 + 2.0) / 3.0)
    assert treatment["root_policy_vs_reference_kappa"][
        "top_action_flip_fraction"
    ] == pytest.approx(1.0 / 3.0)
    assert treatment["search_structure"]["solved_root_fraction"] == (
        pytest.approx(1.0 / 3.0)
    )
    assert treatment["search_structure"][
        "positive_unresolved_root_n_down"
    ]["median"] == 32.0
    assert treatment["search_structure"][
        "implied_local_e_fold_length"
    ]["median"] == pytest.approx(1.0 / np.log1p(8.0 / 32.0))


def test_summary_reports_decisive_oracle_flips_with_explicit_denominators():
    prior = np.full((4, 2), 0.5)
    policies = {
        3.0: np.asarray(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [1.0, 0.0],
                [0.0, 1.0],
            ]
        ),
        8.0: np.asarray(
            [
                [0.0, 1.0],  # Flip to a strictly worse oracle outcome.
                [0.0, 1.0],  # Flip between two oracle-optimal actions.
                [1.0, 0.0],  # No flip.
                [1.0, 0.0],  # Flip to a better oracle outcome.
            ]
        ),
    }
    diagnostics = {
        3.0: _diagnostics(np.zeros(4)),
        8.0: _diagnostics(np.zeros(4)),
    }
    oracle = {
        "available": np.ones(4, dtype=bool),
        "expected_regret": {
            3.0: np.asarray([0.0, 0.0, 0.0, 1.0]),
            8.0: np.asarray([1.0, 0.0, 0.0, 0.0]),
        },
        "top_action_regret": {
            3.0: np.asarray([0.0, 0.0, 0.0, 1.0]),
            8.0: np.asarray([1.0, 0.0, 0.0, 0.0]),
        },
        "top_action_optimal": {
            3.0: np.asarray([1.0, 1.0, 1.0, 0.0]),
            8.0: np.asarray([0.0, 1.0, 1.0, 1.0]),
        },
        "optimal_action_mass": {
            3.0: np.asarray([1.0, 1.0, 1.0, 0.0]),
            8.0: np.asarray([0.0, 1.0, 1.0, 1.0]),
        },
    }
    summary = _summarize(
        stage_ids=np.asarray([0, 0, 1, 2]),
        sample_weights=np.asarray([10.0, 20.0, 30.0, 40.0]),
        prior=prior,
        policies=policies,
        diagnostics=diagnostics,
        reference_kappa=3.0,
        oracle=oracle,
    )

    decisive = summary["overall"]["kappas"]["8"]["exact_oracle"][
        "decisive_flips_vs_reference"
    ]
    assert decisive["oracle_root_denominator"] == 4
    assert decisive["oracle_sample_weight_denominator"] == 100.0
    assert decisive["top_action_flip_count"] == 3
    assert decisive[
        "top_action_flip_fraction_of_oracle_roots"
    ] == pytest.approx(3.0 / 4.0)
    assert decisive["top_action_flip_sample_weight"] == 70.0
    assert decisive[
        "top_action_flip_fraction_of_oracle_sample_weight"
    ] == pytest.approx(0.7)
    assert decisive["strictly_worse_outcome_flip_count"] == 1
    assert decisive[
        "strictly_worse_outcome_flip_fraction_of_oracle_roots"
    ] == pytest.approx(0.25)
    assert decisive[
        "strictly_worse_outcome_flip_fraction_of_flips"
    ] == pytest.approx(1.0 / 3.0)
    assert decisive[
        "strictly_worse_outcome_flip_fraction_of_flipped_sample_weight"
    ] == pytest.approx(1.0 / 7.0)
    assert decisive[
        "positive_normalized_top_action_regret_delta_sum_on_flips"
    ] == 1.0
    assert decisive[
        "positive_normalized_top_action_regret_delta_mean_per_oracle_root"
    ] == pytest.approx(0.25)
    assert decisive[
        "positive_normalized_top_action_regret_delta_mean_per_flip"
    ] == pytest.approx(1.0 / 3.0)
    assert decisive[
        "sample_weighted_positive_normalized_top_action_regret_delta_numerator_on_flips"
    ] == 10.0
    assert decisive[
        "sample_weighted_positive_normalized_top_action_regret_delta_mean_per_oracle_sample"
    ] == pytest.approx(0.1)
    assert decisive[
        "sample_weighted_positive_normalized_top_action_regret_delta_mean_per_flipped_sample"
    ] == pytest.approx(1.0 / 7.0)

    early = summary["by_stage"]["early"]["kappas"]["8"][
        "exact_oracle"
    ]["decisive_flips_vs_reference"]
    assert early["oracle_root_denominator"] == 2
    assert early["top_action_flip_count"] == 2
    assert early["strictly_worse_outcome_flip_count"] == 1
    assert early[
        "sample_weighted_positive_normalized_top_action_regret_delta_mean_per_oracle_sample"
    ] == pytest.approx(1.0 / 3.0)

    reference = summary["overall"]["kappas"]["3"]["exact_oracle"][
        "decisive_flips_vs_reference"
    ]
    assert reference["top_action_flip_count"] == 0
    assert reference["strictly_worse_outcome_flip_count"] == 0
    assert (
        reference[
            "positive_normalized_top_action_regret_delta_mean_per_flip"
        ]
        is None
    )


def test_summary_rejects_invalid_sample_weights():
    prior = np.asarray([[0.5, 0.5]])
    policies = {3.0: np.asarray([[0.5, 0.5]])}
    diagnostics = {3.0: _diagnostics(np.zeros(1))}

    with pytest.raises(ValueError, match="shape"):
        _summarize(
            stage_ids=np.asarray([0]),
            sample_weights=np.asarray([1.0, 2.0]),
            prior=prior,
            policies=policies,
            diagnostics=diagnostics,
            reference_kappa=3.0,
            oracle=None,
        )
    with pytest.raises(ValueError, match="finite non-negative"):
        _summarize(
            stage_ids=np.asarray([0]),
            sample_weights=np.asarray([-1.0]),
            prior=prior,
            policies=policies,
            diagnostics=diagnostics,
            reference_kappa=3.0,
            oracle=None,
        )
