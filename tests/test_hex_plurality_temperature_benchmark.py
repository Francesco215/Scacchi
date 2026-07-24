from __future__ import annotations

import numpy as np
import pytest

from scripts.hex_plurality_temperature_benchmark import (
    _binary_theory,
    _posterior_mean_population,
    cluster_disjoint_split,
    power_policy,
)


def test_power_policy_preserves_support_ranking_and_normalization() -> None:
    policy = np.asarray(
        [
            [0.5, 0.3, 0.2, 0.0],
            [0.1, 0.7, 0.0, 0.2],
        ]
    )
    sharpened = power_policy(policy, 0.5)

    np.testing.assert_allclose(np.sum(sharpened, axis=-1), 1.0)
    np.testing.assert_array_equal(sharpened == 0.0, policy == 0.0)
    np.testing.assert_array_equal(
        np.argsort(sharpened, axis=-1),
        np.argsort(policy, axis=-1),
    )
    np.testing.assert_allclose(power_policy(policy, 1.0), policy)


def test_power_policy_rejects_invalid_temperature() -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        power_policy(np.asarray([0.5, 0.5]), 0.0)


def test_cluster_split_never_separates_one_cluster() -> None:
    cluster = np.asarray([1, 1, 2, 3, 3, 4, 5, 5])
    split = cluster_disjoint_split(cluster)
    assignment = np.stack(list(split.values()), axis=-1)

    np.testing.assert_array_equal(np.sum(assignment, axis=-1), 1)
    for value in np.unique(cluster):
        local = assignment[cluster == value]
        assert np.all(local == local[0])


def test_posterior_mean_population_respects_categorical_outcomes() -> None:
    alpha = np.asarray([[[1.0, 3.0], [3.0, 1.0], [1.0, 1.0]]])
    invalid = np.asarray([[False, False, True]])
    categorical = np.asarray([[-1, 1, -1]], dtype=np.int8)

    policy = _posterior_mean_population(alpha, invalid, categorical)

    np.testing.assert_allclose(policy, [[3.0 / 7.0, 4.0 / 7.0, 0.0]])


def test_binary_theory_exposes_even_vote_tie_asymmetry() -> None:
    result = _binary_theory(32)

    assert result["equal_probability_count_tie_probability"] > 0.1
    assert result["lower_index_win_probability_at_equal_q"] > 0.55
    assert result["power_transform_win_probability_at_equal_q"] == 0.5
    assert result["binary_local_slope_matched_temperature"] == pytest.approx(
        0.2226,
        rel=0.01,
    )
