from __future__ import annotations

import itertools
import math

import jax
import jax.numpy as jnp
import numpy as np

from scripts.hex_plurality_fixed_root_benchmark import (
    _comparison_arrays,
    _eligible_root_mask,
    _make_native_group_sampler,
    _multinomial_calibration,
    _sanitize_policy,
    exact_plurality_law,
)
from scacchi.dirichlet_mctx.outcomes import NO_OUTCOME


def _brute_plurality_law(
    policy: np.ndarray,
    num_votes: int,
) -> np.ndarray:
    policy = np.asarray(policy, dtype=np.float64)
    result = np.zeros_like(policy)
    for count in itertools.product(
        range(num_votes + 1),
        repeat=len(policy),
    ):
        if sum(count) != num_votes:
            continue
        coefficient = math.factorial(num_votes)
        probability = 1.0
        for action_count, action_probability in zip(
            count,
            policy,
            strict=True,
        ):
            coefficient //= math.factorial(action_count)
            probability *= action_probability**action_count
        winner = int(np.argmax(np.asarray(count)))
        result[winner] += coefficient * probability
    return result


def test_exact_plurality_matches_brute_multinomial_enumeration() -> None:
    policies = np.asarray(
        [
            [0.13, 0.31, 0.56],
            [0.40, 0.40, 0.20],
            [0.90, 0.09, 0.01],
        ],
        dtype=np.float64,
    )
    for num_votes in (1, 2, 3, 5):
        actual, normalization_error = exact_plurality_law(
            policies,
            num_votes=num_votes,
        )
        expected = np.stack(
            [
                _brute_plurality_law(policy, num_votes)
                for policy in policies
            ]
        )
        np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=2e-14)
        np.testing.assert_allclose(normalization_error, 0.0, atol=2e-14)


def test_exact_plurality_encodes_lowest_index_count_ties() -> None:
    law, _ = exact_plurality_law(
        np.asarray([0.5, 0.5]),
        num_votes=2,
    )
    # Action zero wins 2--0 and the 1--1 count tie.
    np.testing.assert_allclose(law, [0.75, 0.25], atol=1e-15)


def test_exact_plurality_endpoints_and_one_hot() -> None:
    policy = np.asarray([[0.2, 0.3, 0.5], [0.0, 1.0, 0.0]])
    g1, _ = exact_plurality_law(policy, num_votes=1)
    g32, _ = exact_plurality_law(policy, num_votes=32)
    np.testing.assert_allclose(g1, policy, atol=1e-15)
    np.testing.assert_allclose(g32[1], policy[1], atol=1e-15)
    np.testing.assert_allclose(np.sum(g32, axis=-1), 1.0, atol=1e-15)


def test_exact_plurality_m32_hex_width_is_finite_and_normalized() -> None:
    rng = np.random.default_rng(17)
    policy = rng.dirichlet(np.ones(36), size=4)
    law, normalization_error = exact_plurality_law(
        policy,
        num_votes=32,
    )
    assert np.all(np.isfinite(law))
    assert np.all(law >= 0.0)
    np.testing.assert_allclose(np.sum(law, axis=-1), 1.0, atol=2e-14)
    assert float(np.max(normalization_error)) < 2e-14


def test_sanitize_policy_matches_production_contract() -> None:
    policy = np.asarray(
        [
            [np.nan, -1.0, 2.0, 2.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    legal = np.asarray(
        [
            [True, True, True, False],
            [False, True, False, True],
            [False, False, False, False],
        ]
    )
    actual = _sanitize_policy(policy, legal)
    np.testing.assert_allclose(
        actual,
        [
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.5, 0.0, 0.5],
            [1.0, 0.0, 0.0, 0.0],
        ],
    )


def test_eligible_mask_excludes_categorical_and_guard_fallback() -> None:
    categorical = np.asarray([False, True, False, True])
    unsafe = np.asarray([False, False, True, True])
    np.testing.assert_array_equal(
        _eligible_root_mask(categorical, unsafe),
        [True, False, False, False],
    )


def test_distribution_metrics_use_tv_js_and_lowest_top_index() -> None:
    reference = np.asarray([[0.5, 0.5], [0.8, 0.2]])
    candidate = np.asarray([[0.5, 0.5], [0.6, 0.4]])
    metrics = _comparison_arrays(reference, candidate)
    np.testing.assert_allclose(metrics["l1"], [0.0, 0.4])
    np.testing.assert_allclose(metrics["tv"], [0.0, 0.2])
    assert metrics["js_nats"][0] == 0.0
    np.testing.assert_array_equal(
        metrics["top_action_agreement"],
        [True, True],
    )


def test_multinomial_calibration_is_zero_at_expectation() -> None:
    policy = np.asarray([[0.5, 0.3, 0.2], [0.6, 0.4, 0.0]])
    counts = np.asarray([[50, 30, 20], [60, 40, 0]])
    result = _multinomial_calibration(counts, policy)
    assert result["pearson_chi_square"] == 0.0
    assert result["impossible_observations_under_supplied_policy"] == 0
    assert result["asymptotic_p_value"] == 1.0


def test_native_group_sampler_preserves_int32_scan_carry_under_x64() -> None:
    sampler = _make_native_group_sampler(num_groups=3, num_votes=2)
    alpha = jnp.asarray(
        [
            [[2.0, 3.0], [3.0, 2.0], [1.0, 1.0]],
            [[1.0, 2.0], [2.0, 1.0], [4.0, 4.0]],
        ],
        dtype=jnp.float32,
    )
    invalid = jnp.zeros((2, 3), dtype=bool)
    edge_outcome = jnp.full(
        (2, 3),
        int(NO_OUTCOME),
        dtype=jnp.int8,
    )
    with jax.enable_x64():
        winner_count, plurality_count = jax.block_until_ready(
            sampler(
                jax.random.PRNGKey(19),
                alpha,
                invalid,
                edge_outcome,
            )
        )
    assert winner_count.dtype == jnp.int32
    assert plurality_count.dtype == jnp.int32
    np.testing.assert_array_equal(
        np.sum(np.asarray(winner_count), axis=-1),
        [6, 6],
    )
    np.testing.assert_array_equal(
        np.sum(np.asarray(plurality_count), axis=-1),
        [3, 3],
    )
