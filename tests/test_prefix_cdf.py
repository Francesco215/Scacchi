from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scacchi.dirichlet_mctx.outcomes import NO_OUTCOME
from scacchi.dirichlet_mctx.prefix_cdf import (
    binary_posterior_best_policy_prefix_quadrature,
)


def test_q21_outputs_finite_normalized_policies_and_metadata() -> None:
    alpha = jnp.asarray(
        [
            [
                [1.0, 1.0],
                [2.0, 5.0],
                [4.0, 1.5],
                [0.25, 3.0],
            ],
            [
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
                [1.0, 1.0],
            ],
        ],
        dtype=jnp.float32,
    )
    invalid = jnp.asarray(
        [
            [False, False, True, False],
            [True, True, True, True],
        ]
    )

    result = jax.jit(
        lambda a, mask: (
            binary_posterior_best_policy_prefix_quadrature(
                a,
                mask,
                half_width=10,
            )
        )
    )(alpha, invalid)

    assert result.policy.dtype == jnp.float32
    assert result.raw_policy.dtype == jnp.float32
    assert result.grid_half_range.dtype == jnp.float32
    assert result.density_log_integral.dtype == jnp.float32
    assert result.fallback_interval_count.dtype == jnp.int32
    assert bool(jnp.all(result.finite))
    assert bool(jnp.all(jnp.isfinite(result.policy)))
    assert bool(jnp.all(jnp.isfinite(result.raw_policy)))
    assert bool(jnp.all(jnp.isfinite(result.density_log_integral)))
    assert bool(jnp.all(result.policy >= 0.0))
    np.testing.assert_allclose(
        jnp.sum(result.policy, axis=-1),
        [1.0, 0.0],
        atol=2e-6,
    )
    np.testing.assert_allclose(
        result.normalization_error,
        jnp.abs(result.raw_mass - jnp.asarray([1.0, 0.0])),
        atol=0.0,
    )
    np.testing.assert_array_equal(
        jnp.where(invalid, result.policy, 0.0),
        jnp.zeros_like(result.policy),
    )


def test_q21_is_permutation_equivariant() -> None:
    alpha = jnp.asarray(
        [[[1.0, 4.0], [3.0, 1.0], [0.25, 2.0], [5.0, 0.5]]],
        dtype=jnp.float32,
    )
    invalid = jnp.asarray([[False, True, False, False]])
    permutation = jnp.asarray([2, 0, 3, 1])
    inverse = jnp.argsort(permutation)

    original = binary_posterior_best_policy_prefix_quadrature(
        alpha,
        invalid,
        half_width=10,
    )
    permuted = binary_posterior_best_policy_prefix_quadrature(
        alpha[:, permutation],
        invalid[:, permutation],
        half_width=10,
    )

    np.testing.assert_allclose(
        original.policy,
        permuted.policy[:, inverse],
        rtol=0.0,
        atol=2e-6,
    )
    np.testing.assert_allclose(
        original.raw_mass,
        permuted.raw_mass,
        rtol=0.0,
        atol=2e-6,
    )


def test_categorical_and_invalid_actions_preserve_solved_semantics() -> None:
    alpha = jnp.ones((4, 4, 2), dtype=jnp.float32)
    invalid = jnp.asarray(
        [
            [False, False, False, True],
            [False, False, False, False],
            [False, False, False, False],
            [True, True, True, True],
        ]
    )
    categorical = jnp.asarray(
        [
            [0, NO_OUTCOME, NO_OUTCOME, 1],
            [0, 1, NO_OUTCOME, 1],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
        ],
        dtype=jnp.int8,
    )

    result = binary_posterior_best_policy_prefix_quadrature(
        alpha,
        invalid,
        categorical,
        half_width=10,
    )

    # The invalid certified win is ignored, and a certified loss cannot beat
    # either unresolved action.
    np.testing.assert_allclose(
        result.policy[0],
        [0.0, 0.5, 0.5, 0.0],
        atol=2e-6,
    )
    # The first legal certified win dominates.
    np.testing.assert_array_equal(
        result.policy[1],
        [0.0, 1.0, 0.0, 0.0],
    )
    # All-categorical ties and all-invalid rows preserve native behavior.
    np.testing.assert_array_equal(
        result.policy[2],
        [1.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_array_equal(
        result.policy[3],
        [0.0, 0.0, 0.0, 0.0],
    )


def test_guard_and_fallback_metadata_are_exposed() -> None:
    alpha = jnp.asarray(
        [
            [[0.001, 2.0], [1.0, 1.0], [1.0, 1.0]],
            [[1e-8, 1.0], [1.0, 1.0], [1.0, 1.0]],
            [[10.0, 12.0], [1.0, 1.0], [1.0, 1.0]],
        ],
        dtype=jnp.float32,
    )
    invalid = jnp.asarray(
        [
            [False, True, False],
            [False, True, False],
            [False, True, False],
        ]
    )
    categorical = jnp.asarray(
        [
            [NO_OUTCOME, NO_OUTCOME, 0],
            [NO_OUTCOME, NO_OUTCOME, 0],
            [NO_OUTCOME, NO_OUTCOME, 0],
        ],
        dtype=jnp.int8,
    )

    result = binary_posterior_best_policy_prefix_quadrature(
        alpha,
        invalid,
        categorical,
        half_width=10,
    )

    np.testing.assert_allclose(
        result.grid_half_range,
        [np.arcsinh(8.0 / 0.001), 11.0, 6.0],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        result.tail_range_clipped,
        [False, True, False],
    )
    assert result.density_log_integral.shape == invalid.shape
    assert result.fallback_interval_count.shape == invalid.shape[:-1]
    assert bool(jnp.all(result.fallback_interval_count >= 0))
    assert bool(jnp.all(jnp.isfinite(result.density_log_integral)))
    np.testing.assert_array_equal(
        jnp.where(
            categorical != int(NO_OUTCOME),
            result.density_log_integral,
            0.0,
        ),
        jnp.zeros_like(result.density_log_integral),
    )


def test_q21_agrees_with_closed_form_and_converges_toward_q81() -> None:
    # X_0 ~ Beta(2, 1), X_1 ~ Beta(1, 2) has P(X_0 > X_1) = 5/6.
    alpha = jnp.asarray([[[1.0, 2.0], [2.0, 1.0]]], dtype=jnp.float32)
    invalid = jnp.zeros((1, 2), dtype=jnp.bool_)
    reference = jnp.asarray([[5.0 / 6.0, 1.0 / 6.0]])

    q21 = binary_posterior_best_policy_prefix_quadrature(
        alpha,
        invalid,
        half_width=10,
    )
    q81 = binary_posterior_best_policy_prefix_quadrature(
        alpha,
        invalid,
        half_width=40,
    )

    q21_error = jnp.sum(jnp.abs(q21.policy - reference))
    q81_error = jnp.sum(jnp.abs(q81.policy - reference))
    assert float(q21_error) < 2e-2
    assert float(q81_error) < float(q21_error)
    assert float(q81_error) < 2e-3


def test_q21_hex6_sized_readout_tracks_dense_reference() -> None:
    action = jnp.arange(36, dtype=jnp.float32)
    mean_logit = jnp.sin(1.7 * action) + 0.5 * jnp.cos(0.83 * action)
    mean = jax.nn.sigmoid(mean_logit)
    concentration = 8.0 + 4.0 * jax.nn.sigmoid(
        jnp.sin(0.41 * action)
    )
    alpha = jnp.stack(
        (concentration * (1.0 - mean), concentration * mean),
        axis=-1,
    )[None]
    invalid = jnp.zeros((1, 36), dtype=jnp.bool_)

    q21 = binary_posterior_best_policy_prefix_quadrature(
        alpha,
        invalid,
        half_width=10,
    )
    q321 = binary_posterior_best_policy_prefix_quadrature(
        alpha,
        invalid,
        half_width=160,
    )

    assert not bool(q21.tail_range_clipped[0])
    assert float(jnp.max(jnp.abs(q21.density_log_integral))) < 0.01
    assert float(jnp.sum(jnp.abs(q21.policy - q321.policy))) < 0.1


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"half_width": 0}, "half_width"),
        ({"tail_scale": 0.0}, "tail_scale"),
        ({"tail_scale": float("inf")}, "tail_scale"),
        ({"min_half_range": 0.0}, "min_half_range"),
        (
            {"min_half_range": 7.0, "max_half_range": 6.0},
            "max_half_range",
        ),
    ],
)
def test_validates_grid_parameters(
    kwargs: dict[str, Any],
    message: str,
) -> None:
    alpha = jnp.ones((1, 2, 2))
    invalid = jnp.zeros((1, 2), dtype=jnp.bool_)

    with pytest.raises(ValueError, match=message):
        binary_posterior_best_policy_prefix_quadrature(
            alpha,
            invalid,
            **kwargs,
        )


def test_validates_shapes() -> None:
    alpha = jnp.ones((1, 2, 2))
    invalid = jnp.zeros((1, 2), dtype=jnp.bool_)

    with pytest.raises(ValueError, match="exactly two outcomes"):
        binary_posterior_best_policy_prefix_quadrature(
            jnp.ones((1, 2, 3)),
            invalid,
        )
    with pytest.raises(ValueError, match="invalid_actions"):
        binary_posterior_best_policy_prefix_quadrature(
            alpha,
            jnp.zeros((1, 3), dtype=jnp.bool_),
        )
    with pytest.raises(ValueError, match="categorical_outcome"):
        binary_posterior_best_policy_prefix_quadrature(
            alpha,
            invalid,
            jnp.full((1, 3), int(NO_OUTCOME), dtype=jnp.int8),
        )
