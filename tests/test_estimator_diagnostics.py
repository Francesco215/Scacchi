from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scacchi.dirichlet_mctx.estimator_diagnostics import (
    analytic_cache_noise,
    binary_posterior_best_policy_prefix_quadrature,
    binary_posterior_best_policy_quadrature,
    binary_posterior_best_policy_rao_blackwell,
)
from scacchi.dirichlet_mctx.action_selection import posterior_best_policy
from scacchi.dirichlet_mctx.outcomes import NO_OUTCOME


_PREFIX_TAIL_ALPHA = jnp.asarray(
    [
        [
            [0.019338588, 0.15259013],
            [1.112402, 0.0016965896],
            [1.3755394, 0.0024108132],
            [0.0012154635, 0.00967293],
            [0.041508093, 0.0063411705],
            [0.97789204, 0.0100949705],
            [0.0061329156, 0.54238707],
            [0.010792912, 5.753708],
        ]
    ],
    dtype=jnp.float32,
)


def _exact_binary_reference(
    alpha: jax.Array,
    invalid: jax.Array,
    categorical: jax.Array | None = None,
) -> jax.Array:
    with jax.enable_x64():
        policy = binary_posterior_best_policy_quadrature(
            alpha.astype(jnp.float64),
            invalid,
            categorical,
            half_width=160,
            step=0.1,
        ).policy.astype(jnp.float32)
        return jax.block_until_ready(policy)


def test_binary_quadrature_is_uniform_for_identical_actions() -> None:
    alpha = jnp.asarray(
        [[[2.0, 3.0], [2.0, 3.0], [2.0, 3.0]]],
        dtype=jnp.float32,
    )
    result = binary_posterior_best_policy_quadrature(
        alpha,
        jnp.zeros((1, 3), dtype=jnp.bool_),
    )

    np.testing.assert_allclose(
        result.policy,
        np.full((1, 3), 1.0 / 3.0),
        rtol=2e-4,
        atol=2e-4,
    )
    np.testing.assert_allclose(result.raw_mass, 1.0, atol=5e-4)
    assert bool(result.finite[0])


def test_binary_quadrature_matches_closed_form_beta_pair() -> None:
    # Win probabilities X_0~Beta(2,1), X_1~Beta(1,2) satisfy
    # P(X_0>X_1)=5/6.
    alpha = jnp.asarray([[[1.0, 2.0], [2.0, 1.0]]])
    result = binary_posterior_best_policy_quadrature(
        alpha,
        jnp.zeros((1, 2), dtype=jnp.bool_),
    )
    np.testing.assert_allclose(
        result.policy,
        [[5.0 / 6.0, 1.0 / 6.0]],
        rtol=2e-4,
        atol=2e-4,
    )


def test_binary_quadrature_respects_invalid_and_categorical_actions() -> None:
    alpha = jnp.asarray(
        [
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        ]
    )
    invalid = jnp.asarray(
        [
            [False, True, False],
            [False, False, False],
            [False, False, False],
        ]
    )
    categorical = jnp.asarray(
        [
            [NO_OUTCOME, NO_OUTCOME, NO_OUTCOME],
            [0, NO_OUTCOME, NO_OUTCOME],
            [0, 1, NO_OUTCOME],
        ],
        dtype=jnp.int8,
    )
    result = binary_posterior_best_policy_quadrature(
        alpha,
        invalid,
        categorical,
    )

    np.testing.assert_allclose(result.policy[0], [0.5, 0.0, 0.5], atol=2e-4)
    # A certified loss cannot beat either unresolved continuous action.
    np.testing.assert_allclose(result.policy[1], [0.0, 0.5, 0.5], atol=2e-4)
    # A certified win dominates with probability one.
    np.testing.assert_allclose(result.policy[2], [0.0, 1.0, 0.0], atol=0)


def test_binary_quadrature_preserves_first_index_categorical_ties() -> None:
    alpha = jnp.ones((1, 3, 2))
    categorical = jnp.asarray([[0, 0, 0]], dtype=jnp.int8)
    result = binary_posterior_best_policy_quadrature(
        alpha,
        jnp.zeros((1, 3), dtype=jnp.bool_),
        categorical,
    )
    np.testing.assert_array_equal(result.policy, [[1.0, 0.0, 0.0]])


def test_prefix_quadrature_matches_exact_reference_on_benign_alphas() -> None:
    alpha = jnp.asarray(
        [
            [
                [1.0, 1.0],
                [2.0, 5.0],
                [4.0, 1.5],
                [0.25, 3.0],
            ],
            [
                [8.0, 0.5],
                [0.75, 6.0],
                [3.5, 2.5],
                [1.25, 1.75],
            ],
        ],
        dtype=jnp.float32,
    )
    invalid = jnp.asarray(
        [[False, False, False, False], [False, True, False, False]]
    )
    reference = _exact_binary_reference(alpha, invalid)
    result = binary_posterior_best_policy_prefix_quadrature(alpha, invalid)

    policy_l1 = jnp.sum(jnp.abs(result.policy - reference), axis=-1)
    assert float(jnp.max(policy_l1)) < 1e-2
    assert bool(jnp.all(result.finite))


def test_prefix_quadrature_default_covers_extreme_supported_alphas() -> None:
    invalid = jnp.zeros(_PREFIX_TAIL_ALPHA.shape[:-1], dtype=jnp.bool_)
    reference = _exact_binary_reference(_PREFIX_TAIL_ALPHA, invalid)
    result = binary_posterior_best_policy_prefix_quadrature(
        _PREFIX_TAIL_ALPHA,
        invalid,
    )

    assert float(jnp.min(_PREFIX_TAIL_ALPHA)) >= 1e-3
    policy_l1 = jnp.sum(jnp.abs(result.policy - reference), axis=-1)
    assert float(policy_l1[0]) < 6e-3
    assert float(result.normalization_error[0]) < 2e-2
    assert bool(result.finite[0])
    assert not bool(result.tail_range_clipped[0])


def test_prefix_quadrature_jitted_float32_outputs_finite_simplexes() -> None:
    alpha = jnp.concatenate(
        (
            _PREFIX_TAIL_ALPHA,
            jnp.ones_like(_PREFIX_TAIL_ALPHA),
        ),
        axis=0,
    )
    invalid = jnp.asarray(
        [
            [False, False, False, False, False, False, False, True],
            [True, True, True, True, True, True, True, True],
        ]
    )
    categorical = jnp.full(
        invalid.shape,
        int(NO_OUTCOME),
        dtype=jnp.int8,
    )
    result = jax.jit(
        binary_posterior_best_policy_prefix_quadrature
    )(alpha, invalid, categorical)

    assert result.policy.dtype == jnp.float32
    assert result.raw_policy.dtype == jnp.float32
    assert result.grid_half_range.dtype == jnp.float32
    assert result.density_log_integral.dtype == jnp.float32
    assert bool(jnp.all(result.finite))
    assert bool(jnp.all(jnp.isfinite(result.policy)))
    assert bool(jnp.all(jnp.isfinite(result.raw_policy)))
    assert bool(jnp.all(jnp.isfinite(result.density_log_integral)))
    assert bool(jnp.all(result.policy >= 0.0))
    np.testing.assert_allclose(jnp.sum(result.policy[0]), 1.0, atol=2e-6)
    np.testing.assert_array_equal(result.policy[1], jnp.zeros((8,)))
    np.testing.assert_array_equal(
        jnp.where(invalid, result.raw_policy, 0.0),
        jnp.zeros_like(result.raw_policy),
    )
    np.testing.assert_array_equal(
        jnp.where(invalid, result.density_log_integral, 0.0),
        jnp.zeros_like(result.density_log_integral),
    )
    np.testing.assert_allclose(
        result.normalization_error,
        jnp.abs(result.raw_mass - jnp.asarray([1.0, 0.0])),
        atol=0.0,
    )


def test_prefix_quadrature_is_permutation_equivariant() -> None:
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
    )
    permuted = binary_posterior_best_policy_prefix_quadrature(
        alpha[:, permutation],
        invalid[:, permutation],
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


def test_prefix_quadrature_preserves_native_categorical_behavior() -> None:
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
    )

    # An invalid certified win is ignored; a certified loss cannot beat a
    # remaining continuous action.
    np.testing.assert_allclose(
        result.policy[0],
        [0.0, 0.5, 0.5, 0.0],
        atol=2e-6,
    )
    # The first legal certified win dominates.
    np.testing.assert_array_equal(result.policy[1], [0.0, 1.0, 0.0, 0.0])
    # All-categorical ties and all-invalid rows match production behavior.
    np.testing.assert_array_equal(result.policy[2], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(result.policy[3], [0.0, 0.0, 0.0, 0.0])


def test_prefix_quadrature_reports_adaptive_range_and_tail_clipping() -> None:
    alpha = jnp.asarray(
        [
            [
                [0.001, 2.0],
                [1e-8, 1e-8],
                [1e-8, 1e-8],
            ],
            [
                [1e-5, 1.0],
                [1e-8, 1e-8],
                [1e-8, 1e-8],
            ],
            [
                [10.0, 12.0],
                [1e-8, 1e-8],
                [1e-8, 1e-8],
            ],
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
    )

    expected_middle = np.arcsinh(8.0 / 0.001)
    np.testing.assert_allclose(
        result.grid_half_range,
        [expected_middle, 11.0, 6.0],
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        result.tail_range_clipped,
        [False, True, False],
    )


def test_prefix_quadrature_q41_to_q81_converges_at_adaptive_range() -> None:
    invalid = jnp.zeros(_PREFIX_TAIL_ALPHA.shape[:-1], dtype=jnp.bool_)
    reference = _exact_binary_reference(_PREFIX_TAIL_ALPHA, invalid)
    q41 = binary_posterior_best_policy_prefix_quadrature(
        _PREFIX_TAIL_ALPHA,
        invalid,
        half_width=20,
    )
    q81 = binary_posterior_best_policy_prefix_quadrature(
        _PREFIX_TAIL_ALPHA,
        invalid,
        half_width=40,
    )

    q41_l1 = jnp.sum(jnp.abs(q41.policy - reference), axis=-1)[0]
    q81_l1 = jnp.sum(jnp.abs(q81.policy - reference), axis=-1)[0]
    assert float(q81_l1) < 0.35 * float(q41_l1)
    assert float(q41.normalization_error[0]) < 2e-6
    assert float(q81.normalization_error[0]) < 2e-6


def test_prefix_quadrature_fixed_range_refinement_cannot_recover_tail() -> None:
    invalid = jnp.zeros(_PREFIX_TAIL_ALPHA.shape[:-1], dtype=jnp.bool_)
    reference = _exact_binary_reference(_PREFIX_TAIL_ALPHA, invalid)
    q41_fixed_six = binary_posterior_best_policy_prefix_quadrature(
        _PREFIX_TAIL_ALPHA,
        invalid,
        half_width=20,
        adaptive_range=False,
        fixed_step=0.3,
    )
    q81_fixed_six = binary_posterior_best_policy_prefix_quadrature(
        _PREFIX_TAIL_ALPHA,
        invalid,
        half_width=40,
        adaptive_range=False,
        fixed_step=0.15,
    )
    q81_wide_ten = binary_posterior_best_policy_prefix_quadrature(
        _PREFIX_TAIL_ALPHA,
        invalid,
        half_width=40,
        adaptive_range=False,
        fixed_step=0.25,
    )

    fixed_q41_l1 = jnp.sum(
        jnp.abs(q41_fixed_six.policy - reference),
        axis=-1,
    )[0]
    fixed_q81_l1 = jnp.sum(
        jnp.abs(q81_fixed_six.policy - reference),
        axis=-1,
    )[0]
    wide_q81_l1 = jnp.sum(
        jnp.abs(q81_wide_ten.policy - reference),
        axis=-1,
    )[0]
    assert float(fixed_q41_l1) > 0.5
    assert float(fixed_q81_l1) > 0.5
    assert float(wide_q81_l1) < 1.5e-3
    # Per-action prefix normalization makes the truncated result look finite
    # and can leave a deceptively small winner-mass error.
    assert bool(q81_fixed_six.finite[0])
    assert float(q81_fixed_six.normalization_error[0]) < 0.05


def test_prefix_quadrature_validates_grid_and_shapes() -> None:
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
    with pytest.raises(ValueError, match="half_width"):
        binary_posterior_best_policy_prefix_quadrature(
            alpha,
            invalid,
            half_width=0,
        )
    with pytest.raises(ValueError, match="step"):
        binary_posterior_best_policy_prefix_quadrature(
            alpha,
            invalid,
            adaptive_range=False,
            fixed_step=0.0,
        )


def test_analytic_cache_noise_matches_enumerated_winner_moments() -> None:
    policy = jnp.asarray([[0.25, 0.75]])
    action_alpha = jnp.asarray([[[1.0, 2.0], [4.0, 3.0]]])
    value_prior = jnp.asarray([[2.0, 2.0]])
    result = analytic_cache_noise(
        policy,
        action_alpha,
        value_prior,
        jnp.asarray([3]),
        kappa=3.0,
        num_samples=32,
    )

    actions = np.asarray(action_alpha[0])
    probabilities = np.asarray(policy[0])
    expected = np.sum(probabilities[:, None] * actions, axis=0)
    centered = actions - expected
    covariance = np.einsum(
        "a,ai,aj->ij",
        probabilities,
        centered,
        centered,
    )
    gamma = 0.5
    expected_cache_covariance = gamma**2 * covariance / 32

    np.testing.assert_allclose(
        result.posterior_mean_action_alpha[0],
        expected,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result.winner_action_covariance[0],
        covariance,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result.cache_covariance[0],
        expected_cache_covariance,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result.raw_alpha_mse[0],
        np.trace(expected_cache_covariance),
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        result.concentration_mse[0],
        np.ones(2) @ expected_cache_covariance @ np.ones(2),
        rtol=1e-6,
    )


def test_cache_noise_is_zero_when_every_action_object_is_equal() -> None:
    result = analytic_cache_noise(
        jnp.asarray([[0.2, 0.3, 0.5]]),
        jnp.asarray([[[2.0, 4.0], [2.0, 4.0], [2.0, 4.0]]]),
        jnp.asarray([[1.0, 1.0]]),
        jnp.asarray([8]),
        kappa=3.0,
        num_samples=32,
    )
    np.testing.assert_allclose(result.raw_alpha_mse, 0.0, atol=2e-7)
    np.testing.assert_allclose(result.concentration_mse, 0.0, atol=2e-7)
    np.testing.assert_allclose(result.semantic_mean_delta_mse, 0.0, atol=2e-7)


def test_cache_noise_scales_as_inverse_population_size() -> None:
    args = (
        jnp.asarray([[0.4, 0.6]]),
        jnp.asarray([[[1.0, 5.0], [4.0, 2.0]]]),
        jnp.asarray([[2.0, 2.0]]),
        jnp.asarray([4]),
    )
    m8 = analytic_cache_noise(
        *args,
        kappa=3.0,
        num_samples=8,
    )
    m32 = analytic_cache_noise(
        *args,
        kappa=3.0,
        num_samples=32,
    )
    np.testing.assert_allclose(m8.raw_alpha_mse, 4.0 * m32.raw_alpha_mse)
    np.testing.assert_allclose(
        m8.semantic_mean_delta_mse,
        4.0 * m32.semantic_mean_delta_mse,
    )


def test_cache_noise_separates_intended_repair_from_sampling_step() -> None:
    result = analytic_cache_noise(
        jnp.asarray([[0.5, 0.5]]),
        jnp.asarray([[[1.0, 3.0], [3.0, 1.0]]]),
        jnp.asarray([[2.0, 2.0]]),
        jnp.asarray([6]),
        previous_value_alpha=jnp.asarray([[2.1, 1.9]]),
        kappa=3.0,
        num_samples=32,
    )
    expected_fraction = result.raw_alpha_mse / (
        result.raw_alpha_mse + result.repair_squared_l2
    )
    np.testing.assert_allclose(
        result.noise_fraction_of_expected_repair_step,
        expected_fraction,
    )
    assert float(result.raw_noise_to_repair_ratio[0]) > 0


def test_analytic_cache_noise_validates_arguments() -> None:
    policy = jnp.asarray([[1.0]])
    action_alpha = jnp.asarray([[[1.0, 1.0]]])
    value_prior = jnp.asarray([[1.0, 1.0]])
    n_down = jnp.asarray([1])

    with pytest.raises(ValueError, match="num_samples"):
        analytic_cache_noise(
            policy,
            action_alpha,
            value_prior,
            n_down,
            kappa=3.0,
            num_samples=0,
        )
    with pytest.raises(ValueError, match="kappa"):
        analytic_cache_noise(
            policy,
            action_alpha,
            value_prior,
            n_down,
            kappa=0.0,
            num_samples=1,
        )


def test_quadrature_reference_jits() -> None:
    function = jax.jit(binary_posterior_best_policy_quadrature)
    result = function(
        jnp.asarray([[[1.0, 1.0], [1.0, 1.0]]]),
        jnp.asarray([[False, False]]),
    )
    np.testing.assert_allclose(result.policy, [[0.5, 0.5]], atol=2e-4)


def test_rao_blackwell_samples_are_finite_simplexes_and_cycle_balanced() -> None:
    alpha = jnp.asarray(
        [
            [
                [0.05, 12.0],
                [8.0, 0.1],
                [2.0, 3.0],
                [4.0, 1.0],
            ],
            [
                [1.0, 1.0],
                [3.0, 2.0],
                [2.0, 4.0],
                [1.5, 1.5],
            ],
        ],
        dtype=jnp.float32,
    )
    invalid = jnp.asarray(
        [[False, False, False, True], [False, True, False, False]]
    )

    result = jax.jit(
        lambda key, action_alpha, mask: (
            binary_posterior_best_policy_rao_blackwell(
                key,
                action_alpha,
                mask,
                num_samples=11,
                sample_chunk_size=4,
            )
        )
    )(jax.random.PRNGKey(21), alpha, invalid)

    assert bool(jnp.all(result.finite))
    assert bool(jnp.all(jnp.isfinite(result.policy)))
    assert bool(jnp.all(result.policy >= 0.0))
    assert result.coordinate_counts.dtype == jnp.int32
    np.testing.assert_allclose(
        jnp.sum(result.policy, axis=-1),
        1.0,
        atol=2e-6,
    )
    np.testing.assert_allclose(result.normalization_error, 0.0, atol=2e-6)
    np.testing.assert_array_equal(
        jnp.sum(result.coordinate_counts, axis=-1),
        [11, 11],
    )
    np.testing.assert_array_equal(
        jnp.where(invalid, result.coordinate_counts, 0),
        jnp.zeros_like(result.coordinate_counts),
    )
    for lane in range(2):
        live_counts = np.asarray(result.coordinate_counts[lane])[~np.asarray(
            invalid[lane]
        )]
        assert int(np.max(live_counts) - np.min(live_counts)) <= 1


def test_rao_blackwell_handles_native_categorical_edges_exactly() -> None:
    alpha = jnp.ones((3, 3, 2), dtype=jnp.float32)
    invalid = jnp.zeros((3, 3), dtype=jnp.bool_)
    categorical = jnp.asarray(
        [
            [0, NO_OUTCOME, NO_OUTCOME],
            [0, 1, NO_OUTCOME],
            [0, 0, 0],
        ],
        dtype=jnp.int8,
    )
    result = binary_posterior_best_policy_rao_blackwell(
        jax.random.PRNGKey(7),
        alpha,
        invalid,
        categorical,
        num_samples=12,
        sample_chunk_size=5,
    )

    # A certified loss has exactly zero chance against a continuous
    # unresolved Beta action.
    np.testing.assert_allclose(result.policy[0, 0], 0.0, atol=0.0)
    np.testing.assert_allclose(
        jnp.sum(result.policy[0, 1:]),
        1.0,
        atol=1e-6,
    )
    np.testing.assert_array_equal(
        result.coordinate_counts[0],
        [0, 6, 6],
    )
    # A certified win dominates without requesting estimator samples.
    np.testing.assert_array_equal(result.policy[1], [0.0, 1.0, 0.0])
    np.testing.assert_array_equal(result.coordinate_counts[1], [0, 0, 0])
    # An all-categorical loss node preserves the production first-index tie.
    np.testing.assert_array_equal(result.policy[2], [1.0, 0.0, 0.0])
    np.testing.assert_array_equal(result.coordinate_counts[2], [0, 0, 0])


def test_rao_blackwell_is_unbiased_against_exact_beta_reference() -> None:
    alpha = jnp.asarray(
        [[[1.0, 2.0], [2.0, 1.0], [1.5, 1.5]]],
        dtype=jnp.float32,
    )
    invalid = jnp.zeros((1, 3), dtype=jnp.bool_)
    reference = binary_posterior_best_policy_quadrature(
        alpha,
        invalid,
    ).policy[0]
    keys = jax.random.split(jax.random.PRNGKey(101), 4096)

    estimates = jax.jit(
        jax.vmap(
            lambda key: binary_posterior_best_policy_rao_blackwell(
                key,
                alpha,
                invalid,
                num_samples=4,
                sample_chunk_size=2,
            ).policy[0]
        )
    )(keys)
    empirical_mean = jnp.mean(estimates, axis=0)

    np.testing.assert_allclose(
        empirical_mean,
        reference,
        rtol=0.0,
        atol=8e-3,
    )


def test_rao_blackwell_equal_case_and_permutation_equivariance_in_mean() -> None:
    equal_alpha = jnp.full((1, 3, 2), 2.0, dtype=jnp.float32)
    invalid = jnp.zeros((1, 3), dtype=jnp.bool_)
    keys = jax.random.split(jax.random.PRNGKey(202), 2048)

    equal_estimates = jax.jit(
        jax.vmap(
            lambda key: binary_posterior_best_policy_rao_blackwell(
                key,
                equal_alpha,
                invalid,
                num_samples=6,
                sample_chunk_size=3,
            ).policy[0]
        )
    )(keys)
    np.testing.assert_allclose(
        jnp.mean(equal_estimates, axis=0),
        jnp.full((3,), 1.0 / 3.0),
        atol=8e-3,
    )

    alpha = jnp.asarray(
        [[[1.0, 4.0], [3.0, 1.0], [2.0, 2.0]]],
        dtype=jnp.float32,
    )
    permutation = jnp.asarray([2, 0, 1])
    inverse = jnp.argsort(permutation)
    permuted_alpha = alpha[:, permutation]

    def empirical_mean(action_alpha: jax.Array) -> jax.Array:
        estimates = jax.jit(
            jax.vmap(
                lambda key: binary_posterior_best_policy_rao_blackwell(
                    key,
                    action_alpha,
                    invalid,
                    num_samples=4,
                    sample_chunk_size=2,
                ).policy[0]
            )
        )(keys)
        return jnp.mean(estimates, axis=0)

    original_mean = empirical_mean(alpha)
    restored_permuted_mean = empirical_mean(permuted_alpha)[inverse]
    np.testing.assert_allclose(
        original_mean,
        restored_permuted_mean,
        atol=1.2e-2,
    )


def test_rao_blackwell_empirically_reduces_winner_count_variance() -> None:
    alpha = jnp.ones((1, 2, 2), dtype=jnp.float32)
    invalid = jnp.zeros((1, 2), dtype=jnp.bool_)
    keys = jax.random.split(jax.random.PRNGKey(303), 4096)
    num_samples = 4

    rao_blackwell = jax.jit(
        jax.vmap(
            lambda key: binary_posterior_best_policy_rao_blackwell(
                key,
                alpha,
                invalid,
                num_samples=num_samples,
                sample_chunk_size=2,
            ).policy[0, 0]
        )
    )(keys)
    winner_count = jax.jit(
        jax.vmap(
            lambda key: posterior_best_policy(
                key,
                alpha,
                invalid,
                num_samples,
                chunk_size=2,
            )[0, 0]
        )
    )(keys)

    rb_variance = float(jnp.var(rao_blackwell, ddof=1))
    winner_variance = float(jnp.var(winner_count, ddof=1))
    np.testing.assert_allclose(jnp.mean(rao_blackwell), 0.5, atol=8e-3)
    np.testing.assert_allclose(jnp.mean(winner_count), 0.5, atol=1.2e-2)
    # For two uniform actions the theoretical ratio is 1/3.
    assert rb_variance < 0.5 * winner_variance


def test_rao_blackwell_validates_arguments() -> None:
    alpha = jnp.ones((1, 2, 2))
    invalid = jnp.zeros((1, 2), dtype=jnp.bool_)

    with pytest.raises(ValueError, match="num_samples"):
        binary_posterior_best_policy_rao_blackwell(
            jax.random.PRNGKey(0),
            alpha,
            invalid,
            num_samples=0,
        )
    with pytest.raises(ValueError, match="sample_chunk_size"):
        binary_posterior_best_policy_rao_blackwell(
            jax.random.PRNGKey(0),
            alpha,
            invalid,
            num_samples=1,
            sample_chunk_size=0,
        )
    with pytest.raises(ValueError, match="exactly two outcomes"):
        binary_posterior_best_policy_rao_blackwell(
            jax.random.PRNGKey(0),
            jnp.ones((1, 2, 3)),
            invalid,
            num_samples=1,
        )
