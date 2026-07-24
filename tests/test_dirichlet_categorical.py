import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
import pytest

from scacchi.dirichlet_mctx.native_targets import (
    TARGET_CATEGORICAL,
    TARGET_DIRICHLET,
    TARGET_PAD,
    categorical_point,
    dirichlet_nll_at_categorical,
    native_fields_from_beta,
)
from scacchi.dirichlet_mctx.outcomes import NO_DISTANCE, NO_OUTCOME


def test_categorical_point_is_positive_normalized_and_batched():
    point = categorical_point(
        jnp.array([0, 2]),
        num_outcomes=3,
        epsilon=0.01,
    )

    assert point.shape == (2, 3)
    assert jnp.allclose(point[0], jnp.array([0.98, 0.01, 0.01]))
    assert jnp.allclose(point[1], jnp.array([0.01, 0.01, 0.98]))
    assert jnp.all(point > 0.0)
    assert jnp.allclose(jnp.sum(point, axis=-1), 1.0)


@pytest.mark.parametrize("epsilon", [0.0, -0.01, 1.0 / 3.0, 0.5])
def test_categorical_point_rejects_invalid_epsilon(epsilon: float):
    with pytest.raises(ValueError, match="epsilon must be"):
        categorical_point(jnp.asarray(2), num_outcomes=3, epsilon=epsilon)


@pytest.mark.parametrize("outcome", [-1, 3])
def test_invalid_categorical_outcome_is_nonfinite(outcome: int):
    point = categorical_point(
        jnp.asarray(outcome),
        num_outcomes=3,
        epsilon=0.01,
    )
    nll = dirichlet_nll_at_categorical(
        jnp.ones((3,), dtype=jnp.float32),
        jnp.asarray(outcome),
        epsilon=0.01,
    )

    assert jnp.all(jnp.isnan(point))
    assert jnp.isnan(nll)


def test_dirichlet_categorical_nll_matches_log_density_and_has_finite_gradients():
    alpha = jnp.array([[1.5, 2.0, 4.0], [3.0, 2.5, 1.0]])
    outcome = jnp.array([2, 0])
    epsilon = 0.01

    actual = jax.jit(
        lambda candidate, target: dirichlet_nll_at_categorical(
            candidate,
            target,
            epsilon,
        )
    )(alpha, outcome)
    point = categorical_point(outcome, 3, epsilon)
    alpha_sum = jnp.sum(alpha, axis=-1)
    expected = (
        -gammaln(alpha_sum)
        + jnp.sum(gammaln(alpha), axis=-1)
        - jnp.sum((alpha - 1.0) * jnp.log(point), axis=-1)
    )
    gradient = jax.grad(
        lambda candidate: jnp.sum(
            dirichlet_nll_at_categorical(candidate, outcome, epsilon)
        )
    )(alpha)

    assert actual.shape == (2,)
    assert jnp.allclose(actual, expected)
    assert jnp.all(jnp.isfinite(gradient))


def test_native_fields_match_beta_shapes_dtypes_and_sentinels():
    fields = native_fields_from_beta(
        jnp.ones((2, 3, 4), dtype=jnp.float32),
        jnp.ones((2, 4), dtype=jnp.float32),
    )

    assert fields["q_target_kind"].shape == (2, 3)
    assert fields["v_target_kind"].shape == (2,)
    assert fields["q_target_kind"].dtype == jnp.int8
    assert fields["q_target_weight"].dtype == jnp.float32
    assert fields["q_target_outcome"].dtype == jnp.int8
    assert fields["q_target_distance"].dtype == jnp.int32
    assert jnp.all(fields["q_target_kind"] == int(TARGET_DIRICHLET))
    assert jnp.all(fields["v_target_kind"] == int(TARGET_DIRICHLET))
    assert jnp.all(fields["q_target_weight"] == 1.0)
    assert jnp.all(fields["v_target_weight"] == 1.0)
    assert jnp.all(fields["q_target_outcome"] == int(NO_OUTCOME))
    assert jnp.all(fields["v_target_distance"] == int(NO_DISTANCE))
