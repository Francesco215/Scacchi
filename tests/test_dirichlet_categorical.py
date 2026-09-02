import jax
import jax.numpy as jnp

from scacchi.dirichlet_mctx.native_targets import (
    TARGET_DIRICHLET,
    native_fields_from_beta,
)
from scacchi.dirichlet_mctx.outcomes import NO_DISTANCE, NO_OUTCOME
from scacchi.loss import (
    _categorical_dispersion_loss,
    _dirichlet_dispersion_loss,
)
from scacchi.network import dirichlet_from_logits


def test_dirichlet_dispersion_loss_is_zero_only_at_complete_target():
    beta = jnp.array([[1.2, 1.8], [0.5, 1.0]])
    same = _dirichlet_dispersion_loss(beta, beta)
    wrong_mean = _dirichlet_dispersion_loss(
        beta,
        jnp.array([[1.8, 1.2], [1.0, 0.5]]),
    )
    wrong_concentration = _dirichlet_dispersion_loss(beta, beta * 2.0)

    assert jnp.allclose(same, 0.0, atol=1e-6)
    assert jnp.all(wrong_mean > 0.0)
    assert jnp.all(wrong_concentration > 0.0)


def test_log_concentration_gradient_pushes_two_to_target_three():
    mean = jnp.array([0.4, 0.6])
    beta = 3.0 * mean

    def loss(log_concentration):
        alpha = jnp.exp(log_concentration) * mean
        return _dirichlet_dispersion_loss(beta, alpha)

    gradients = jnp.asarray(
        [
            jax.grad(loss)(jnp.log(jnp.asarray(concentration)))
            for concentration in (2.0, 3.0, 4.0)
        ]
    )

    assert jnp.allclose(
        gradients,
        jnp.array([-1.0 / 3.0, 0.0, 1.0 / 3.0]),
        atol=1e-6,
    )


def test_wrong_mean_has_closed_form_lower_optimal_concentration():
    target_mean = jnp.array([0.4, 0.6])
    prediction_mean = jnp.array([0.6, 0.4])
    beta = 3.0 * target_mean
    mean_kl = jnp.sum(
        target_mean * (jnp.log(target_mean) - jnp.log(prediction_mean))
    )
    expected_concentration = 1.0 / (1.0 / 3.0 + mean_kl)

    def loss(log_concentration):
        return _dirichlet_dispersion_loss(
            beta,
            jnp.exp(log_concentration) * prediction_mean,
        )

    gradient = jax.grad(loss)(jnp.log(expected_concentration))
    expected_minimum = jnp.log1p(3.0 * mean_kl)

    assert expected_concentration < 3.0
    assert jnp.allclose(gradient, 0.0, atol=1e-6)
    assert jnp.allclose(
        loss(jnp.log(expected_concentration)),
        expected_minimum,
        atol=1e-6,
    )


def test_dispersion_autodiff_matches_closed_form_gradients():
    target_mean = jnp.array([0.25, 0.75])
    beta = 4.0 * target_mean
    logits = jnp.array([-0.3, 0.7])
    log_concentration = jnp.log(jnp.asarray(2.5))

    def loss(candidate_logits, candidate_log_concentration):
        mean = jax.nn.softmax(candidate_logits)
        alpha = jnp.exp(candidate_log_concentration) * mean
        return _dirichlet_dispersion_loss(beta, alpha)

    logit_gradient, concentration_gradient = jax.grad(
        loss,
        argnums=(0, 1),
    )(logits, log_concentration)
    prediction_mean = jax.nn.softmax(logits)
    mean_kl = jnp.sum(
        target_mean
        * (jnp.log(target_mean) - jnp.log(prediction_mean))
    )
    concentration = jnp.exp(log_concentration)

    assert jnp.allclose(
        logit_gradient,
        concentration * (prediction_mean - target_mean),
        atol=1e-6,
    )
    assert jnp.allclose(
        concentration_gradient,
        concentration * (1.0 / 4.0 + mean_kl) - 1.0,
        atol=1e-6,
    )


def test_categorical_dispersion_uses_finite_reference_and_rejects_bad_tag():
    reference = 16.0
    nearly_exact_mean = jnp.array([1e-6, 1.0 - 1e-6])
    mean_nll = -jnp.log(nearly_exact_mean[1])
    optimal_concentration = 1.0 / (1.0 / reference + mean_nll)
    alpha = optimal_concentration * nearly_exact_mean

    radial_gradient = jax.grad(
        lambda eta: _categorical_dispersion_loss(
            jnp.exp(eta) * nearly_exact_mean,
            jnp.asarray(1),
            reference,
        )
    )(jnp.log(optimal_concentration))
    invalid = _categorical_dispersion_loss(
        alpha,
        jnp.asarray(-1),
        reference,
    )

    assert optimal_concentration < reference
    assert jnp.allclose(optimal_concentration, reference, rtol=2e-5)
    assert jnp.allclose(radial_gradient, 0.0, atol=1e-6)
    assert jnp.isnan(invalid)


def test_very_low_log_concentration_has_recovery_gradient():
    mean = jnp.array([0.4, 0.6])
    beta = 3.0 * mean

    def loss(eta):
        alpha = dirichlet_from_logits(jnp.log(mean), eta)
        return _dirichlet_dispersion_loss(beta, alpha)

    low_eta = jnp.asarray(-81.0)
    high_eta = jnp.asarray(81.0)
    low_value = loss(low_eta)
    high_value = loss(high_eta)
    low_gradient = jax.grad(loss)(low_eta)
    high_gradient = jax.grad(loss)(high_eta)

    assert jnp.isfinite(low_value)
    assert jnp.isfinite(high_value)
    assert jnp.allclose(low_gradient, -1.0, atol=1e-5)
    assert jnp.isfinite(high_gradient)
    assert high_gradient > 0.0


def test_joint_training_converges_from_initial_concentrations_two_and_three():
    beta = jnp.array([1.2, 1.8])

    def loss(parameters):
        mean = jax.nn.softmax(parameters[:2])
        concentration = jnp.exp(parameters[2])
        return _dirichlet_dispersion_loss(
            beta,
            concentration * mean,
        )

    solutions = []
    for initial_concentration in (2.0, 3.0):
        parameters = jnp.array(
            [0.0, 0.0, jnp.log(initial_concentration)]
        )
        for _ in range(300):
            parameters = parameters - 0.1 * jax.grad(loss)(parameters)
        solutions.append(
            (
                jax.nn.softmax(parameters[:2]),
                jnp.exp(parameters[2]),
            )
        )

    for mean, concentration in solutions:
        assert jnp.allclose(mean, jnp.array([0.4, 0.6]), atol=1e-5)
        assert jnp.allclose(concentration, 3.0, atol=1e-5)


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
