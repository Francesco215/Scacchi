from __future__ import annotations

import jax
import jax.numpy as jnp

from scacchi import play_search
from scacchi.dirichlet_mctx.action_selection import posterior_best_policy
from scacchi.dirichlet_mctx.outcomes import NO_OUTCOME
from scacchi.dirichlet_mctx.prefix_cdf import (
    binary_posterior_best_policy_prefix_quadrature,
)
from scacchi.dirichlet_mctx.posterior_updates import (
    DEFAULT_PREFIX_DENSITY_LOG_INTEGRAL_TOLERANCE,
)
from scacchi.play_search import (
    PosteriorPrediction,
    PosteriorTargets,
    TargetMetadata,
    _dirichlet_commitment_policy,
    _normalize_policy_on_support,
)
from scacchi.types import (
    DirichletThompsonSearchConfig,
    MonteCarloPosteriorUpdateConfig,
    NumericalPosteriorUpdateConfig,
    PosteriorUpdateConfig,
    PosteriorUpdateKind,
)


def _search_config():
    return DirichletThompsonSearchConfig(
        posterior_update=PosteriorUpdateConfig(
            kind=PosteriorUpdateKind.numerical,
            monte_carlo=MonteCarloPosteriorUpdateConfig(
                policy_samples=8,
                policy_sample_chunk_size=2,
            ),
            numerical=NumericalPosteriorUpdateConfig(
                fallback_policy_samples=8,
                fallback_policy_sample_chunk_size=2,
            ),
        )
    )


def _safe_posterior():
    alpha = jnp.broadcast_to(
        jnp.asarray([[2.0, 3.0], [4.0, 1.0], [1.0, 2.0]]),
        (4, 3, 2),
    )
    return PosteriorTargets(
        prediction=PosteriorPrediction(
            policy=jnp.full((4, 3), 1.0 / 3.0),
            alpha_q=alpha,
        ),
        metadata=TargetMetadata(
            search_action=jnp.asarray([0, 1, 0, 0], dtype=jnp.int32),
            q_target_outcome=jnp.full((4, 3), int(NO_OUTCOME), jnp.int8),
            v_target_outcome=jnp.full((4,), int(NO_OUTCOME), jnp.int8),
        ),
    )


def _mixed_posterior():
    posterior = _safe_posterior()
    # The solved row has two certified wins and an unsafe unresolved action.
    # Commitment must retain the backend's chosen win at index 1.
    alpha = posterior.prediction.alpha_q.at[1, 2, 0].set(1e-5)
    alpha = alpha.at[3, 0, 0].set(1e-5)
    return posterior._replace(
        prediction=posterior.prediction._replace(alpha_q=alpha),
        metadata=posterior.metadata._replace(
            q_target_outcome=posterior.metadata.q_target_outcome.at[
                1, :2
            ].set(1),
            v_target_outcome=posterior.metadata.v_target_outcome.at[1].set(1),
        ),
    )


def _eager_commitment_policy(posterior, support, key, *, numerical):
    """Previous eager path, with the same key and full-batch sample shape."""
    alpha = posterior.prediction.alpha_q
    metadata = posterior.metadata
    native = posterior_best_policy(
        key,
        alpha,
        ~support,
        8,
        chunk_size=2,
        categorical_outcome=metadata.q_target_outcome,
    )
    categorical = metadata.v_target_outcome != int(NO_OUTCOME)
    native = jnp.where(
        categorical[:, None],
        jax.nn.one_hot(metadata.search_action, alpha.shape[-2]),
        native,
    )
    native = _normalize_policy_on_support(native, support)
    if not numerical:
        return native
    estimate = binary_posterior_best_policy_prefix_quadrature(
        alpha, ~support, metadata.q_target_outcome
    )
    unsafe = (
        estimate.tail_range_clipped
        | ~estimate.finite
        | (
            jnp.max(jnp.abs(estimate.density_log_integral), axis=-1)
            > DEFAULT_PREFIX_DENSITY_LOG_INTEGRAL_TOLERANCE
        )
    )
    accepted = jnp.any(support, axis=-1) & ~categorical & ~unsafe
    return jnp.where(accepted[:, None], estimate.policy, native)


def _record_sampling(monkeypatch):
    calls = []

    def recorded_sampler(*args, **kwargs):
        # Count executed branches under JIT, independently of Python tracing.
        jax.debug.callback(lambda: calls.append(True), ordered=True)
        return posterior_best_policy(*args, **kwargs)

    monkeypatch.setattr(play_search, "posterior_best_policy", recorded_sampler)
    return calls


def test_numerical_commitment_samples_only_for_unsafe_unresolved_rows(monkeypatch):
    key = jax.random.PRNGKey(917)
    config = _search_config()
    all_legal = jnp.ones((4, 3), dtype=jnp.bool_)
    mixed_legal = all_legal.at[3].set(False)
    safe = _safe_posterior()
    mixed = _mixed_posterior()
    unsafe = mixed._replace(
        prediction=mixed.prediction._replace(
            alpha_q=mixed.prediction.alpha_q.at[2, 0, 0].set(1e-5)
        )
    )
    eager = jax.jit(
        lambda posterior, support: _eager_commitment_policy(
            posterior, support, key, numerical=True
        )
    )
    expected_safe = eager(safe, all_legal)
    expected_mixed = eager(mixed, mixed_legal)
    expected_unsafe = eager(unsafe, mixed_legal)
    estimate = binary_posterior_best_policy_prefix_quadrature(
        unsafe.prediction.alpha_q,
        ~mixed_legal,
        unsafe.metadata.q_target_outcome,
    )
    assert jnp.array_equal(
        estimate.tail_range_clipped, jnp.asarray([False, True, True, False])
    )

    calls = _record_sampling(monkeypatch)
    commitment = jax.jit(
        lambda posterior, legal: _dirichlet_commitment_policy(
            posterior, legal, key, config, PosteriorUpdateKind.numerical
        )
    )
    actual_safe = jax.block_until_ready(commitment(safe, all_legal))
    jax.effects_barrier()
    assert calls == []
    assert jnp.allclose(actual_safe, expected_safe, atol=1e-6)

    actual_mixed = jax.block_until_ready(commitment(mixed, mixed_legal))
    jax.effects_barrier()
    assert calls == []
    assert jnp.allclose(actual_mixed, expected_mixed, atol=1e-6)
    assert jnp.array_equal(actual_mixed[1], jnp.asarray([0.0, 1.0, 0.0]))
    assert jnp.array_equal(actual_mixed[3], jnp.zeros((3,)))

    # Changing only the unresolved row reuses the compiled function and key.
    actual_unsafe = jax.block_until_ready(commitment(unsafe, mixed_legal))
    jax.effects_barrier()
    assert calls == [True]
    assert jnp.array_equal(actual_unsafe[0], actual_mixed[0])
    assert jnp.array_equal(actual_unsafe[2], expected_unsafe[2])
    assert jnp.allclose(actual_unsafe, expected_unsafe, atol=1e-6)


def test_monte_carlo_commitment_keeps_native_sampling_and_certified_action(
    monkeypatch,
):
    key = jax.random.PRNGKey(918)
    config = _search_config()
    posterior = _mixed_posterior()
    legal = jnp.ones((4, 3), dtype=jnp.bool_).at[3].set(False)
    expected = jax.jit(
        lambda posterior, support: _eager_commitment_policy(
            posterior, support, key, numerical=False
        )
    )(posterior, legal)

    calls = _record_sampling(monkeypatch)
    actual = jax.block_until_ready(
        jax.jit(
            lambda posterior, legal: _dirichlet_commitment_policy(
                posterior, legal, key, config, PosteriorUpdateKind.monte_carlo
            )
        )(posterior, legal)
    )
    jax.effects_barrier()

    assert calls == [True]
    assert jnp.array_equal(actual, expected)
    assert jnp.array_equal(actual[1], jnp.asarray([0.0, 1.0, 0.0]))
    assert jnp.array_equal(actual[3], jnp.zeros((3,)))
