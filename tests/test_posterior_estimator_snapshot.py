import math

import jax
import jax.numpy as jnp
import pytest

from scacchi.dirichlet_mctx import posterior_updates as posterior_updates_module
from scacchi.dirichlet_mctx.action_selection import posterior_best_policy
from scacchi.dirichlet_mctx.estimator_diagnostics import (
    binary_posterior_best_policy_prefix_quadrature,
)
from scacchi.dirichlet_mctx.outcomes import NO_OUTCOME
from scacchi.dirichlet_mctx.posterior_updates import (
    PosteriorEstimatorSnapshot,
    mix_value_prior,
    posterior_estimator_snapshot,
    update_posterior,
    update_posterior_prefix_cdf,
    update_posterior_with_estimator,
)
from scacchi.dirichlet_mctx.tree import (
    ChildrenView,
    LeafView,
    NodeView,
    PosteriorUpdateContext,
)


def _mixed_context() -> PosteriorUpdateContext:
    """Build active/inactive lanes with unresolved and categorical edges."""

    node = NodeView(
        index=jnp.array([0, 0], dtype=jnp.int32),
        embedding=jnp.zeros((2,), dtype=jnp.int32),
        value_prior=jnp.array([[4.0, 2.0], [1.0, 3.0]], dtype=jnp.float32),
        value_alpha=jnp.array([[9.0, 1.0], [2.0, 8.0]], dtype=jnp.float32),
        node_payload=jnp.array([5, 0], dtype=jnp.int32),
        edge_alpha=jnp.array(
            [
                [[1.0, 1.0], [2.0, 3.0], [7.0, 1.0]],
                [[3.0, 1.0], [2.0, 5.0], [4.0, 4.0]],
            ],
            dtype=jnp.float32,
        ),
        edge_payload=jnp.array([[0, 2, 9], [0, 4, 0]], dtype=jnp.int32),
        edge_categorical_outcome=jnp.array(
            [
                [int(NO_OUTCOME), int(NO_OUTCOME), 0],
                [int(NO_OUTCOME), 1, int(NO_OUTCOME)],
            ],
            dtype=jnp.int8,
        ),
        to_play=jnp.zeros((2,), dtype=jnp.int32),
        invalid_actions=jnp.array(
            [[False, False, True], [False, True, False]],
            dtype=bool,
        ),
    )
    children = ChildrenView(
        index=jnp.array([[-1, 1, 2], [1, 2, 3]], dtype=jnp.int32),
        visited=jnp.array(
            [[False, True, True], [True, True, True]],
            dtype=bool,
        ),
        embedding_table=jnp.zeros((2, 4), dtype=jnp.int32),
        value_prior=jnp.array(
            [
                [[8.0, 2.0], [4.0, 2.0], [3.0, 3.0]],
                [[6.0, 2.0], [1.0, 7.0], [2.0, 6.0]],
            ],
            dtype=jnp.float32,
        ),
        value_alpha=jnp.array(
            [
                [[8.0, 2.0], [5.0, 1.0], [9.0, 1.0]],
                [[7.0, 1.0], [1.0, 8.0], [3.0, 5.0]],
            ],
            dtype=jnp.float32,
        ),
        node_payload=jnp.array([[0, 2, 4], [3, 2, 4]], dtype=jnp.int32),
        categorical_outcome=jnp.array(
            [
                [int(NO_OUTCOME), int(NO_OUTCOME), 0],
                [int(NO_OUTCOME), 1, int(NO_OUTCOME)],
            ],
            dtype=jnp.int8,
        ),
        to_play=jnp.zeros((2, 3), dtype=jnp.int32),
    )
    return PosteriorUpdateContext(
        node=node,
        children=children,
        leaf=LeafView(
            action=jnp.array([0, 2], dtype=jnp.int32),
            value_alpha=jnp.array([[6.0, 2.0], [9.0, 1.0]], dtype=jnp.float32),
            to_play=jnp.zeros((2,), dtype=jnp.int32),
            active=jnp.array([True, False]),
        ),
        active=jnp.array([True, False]),
    )


def test_snapshot_exposes_exact_post_message_pre_estimator_operands():
    context = _mixed_context()

    snapshot = jax.jit(
        lambda c: posterior_estimator_snapshot(c, kappa=4.0)
    )(context)

    assert jnp.array_equal(
        snapshot.effective_alpha,
        jnp.array(
            [
                [[6.0, 2.0], [5.0, 1.0], [7.0, 1.0]],
                [[6.0, 2.0], [2.0, 5.0], [2.0, 6.0]],
            ],
            dtype=jnp.float32,
        ),
    )
    assert jnp.array_equal(
        snapshot.cache_alpha,
        jnp.array(
            [
                [[6.0, 2.0], [5.0, 1.0], [8.0, 0.0]],
                [[6.0, 2.0], [0.0, 7.0], [2.0, 6.0]],
            ],
            dtype=jnp.float32,
        ),
    )
    assert jnp.array_equal(snapshot.invalid_actions, context.node.invalid_actions)
    assert jnp.array_equal(
        snapshot.categorical_mask,
        context.node.edge_categorical_outcome != int(NO_OUTCOME),
    )
    assert jnp.array_equal(
        snapshot.categorical_outcome,
        context.node.edge_categorical_outcome,
    )
    assert jnp.array_equal(snapshot.n_down, jnp.array([7, 0], dtype=jnp.int32))
    assert jnp.allclose(snapshot.gamma, jnp.array([7.0 / 11.0, 0.0]))
    assert snapshot.kappa == jnp.asarray(4.0, dtype=jnp.float32)
    assert jnp.array_equal(snapshot.value_prior, context.node.value_prior)
    assert jnp.array_equal(
        snapshot.previous_value_alpha,
        context.node.value_alpha,
    )
    assert jnp.array_equal(snapshot.active, context.active)


def test_update_consumes_the_exposed_operands_without_changing_semantics():
    context = _mixed_context()
    key = jax.random.PRNGKey(17)
    snapshot = posterior_estimator_snapshot(context, kappa=4.0)

    search_policy = posterior_best_policy(
        key,
        snapshot.effective_alpha,
        snapshot.invalid_actions,
        16,
        chunk_size=4,
        categorical_outcome=snapshot.categorical_outcome,
    )
    repaired_value = mix_value_prior(
        snapshot.value_prior,
        snapshot.cache_alpha,
        search_policy,
        snapshot.n_down,
        kappa=4.0,
    )
    expected_value = jnp.where(
        (snapshot.active & (snapshot.n_down > 0))[..., None],
        repaired_value,
        context.node.value_alpha,
    )

    update = update_posterior(
        key,
        context,
        kappa=4.0,
        policy_samples=16,
        policy_sample_chunk_size=4,
    )

    assert jnp.array_equal(
        update.edge_alpha,
        jnp.array(
            [
                [[6.0, 2.0], [5.0, 1.0], [7.0, 1.0]],
                [[3.0, 1.0], [2.0, 5.0], [4.0, 4.0]],
            ],
            dtype=jnp.float32,
        ),
    )
    assert jnp.array_equal(
        update.edge_payload,
        jnp.array([[1, 3, 9], [0, 4, 0]], dtype=jnp.int32),
    )
    assert jnp.array_equal(update.value_alpha, expected_value)


def test_update_reports_exact_kappa_innovation_and_local_leverage():
    context = _mixed_context()
    snapshot = posterior_estimator_snapshot(context, kappa=4.0)
    policy = jnp.asarray(
        [[0.25, 0.75, 0.0], [0.5, 0.0, 0.5]],
        dtype=jnp.float32,
    )

    update = update_posterior_with_estimator(
        jax.random.PRNGKey(31),
        context,
        lambda _key, _snapshot: policy,
        kappa=4.0,
    )

    diagnostics = update.diagnostics
    assert diagnostics is not None
    assert jnp.array_equal(
        diagnostics.numeric,
        jnp.asarray([True, False]),
    )
    weighted = jnp.sum(
        policy[..., None] * snapshot.cache_alpha,
        axis=-2,
    )
    innovation = weighted - snapshot.value_prior
    expected_raw = jnp.linalg.vector_norm(innovation, axis=-1)
    weighted_mean = weighted / jnp.sum(weighted, axis=-1, keepdims=True)
    prior_mean = snapshot.value_prior / jnp.sum(
        snapshot.value_prior,
        axis=-1,
        keepdims=True,
    )
    expected_semantic = jnp.linalg.vector_norm(
        weighted_mean - prior_mean,
        axis=-1,
    )
    expected_concentration = jnp.abs(
        jnp.sum(weighted, axis=-1)
        - jnp.sum(snapshot.value_prior, axis=-1)
    )
    assert jnp.allclose(diagnostics.raw_innovation_l2[0], expected_raw[0])
    assert jnp.allclose(
        diagnostics.semantic_innovation_l2[0],
        expected_semantic[0],
    )
    assert jnp.allclose(
        diagnostics.concentration_innovation_abs[0],
        expected_concentration[0],
    )

    exact_leverage = jnp.linalg.vector_norm(
        (
            -snapshot.gamma * (1.0 - snapshot.gamma)
        )[..., None] * innovation,
        axis=-1,
    )
    assert jnp.allclose(
        diagnostics.raw_dcache_dlogkappa_l2[0],
        exact_leverage[0],
        rtol=1e-6,
        atol=1e-7,
    )
    cache = update.value_alpha
    cache_mean = cache / jnp.sum(cache, axis=-1, keepdims=True)
    cache_mass = jnp.sum(cache, axis=-1, keepdims=True)
    exact_mean_leverage = (
        (
            -snapshot.gamma * (1.0 - snapshot.gamma)
        )[..., None]
        * innovation
    )
    exact_mean_leverage = (
        exact_mean_leverage
        - cache_mean
        * jnp.sum(exact_mean_leverage, axis=-1, keepdims=True)
    ) / cache_mass
    assert jnp.allclose(
        diagnostics.mean_dcache_dlogkappa_l2[0],
        jnp.linalg.vector_norm(exact_mean_leverage, axis=-1)[0],
        rtol=1e-6,
        atol=1e-7,
    )
    exact_log_concentration_leverage = jnp.abs(
        jnp.sum(
            (
                -snapshot.gamma * (1.0 - snapshot.gamma)
            )[..., None]
            * innovation,
            axis=-1,
        )
        / jnp.sum(cache, axis=-1)
    )
    assert jnp.allclose(
        diagnostics.log_concentration_dcache_dlogkappa_abs[0],
        exact_log_concentration_leverage[0],
        rtol=1e-6,
        atol=1e-7,
    )

    epsilon = 1e-2
    plus = mix_value_prior(
        snapshot.value_prior,
        snapshot.cache_alpha,
        policy,
        snapshot.n_down,
        kappa=4.0 * math.exp(epsilon),
    )
    minus = mix_value_prior(
        snapshot.value_prior,
        snapshot.cache_alpha,
        policy,
        snapshot.n_down,
        kappa=4.0 * math.exp(-epsilon),
    )
    finite_difference = (plus - minus) / (2.0 * epsilon)
    expected_leverage = jnp.linalg.vector_norm(
        finite_difference,
        axis=-1,
    )
    assert jnp.allclose(
        diagnostics.raw_dcache_dlogkappa_l2[0],
        expected_leverage[0],
        rtol=1e-3,
        atol=2e-5,
    )
    plus_mean = plus / jnp.sum(plus, axis=-1, keepdims=True)
    minus_mean = minus / jnp.sum(minus, axis=-1, keepdims=True)
    finite_mean_difference = (plus_mean - minus_mean) / (
        2.0 * epsilon
    )
    assert jnp.allclose(
        diagnostics.mean_dcache_dlogkappa_l2[0],
        jnp.linalg.vector_norm(finite_mean_difference, axis=-1)[0],
        rtol=1e-3,
        atol=2e-5,
    )
    finite_log_concentration_difference = (
        jnp.log(jnp.sum(plus, axis=-1))
        - jnp.log(jnp.sum(minus, axis=-1))
    ) / (2.0 * epsilon)
    assert jnp.allclose(
        diagnostics.log_concentration_dcache_dlogkappa_abs[0],
        jnp.abs(finite_log_concentration_difference[0]),
        rtol=1e-3,
        atol=2e-5,
    )
    assert jnp.all(diagnostics.gamma[1:] == 0)
    assert jnp.all(diagnostics.raw_innovation_l2[1:] == 0)
    assert jnp.all(diagnostics.semantic_innovation_l2[1:] == 0)
    assert jnp.all(diagnostics.mean_dcache_dlogkappa_l2[1:] == 0)
    assert jnp.all(
        diagnostics.log_concentration_dcache_dlogkappa_abs[1:] == 0
    )


def test_estimator_hook_prepares_once_and_passes_original_key(monkeypatch):
    context = _mixed_context()
    key = jax.random.PRNGKey(23)
    expected_snapshot = posterior_estimator_snapshot(context, kappa=4.0)
    fixed_policy = jnp.array(
        [[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=jnp.float32,
    )

    preparation_calls = 0
    estimator_calls = 0
    original_prepare = (
        posterior_updates_module._prepare_posterior_estimator_snapshot
    )

    def counted_prepare(context, *, kappa):
        nonlocal preparation_calls
        preparation_calls += 1
        return original_prepare(context, kappa=kappa)

    def estimator(
        estimator_key: jax.Array,
        snapshot: PosteriorEstimatorSnapshot,
    ) -> jax.Array:
        nonlocal estimator_calls
        estimator_calls += 1
        assert jnp.array_equal(estimator_key, key)
        assert jnp.array_equal(
            snapshot.effective_alpha,
            expected_snapshot.effective_alpha,
        )
        assert jnp.array_equal(snapshot.cache_alpha, expected_snapshot.cache_alpha)
        return fixed_policy

    monkeypatch.setattr(
        posterior_updates_module,
        "_prepare_posterior_estimator_snapshot",
        counted_prepare,
    )
    update = update_posterior_with_estimator(
        key,
        context,
        estimator,
        kappa=4.0,
    )

    expected_repaired_value = mix_value_prior(
        expected_snapshot.value_prior,
        expected_snapshot.cache_alpha,
        fixed_policy,
        expected_snapshot.n_down,
        kappa=4.0,
    )
    expected_value = jnp.where(
        (
            expected_snapshot.active
            & (expected_snapshot.n_down > 0)
        )[..., None],
        expected_repaired_value,
        expected_snapshot.previous_value_alpha,
    )
    assert preparation_calls == 1
    assert estimator_calls == 1
    assert jnp.array_equal(
        update.edge_payload,
        jnp.array([[1, 3, 9], [0, 4, 0]], dtype=jnp.int32),
    )
    assert jnp.array_equal(update.value_alpha, expected_value)


def test_default_update_is_bitwise_identical_to_legacy_estimator_hook():
    context = _mixed_context()
    key = jax.random.PRNGKey(29)

    def legacy_estimator(
        estimator_key: jax.Array,
        snapshot: PosteriorEstimatorSnapshot,
    ) -> jax.Array:
        return posterior_best_policy(
            estimator_key,
            snapshot.effective_alpha,
            snapshot.invalid_actions,
            16,
            chunk_size=4,
            categorical_outcome=snapshot.categorical_outcome,
        )

    expected = update_posterior_with_estimator(
        key,
        context,
        legacy_estimator,
        kappa=4.0,
    )
    actual = update_posterior(
        key,
        context,
        kappa=4.0,
        policy_samples=16,
        policy_sample_chunk_size=4,
    )

    assert jnp.array_equal(actual.edge_alpha, expected.edge_alpha)
    assert jnp.array_equal(actual.edge_payload, expected.edge_payload)
    assert jnp.array_equal(actual.value_alpha, expected.value_alpha)


def test_prefix_update_uses_mass_conserving_policy_inside_same_cache_rule():
    context = _mixed_context()
    key = jax.random.PRNGKey(31)
    snapshot = posterior_estimator_snapshot(context, kappa=4.0)
    policy = binary_posterior_best_policy_prefix_quadrature(
        snapshot.effective_alpha,
        snapshot.invalid_actions,
        snapshot.categorical_outcome,
    ).policy
    expected = update_posterior_with_estimator(
        key,
        context,
        lambda _key, _snapshot: policy,
        kappa=4.0,
    )

    actual = jax.jit(
        lambda update_key, update_context: update_posterior_prefix_cdf(
            update_key,
            update_context,
            kappa=4.0,
        )
    )(key, context)

    assert jnp.array_equal(actual.edge_alpha, expected.edge_alpha)
    assert jnp.array_equal(actual.edge_payload, expected.edge_payload)
    assert jnp.allclose(actual.value_alpha, expected.value_alpha, atol=1e-7)


def test_prefix_update_falls_back_bitwise_outside_density_gate():
    context = _mixed_context()
    key = jax.random.PRNGKey(37)
    expected = update_posterior(
        key,
        context,
        kappa=4.0,
        policy_samples=16,
        policy_sample_chunk_size=4,
    )

    actual = jax.jit(
        lambda update_key, update_context: update_posterior_prefix_cdf(
            update_key,
            update_context,
            kappa=4.0,
            fallback_policy_samples=16,
            fallback_policy_sample_chunk_size=4,
            density_log_integral_tolerance=1e-12,
        )
    )(key, context)

    assert jnp.array_equal(actual.edge_alpha, expected.edge_alpha)
    assert jnp.array_equal(actual.edge_payload, expected.edge_payload)
    assert jnp.array_equal(actual.value_alpha, expected.value_alpha)


@pytest.mark.parametrize("kappa", [0.0, -1.0, float("inf"), float("nan")])
def test_snapshot_rejects_invalid_kappa(kappa: float):
    with pytest.raises(ValueError, match="kappa must be finite and > 0"):
        posterior_estimator_snapshot(_mixed_context(), kappa=kappa)
