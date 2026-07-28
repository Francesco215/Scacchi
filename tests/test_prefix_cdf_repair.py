from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from scacchi import dirichlet_mctx
from scacchi.dirichlet_mctx.action_selection import (
    categorical_action_population,
    posterior_best_policy,
)
from scacchi.dirichlet_mctx.outcomes import NO_OUTCOME
from scacchi.dirichlet_mctx.prefix_cdf import (
    binary_posterior_best_policy_prefix_quadrature,
)
from scacchi.dirichlet_mctx.posterior_updates import mix_value_prior
from scacchi.dirichlet_mctx.tree import ChildrenView, LeafView, NodeView


def _context(
    edge_alpha: jax.Array,
    *,
    edge_payload: jax.Array | None = None,
    edge_outcome: jax.Array | None = None,
    invalid_actions: jax.Array | None = None,
    node_payload: jax.Array | None = None,
    value_prior: jax.Array | None = None,
) -> dirichlet_mctx.PosteriorUpdateContext:
    edge_alpha = jnp.asarray(edge_alpha, dtype=jnp.float32)
    batch_size, num_actions, num_outcomes = edge_alpha.shape
    if edge_payload is None:
        edge_payload = jnp.ones(
            (batch_size, num_actions),
            dtype=jnp.int32,
        )
    if edge_outcome is None:
        edge_outcome = jnp.full(
            (batch_size, num_actions),
            int(NO_OUTCOME),
            dtype=jnp.int8,
        )
    if invalid_actions is None:
        invalid_actions = jnp.zeros(
            (batch_size, num_actions),
            dtype=jnp.bool_,
        )
    if node_payload is None:
        unresolved = edge_outcome == int(NO_OUTCOME)
        node_payload = jnp.sum(
            jnp.where(unresolved & ~invalid_actions, edge_payload, 0),
            axis=-1,
            dtype=jnp.int32,
        )
    if value_prior is None:
        value_prior = jnp.ones(
            (batch_size, num_outcomes),
            dtype=jnp.float32,
        )

    return dirichlet_mctx.PosteriorUpdateContext(
        node=NodeView(
            index=jnp.zeros((batch_size,), dtype=jnp.int32),
            embedding=jnp.zeros((batch_size,), dtype=jnp.int32),
            value_prior=jnp.asarray(value_prior, dtype=jnp.float32),
            value_alpha=jnp.full(
                (batch_size, num_outcomes),
                3.0,
                dtype=jnp.float32,
            ),
            node_payload=jnp.asarray(node_payload, dtype=jnp.int32),
            edge_alpha=edge_alpha,
            edge_payload=jnp.asarray(edge_payload, dtype=jnp.int32),
            edge_categorical_outcome=jnp.asarray(
                edge_outcome,
                dtype=jnp.int8,
            ),
            to_play=jnp.zeros((batch_size,), dtype=jnp.int32),
            invalid_actions=jnp.asarray(
                invalid_actions,
                dtype=jnp.bool_,
            ),
        ),
        children=ChildrenView(
            index=jnp.full(
                (batch_size, num_actions),
                -1,
                dtype=jnp.int32,
            ),
            visited=jnp.zeros(
                (batch_size, num_actions),
                dtype=jnp.bool_,
            ),
            embedding_table=jnp.zeros(
                (batch_size, 1),
                dtype=jnp.int32,
            ),
            value_prior=jnp.ones_like(edge_alpha),
            value_alpha=jnp.ones_like(edge_alpha),
            node_payload=jnp.zeros(
                (batch_size, num_actions),
                dtype=jnp.int32,
            ),
            categorical_outcome=jnp.full(
                (batch_size, num_actions),
                int(NO_OUTCOME),
                dtype=jnp.int8,
            ),
            to_play=jnp.zeros(
                (batch_size, num_actions),
                dtype=jnp.int32,
            ),
        ),
        leaf=LeafView(
            action=jnp.zeros((batch_size,), dtype=jnp.int32),
            value_alpha=jnp.ones(
                (batch_size, num_outcomes),
                dtype=jnp.float32,
            ),
            to_play=jnp.zeros((batch_size,), dtype=jnp.int32),
            active=jnp.zeros((batch_size,), dtype=jnp.bool_),
        ),
        active=jnp.ones((batch_size,), dtype=jnp.bool_),
    )


def test_categorical_action_population_matches_native_ties_and_invalid_mask():
    node_outcome = jnp.asarray([2, 0, 1, 1], dtype=jnp.int8)
    edge_outcome = jnp.asarray(
        [
            [2, 2, 2, 0, 1],
            [0, 0, 0, 1, 2],
            [1, 0, 1, 2, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=jnp.int8,
    )
    edge_distance = jnp.asarray(
        [
            [1, 1, 1, 99, 0],
            [5, 5, 5, 0, 0],
            [8, 2, 1, 3, 5],
            [1, 2, 3, 4, 5],
        ],
        dtype=jnp.int32,
    )
    invalid_actions = jnp.asarray(
        [
            [False, False, True, False, False],
            [False, False, True, False, False],
            [False, False, False, False, True],
            [True, True, True, True, True],
        ],
        dtype=jnp.bool_,
    )

    population = categorical_action_population(
        node_outcome,
        edge_outcome,
        edge_distance,
        invalid_actions,
        num_outcomes=3,
    )

    assert jnp.array_equal(
        population,
        jnp.asarray(
            [
                [0.5, 0.5, 0.0, 0.0, 0.0],
                [0.5, 0.5, 0.0, 0.0, 0.0],
                [0.5, 0.0, 0.5, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        ),
    )

    binary_population = categorical_action_population(
        node_outcome=jnp.asarray([1, 0], dtype=jnp.int8),
        edge_outcome=jnp.asarray([[1, 1, 0], [0, 0, 1]], dtype=jnp.int8),
        edge_distance=jnp.asarray([[2, 2, 9], [3, 5, 1]], dtype=jnp.int32),
        invalid_actions=jnp.zeros((2, 3), dtype=jnp.bool_),
        num_outcomes=2,
    )
    assert jnp.array_equal(
        binary_population,
        jnp.asarray(
            [[0.5, 0.5, 0.0], [0.0, 1.0, 0.0]],
            dtype=jnp.float32,
        ),
    )


def test_native_update_keeps_original_key_and_winner_population_path():
    context = _context(
        jnp.asarray(
            [[[4.0, 1.0], [1.0, 4.0], [2.0, 3.0]]],
            dtype=jnp.float32,
        )
    )
    key = jax.random.PRNGKey(913)
    samples = 7
    chunk_size = 3
    expected_policy = posterior_best_policy(
        key,
        context.node.edge_alpha,
        context.node.invalid_actions,
        samples,
        chunk_size=chunk_size,
        categorical_outcome=context.node.edge_categorical_outcome,
    )
    expected_value = mix_value_prior(
        context.node.value_prior,
        context.node.edge_alpha,
        expected_policy,
        context.node.node_payload,
    )

    actual = dirichlet_mctx.update_posterior(
        key,
        context,
        policy_samples=samples,
        policy_sample_chunk_size=chunk_size,
    )

    assert jnp.array_equal(actual.edge_alpha, context.node.edge_alpha)
    assert jnp.array_equal(actual.edge_payload, context.node.edge_payload)
    assert jnp.array_equal(actual.value_alpha, expected_value)


def test_native_and_prefix_share_direct_leaf_message_preparation():
    context = _context(
        jnp.ones((1, 2, 2), dtype=jnp.float32),
        edge_payload=jnp.zeros((1, 2), dtype=jnp.int32),
        node_payload=jnp.zeros((1,), dtype=jnp.int32),
    )
    context = context.replace(
        leaf=context.leaf.replace(
            action=jnp.asarray([1], dtype=jnp.int32),
            value_alpha=jnp.asarray([[2.0, 6.0]], dtype=jnp.float32),
            active=jnp.asarray([True]),
        )
    )
    key = jax.random.PRNGKey(914)

    native = dirichlet_mctx.update_posterior(
        key,
        context,
        policy_samples=11,
        policy_sample_chunk_size=4,
    )
    prefix = dirichlet_mctx.update_posterior_prefix_cdf(
        key,
        context,
        fallback_policy_samples=11,
        fallback_policy_sample_chunk_size=4,
    )
    expected_alpha = context.node.edge_alpha.at[0, 1].set(
        jnp.asarray([2.0, 6.0], dtype=jnp.float32)
    )
    expected_policy = posterior_best_policy(
        key,
        expected_alpha,
        context.node.invalid_actions,
        11,
        chunk_size=4,
        categorical_outcome=context.node.edge_categorical_outcome,
    )
    expected_value = mix_value_prior(
        context.node.value_prior,
        expected_alpha,
        expected_policy,
        jnp.asarray([1], dtype=jnp.int32),
    )

    assert jnp.array_equal(native.edge_alpha, prefix.edge_alpha)
    assert jnp.array_equal(native.edge_payload, prefix.edge_payload)
    assert jnp.array_equal(native.edge_alpha[0, 1], jnp.asarray([2.0, 6.0]))
    assert int(native.edge_payload[0, 1]) == 1
    assert jnp.array_equal(native.value_alpha, expected_value)


def test_native_and_prefix_share_refreshed_child_message_preparation():
    context = _context(
        jnp.ones((1, 2, 2), dtype=jnp.float32),
        edge_payload=jnp.zeros((1, 2), dtype=jnp.int32),
        node_payload=jnp.zeros((1,), dtype=jnp.int32),
    )
    context = context.replace(
        children=context.children.replace(
            visited=jnp.asarray([[True, False]]),
            value_alpha=jnp.asarray(
                [[[3.0, 7.0], [1.0, 1.0]]],
                dtype=jnp.float32,
            ),
            node_payload=jnp.asarray([[2, 0]], dtype=jnp.int32),
        )
    )
    key = jax.random.PRNGKey(915)

    native = dirichlet_mctx.update_posterior(
        key,
        context,
        policy_samples=11,
        policy_sample_chunk_size=4,
    )
    prefix = dirichlet_mctx.update_posterior_prefix_cdf(
        key,
        context,
        fallback_policy_samples=11,
        fallback_policy_sample_chunk_size=4,
    )
    expected_alpha = context.node.edge_alpha.at[0, 0].set(
        jnp.asarray([3.0, 7.0], dtype=jnp.float32)
    )
    expected_policy = posterior_best_policy(
        key,
        expected_alpha,
        context.node.invalid_actions,
        11,
        chunk_size=4,
        categorical_outcome=context.node.edge_categorical_outcome,
    )
    expected_value = mix_value_prior(
        context.node.value_prior,
        expected_alpha,
        expected_policy,
        jnp.asarray([3], dtype=jnp.int32),
    )

    assert jnp.array_equal(native.edge_alpha, prefix.edge_alpha)
    assert jnp.array_equal(native.edge_payload, prefix.edge_payload)
    assert jnp.array_equal(native.edge_alpha[0, 0], jnp.asarray([3.0, 7.0]))
    assert int(native.edge_payload[0, 0]) == 3
    assert jnp.array_equal(native.value_alpha, expected_value)


def test_safe_prefix_repair_is_deterministic_and_uses_q21_policy():
    context = _context(
        jnp.asarray(
            [[[5.0, 1.0], [1.0, 5.0], [2.0, 2.0]]],
            dtype=jnp.float32,
        )
    )
    estimate = binary_posterior_best_policy_prefix_quadrature(
        context.node.edge_alpha,
        context.node.invalid_actions,
        context.node.edge_categorical_outcome,
    )
    expected_value = mix_value_prior(
        context.node.value_prior,
        context.node.edge_alpha,
        estimate.policy,
        context.node.node_payload,
    )

    first = dirichlet_mctx.update_posterior_prefix_cdf(
        jax.random.PRNGKey(1),
        context,
    )
    second = dirichlet_mctx.update_posterior_prefix_cdf(
        jax.random.PRNGKey(2),
        context,
    )

    assert not bool(jnp.any(estimate.tail_range_clipped))
    assert jnp.max(jnp.abs(estimate.density_log_integral)) < 0.01
    assert jnp.array_equal(first.edge_alpha, context.node.edge_alpha)
    assert jnp.array_equal(first.edge_payload, context.node.edge_payload)
    assert jnp.allclose(first.value_alpha, expected_value, atol=1e-6)
    assert jnp.array_equal(first.value_alpha, second.value_alpha)


def test_one_unsafe_lane_falls_back_without_changing_safe_lane():
    context = _context(
        jnp.asarray(
            [
                [[2.0, 3.0], [4.0, 1.0], [1.0, 2.0]],
                [[1e-5, 1.0], [2.0, 1.0], [1.0, 3.0]],
            ],
            dtype=jnp.float32,
        )
    )
    estimate = binary_posterior_best_policy_prefix_quadrature(
        context.node.edge_alpha,
        context.node.invalid_actions,
        context.node.edge_categorical_outcome,
    )
    key = jax.random.PRNGKey(441)
    samples = 11
    chunk_size = 4

    guarded = dirichlet_mctx.update_posterior_prefix_cdf(
        key,
        context,
        fallback_policy_samples=samples,
        fallback_policy_sample_chunk_size=chunk_size,
    )
    native = dirichlet_mctx.update_posterior(
        key,
        context,
        policy_samples=samples,
        policy_sample_chunk_size=chunk_size,
    )
    expected_prefix_value = mix_value_prior(
        context.node.value_prior,
        context.node.edge_alpha,
        estimate.policy,
        context.node.node_payload,
    )

    assert jnp.array_equal(
        estimate.tail_range_clipped,
        jnp.asarray([False, True]),
    )
    assert jnp.allclose(
        guarded.value_alpha[0],
        expected_prefix_value[0],
        atol=1e-6,
    )
    assert jnp.array_equal(guarded.value_alpha[1], native.value_alpha[1])
    assert jnp.array_equal(guarded.edge_alpha, native.edge_alpha)
    assert jnp.array_equal(guarded.edge_payload, native.edge_payload)


def test_prefix_repair_projects_categorical_cache_without_mutating_edge():
    edge_alpha = jnp.asarray(
        [[[2.0, 5.0], [5.0, 2.0]]],
        dtype=jnp.float32,
    )
    context = _context(
        edge_alpha,
        edge_outcome=jnp.asarray(
            [[int(NO_OUTCOME), 1]],
            dtype=jnp.int8,
        ),
        edge_payload=jnp.asarray([[1, 1]], dtype=jnp.int32),
        node_payload=jnp.asarray([1], dtype=jnp.int32),
    )
    expected_cache_alpha = edge_alpha.at[0, 1].set(
        jnp.asarray([0.0, 7.0], dtype=jnp.float32)
    )
    expected_value = mix_value_prior(
        context.node.value_prior,
        expected_cache_alpha,
        jnp.asarray([[0.0, 1.0]], dtype=jnp.float32),
        context.node.node_payload,
    )

    update = dirichlet_mctx.update_posterior_prefix_cdf(
        jax.random.PRNGKey(71),
        context,
    )

    assert jnp.array_equal(update.edge_alpha, edge_alpha)
    assert jnp.array_equal(update.edge_payload, context.node.edge_payload)
    assert jnp.allclose(update.value_alpha, expected_value, atol=1e-6)


def test_prefix_repair_rejects_nonbinary_outcomes_and_invalid_guard():
    three_outcome = _context(
        jnp.ones((1, 2, 3), dtype=jnp.float32)
    )
    with pytest.raises(ValueError, match="exactly two outcomes"):
        dirichlet_mctx.update_posterior_prefix_cdf(
            jax.random.PRNGKey(0),
            three_outcome,
        )

    binary = _context(jnp.ones((1, 2, 2), dtype=jnp.float32))
    with pytest.raises(
        ValueError,
        match="density_log_integral_tolerance",
    ):
        dirichlet_mctx.update_posterior_prefix_cdf(
            jax.random.PRNGKey(0),
            binary,
            density_log_integral_tolerance=0.0,
        )
