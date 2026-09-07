"""Parity with the previous scatter-based posterior-message preparation."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scacchi.dirichlet_mctx.outcomes import NO_OUTCOME, align_outcome
from scacchi.dirichlet_mctx.posterior_updates import (
    _RepairInputs,
    _repair_from_policy,
    _repair_inputs,
)
from scacchi.dirichlet_mctx.tree import (
    ChildrenView,
    LeafView,
    NodeView,
    PosteriorUpdateContext,
)


def _scatter_reference(context: PosteriorUpdateContext) -> _RepairInputs:
    """Frozen pre-optimization formulas, including the direct-edge scatters."""
    node, children, leaf = context.node, context.children, context.leaf
    unresolved = node.edge_categorical_outcome == int(NO_OUTCOME)
    batch = jnp.arange(node.edge_payload.shape[0])
    action = jnp.where(leaf.active, leaf.action, 0)
    direct = leaf.active & unresolved[batch, action]
    aligned_leaf = align_outcome(leaf.value_alpha, leaf.to_play, node.to_play)
    alpha = node.edge_alpha.at[batch, action].set(
        jnp.where(direct[:, None], aligned_leaf, node.edge_alpha[batch, action])
    )
    payload = node.edge_payload.at[batch, action].set(
        node.edge_payload[batch, action] + direct.astype(node.edge_payload.dtype)
    )
    refresh = (
        context.active[:, None]
        & children.visited
        & (children.categorical_outcome == int(NO_OUTCOME))
        & (children.node_payload > 0)
        & unresolved
    )
    child_value = align_outcome(
        children.value_alpha, children.to_play, node.to_play[:, None]
    )
    alpha = jnp.where(refresh[:, :, None], child_value, alpha)
    payload = jnp.where(refresh, 1 + children.node_payload, payload)
    child_prior = align_outcome(
        children.value_prior, children.to_play, node.to_play[:, None]
    )
    use_prior = children.visited & unresolved & (payload <= 0)
    effective = jnp.where(use_prior[:, :, None], child_prior, alpha)
    categorical = ~unresolved
    outcome = jnp.where(categorical, node.edge_categorical_outcome, 0)
    categorical_alpha = jnp.sum(effective, axis=-1, keepdims=True) * jax.nn.one_hot(
        outcome, effective.shape[-1], dtype=effective.dtype
    )
    cache = jnp.where(categorical[:, :, None], categorical_alpha, effective)
    delta = jnp.sum(
        jnp.where(
            unresolved & ~node.invalid_actions,
            payload - node.edge_payload,
            0,
        ),
        axis=-1,
    )
    return _RepairInputs(alpha, payload, effective, cache, node.node_payload + delta)


def _mixed_context(num_outcomes: int, *, extreme: bool) -> PosteriorUpdateContext:
    """Cover independent leaf/activity flags and every child-refresh predicate."""
    batch, actions = 48, 7
    rng = np.random.default_rng(1709 + num_outcomes)

    def alphas(shape: tuple[int, ...]) -> jax.Array:
        if extreme:
            # Include the float32 normal floor and very high finite masses.
            options = np.asarray(
                [np.finfo(np.float32).tiny, 1e-12, 0.25, 1.0, 1e12, 1e30],
                dtype=np.float32,
            )
            values = rng.choice(options, size=shape)
        else:
            values = rng.uniform(0.1, 20.0, size=shape).astype(np.float32)
        return jnp.asarray(values)

    edge_shape = (batch, actions)
    outcome_shape = (*edge_shape, num_outcomes)
    edge_tags = rng.integers(-1, num_outcomes, size=edge_shape, dtype=np.int8)
    edge_tags[::3, :] = int(NO_OUTCOME)
    # Negative and out-of-bounds actions retain the old JAX indexing semantics;
    # actual search actions are valid, but inactive lanes may carry arbitrary data.
    action = np.resize(np.asarray([0, 3, 6, -1, -7, -8, 7, 50]), batch)
    return PosteriorUpdateContext(
        node=NodeView(
            index=jnp.zeros(batch, jnp.int32),
            embedding=jnp.zeros(batch, jnp.int32),
            value_prior=alphas((batch, num_outcomes)),
            value_alpha=alphas((batch, num_outcomes)),
            node_payload=jnp.asarray(rng.integers(0, 25, batch), jnp.int32),
            edge_alpha=alphas(outcome_shape),
            edge_payload=jnp.asarray(rng.integers(0, 4, edge_shape), jnp.int32),
            edge_categorical_outcome=jnp.asarray(edge_tags),
            to_play=jnp.asarray(rng.integers(0, 2, batch), jnp.int32),
            invalid_actions=jnp.asarray(rng.random(edge_shape) < 0.25),
        ),
        children=ChildrenView(
            index=jnp.zeros(edge_shape, jnp.int32),
            visited=jnp.asarray(rng.random(edge_shape) < 0.75),
            embedding_table=jnp.zeros((batch, 1), jnp.int32),
            value_prior=alphas(outcome_shape),
            value_alpha=alphas(outcome_shape),
            node_payload=jnp.asarray(rng.integers(0, 4, edge_shape), jnp.int32),
            categorical_outcome=jnp.asarray(
                rng.integers(-1, num_outcomes, edge_shape), jnp.int8
            ),
            to_play=jnp.asarray(rng.integers(0, 2, edge_shape), jnp.int32),
        ),
        leaf=LeafView(
            action=jnp.asarray(action, jnp.int32),
            value_alpha=alphas((batch, num_outcomes)),
            to_play=jnp.asarray(rng.integers(0, 2, batch), jnp.int32),
            active=jnp.asarray(np.arange(batch) % 3 != 0),
        ),
        active=jnp.asarray(np.arange(batch) % 4 != 0),
    )


def _assert_exact(left, right) -> None:
    for old, new in zip(jax.tree.leaves(left), jax.tree.leaves(right), strict=True):
        np.testing.assert_array_equal(np.asarray(old), np.asarray(new))


@pytest.mark.parametrize("num_outcomes", [2, 3])
@pytest.mark.parametrize("extreme", [False, True])
def test_message_preparation_matches_previous_scatter(num_outcomes, extreme):
    context = _mixed_context(num_outcomes, extreme=extreme)
    expected = jax.jit(_scatter_reference)(context)
    actual = jax.jit(_repair_inputs)(context)
    _assert_exact(expected, actual)

    # Keep the final mixture and inactive-value selection in the comparison:
    # prepared cache direction and support count both affect this output.
    weights = np.arange(1, context.node.edge_payload.shape[-1] + 1, dtype=np.float32)
    policy = jnp.broadcast_to(
        jnp.asarray(weights / weights.sum()), context.node.edge_payload.shape
    )
    for kappa in (1e-6, 4.0, 1e300):
        finish = jax.jit(
            lambda inputs: _repair_from_policy(context, inputs, policy, kappa=kappa)
        )
        _assert_exact(finish(expected), finish(actual))


@pytest.mark.parametrize("edge_dtype", [jnp.bfloat16, jnp.float16])
@pytest.mark.parametrize("refresh", [False, True])
def test_direct_leaf_preserves_mixed_precision_callback_behavior(edge_dtype, refresh):
    context = _mixed_context(2, extreme=False)
    batch, actions = context.node.edge_payload.shape
    leaf_values = jnp.broadcast_to(jnp.asarray([1.234567, 2.345678]), (batch, 2))
    child_values = jnp.broadcast_to(jnp.asarray([3.141593, 4.567891]), (batch, actions, 2))
    leaf_action = jnp.arange(batch, dtype=jnp.int32) % actions
    context = replace(
        context,
        node=replace(context.node,
            edge_alpha=context.node.edge_alpha.astype(edge_dtype),
            edge_categorical_outcome=jnp.full((batch, actions), int(NO_OUTCOME), jnp.int8),
            to_play=jnp.zeros(batch, jnp.int32)),
        leaf=replace(context.leaf, value_alpha=leaf_values, action=leaf_action,
            active=jnp.ones(batch, jnp.bool_), to_play=jnp.zeros(batch, jnp.int32)),
        children=replace(context.children, value_alpha=child_values,
            visited=jnp.full((batch, actions), refresh),
            node_payload=jnp.full((batch, actions), 2, jnp.int32),
            categorical_outcome=jnp.full((batch, actions), int(NO_OUTCOME), jnp.int8),
            to_play=jnp.zeros((batch, actions), jnp.int32)),
        active=jnp.ones(batch, jnp.bool_),
    )
    expected = jax.jit(_scatter_reference)(context)
    actual = jax.jit(_repair_inputs)(context)
    _assert_exact(expected, actual)
    for left, right in zip(expected, actual, strict=True):
        assert left.dtype == right.dtype
    if refresh:
        selected = actual.edge_alpha[jnp.arange(batch), leaf_action]
        np.testing.assert_array_equal(selected, child_values[:, 0])


@pytest.mark.parametrize("num_outcomes", [2, 3])
def test_child_refresh_still_overrides_direct_leaf_and_categorical_stays_frozen(num_outcomes):
    context = _mixed_context(num_outcomes, extreme=False)
    batch, actions = context.node.edge_payload.shape
    tags = jnp.full((batch, actions), int(NO_OUTCOME), jnp.int8)
    # Action 0 receives a direct message followed by an authoritative child
    # refresh. Action 1 has a categorical certificate and must not be refreshed.
    tags = tags.at[:, 1].set(1)
    context = replace(
        context,
        node=replace(context.node, edge_categorical_outcome=tags),
        children=replace(
            context.children,
            visited=jnp.ones((batch, actions), jnp.bool_),
            node_payload=jnp.full((batch, actions), 8, jnp.int32),
            categorical_outcome=jnp.full((batch, actions), int(NO_OUTCOME), jnp.int8),
        ),
        leaf=replace(
            context.leaf,
            action=jnp.arange(batch, dtype=jnp.int32) % 2,
            active=jnp.ones(batch, jnp.bool_),
        ),
        active=jnp.ones(batch, jnp.bool_),
    )
    result = jax.jit(_repair_inputs)(context)
    _assert_exact(jax.jit(_scatter_reference)(context), result)
    np.testing.assert_array_equal(result.edge_payload[:, 0], 9)
    np.testing.assert_array_equal(result.edge_alpha[:, 1], context.node.edge_alpha[:, 1])
    np.testing.assert_array_equal(result.edge_payload[:, 1], context.node.edge_payload[:, 1])
