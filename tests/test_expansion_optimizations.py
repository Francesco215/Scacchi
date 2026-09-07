"""Expansion's common-slot slice writes must preserve the batched scatter rule."""

from dataclasses import replace
import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scacchi.dirichlet_mctx import base, utils
from scacchi.dirichlet_mctx.outcomes import NO_OUTCOME
from scacchi.dirichlet_mctx.tree import PosteriorUpdate, instantiate_tree_from_root

search_module = importlib.import_module("scacchi.dirichlet_mctx.search")


def _reference_set_expanded_node(array, index, value, active):
    return utils._set_node(array, jnp.broadcast_to(index, active.shape), value, active)


def _assert_same_tree(actual, expected):
    left, left_structure = jax.tree.flatten(actual)
    right, right_structure = jax.tree.flatten(expected)
    assert left_structure == right_structure
    for a, b in zip(left, right, strict=True):
        a, b = np.asarray(a), np.asarray(b)
        assert a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))


@pytest.mark.parametrize("dtype", [jnp.bool_, jnp.int8, jnp.int32, jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("payload", [(), (3,), (2, 3, 2)])
def test_expansion_slice_preserves_payloads_masks_and_scalar_index_bounds(dtype, payload):
    shape = (3, 5, *payload)
    values = jnp.arange(np.prod(shape), dtype=jnp.int32).reshape(shape)
    array = (values % 2 == 0) if dtype == jnp.bool_ else values.astype(dtype)
    value = jnp.full((3, *payload), 7, dtype=dtype)
    before = jax.jit(_reference_set_expanded_node)
    after = jax.jit(search_module._set_expanded_node)
    for index in [0, 4, -1, -5, 5, -6]:
        for mask in [[True, True, True], [True, False, True], [False, False, False]]:
            args = array, jnp.asarray(index, jnp.int32), value, jnp.asarray(mask)
            _assert_same_tree(after(*args), before(*args))


def test_expansion_slice_preserves_custom_dtype_promotion():
    array = jnp.full((2, 3, 2), 2**24 + 1, jnp.int32)
    value = jnp.asarray([[1.5, 2.5], [3.5, 4.5]], jnp.float32)
    args = array, jnp.asarray(1, jnp.int32), value, jnp.asarray([True, False])
    before = jax.jit(_reference_set_expanded_node)(*args)
    after = jax.jit(search_module._set_expanded_node)(*args)
    _assert_same_tree(after, before)
    assert int(after[1, 1, 0]) == 2**24


@pytest.mark.parametrize("stored_dtype", [jnp.float32, jnp.bfloat16])
def test_expand_preserves_every_tree_leaf_for_custom_recurrent_callback(monkeypatch, stored_dtype):
    batch_size, actions = 4, 3
    root = base.RootFnOutput(
        prior_logits=jnp.zeros((batch_size, actions)),
        value=jnp.ones((batch_size, 2), dtype=stored_dtype),
        action_values=jnp.ones((batch_size, actions, 2), dtype=stored_dtype),
        embedding={
            "counter": jnp.arange(batch_size, dtype=jnp.int32),
            "board": jnp.arange(batch_size * 12, dtype=jnp.int8).reshape(batch_size, 3, 4),
        },
        terminal_outcome=jnp.full((batch_size,), int(NO_OUTCOME), jnp.int8),
        to_play=jnp.zeros(batch_size, jnp.int32),
    )
    tree = instantiate_tree_from_root(root, 4, jnp.zeros((batch_size, actions), bool))
    tree = replace(tree, children_index=tree.children_index.at[1, 0, 1].set(1))
    simulation = base.Simulation(
        parent_index=jnp.zeros(batch_size, jnp.int32),
        action=jnp.asarray([0, 1, 2, 0], jnp.int32),
        active=jnp.asarray([True, True, False, True]),
    )

    def recurrent(params, key, action, embedding):
        del params
        value = jax.random.uniform(key, (batch_size, 2)) + 1.0
        counter = embedding["counter"] + 1
        return base.RecurrentFnOutput(
            value=value,
            action_values=jnp.broadcast_to(value[:, None, :], (batch_size, actions, 2)),
            invalid_actions=jnp.broadcast_to((action == 0)[:, None], (batch_size, actions)),
            terminal_outcome=jnp.where(action == 0, 1, int(NO_OUTCOME)).astype(jnp.int8),
            to_play=(counter % 2).astype(jnp.int32),
        ), {
            "counter": counter,
            "board": embedding["board"] + jnp.asarray(1, jnp.int8),
        }

    def run(key, candidate_tree):
        return search_module.expand(
            (), key, candidate_tree, recurrent, simulation, jnp.asarray(2, jnp.int32)
        )

    key = jax.random.PRNGKey(31)
    compiled = jax.jit(run).lower(key, tree).compile()
    actual = compiled(key, tree)
    with monkeypatch.context() as patch:
        patch.setattr(search_module, "_set_expanded_node", _reference_set_expanded_node)
        # A fresh wrapper forces tracing with the original write helper.
        expected = jax.jit(lambda k, t: run(k, t))(key, tree)
    _assert_same_tree(actual, expected)


@pytest.mark.parametrize("inactive", [False, True])
@pytest.mark.parametrize("promote_embedding", [False, True])
def test_search_recurrent_guard_preserves_tree_rng_and_callback_execution(monkeypatch, inactive, promote_embedding):
    executions = []
    batch_size, actions = 4, 2
    counter = jnp.full(batch_size, 2**24 + 1, jnp.int32) if promote_embedding else jnp.zeros(batch_size, jnp.int32)
    root = base.RootFnOutput(
        prior_logits=jnp.zeros((batch_size, actions)),
        value=jnp.ones((batch_size, 2)),
        action_values=jnp.ones((batch_size, actions, 2)),
        embedding={"counter": counter},
        terminal_outcome=jnp.full((batch_size,), 0 if inactive else int(NO_OUTCOME), jnp.int8),
        to_play=jnp.zeros(batch_size, jnp.int32),
    )

    def selector(key, tree, node):
        scores = jax.random.uniform(key, (actions,))
        return jnp.argmax(jnp.where(tree.invalid_actions[node], -jnp.inf, scores)).astype(jnp.int32)

    def repair(key, context):
        return PosteriorUpdate(
            edge_alpha=context.node.edge_alpha,
            edge_payload=context.node.edge_payload + context.active[:, None].astype(jnp.int32),
            value_alpha=context.node.value_alpha + jax.random.uniform(key, (batch_size, 2)),
        )

    def run(key, weights):
        # Capture dynamic outer tracers just as a neural evaluator captures its
        # model weights; shape inference must not concretize those values.
        def recurrent(params, recurrent_key, action, embedding):
            del params
            jax.debug.callback(lambda k: executions.append(np.asarray(k).copy()), recurrent_key, ordered=True)
            depth = embedding["counter"] + 1
            child_counter = depth.astype(jnp.float32) if promote_embedding else depth
            terminal = depth >= 2 + jnp.arange(batch_size) % 2
            value = weights + jax.random.uniform(recurrent_key, (batch_size, 2))
            return base.RecurrentFnOutput(
                value=value,
                action_values=jnp.broadcast_to(value[:, None, :], (batch_size, actions, 2)),
                invalid_actions=jnp.broadcast_to(terminal[:, None], (batch_size, actions)),
                terminal_outcome=jnp.where(terminal, action % 2, int(NO_OUTCOME)).astype(jnp.int8),
                to_play=(depth % 2).astype(jnp.int32),
            ), {"counter": child_counter}

        return search_module.search(
            (), key, root=root, recurrent_fn=recurrent,
            action_selection_fn=selector, posterior_update=repair,
            num_simulations=12, max_depth=4,
        )

    key, weights = jax.random.PRNGKey(47), jnp.full((batch_size, 2), 1.5)
    actual = jax.block_until_ready(jax.jit(run)(key, weights))
    jax.effects_barrier()
    actual_executions = list(executions)
    executions.clear()
    with monkeypatch.context() as patch:
        patch.setattr(search_module, "_recurrent_dtypes_fit_storage", lambda *_: False)
        expected = jax.block_until_ready(jax.jit(lambda k, w: run(k, w))(key, weights))
        jax.effects_barrier()
    _assert_same_tree(actual, expected)
    assert len(actual_executions) == len(executions)
    for before_key, after_key in zip(actual_executions, executions, strict=True):
        np.testing.assert_array_equal(before_key, after_key)
    assert (len(actual_executions) == 0) == inactive


def test_search_shape_inference_preserves_static_callback_parameters():
    root = base.RootFnOutput(
        prior_logits=jnp.zeros((1, 1)),
        value=jnp.ones((1, 2)),
        action_values=jnp.ones((1, 1, 2)),
        embedding=jnp.zeros(1, jnp.int32),
        terminal_outcome=jnp.full(1, int(NO_OUTCOME), jnp.int8),
        to_play=jnp.zeros(1, jnp.int32),
    )

    def recurrent(params, key, action, embedding):
        del key, action
        # Both the Python branch and non-array string worked with closed-over
        # params before introducing recurrent output shape inference.
        if params["mode"] and params["tag"] == "custom":
            prior = 7.0
        else:
            prior = 3.0
        return base.RecurrentFnOutput(
            value=jnp.full((1, 2), prior),
            action_values=jnp.ones((1, 1, 2)),
            invalid_actions=jnp.zeros((1, 1), bool),
            terminal_outcome=jnp.full(1, int(NO_OUTCOME), jnp.int8),
            to_play=jnp.ones(1, jnp.int32),
        ), embedding + 1

    def repair(key, context):
        del key
        return PosteriorUpdate(
            edge_alpha=context.node.edge_alpha,
            edge_payload=context.node.edge_payload,
            value_alpha=context.node.value_alpha,
        )

    tree = jax.jit(lambda key: search_module.search(
        {"mode": True, "tag": "custom"}, key, root=root,
        recurrent_fn=recurrent,
        action_selection_fn=lambda *_: jnp.asarray(0, jnp.int32),
        posterior_update=repair, num_simulations=1,
    ))(jax.random.PRNGKey(71))
    np.testing.assert_array_equal(tree.node_value_priors[:, 1], [[7.0, 7.0]])
    np.testing.assert_array_equal(tree.embeddings[:, 1], [1])


def test_search_shape_inference_uses_stored_embedding_strength():
    weak_embedding = jnp.broadcast_to(jnp.asarray(1.0), (1,))
    assert weak_embedding.weak_type
    root = base.RootFnOutput(
        prior_logits=jnp.zeros((1, 1)),
        value=jnp.ones((1, 2)),
        action_values=jnp.ones((1, 1, 2)),
        embedding=weak_embedding,
        terminal_outcome=jnp.full(1, int(NO_OUTCOME), jnp.int8),
        to_play=jnp.zeros(1, jnp.int32),
    )

    def recurrent(params, key, action, embedding):
        del params, key, action
        # Weak f32 plus strong bf16 infers bf16, but the tree allocates a
        # strong f32 embedding: the actual recurrent result must remain f32.
        value = embedding[:, None] + jnp.ones((1, 2), jnp.bfloat16)
        return base.RecurrentFnOutput(
            value=value,
            action_values=value[:, None, :],
            invalid_actions=jnp.zeros((1, 1), bool),
            terminal_outcome=jnp.full(1, int(NO_OUTCOME), jnp.int8),
            to_play=jnp.ones(1, jnp.int32),
        ), embedding + jnp.asarray(1.0, jnp.bfloat16)

    def repair(key, context):
        del key
        return PosteriorUpdate(
            edge_alpha=context.node.edge_alpha,
            edge_payload=context.node.edge_payload,
            value_alpha=context.node.value_alpha,
        )

    tree = jax.jit(lambda key: search_module.search(
        (), key, root=root, recurrent_fn=recurrent,
        action_selection_fn=lambda *_: jnp.asarray(0, jnp.int32),
        posterior_update=repair, num_simulations=1,
    ))(jax.random.PRNGKey(73))
    assert tree.node_value_priors.dtype == jnp.float32
    np.testing.assert_array_equal(tree.node_value_priors[:, 1], [[2.0, 2.0]])
    np.testing.assert_array_equal(tree.embeddings[:, 1], [2.0])
