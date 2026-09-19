"""Preserve selector callbacks and certificate publication across fast paths."""

from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scacchi.dirichlet_mctx import action_selection, base, utils
from scacchi.dirichlet_mctx.outcomes import NO_OUTCOME
from scacchi.dirichlet_mctx.search import simulate
from scacchi.dirichlet_mctx.tree import Tree, instantiate_tree_from_root


def _tree(batch_size=1, num_actions=4, num_outcomes=2):
    root = base.RootFnOutput(
        prior_logits=jnp.zeros((batch_size, num_actions)),
        value=jnp.ones((batch_size, num_outcomes)),
        action_values=jnp.ones((batch_size, num_actions, num_outcomes)),
        embedding=jnp.zeros((batch_size,), dtype=jnp.int32),
        terminal_outcome=jnp.full((batch_size,), int(NO_OUTCOME), dtype=jnp.int8),
        to_play=jnp.zeros((batch_size,), dtype=jnp.int32),
    )
    return instantiate_tree_from_root(
        root, num_simulations=8,
        root_invalid_actions=jnp.zeros((batch_size, num_actions), dtype=bool),
    )


def _assert_same_tree(actual, expected):
    for actual_leaf, expected_leaf in zip(
        jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True
    ):
        np.testing.assert_array_equal(actual_leaf, expected_leaf)


def _sampled_publication_reference(key, tree, node_indices, active):
    """Original sampled-distance rule, with host-side publication for clarity."""
    batch = np.arange(len(node_indices))
    indices = np.asarray(node_indices)
    outcomes = np.asarray(tree.edge_categorical_outcome)[batch, indices]
    distances = np.asarray(tree.edge_payload)[batch, indices]
    invalid = np.asarray(tree.invalid_actions)[batch, indices]
    num_outcomes = tree.node_value_alpha.shape[-1]
    candidates = np.full(len(indices), int(NO_OUTCOME), dtype=np.int8)
    for lane in batch:
        legal_outcomes = outcomes[lane, ~invalid[lane]]
        if np.any(legal_outcomes == num_outcomes - 1):
            candidates[lane] = num_outcomes - 1
        elif len(legal_outcomes) and np.all(legal_outcomes != int(NO_OUTCOME)):
            candidates[lane] = int(num_outcomes == 3 and np.any(legal_outcomes == 1))
    actions = np.asarray(action_selection.categorical_action(
        key, jnp.asarray(candidates), jnp.asarray(outcomes),
        jnp.asarray(distances), jnp.asarray(invalid), num_outcomes=num_outcomes,
    ))
    node_outcomes = np.asarray(tree.node_categorical_outcome).copy()
    node_payload = np.asarray(tree.node_payload).copy()
    edge_outcomes = np.asarray(tree.edge_categorical_outcome).copy()
    edge_payload = np.asarray(tree.edge_payload).copy()
    for lane, index in enumerate(indices):
        outcome = int(candidates[lane])
        if not active[lane] or node_outcomes[lane, index] != int(NO_OUTCOME) or outcome == int(NO_OUTCOME):
            continue
        distance = int(distances[lane, actions[lane]])
        parent = int(tree.parents[lane, index])
        incoming = np.flatnonzero(np.asarray(tree.children_index[lane, max(parent, 0)]) == index)
        if parent != Tree.NO_PARENT and len(incoming):
            action = int(incoming[0])
            if edge_outcomes[lane, parent, action] == int(NO_OUTCOME):
                node_payload[lane, parent] += 1 + node_payload[lane, index] - edge_payload[lane, parent, action]
                flip = tree.node_to_play[lane, index] != tree.node_to_play[lane, parent]
                edge_outcomes[lane, parent, action] = num_outcomes - 1 - outcome if flip else outcome
                edge_payload[lane, parent, action] = distance + 1
        node_outcomes[lane, index] = outcome
        node_payload[lane, index] = distance
    return replace(
        tree, node_categorical_outcome=jnp.asarray(node_outcomes),
        node_payload=jnp.asarray(node_payload),
        edge_categorical_outcome=jnp.asarray(edge_outcomes),
        edge_payload=jnp.asarray(edge_payload),
    )


def test_custom_selector_receives_only_current_nodes_search_mask():
    tree = _tree(num_actions=3)
    tree = replace(
        tree,
        children_index=tree.children_index.at[0, 0, 1].set(1),
        invalid_actions=tree.invalid_actions.at[0, 1].set(jnp.asarray([False, False, True])),
        edge_categorical_outcome=tree.edge_categorical_outcome.at[0, 0, 0].set(0).at[0, 1, 1].set(0),
    )

    def custom_selector(key, candidate_tree, node_index):
        del key
        expected = tree.invalid_actions[0].at[node_index].set(~tree.searchable_actions[0, node_index])
        action = action_selection.masked_argmax(
            jnp.asarray([2.0, 1.0, 0.0]), candidate_tree.invalid_actions[node_index]
        )
        # Inspect other rows too: the callback must receive a whole valid tree.
        return jnp.where(jnp.all(candidate_tree.invalid_actions == expected), action, 2)

    result = jax.jit(lambda key: simulate(key, tree, custom_selector, 3))(
        jax.random.split(jax.random.PRNGKey(4), 1)
    )
    np.testing.assert_array_equal(result.parent_index, [1])
    np.testing.assert_array_equal(result.action, [0])
    np.testing.assert_array_equal(result.active, [True])


def test_builtin_selection_matches_wrapped_callback_random_stream():
    tree = _tree(batch_size=16)
    tree = replace(
        tree,
        children_index=tree.children_index.at[:, 0, 1].set(1).at[:, 0, 2].set(2),
        invalid_actions=tree.invalid_actions.at[:, 1:3].set(False),
        edge_categorical_outcome=tree.edge_categorical_outcome.at[:, 0, 0].set(0).at[:, 1, 1].set(0),
    )

    def wrapped(key, candidate_tree, node_index):
        return action_selection.thompson_action_selection(key, candidate_tree, node_index)

    keys = jax.random.split(jax.random.PRNGKey(19), 16)
    builtin = jax.jit(lambda k: simulate(k, tree, action_selection.thompson_action_selection, 3))(keys)
    generic = jax.jit(lambda k: simulate(k, tree, wrapped, 3))(keys)
    for actual, expected in zip(jax.tree.leaves(builtin), jax.tree.leaves(generic), strict=True):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("seed", [0, 13, 37])
def test_binary_publication_matches_sampled_reference(seed):
    tree = _tree(batch_size=12)
    batch = jnp.arange(12)
    nodes = jnp.ones(12, dtype=jnp.int32).at[0].set(0)
    outcomes = jnp.asarray([
        [1, -1, 1, 0], [1, 1, 0, -1], [0, 0, 0, -1],
        [-1, 0, -1, -1], [1, 0, 1, 0], [1, 1, 0, -1],
        [1, 1, 0, -1], [1, 0, 0, 0], [0, 0, 0, -1],
        [1, 1, 0, -1], [1, 1, 0, -1], [1, 1, 0, -1],
    ], dtype=jnp.int8)
    distances = jnp.asarray([
        [4, 0, 2, 7], [3, 3, 7, 0], [2, 5, 5, 0],
        [0, 2, 0, 0], [2, 3, 4, 5], [3, 3, 7, 0],
        [3, 3, 7, 0], [1, 2, 5, 7], [2, 5, 5, 0],
        [3, 3, 7, 0], [3, 3, 7, 0], [3, 3, 7, 0],
    ], dtype=jnp.int32)
    invalid = jnp.zeros((12, 4), dtype=bool).at[2, 3].set(True).at[4].set(True).at[7, 0].set(True).at[8, 3].set(True)
    tree = replace(
        tree,
        parents=tree.parents.at[1:11, 1].set(0),
        children_index=tree.children_index.at[1:10, 0, 3].set(1),
        node_to_play=tree.node_to_play.at[:, 1].set(batch % 2),
        node_payload=tree.node_payload.at[:, 0].set(17).at[1:, 1].set(5),
        node_categorical_outcome=tree.node_categorical_outcome.at[6, 1].set(0),
        invalid_actions=tree.invalid_actions.at[batch, nodes].set(invalid),
        edge_categorical_outcome=tree.edge_categorical_outcome.at[batch, nodes].set(outcomes).at[9, 0, 3].set(0),
        edge_payload=tree.edge_payload.at[1:, 0, 3].set(3).at[batch, nodes].set(distances),
    )
    active = jnp.ones(12, dtype=bool).at[5].set(False)
    key = jax.random.PRNGKey(seed)
    expected = _sampled_publication_reference(key, tree, nodes, active)
    actual = jax.jit(utils._categorize_node_and_publish)(key, tree, nodes, active)
    _assert_same_tree(actual, expected)


def test_draw_publication_preserves_sampled_distance_and_key():
    tree = _tree(num_outcomes=3)
    tree = replace(
        tree,
        edge_categorical_outcome=tree.edge_categorical_outcome.at[0, 0].set(jnp.asarray([1, 0, 1, 0], dtype=jnp.int8)),
        edge_payload=tree.edge_payload.at[0, 0].set(jnp.asarray([2, 8, 7, 5], dtype=jnp.int32)),
    )
    nodes, active = jnp.asarray([0]), jnp.asarray([True])
    keys = jax.random.split(jax.random.PRNGKey(7), 32)
    actual = jax.jit(jax.vmap(lambda key: utils._categorize_node_and_publish(key, tree, nodes, active)))(keys)
    chosen = action_selection.categorical_action
    expected_actions = jax.vmap(lambda key: chosen(
        key, jnp.asarray([1], dtype=jnp.int8), tree.edge_categorical_outcome[:, 0],
        tree.edge_payload[:, 0], tree.invalid_actions[:, 0], num_outcomes=3,
    ))(keys)
    expected_distances = tree.edge_payload[0, 0, expected_actions[:, 0]]
    np.testing.assert_array_equal(actual.node_payload[:, 0, 0], expected_distances)
    assert set(np.asarray(expected_distances).tolist()) == {2, 7}


def test_huge_binary_tree_retains_sampled_distance(monkeypatch):
    tree = _tree()
    # Trace an abstract large capacity without allocating the parent table.
    tree = replace(tree, parents=jax.ShapeDtypeStruct((1, 2**24 + 1), jnp.int32))
    calls = []

    def sampled_distance(key, *args, **kwargs):
        calls.append(True)
        return jnp.zeros((1,), dtype=jnp.int32)

    monkeypatch.setattr(action_selection, "categorical_action", sampled_distance)
    jax.eval_shape(
        utils._categorize_node_and_publish, jax.random.PRNGKey(2), tree,
        jnp.asarray([0]), jnp.asarray([True]),
    )
    assert calls == [True]
