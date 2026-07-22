"""Internal helpers shared by Dirichlet MCTS search operations."""

from __future__ import annotations

from dataclasses import replace

import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int32, Shaped

from . import action_selection, base
from .categorical import NO_OUTCOME
from .tree import ChildrenView, NodeView, PosteriorUpdate, Tree


def _set_scalar_node(array: Shaped[Array, "batch node"], node_index: Int32[Array, "batch"], value: Shaped[Array, "batch"], active: Bool[Array, "batch"]) -> Shaped[Array, "batch node"]:
    batch = jnp.arange(array.shape[0])
    old = array[batch, node_index]
    return array.at[batch, node_index].set(jnp.where(active, value, old))


def _set_action_node(array: Shaped[Array, "batch node action"], node_index: Int32[Array, "batch"], value: Shaped[Array, "batch action"], active: Bool[Array, "batch"]) -> Shaped[Array, "batch node action"]:
    batch = jnp.arange(array.shape[0])
    old = array[batch, node_index]
    return array.at[batch, node_index].set(jnp.where(active[:, None], value, old))


def _set_outcome_node(array: Shaped[Array, "batch node outcome"], node_index: Int32[Array, "batch"], value: Shaped[Array, "batch outcome"], active: Bool[Array, "batch"]) -> Shaped[Array, "batch node outcome"]:
    batch = jnp.arange(array.shape[0])
    old = array[batch, node_index]
    return array.at[batch, node_index].set(jnp.where(active[:, None], value, old))


def _set_action_outcome_node(array: Shaped[Array, "batch node action outcome"], node_index: Int32[Array, "batch"], value: Shaped[Array, "batch action outcome"], active: Bool[Array, "batch"]) -> Shaped[Array, "batch node action outcome"]:
    batch = jnp.arange(array.shape[0])
    old = array[batch, node_index]
    return array.at[batch, node_index].set(jnp.where(active[:, None, None], value, old))


def _set_embedding_node(array: Shaped[Array, "batch node *embedding_axes"], node_index: Int32[Array, "batch"], value: Shaped[Array, "batch *embedding_axes"], active: Bool[Array, "batch"]) -> Shaped[Array, "batch node *embedding_axes"]:
    batch = jnp.arange(array.shape[0])
    old = array[batch, node_index]
    mask = active.reshape(active.shape + (1,) * (old.ndim - 1))
    return array.at[batch, node_index].set(jnp.where(mask, value, old))


def _set_edge(array: Shaped[Array, "batch node action"], node_index: Int32[Array, "batch"], action: Int32[Array, "batch"], value: Shaped[Array, "batch"], active: Bool[Array, "batch"]) -> Shaped[Array, "batch node action"]:
    batch = jnp.arange(array.shape[0])
    old = array[batch, node_index, action]
    return array.at[batch, node_index, action].set(jnp.where(active, value, old))


def _gather_node(tree: Tree, node_index: Int32[Array, "batch"]) -> NodeView:
    batch = jnp.arange(tree.parents.shape[0])

    def gather_node_embedding(table: Shaped[Array, "batch node *embedding_axes"]) -> Shaped[Array, "batch *embedding_axes"]:
        return table[batch, node_index]

    return NodeView(index=node_index, embedding=jax.tree.map(gather_node_embedding, tree.embeddings), value_prior=tree.node_value_priors[batch, node_index], value_alpha=tree.node_value_alpha[batch, node_index], node_payload=tree.node_payload[batch, node_index], edge_alpha=tree.edge_alpha[batch, node_index], edge_payload=tree.edge_payload[batch, node_index], edge_categorical_outcome=tree.edge_categorical_outcome[batch, node_index], to_play=tree.node_to_play[batch, node_index], invalid_actions=tree.invalid_actions[batch, node_index])


def _gather_children(tree: Tree, node_index: Int32[Array, "batch"]) -> ChildrenView:
    batch = jnp.arange(tree.parents.shape[0])
    child_index = tree.children_index[batch, node_index]
    visited = child_index != Tree.UNVISITED
    safe_child = jnp.where(visited, child_index, Tree.ROOT_INDEX)
    batch_actions = batch[:, None]
    return ChildrenView(index=child_index, visited=visited, embedding_table=tree.embeddings, value_prior=tree.node_value_priors[batch_actions, safe_child], value_alpha=tree.node_value_alpha[batch_actions, safe_child], node_payload=tree.node_payload[batch_actions, safe_child], categorical_outcome=tree.node_categorical_outcome[batch_actions, safe_child], to_play=tree.node_to_play[batch_actions, safe_child])


def _set_node_update(tree: Tree, node_index: Int32[Array, "batch"], update: PosteriorUpdate, active: Bool[Array, "batch"]) -> Tree:
    """Store a repair and increment unresolved node support by count delta."""
    batch = jnp.arange(tree.parents.shape[0])
    edge_outcome = tree.edge_categorical_outcome[batch, node_index]
    unresolved = edge_outcome == int(NO_OUTCOME)
    legal = ~tree.invalid_actions[batch, node_index]
    old_payload = tree.edge_payload[batch, node_index]
    new_payload = jnp.where(unresolved, update.edge_payload, old_payload)
    count_delta = jnp.sum(jnp.where(legal, new_payload - old_payload, 0), axis=-1)
    new_node_payload = tree.node_payload[batch, node_index] + count_delta
    edge_alpha = jnp.where(unresolved[..., None], update.edge_alpha, tree.edge_alpha[batch, node_index])
    return replace(tree, node_payload=_set_scalar_node(tree.node_payload, node_index, new_node_payload, active), edge_alpha=_set_action_outcome_node(tree.edge_alpha, node_index, edge_alpha, active), edge_payload=_set_action_node(tree.edge_payload, node_index, new_payload, active), node_value_alpha=_set_outcome_node(tree.node_value_alpha, node_index, update.value_alpha, active))


def _categorize_node_and_publish(rng_key: base.PRNGKey, tree: Tree, node_index: Int32[Array, "batch"], active: Bool[Array, "batch"]) -> Tree:
    """Publish a node certificate after preserving its final parent support."""
    batch = jnp.arange(tree.parents.shape[0])
    edge_outcome = tree.edge_categorical_outcome[batch, node_index]
    edge_distance = tree.edge_payload[batch, node_index]
    invalid_actions = tree.invalid_actions[batch, node_index]
    legal = ~invalid_actions
    known = legal & (edge_outcome != int(NO_OUTCOME))
    has_legal = jnp.any(legal, axis=-1)
    all_categorical = has_legal & jnp.all(~legal | known, axis=-1)
    num_outcomes = tree.node_value_alpha.shape[-1]
    win_index = num_outcomes - 1
    has_win = jnp.any(known & (edge_outcome == win_index), axis=-1)
    has_draw = jnp.any(known & (edge_outcome == 1), axis=-1) if num_outcomes == 3 else jnp.zeros_like(has_win)
    candidate_outcome = jnp.where(has_win, jnp.asarray(win_index, dtype=jnp.int8), jnp.where(all_categorical & has_draw, jnp.asarray(1, dtype=jnp.int8), jnp.where(all_categorical, jnp.asarray(0, dtype=jnp.int8), jnp.asarray(int(NO_OUTCOME), dtype=jnp.int8))))
    candidate_action = action_selection.categorical_action(rng_key, candidate_outcome, edge_outcome, edge_distance, invalid_actions, num_outcomes=num_outcomes)
    candidate_distance = edge_distance[batch, candidate_action]
    old_outcome = tree.node_categorical_outcome[batch, node_index]
    old_support = tree.node_payload[batch, node_index]
    publish_node = active & (old_outcome == int(NO_OUTCOME)) & (candidate_outcome != int(NO_OUTCOME))
    parent = tree.parents[batch, node_index]
    has_parent = parent != Tree.NO_PARENT
    safe_parent = jnp.where(has_parent, parent, Tree.ROOT_INDEX)
    incoming = tree.children_index[batch, safe_parent] == node_index[:, None]
    has_incoming = jnp.any(incoming, axis=-1)
    incoming_action = jnp.argmax(incoming, axis=-1).astype(jnp.int32)
    old_edge_outcome = tree.edge_categorical_outcome[batch, safe_parent, incoming_action]
    publish_edge = publish_node & has_parent & has_incoming & (old_edge_outcome == int(NO_OUTCOME))
    old_edge_count = tree.edge_payload[batch, safe_parent, incoming_action]
    new_parent_support = tree.node_payload[batch, safe_parent] + 1 + old_support - old_edge_count
    node_player = tree.node_to_play[batch, node_index]
    parent_player = tree.node_to_play[batch, safe_parent]
    aligned_outcome = jnp.where(node_player == parent_player, candidate_outcome, (num_outcomes - 1 - candidate_outcome).astype(jnp.int8))
    node_payload = _set_scalar_node(tree.node_payload, safe_parent, new_parent_support, publish_edge)
    node_payload = _set_scalar_node(node_payload, node_index, candidate_distance, publish_node)
    return replace(tree, node_payload=node_payload, node_categorical_outcome=_set_scalar_node(tree.node_categorical_outcome, node_index, candidate_outcome, publish_node), edge_categorical_outcome=_set_edge(tree.edge_categorical_outcome, safe_parent, incoming_action, aligned_outcome, publish_edge), edge_payload=_set_edge(tree.edge_payload, safe_parent, incoming_action, candidate_distance + jnp.asarray(1, dtype=jnp.int32), publish_edge))
