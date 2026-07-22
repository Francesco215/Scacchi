"""Batched simulate-expand-repair search for Dirichlet posteriors."""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Int, Shaped

from . import action_selection, base
from .categorical import NO_OUTCOME
from .tree import (
    ChildrenView,
    LeafView,
    NodeView,
    PosteriorUpdate,
    PosteriorUpdateContext,
    Tree,
)


class Simulation(NamedTuple):
    parent_index: Int[Array, "batch"]
    action: Int[Array, "batch"]
    active: Bool[Array, "batch"]


class _SimulationState(NamedTuple):
    rng_key: base.PRNGKey
    node_index: Int[Array, ""]
    action: Int[Array, ""]
    next_node_index: Int[Array, ""]
    depth: Int[Array, ""]
    is_continuing: Bool[Array, ""]


type _BackwardState = tuple[base.PRNGKey, Tree, Int[Array, "batch"], Bool[Array, "batch"], Bool[Array, "batch"]]
type _SearchState = tuple[base.PRNGKey, Tree]


def _set_node(array: Shaped[Array, "batch node *value_axes"], node_index: Int[Array, ""] | Int[Array, "batch"], value: Shaped[Array, "batch *value_axes"], active: Bool[Array, "batch"]) -> Shaped[Array, "batch node *value_axes"]:
    batch = jnp.arange(array.shape[0])
    old = array[batch, node_index]
    mask = active.reshape(active.shape + (1,) * (old.ndim - 1))
    return array.at[batch, node_index].set(jnp.where(mask, value, old))


def _set_edge(array: Shaped[Array, "batch node action *value_axes"], node_index: Int[Array, "batch"], action: Int[Array, "batch"], value: Shaped[Array, ""] | Shaped[Array, "batch *value_axes"], active: Bool[Array, "batch"]) -> Shaped[Array, "batch node action *value_axes"]:
    batch = jnp.arange(array.shape[0])
    old = array[batch, node_index, action]
    mask = active.reshape(active.shape + (1,) * (old.ndim - 1))
    return array.at[batch, node_index, action].set(jnp.where(mask, value, old))


def instantiate_tree_from_root(root: base.RootFnOutput, num_simulations: int, root_invalid_actions: Bool[Array, "batch action"]) -> Tree:
    """Allocate compact fixed-capacity storage and initialize the root."""

    chex.assert_rank(root.prior_logits, 2)
    batch_size, num_actions = root.prior_logits.shape
    num_outcomes = root.action_values.shape[-1]
    chex.assert_shape(root.value, (batch_size, num_outcomes))
    chex.assert_shape(root.action_values, (batch_size, num_actions, num_outcomes))
    chex.assert_shape(root.terminal_outcome, (batch_size,))
    chex.assert_shape(root_invalid_actions, (batch_size, num_actions))
    num_nodes = num_simulations + 1
    batch_node = (batch_size, num_nodes)
    batch_node_action = (batch_size, num_nodes, num_actions)

    def allocate_embedding(value: Shaped[Array, "batch *embedding_axes"]) -> Shaped[Array, "batch node *embedding_axes"]:
        table = jnp.zeros((batch_size, num_nodes, *value.shape[1:]), dtype=value.dtype)
        return table.at[:, Tree.ROOT_INDEX].set(value)

    edge_alpha = jnp.ones((*batch_node_action, num_outcomes), dtype=root.action_values.dtype).at[:, Tree.ROOT_INDEX].set(root.action_values)
    node_value_alpha = jnp.ones((*batch_node, num_outcomes), dtype=root.value.dtype).at[:, Tree.ROOT_INDEX].set(root.value)
    return Tree(
        parents=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
        children_index=jnp.full(batch_node_action, Tree.UNVISITED, dtype=jnp.int32),
        node_to_play=jnp.zeros(batch_node, dtype=root.to_play.dtype).at[:, Tree.ROOT_INDEX].set(root.to_play),
        node_categorical_outcome=jnp.full(batch_node, int(NO_OUTCOME), dtype=jnp.int8).at[:, Tree.ROOT_INDEX].set(root.terminal_outcome.astype(jnp.int8)),
        # Zero means unresolved support zero or categorical terminal distance
        # zero; the outcome tag is the required discriminant.
        node_payload=jnp.zeros(batch_node, dtype=jnp.int32),
        edge_categorical_outcome=jnp.full(batch_node_action, int(NO_OUTCOME), dtype=jnp.int8),
        edge_payload=jnp.zeros(batch_node_action, dtype=jnp.int32),
        node_value_priors=node_value_alpha,
        node_value_alpha=node_value_alpha,
        edge_alpha=edge_alpha,
        invalid_actions=jnp.ones(batch_node_action, dtype=bool).at[:, Tree.ROOT_INDEX].set(root_invalid_actions),
        embeddings=jax.tree.map(allocate_embedding, root.embedding),
    )


@functools.partial(jax.vmap, in_axes=(0, 0, None, None), out_axes=0)
def simulate(rng_key: base.BatchedPRNGKey, tree: Tree, action_selection_fn: base.ActionSelectionFn, max_depth: int) -> Simulation:
    """Traverse one unbatched tree to an unvisited or cutoff edge."""

    root_index = jnp.asarray(Tree.ROOT_INDEX, dtype=jnp.int32)
    root_searchable_actions = tree.searchable_actions[root_index]
    root_active = (tree.node_categorical_outcome[root_index] == int(NO_OUTCOME)) & jnp.any(root_searchable_actions)

    def body_fn(state: _SimulationState) -> _SimulationState:
        next_key, selection_key = jax.random.split(state.rng_key)
        node_index = state.next_node_index

        searchable_actions = tree.searchable_actions[node_index]
        selection_tree = replace(tree, invalid_actions=tree.invalid_actions.at[node_index].set(~searchable_actions))
        action = action_selection_fn(selection_key, selection_tree, node_index)
        child_index = tree.children_index[node_index, action]

        visited = child_index != Tree.UNVISITED
        safe_child = jnp.where(visited, child_index, Tree.ROOT_INDEX)
        depth = state.depth + 1
        child_has_searchable_action = jnp.any(tree.searchable_actions[safe_child])
        child_selectable = (tree.node_categorical_outcome[safe_child] == int(NO_OUTCOME)) & child_has_searchable_action
        continuing = visited & child_selectable & (depth < max_depth)

        return _SimulationState(
            rng_key=next_key,
            node_index=node_index,
            action=action,
            next_node_index=child_index,
            depth=depth,
            is_continuing=continuing,
        )

    initial = _SimulationState(
        rng_key=rng_key,
        node_index=root_index,
        action=jnp.asarray(0, dtype=jnp.int32),
        next_node_index=root_index,
        depth=jnp.asarray(0, dtype=jnp.int32),
        is_continuing=root_active,
    )
    end = jax.lax.while_loop(lambda state: state.is_continuing, body_fn, initial)

    return Simulation(parent_index=end.node_index, action=end.action, active=root_active)


def expand(params: base.Params, rng_key: base.PRNGKey, tree: Tree, recurrent_fn: base.RecurrentFn, simulation: Simulation, new_node_index: Int[Array, ""]) -> tuple[Tree, base.RecurrentFnOutput]:
    """Evaluate selected edges and initialize genuinely new child nodes."""

    batch = jnp.arange(tree.parents.shape[0])
    parent_index = jnp.where(simulation.active, simulation.parent_index, 0)
    action = jnp.where(simulation.active, simulation.action, 0)
    embedding = jax.tree.map(lambda value: value[batch, parent_index],tree.embeddings)
    step, child_embedding = recurrent_fn(params, rng_key, action, embedding) #TODO: remove the params argument from recurrent_fn. just use nnx (low-priority)
    child_index = tree.children_index[batch, parent_index, action]
    initialize = simulation.active & (child_index == Tree.UNVISITED)
    child_outcome = step.terminal_outcome.astype(jnp.int8)
    parent_player = tree.node_to_play[batch, parent_index]
    num_outcomes = tree.node_value_alpha.shape[-1]
    aligned_outcome = jnp.where(step.to_play == parent_player, child_outcome, (num_outcomes - 1 - child_outcome).astype(jnp.int8))
    publish_terminal = (simulation.active & (child_outcome != int(NO_OUTCOME)))

    # A terminal edge's unresolved count is consumed before its payload is
    # overwritten by distance one. The parent's support keeps that final unit.
    old_edge_count = tree.edge_payload[batch, parent_index, action]
    old_parent_support = tree.node_payload[batch, parent_index]
    parent_support = old_parent_support + (1 - old_edge_count)

    tree = replace(
        tree,
        parents=_set_node(tree.parents, new_node_index, parent_index, initialize),
        children_index=_set_edge(tree.children_index, parent_index, action, new_node_index, initialize),
        node_to_play=_set_node(tree.node_to_play, new_node_index, step.to_play, initialize),
        node_categorical_outcome=_set_node(tree.node_categorical_outcome, new_node_index, child_outcome, initialize),
        node_payload=_set_node(tree.node_payload, parent_index, parent_support, publish_terminal),
        edge_categorical_outcome=_set_edge(tree.edge_categorical_outcome, parent_index, action, aligned_outcome, publish_terminal),
        edge_payload=_set_edge(tree.edge_payload, parent_index, action, jnp.ones_like(aligned_outcome, dtype=jnp.int32), publish_terminal),
        node_value_priors=_set_node(tree.node_value_priors, new_node_index, step.value, initialize),
        node_value_alpha=_set_node(tree.node_value_alpha, new_node_index, step.value, initialize),
        edge_alpha=_set_node(tree.edge_alpha, new_node_index, step.action_values, initialize),
        invalid_actions=_set_node(tree.invalid_actions, new_node_index, step.invalid_actions, initialize),
        embeddings=jax.tree.map(lambda table, value: _set_node(table, new_node_index, value, initialize), tree.embeddings, child_embedding),
    )
    return tree, step


def _gather_node(tree: Tree, node_index: Int[Array, "batch"]) -> NodeView:
    batch = jnp.arange(tree.parents.shape[0])
    return NodeView(
        index=node_index,
        embedding=jax.tree.map(lambda value: value[batch, node_index], tree.embeddings),
        value_prior=tree.node_value_priors[batch, node_index],
        value_alpha=tree.node_value_alpha[batch, node_index],
        node_payload=tree.node_payload[batch, node_index],
        edge_alpha=tree.edge_alpha[batch, node_index],
        edge_payload=tree.edge_payload[batch, node_index],
        edge_categorical_outcome=tree.edge_categorical_outcome[batch, node_index],
        to_play=tree.node_to_play[batch, node_index],
        invalid_actions=tree.invalid_actions[batch, node_index],
    )


def _gather_children(tree: Tree, node_index: Int[Array, "batch"]) -> ChildrenView:
    batch = jnp.arange(tree.parents.shape[0])
    child_index = tree.children_index[batch, node_index]
    visited = child_index != Tree.UNVISITED
    safe_child = jnp.where(visited, child_index, Tree.ROOT_INDEX)
    batch_actions = batch[:, None]
    return ChildrenView(
        index=child_index,
        visited=visited,
        embedding_table=tree.embeddings,
        value_prior=tree.node_value_priors[batch_actions, safe_child],
        value_alpha=tree.node_value_alpha[batch_actions, safe_child],
        node_payload=tree.node_payload[batch_actions, safe_child],
        categorical_outcome=tree.node_categorical_outcome[batch_actions, safe_child],
        to_play=tree.node_to_play[batch_actions, safe_child],
    )


def _set_node_update(tree: Tree, node_index: Int[Array, "batch"], update: PosteriorUpdate, active: Bool[Array, "batch"]) -> Tree:
    """Store a repair and increment unresolved node support by count delta."""

    batch = jnp.arange(tree.parents.shape[0])
    edge_outcome = tree.edge_categorical_outcome[batch, node_index]
    unresolved = edge_outcome == int(NO_OUTCOME)
    legal = ~tree.invalid_actions[batch, node_index]
    old_payload = tree.edge_payload[batch, node_index]
    new_payload = jnp.where(unresolved, update.edge_payload, old_payload)
    count_delta = jnp.sum(jnp.where(legal, new_payload - old_payload, 0), axis=-1)
    old_node_payload = tree.node_payload[batch, node_index]
    new_node_payload = old_node_payload + count_delta
    edge_alpha = jnp.where(unresolved[..., None], update.edge_alpha, tree.edge_alpha[batch, node_index])
    return replace(
        tree,
        node_payload=_set_node(tree.node_payload, node_index, new_node_payload, active),
        edge_alpha=_set_node(tree.edge_alpha, node_index, edge_alpha, active),
        edge_payload=_set_node(tree.edge_payload, node_index, new_payload, active),
        node_value_alpha=_set_node(tree.node_value_alpha, node_index, update.value_alpha, active),
    )


def _categorize_node_and_publish(rng_key: base.PRNGKey, tree: Tree, node_index: Int[Array, "batch"], active: Bool[Array, "batch"]) -> Tree:
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
    if num_outcomes == 3:
        has_draw = jnp.any(known & (edge_outcome == 1), axis=-1)
    else:
        has_draw = jnp.zeros_like(has_win)

    candidate_outcome = jnp.where(has_win, jnp.asarray(win_index, dtype=jnp.int8), jnp.where(all_categorical & has_draw, jnp.asarray(1, dtype=jnp.int8), jnp.where(all_categorical, jnp.asarray(0, dtype=jnp.int8), jnp.asarray(int(NO_OUTCOME), dtype=jnp.int8))))
    candidate_action = action_selection.categorical_action(rng_key, candidate_outcome, edge_outcome, edge_distance, invalid_actions, num_outcomes=num_outcomes)
    candidate_distance = edge_distance[batch, candidate_action]
    old_outcome = tree.node_categorical_outcome[batch, node_index]
    old_support = tree.node_payload[batch, node_index]
    publish_node = (
        active
        & (old_outcome == int(NO_OUTCOME))
        & (candidate_outcome != int(NO_OUTCOME))
    )

    parent = tree.parents[batch, node_index]
    has_parent = parent != Tree.NO_PARENT
    safe_parent = jnp.where(has_parent, parent, Tree.ROOT_INDEX)
    parent_children = tree.children_index[batch, safe_parent]
    incoming = parent_children == node_index[:, None]
    has_incoming = jnp.any(incoming, axis=-1)
    incoming_action = jnp.argmax(incoming, axis=-1).astype(jnp.int32)
    old_edge_outcome = tree.edge_categorical_outcome[
        batch,
        safe_parent,
        incoming_action,
    ]
    publish_edge = (
        publish_node
        & has_parent
        & has_incoming
        & (old_edge_outcome == int(NO_OUTCOME))
    )

    # Commit 1 + the child's final support before reinterpreting either the
    # incoming edge payload or the child's node payload as distance.
    old_edge_count = tree.edge_payload[
        batch,
        safe_parent,
        incoming_action,
    ]
    old_parent_support = tree.node_payload[batch, safe_parent]
    final_edge_count = 1 + old_support
    new_parent_support = old_parent_support + final_edge_count - old_edge_count
    node_player = tree.node_to_play[batch, node_index]
    parent_player = tree.node_to_play[batch, safe_parent]
    aligned_outcome = jnp.where(node_player == parent_player, candidate_outcome, (num_outcomes - 1 - candidate_outcome).astype(jnp.int8))

    node_payload = _set_node(tree.node_payload, safe_parent, new_parent_support, publish_edge)
    node_payload = _set_node(node_payload, node_index, candidate_distance, publish_node)
    return replace(
        tree,
        node_payload=node_payload,
        node_categorical_outcome=_set_node(tree.node_categorical_outcome, node_index, candidate_outcome, publish_node),
        edge_categorical_outcome=_set_edge(tree.edge_categorical_outcome, safe_parent, incoming_action, aligned_outcome, publish_edge),
        edge_payload=_set_edge(tree.edge_payload, safe_parent, incoming_action, candidate_distance + jnp.asarray(1, dtype=jnp.int32), publish_edge),
    )


def backward(rng_key: base.PRNGKey, tree: Tree, simulation: Simulation, step: base.RecurrentFnOutput, posterior_update: base.PosteriorUpdateFn) -> Tree:
    """Repair uncertain posteriors and propagate exact certificates upward."""

    batch = jnp.arange(tree.parents.shape[0])
    active = simulation.active
    node_index = jnp.where(active, simulation.parent_index, Tree.ROOT_INDEX)
    leaf_active = active

    def cond_fn(state: _BackwardState) -> Bool[Array, ""]:
        return jnp.any(state[3])

    def body_fn(state: _BackwardState) -> _BackwardState:
        key, tree, node_index, active, leaf_active = state
        key, update_key, tie_break_key = jax.random.split(key, 3)
        leaf = LeafView(action=simulation.action, value_alpha=step.value, to_play=step.to_play, active=leaf_active)
        context = PosteriorUpdateContext(node=_gather_node(tree, node_index), children=_gather_children(tree, node_index), leaf=leaf, active=active)
        update = posterior_update(update_key, context)
        tree = _set_node_update(tree, node_index, update, active)
        tree = _categorize_node_and_publish(tie_break_key, tree, node_index, active)
        continue_up = active & (node_index != Tree.ROOT_INDEX)
        parent = tree.parents[batch, node_index]
        next_node = jnp.where(continue_up, parent, Tree.ROOT_INDEX)
        return (key, tree, next_node, continue_up, jnp.zeros_like(leaf_active))

    _, tree, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, (rng_key, tree, node_index, active, leaf_active))
    return tree


def search(params: base.Params, rng_key: base.PRNGKey, *, root: base.RootFnOutput, recurrent_fn: base.RecurrentFn, action_selection_fn: base.ActionSelectionFn, posterior_update: base.PosteriorUpdateFn, num_simulations: int, max_depth: int | None = None, invalid_actions: Bool[Array, "batch action"] | None = None, loop_fn: base.LoopFn = jax.lax.fori_loop) -> Tree:
    """Run ``simulate -> expand -> bottom-up repair`` a fixed number of times."""

    if num_simulations < 0:
        raise ValueError(f"num_simulations must be >= 0, got {num_simulations}")
    if max_depth is None:
        max_depth = num_simulations
    if num_simulations > 0 and max_depth < 1:
        raise ValueError(f"max_depth must be >= 1 when num_simulations is positive, got {max_depth}")
    max_depth = max(1, int(max_depth))
    if invalid_actions is None:
        invalid_actions = ~jnp.isfinite(root.prior_logits)

    tree = instantiate_tree_from_root(root, num_simulations, invalid_actions)
    batch_size = root.prior_logits.shape[0]

    def body_fn(simulation_index: int | Int[Array, ""], state: _SearchState) -> _SearchState:
        key, tree = state
        key, simulate_key, expand_key, backward_key = jax.random.split(key, 4)
        simulation = simulate(jax.random.split(simulate_key, batch_size), tree, action_selection_fn, max_depth)

        def run_active_simulation(tree: Tree) -> Tree:
            new_node = jnp.asarray(simulation_index + 1, dtype=jnp.int32)
            tree, step = expand(params, expand_key, tree, recurrent_fn, simulation, new_node)
            return backward(backward_key, tree, simulation, step, posterior_update)

        tree = jax.lax.cond(jnp.any(simulation.active), run_active_simulation, lambda current_tree: current_tree, tree)
        return key, tree

    _, tree = loop_fn(0, num_simulations, body_fn, (rng_key, tree))
    return tree
