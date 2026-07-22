"""Batched simulate-expand-repair search for Dirichlet posteriors."""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, Int32, Shaped

from . import base, utils
from .categorical import NO_OUTCOME
from .tree import (
    LeafView,
    PosteriorUpdateContext,
    Tree,
    UnbatchedTree,
)


class _ScalarSimulation(NamedTuple):
    parent_index: Int32[Array, ""]
    action: Int32[Array, ""]
    active: Bool[Array, ""]


class Simulation(NamedTuple):
    parent_index: Int32[Array, "batch"]
    action: Int32[Array, "batch"]
    active: Bool[Array, "batch"]


class _SimulationState(NamedTuple):
    rng_key: base.PRNGKey
    node_index: Int32[Array, ""]
    action: Int32[Array, ""]
    next_node_index: Int32[Array, ""]
    depth: Int32[Array, ""]
    is_continuing: Bool[Array, ""]


type _BackwardState = tuple[base.PRNGKey, Tree, Int32[Array, "batch"], Bool[Array, "batch"], Bool[Array, "batch"]]
type _SearchState = tuple[base.PRNGKey, Tree]


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


def _simulate_one(rng_key: base.PRNGKey, tree: UnbatchedTree, action_selection_fn: base.ActionSelectionFn, max_depth: int) -> _ScalarSimulation:
    """Traverse one unbatched tree to an unvisited or cutoff edge."""

    root_index = jnp.asarray(Tree.ROOT_INDEX, dtype=jnp.int32)
    root_searchable_actions = tree.searchable_actions[root_index]
    root_active = (tree.node_categorical_outcome[root_index] == int(NO_OUTCOME)) & jnp.any(root_searchable_actions)

    def body_fn(state: _SimulationState) -> _SimulationState:
        next_key, selection_key = jax.random.split(state.rng_key)
        node_index = state.next_node_index

        searchable_actions = tree.searchable_actions[node_index]
        selection_tree = tree.replace(invalid_actions=tree.invalid_actions.at[node_index].set(~searchable_actions))
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

    def cond_fn(state: _SimulationState) -> Bool[Array, ""]:
        return state.is_continuing

    end = jax.lax.while_loop(cond_fn, body_fn, initial)

    return _ScalarSimulation(parent_index=end.node_index, action=end.action, active=root_active)


def simulate(rng_key: base.BatchedPRNGKey, tree: Tree, action_selection_fn: base.ActionSelectionFn, max_depth: int) -> Simulation:
    """Traverse every lane of a batched tree with exact per-lane types."""

    def simulate_one(key: base.PRNGKey, lane_tree: UnbatchedTree) -> _ScalarSimulation:
        return _simulate_one(key, lane_tree, action_selection_fn, max_depth)

    result = jax.vmap(simulate_one)(rng_key, tree)
    return Simulation(parent_index=result.parent_index, action=result.action, active=result.active)


def expand(params: base.Params, rng_key: base.PRNGKey, tree: Tree, recurrent_fn: base.RecurrentFn, simulation: Simulation, new_node_index: Int32[Array, ""]) -> tuple[Tree, base.RecurrentFnOutput]:
    """Evaluate selected edges and initialize genuinely new child nodes."""

    batch = jnp.arange(tree.parents.shape[0])
    parent_index = jnp.where(simulation.active, simulation.parent_index, 0)
    action = jnp.where(simulation.active, simulation.action, 0)

    def gather_parent_embedding(table: Shaped[Array, "batch node *embedding_axes"]) -> Shaped[Array, "batch *embedding_axes"]:
        return table[batch, parent_index]

    embedding = jax.tree.map(gather_parent_embedding, tree.embeddings)
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
    new_node_indices = jnp.broadcast_to(new_node_index, parent_index.shape)

    def set_child_embedding(table: Shaped[Array, "batch node *embedding_axes"], value: Shaped[Array, "batch *embedding_axes"]) -> Shaped[Array, "batch node *embedding_axes"]:
        return utils._set_embedding_node(table, new_node_indices, value, initialize)

    tree = replace(
        tree,
        parents=utils._set_scalar_node(tree.parents, new_node_indices, parent_index, initialize),
        children_index=utils._set_edge(tree.children_index, parent_index, action, new_node_indices, initialize),
        node_to_play=utils._set_scalar_node(tree.node_to_play, new_node_indices, step.to_play, initialize),
        node_categorical_outcome=utils._set_scalar_node(tree.node_categorical_outcome, new_node_indices, child_outcome, initialize),
        node_payload=utils._set_scalar_node(tree.node_payload, parent_index, parent_support, publish_terminal),
        edge_categorical_outcome=utils._set_edge(tree.edge_categorical_outcome, parent_index, action, aligned_outcome, publish_terminal),
        edge_payload=utils._set_edge(tree.edge_payload, parent_index, action, jnp.ones_like(aligned_outcome, dtype=jnp.int32), publish_terminal),
        node_value_priors=utils._set_outcome_node(tree.node_value_priors, new_node_indices, step.value, initialize),
        node_value_alpha=utils._set_outcome_node(tree.node_value_alpha, new_node_indices, step.value, initialize),
        edge_alpha=utils._set_action_outcome_node(tree.edge_alpha, new_node_indices, step.action_values, initialize),
        invalid_actions=utils._set_action_node(tree.invalid_actions, new_node_indices, step.invalid_actions, initialize),
        embeddings=jax.tree.map(set_child_embedding, tree.embeddings, child_embedding),
    )
    return tree, step


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
        context = PosteriorUpdateContext(node=utils._gather_node(tree, node_index), children=utils._gather_children(tree, node_index), leaf=leaf, active=active)
        update = posterior_update(update_key, context)
        tree = utils._set_node_update(tree, node_index, update, active)
        tree = utils._categorize_node_and_publish(tie_break_key, tree, node_index, active)
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

    def body_fn(simulation_index: Int[Array, ""], state: _SearchState) -> _SearchState:
        key, tree = state
        key, simulate_key, expand_key, backward_key = jax.random.split(key, 4)
        simulation = simulate(jax.random.split(simulate_key, batch_size), tree, action_selection_fn, max_depth)

        def run_active_simulation(tree: Tree) -> Tree:
            new_node = jnp.asarray(simulation_index + 1, dtype=jnp.int32)
            tree, step = expand(params, expand_key, tree, recurrent_fn, simulation, new_node)
            return backward(backward_key, tree, simulation, step, posterior_update)

        def keep_tree(current_tree: Tree) -> Tree:
            return current_tree

        tree = jax.lax.cond(jnp.any(simulation.active), run_active_simulation, keep_tree, tree)
        return key, tree

    _, tree = loop_fn(0, num_simulations, body_fn, (rng_key, tree))
    return tree
