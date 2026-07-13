"""Batched simulate-expand-backward search for Dirichlet posteriors."""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import Any, NamedTuple

import chex
import jax
import jax.numpy as jnp

from . import action_selection
from . import base
from .tree import Posterior, Tree


class Simulation(NamedTuple):
    parent_index: jax.Array
    action: jax.Array
    active: jax.Array


class _SimulationState(NamedTuple):
    rng_key: chex.PRNGKey
    node_index: jax.Array
    action: jax.Array
    next_node_index: jax.Array
    depth: jax.Array
    is_continuing: jax.Array


def _set_node(
    array: jax.Array,
    node_index: jax.Array,
    value: jax.Array,
    active: jax.Array,
) -> jax.Array:
    batch = jnp.arange(array.shape[0])
    old = array[batch, node_index]
    mask = active.reshape(active.shape + (1,) * (old.ndim - 1))
    return array.at[batch, node_index].set(jnp.where(mask, value, old))


def _set_edge(
    array: jax.Array,
    node_index: jax.Array,
    action: jax.Array,
    value: jax.Array,
    active: jax.Array,
) -> jax.Array:
    batch = jnp.arange(array.shape[0])
    old = array[batch, node_index, action]
    mask = active.reshape(active.shape + (1,) * (old.ndim - 1))
    return array.at[batch, node_index, action].set(jnp.where(mask, value, old))


def instantiate_tree_from_root(
    root: base.RootFnOutput,
    num_simulations: int,
    root_invalid_actions: jax.Array,
    posterior: Posterior | None = None,
) -> Tree:
    """Allocate fixed-capacity storage and place the root at node zero."""

    chex.assert_rank(root.prior_logits, 2)
    batch_size, num_actions = root.prior_logits.shape
    chex.assert_shape(root.value, (batch_size, root.action_values.shape[-1]))
    chex.assert_shape(
        root.action_values,
        (batch_size, num_actions, root.action_values.shape[-1]),
    )
    num_nodes = num_simulations + 1
    batch_node = (batch_size, num_nodes)
    batch_node_action = (batch_size, num_nodes, num_actions)

    def allocate_embedding(value: jax.Array) -> jax.Array:
        table = jnp.zeros((batch_size, num_nodes, *value.shape[1:]), dtype=value.dtype)
        return table.at[:, Tree.ROOT_INDEX].set(value)

    if posterior is None:
        posterior = Posterior(
            base=root.action_values,
            evidence=jnp.zeros_like(root.action_values),
            explored=jnp.zeros(root.action_values.shape[:-1], dtype=bool),
        )

    return Tree(
        parents=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
        action_from_parent=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
        children_index=jnp.full(
            batch_node_action,
            Tree.UNVISITED,
            dtype=jnp.int32,
        ),
        children_prior_logits=jnp.zeros(
            batch_node_action,
            dtype=root.prior_logits.dtype,
        ).at[:, Tree.ROOT_INDEX].set(root.prior_logits),
        children_visits=jnp.zeros(batch_node_action, dtype=jnp.int32),
        node_to_play=jnp.zeros(batch_node, dtype=root.to_play.dtype).at[
            :, Tree.ROOT_INDEX
        ].set(root.to_play),
        node_terminal=jnp.zeros(batch_node, dtype=bool).at[:, Tree.ROOT_INDEX].set(
            root.terminal
        ),
        embeddings=jax.tree.map(allocate_embedding, root.embedding),
        root_invalid_actions=root_invalid_actions,
        posterior=posterior,
    )


@functools.partial(jax.vmap, in_axes=(0, 0, None, None, None), out_axes=0)
def simulate(
    rng_key: chex.PRNGKey,
    tree: Tree,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    max_depth: int,
) -> Simulation:
    """Traverse one unbatched tree to an unvisited or cutoff edge."""

    root_active = (~tree.node_terminal[Tree.ROOT_INDEX]) & jnp.any(
        ~tree.root_invalid_actions
    )

    root_index = jnp.asarray(Tree.ROOT_INDEX, dtype=jnp.int32)
    interior_key, root_key = jax.random.split(rng_key)
    root_action = root_action_selection_fn(root_key, tree, root_index)
    root_child = tree.children_index[root_index, root_action]
    root_child_visited = root_child != Tree.UNVISITED
    safe_root_child = jnp.where(root_child_visited, root_child, root_index)

    def body_fn(state: _SimulationState) -> _SimulationState:
        node_index = state.next_node_index
        next_key, selection_key = jax.random.split(state.rng_key)
        action = interior_action_selection_fn(
            selection_key,
            tree,
            node_index,
            state.depth,
        )
        child_index = tree.children_index[node_index, action]
        visited = child_index != Tree.UNVISITED
        safe_child = jnp.where(visited, child_index, Tree.ROOT_INDEX)
        depth = state.depth + 1
        continuing = (
            visited
            & ~tree.node_terminal[safe_child]
            & (depth < max_depth)
        )
        return _SimulationState(
            rng_key=next_key,
            node_index=node_index,
            action=action,
            next_node_index=child_index,
            depth=depth,
            is_continuing=continuing,
        )

    initial = _SimulationState(
        rng_key=interior_key,
        node_index=root_index,
        action=root_action,
        next_node_index=root_child,
        depth=jnp.asarray(1, dtype=jnp.int32),
        is_continuing=(
            root_active
            & root_child_visited
            & ~tree.node_terminal[safe_root_child]
            & (1 < max_depth)
        ),
    )
    end = jax.lax.while_loop(lambda state: state.is_continuing, body_fn, initial)
    return Simulation(
        parent_index=end.node_index,
        action=end.action,
        active=root_active,
    )


def expand(
    params: base.Params,
    rng_key: chex.PRNGKey,
    tree: Tree,
    recurrent_fn: base.RecurrentFn,
    simulation: Simulation,
    next_node_index: jax.Array,
) -> tuple[Tree, base.RecurrentFnOutput]:
    """Evaluate the selected edges and update node/topology storage."""

    batch = jnp.arange(tree.parents.shape[0])
    parent_index = jnp.where(simulation.active, simulation.parent_index, 0)
    action = jnp.where(simulation.active, simulation.action, 0)
    embedding = jax.tree.map(
        lambda value: value[batch, parent_index],
        tree.embeddings,
    )
    step, child_embedding = recurrent_fn(params, rng_key, action, embedding)

    tree = replace(
        tree,
        parents=_set_node(
            tree.parents,
            next_node_index,
            parent_index,
            simulation.active,
        ),
        action_from_parent=_set_node(
            tree.action_from_parent,
            next_node_index,
            action,
            simulation.active,
        ),
        children_index=_set_edge(
            tree.children_index,
            parent_index,
            action,
            next_node_index,
            simulation.active,
        ),
        children_prior_logits=_set_node(
            tree.children_prior_logits,
            next_node_index,
            step.prior_logits,
            simulation.active,
        ),
        node_to_play=_set_node(
            tree.node_to_play,
            next_node_index,
            step.to_play,
            simulation.active,
        ),
        node_terminal=_set_node(
            tree.node_terminal,
            next_node_index,
            step.terminal,
            simulation.active,
        ),
        embeddings=jax.tree.map(
            lambda table, value: _set_node(
                table,
                next_node_index,
                value,
                simulation.active,
            ),
            tree.embeddings,
            child_embedding,
        ),
    )
    return tree, step


def backward(
    tree: Tree,
    simulation: Simulation,
    leaf_index: jax.Array,
    step: base.RecurrentFnOutput,
    posterior_update: base.PosteriorUpdateFn,
) -> Tree:
    """Update path visit counts, align evidence, and update the posterior."""

    batch = jnp.arange(tree.parents.shape[0])
    root_player = tree.node_to_play[:, Tree.ROOT_INDEX]
    active = simulation.active & (leaf_index != Tree.ROOT_INDEX)
    root_action = jnp.zeros_like(simulation.action)

    def cond_fn(state) -> jax.Array:
        return jnp.any(state[4])

    def body_fn(state):
        tree, outcome, player, node_index, active, root_action = state
        safe_node = jnp.where(active, node_index, Tree.ROOT_INDEX)
        parent = tree.parents[batch, safe_node]
        safe_parent = jnp.where(active, parent, Tree.ROOT_INDEX)
        action = tree.action_from_parent[batch, safe_node]
        safe_action = jnp.where(active, action, 0)
        parent_player = tree.node_to_play[batch, safe_parent]
        aligned = action_selection.align_outcome(outcome, player, parent_player)

        visits = tree.children_visits[batch, safe_parent, safe_action]
        visits = visits + active.astype(visits.dtype)
        children_visits = tree.children_visits.at[
            batch, safe_parent, safe_action
        ].set(visits)
        at_root = active & (safe_parent == Tree.ROOT_INDEX)
        root_action = jnp.where(at_root, safe_action, root_action)
        next_active = active & ~at_root
        return (
            replace(tree, children_visits=children_visits),
            jnp.where(active[..., None], aligned, outcome),
            jnp.where(active, parent_player, player),
            safe_parent,
            next_active,
            root_action,
        )

    tree, root_evidence, _, _, _, root_action = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (
            tree,
            step.outcome,
            step.to_play,
            leaf_index,
            active,
            root_action,
        ),
    )
    aligned_action_value = action_selection.align_outcome(
        step.value,
        step.to_play,
        root_player,
    )
    posterior = posterior_update(
        tree.posterior,
        action=root_action,
        action_value_prior=aligned_action_value,
        has_action_value_prior=simulation.parent_index == Tree.ROOT_INDEX,
        evidence=root_evidence,
        evidence_weight=step.evidence_weight,
        active=simulation.active,
    )
    return replace(tree, posterior=posterior)


def search(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn,
    posterior_update: base.PosteriorUpdateFn,
    num_simulations: int,
    max_depth: int | None = None,
    invalid_actions: jax.Array | None = None,
    posterior: Posterior | None = None,
    loop_fn: base.LoopFn = jax.lax.fori_loop,
) -> Tree:
    """Run ``simulate -> expand -> backward`` for a fixed number of steps."""

    if num_simulations < 0:
        raise ValueError(f"num_simulations must be >= 0, got {num_simulations}")
    if max_depth is None:
        max_depth = num_simulations
    max_depth = max(1, int(max_depth))
    if invalid_actions is None:
        invalid_actions = jnp.zeros_like(root.prior_logits, dtype=bool)

    tree = instantiate_tree_from_root(
        root,
        num_simulations,
        invalid_actions,
        posterior,
    )
    batch_size = root.prior_logits.shape[0]
    batch = jnp.arange(batch_size)

    def body_fn(simulation_index: int, state):
        key, tree = state
        key, simulate_key, expand_key = jax.random.split(key, 3)
        simulation = simulate(
            jax.random.split(simulate_key, batch_size),
            tree,
            root_action_selection_fn,
            interior_action_selection_fn,
            max_depth,
        )
        parent = jnp.where(simulation.active, simulation.parent_index, 0)
        action = jnp.where(simulation.active, simulation.action, 0)
        child = tree.children_index[batch, parent, action]
        new_node = jnp.asarray(simulation_index + 1, dtype=jnp.int32)
        next_node = jnp.where(
            simulation.active & (child == Tree.UNVISITED),
            new_node,
            jnp.where(simulation.active, child, Tree.ROOT_INDEX),
        )
        tree, step = expand(
            params,
            expand_key,
            tree,
            recurrent_fn,
            simulation,
            next_node,
        )
        tree = backward(tree, simulation, next_node, step, posterior_update)
        return key, tree

    _, tree = loop_fn(0, num_simulations, body_fn, (rng_key, tree))
    return tree
