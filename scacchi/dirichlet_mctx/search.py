"""Batched simulate-expand-repair search for Dirichlet posteriors."""

from __future__ import annotations

import functools
from dataclasses import replace
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp

from . import action_selection, base
from .categorical import NO_DISTANCE, NO_OUTCOME
from .tree import (
    ChildrenView,
    LeafView,
    NodePosterior,
    NodeView,
    PosteriorUpdateContext,
    Tree,
)


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


def _set_new_node(
    array: jax.Array,
    node_index: jax.Array,
    value: jax.Array,
    active: jax.Array,
) -> jax.Array:
    """Initialize the one scalar slot reserved for this simulation."""

    old = array[:, node_index]
    mask = active.reshape(active.shape + (1,) * (old.ndim - 1))
    return array.at[:, node_index].set(jnp.where(mask, value, old))


def instantiate_tree_from_root(
    root: base.RootFnOutput,
    num_simulations: int,
    root_invalid_actions: jax.Array,
) -> Tree:
    """Allocate fixed-capacity storage and place the root at node zero."""

    chex.assert_rank(root.prior_logits, 2)
    batch_size, num_actions = root.prior_logits.shape
    num_outcomes = root.action_values.shape[-1]
    chex.assert_shape(root.value, (batch_size, num_outcomes))
    chex.assert_shape(
        root.action_values,
        (batch_size, num_actions, num_outcomes),
    )
    chex.assert_shape(root_invalid_actions, (batch_size, num_actions))
    num_nodes = num_simulations + 1
    batch_node = (batch_size, num_nodes)
    batch_node_action = (batch_size, num_nodes, num_actions)

    def allocate_embedding(value: jax.Array) -> jax.Array:
        table = jnp.zeros((batch_size, num_nodes, *value.shape[1:]), dtype=value.dtype)
        return table.at[:, Tree.ROOT_INDEX].set(value)

    action_alpha = jnp.ones(
        (*batch_node_action, num_outcomes),
        dtype=root.action_values.dtype,
    ).at[:, Tree.ROOT_INDEX].set(root.action_values)
    action_count = jnp.zeros(batch_node_action, dtype=jnp.int32)
    value_alpha = jnp.ones(
        (*batch_node, num_outcomes),
        dtype=root.value.dtype,
    ).at[:, Tree.ROOT_INDEX].set(root.value)
    return Tree(
        parents=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
        children_index=jnp.full(
            batch_node_action,
            Tree.UNVISITED,
            dtype=jnp.int32,
        ),
        node_to_play=jnp.zeros(batch_node, dtype=root.to_play.dtype).at[
            :, Tree.ROOT_INDEX
        ].set(root.to_play),
        node_terminal=jnp.zeros(batch_node, dtype=bool).at[:, Tree.ROOT_INDEX].set(
            root.terminal
        ),
        node_prior_logits=jnp.zeros(
            batch_node_action,
            dtype=root.prior_logits.dtype,
        ).at[:, Tree.ROOT_INDEX].set(root.prior_logits),
        node_categorical_outcome=jnp.full(
            batch_node,
            int(NO_OUTCOME),
            dtype=jnp.int8,
        ),
        node_categorical_distance=jnp.full(
            batch_node,
            int(NO_DISTANCE),
            dtype=jnp.int32,
        ),
        edge_categorical_outcome=jnp.full(
            batch_node_action,
            int(NO_OUTCOME),
            dtype=jnp.int8,
        ),
        edge_categorical_distance=jnp.full(
            batch_node_action,
            int(NO_DISTANCE),
            dtype=jnp.int32,
        ),
        node_value_priors=jnp.ones(
            (*batch_node, num_outcomes),
            dtype=root.value.dtype,
        ).at[:, Tree.ROOT_INDEX].set(root.value),
        node_n_down=jnp.zeros(batch_node, dtype=jnp.int32),
        invalid_actions=jnp.ones(batch_node_action, dtype=bool).at[
            :, Tree.ROOT_INDEX
        ].set(root_invalid_actions),
        posterior=NodePosterior(
            action_alpha=action_alpha,
            action_count=action_count,
            value_alpha=value_alpha,
        ),
        embeddings=jax.tree.map(allocate_embedding, root.embedding),
    )


@functools.partial(jax.vmap, in_axes=(0, 0, None, None), out_axes=0)
def simulate(
    rng_key: chex.PRNGKey,
    tree: Tree,
    action_selection_fn: base.ActionSelectionFn,
    max_depth: int,
) -> Simulation:
    """Traverse one unbatched tree to an unvisited, terminal, or cutoff edge."""

    root_index = jnp.asarray(Tree.ROOT_INDEX, dtype=jnp.int32)
    root_searchable_actions = (
        ~tree.invalid_actions[root_index]
        & (
            tree.edge_categorical_outcome[root_index]
            == int(NO_OUTCOME)
        )
    )
    root_active = (
        ~tree.node_terminal[root_index]
        & (tree.node_categorical_outcome[root_index] == int(NO_OUTCOME))
        & jnp.any(root_searchable_actions)
    )

    def body_fn(state: _SimulationState) -> _SimulationState:
        node_index = state.next_node_index
        next_key, selection_key = jax.random.split(state.rng_key)
        searchable_actions = (
            ~tree.invalid_actions[node_index]
            & (
                tree.edge_categorical_outcome[node_index]
                == int(NO_OUTCOME)
            )
        )
        # Categorical exclusion is part of traversal, not a convention that
        # only the default selector knows about. Existing custom selectors see
        # the certified edges through the same invalid-action contract.
        selection_tree = replace(
            tree,
            invalid_actions=tree.invalid_actions.at[node_index].set(
                ~searchable_actions
            ),
        )
        proposed_action = action_selection_fn(
            selection_key,
            selection_tree,
            node_index,
        )
        # Selectors are expected to honor the mask. The fallback also makes
        # the categorical invariant robust to an older selector that does not.
        fallback_action = jnp.argmax(searchable_actions).astype(jnp.int32)
        action = jnp.where(
            searchable_actions[proposed_action],
            proposed_action,
            fallback_action,
        )
        child_index = tree.children_index[node_index, action]
        visited = child_index != Tree.UNVISITED
        safe_child = jnp.where(visited, child_index, Tree.ROOT_INDEX)
        depth = state.depth + 1
        child_has_searchable_action = jnp.any(
            ~tree.invalid_actions[safe_child]
            & (
                tree.edge_categorical_outcome[safe_child]
                == int(NO_OUTCOME)
            )
        )
        child_selectable = (
            ~tree.node_terminal[safe_child]
            & (
                tree.node_categorical_outcome[safe_child]
                == int(NO_OUTCOME)
            )
            & child_has_searchable_action
        )
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
    new_node_index: jax.Array,
    is_new: jax.Array,
) -> tuple[Tree, base.RecurrentFnOutput]:
    """Evaluate selected edges and initialize genuinely new child nodes."""

    batch = jnp.arange(tree.parents.shape[0])
    parent_index = jnp.where(simulation.active, simulation.parent_index, 0)
    action = jnp.where(simulation.active, simulation.action, 0)
    embedding = jax.tree.map(
        lambda value: value[batch, parent_index],
        tree.embeddings,
    )
    step, child_embedding = recurrent_fn(params, rng_key, action, embedding)
    child_invalid_actions = ~jnp.isfinite(step.prior_logits)
    initialize = simulation.active & is_new
    num_outcomes = tree.posterior.value_alpha.shape[-1]
    child_terminal_outcome = step.terminal_outcome.astype(jnp.int8)
    child_terminal = child_terminal_outcome != int(NO_OUTCOME)
    parent_player = tree.node_to_play[batch, parent_index]
    parent_terminal_outcome = jnp.where(
        step.to_play == parent_player,
        child_terminal_outcome,
        (num_outcomes - 1 - child_terminal_outcome).astype(jnp.int8),
    )
    old_edge_categorical = tree.edge_categorical_outcome[
        batch,
        parent_index,
        action,
    ]
    publish_terminal = (
        simulation.active
        & child_terminal
        & (old_edge_categorical == int(NO_OUTCOME))
    )
    posterior = tree.posterior

    tree = replace(
        tree,
        parents=_set_new_node(
            tree.parents,
            new_node_index,
            parent_index,
            initialize,
        ),
        children_index=_set_edge(
            tree.children_index,
            parent_index,
            action,
            new_node_index,
            initialize,
        ),
        node_to_play=_set_new_node(
            tree.node_to_play,
            new_node_index,
            step.to_play,
            initialize,
        ),
        node_terminal=_set_new_node(
            tree.node_terminal,
            new_node_index,
            child_terminal,
            initialize,
        ),
        node_prior_logits=_set_new_node(
            tree.node_prior_logits,
            new_node_index,
            step.prior_logits,
            initialize,
        ),
        node_categorical_outcome=_set_new_node(
            tree.node_categorical_outcome,
            new_node_index,
            jnp.where(
                child_terminal,
                child_terminal_outcome,
                jnp.asarray(int(NO_OUTCOME), dtype=jnp.int8),
            ),
            initialize,
        ),
        node_categorical_distance=_set_new_node(
            tree.node_categorical_distance,
            new_node_index,
            jnp.where(
                child_terminal,
                jnp.zeros_like(child_terminal, dtype=jnp.int32),
                jnp.full_like(
                    child_terminal,
                    int(NO_DISTANCE),
                    dtype=jnp.int32,
                ),
            ),
            initialize,
        ),
        edge_categorical_outcome=_set_edge(
            tree.edge_categorical_outcome,
            parent_index,
            action,
            parent_terminal_outcome,
            publish_terminal,
        ),
        edge_categorical_distance=_set_edge(
            tree.edge_categorical_distance,
            parent_index,
            action,
            jnp.ones_like(parent_terminal_outcome, dtype=jnp.int32),
            publish_terminal,
        ),
        node_value_priors=_set_new_node(
            tree.node_value_priors,
            new_node_index,
            step.value,
            initialize,
        ),
        invalid_actions=_set_new_node(
            tree.invalid_actions,
            new_node_index,
            child_invalid_actions,
            initialize,
        ),
        posterior=NodePosterior(
            action_alpha=_set_new_node(
                posterior.action_alpha,
                new_node_index,
                step.action_values,
                initialize,
            ),
            action_count=_set_new_node(
                posterior.action_count,
                new_node_index,
                jnp.zeros_like(step.prior_logits, dtype=jnp.int32),
                initialize,
            ),
            value_alpha=_set_new_node(
                posterior.value_alpha,
                new_node_index,
                step.value,
                initialize,
            ),
        ),
        embeddings=jax.tree.map(
            lambda table, value: _set_new_node(
                table,
                new_node_index,
                value,
                initialize,
            ),
            tree.embeddings,
            child_embedding,
        ),
    )
    return tree, step


def _gather_node(tree: Tree, node_index: jax.Array) -> NodeView:
    batch = jnp.arange(tree.parents.shape[0])
    posterior = tree.posterior
    gathered = NodePosterior(
        action_alpha=posterior.action_alpha[batch, node_index],
        action_count=posterior.action_count[batch, node_index],
        value_alpha=posterior.value_alpha[batch, node_index],
    )
    return NodeView(
        index=node_index,
        embedding=jax.tree.map(
            lambda value: value[batch, node_index],
            tree.embeddings,
        ),
        value_prior=tree.node_value_priors[batch, node_index],
        to_play=tree.node_to_play[batch, node_index],
        terminal=tree.node_terminal[batch, node_index],
        invalid_actions=tree.invalid_actions[batch, node_index],
        posterior=gathered,
    )


def _gather_children(tree: Tree, node_index: jax.Array) -> ChildrenView:
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
        value_alpha=tree.posterior.value_alpha[batch_actions, safe_child],
        count=tree.node_n_down[batch_actions, safe_child],
        to_play=tree.node_to_play[batch_actions, safe_child],
        terminal=tree.node_terminal[batch_actions, safe_child],
    )


def _set_node_posterior(
    tree: Tree,
    node_index: jax.Array,
    posterior: NodePosterior,
    active: jax.Array,
) -> Tree:
    old = tree.posterior
    batch = jnp.arange(tree.parents.shape[0])
    legal_actions = ~tree.invalid_actions[batch, node_index]
    n_down = jnp.sum(
        jnp.where(legal_actions, posterior.action_count, 0),
        axis=-1,
    )
    return replace(
        tree,
        node_n_down=_set_node(
            tree.node_n_down,
            node_index,
            n_down,
            active,
        ),
        posterior=NodePosterior(
            action_alpha=_set_node(
                old.action_alpha,
                node_index,
                posterior.action_alpha,
                active,
            ),
            action_count=_set_node(
                old.action_count,
                node_index,
                posterior.action_count,
                active,
            ),
            value_alpha=_set_node(
                old.value_alpha,
                node_index,
                posterior.value_alpha,
                active,
            ),
        ),
    )


def _categorize_node_and_publish(
    tree: Tree,
    node_index: jax.Array,
    active: jax.Array,
    categorical_draw_rule: str,
) -> Tree:
    """Publish an absorbing node certificate and its incoming edge."""

    batch = jnp.arange(tree.parents.shape[0])
    edge_outcome = tree.edge_categorical_outcome[batch, node_index]
    edge_distance = tree.edge_categorical_distance[batch, node_index]
    invalid_actions = tree.invalid_actions[batch, node_index]
    legal = ~invalid_actions
    known = legal & (edge_outcome != int(NO_OUTCOME))
    has_legal = jnp.any(legal, axis=-1)
    all_categorical = has_legal & jnp.all(~legal | known, axis=-1)
    num_outcomes = tree.posterior.value_alpha.shape[-1]
    win_index = num_outcomes - 1
    has_win = jnp.any(known & (edge_outcome == win_index), axis=-1)
    if num_outcomes == 3:
        has_draw = jnp.any(known & (edge_outcome == 1), axis=-1)
    else:
        has_draw = jnp.zeros_like(has_win)

    candidate_outcome = jnp.where(
        has_win,
        jnp.asarray(win_index, dtype=jnp.int8),
        jnp.where(
            all_categorical & has_draw,
            jnp.asarray(1, dtype=jnp.int8),
            jnp.where(
                all_categorical,
                jnp.asarray(0, dtype=jnp.int8),
                jnp.asarray(int(NO_OUTCOME), dtype=jnp.int8),
            ),
        ),
    )
    candidate_action = action_selection.categorical_action(
        candidate_outcome,
        edge_outcome,
        edge_distance,
        tree.node_prior_logits[batch, node_index],
        invalid_actions,
        num_outcomes=num_outcomes,
        draw_rule=categorical_draw_rule,
    )
    candidate_distance = jnp.take_along_axis(
        edge_distance,
        candidate_action[:, None],
        axis=-1,
    )[:, 0]
    old_outcome = tree.node_categorical_outcome[batch, node_index]
    old_distance = tree.node_categorical_distance[batch, node_index]
    publish_node = (
        active
        & (old_outcome == int(NO_OUTCOME))
        & (candidate_outcome != int(NO_OUTCOME))
    )
    new_outcome = jnp.where(publish_node, candidate_outcome, old_outcome)
    new_distance = jnp.where(publish_node, candidate_distance, old_distance)
    tree = replace(
        tree,
        node_categorical_outcome=_set_node(
            tree.node_categorical_outcome,
            node_index,
            new_outcome,
            publish_node,
        ),
        node_categorical_distance=_set_node(
            tree.node_categorical_distance,
            node_index,
            new_distance,
            publish_node,
        ),
    )

    # A node certificate induces an exact edge certificate one ply farther
    # from terminal, aligned to the parent node's player perspective.
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
        active
        & has_parent
        & has_incoming
        & (new_outcome != int(NO_OUTCOME))
        & (old_edge_outcome == int(NO_OUTCOME))
    )
    node_player = tree.node_to_play[batch, node_index]
    parent_player = tree.node_to_play[batch, safe_parent]
    aligned_outcome = jnp.where(
        node_player == parent_player,
        new_outcome,
        (num_outcomes - 1 - new_outcome).astype(jnp.int8),
    )
    return replace(
        tree,
        edge_categorical_outcome=_set_edge(
            tree.edge_categorical_outcome,
            safe_parent,
            incoming_action,
            aligned_outcome,
            publish_edge,
        ),
        edge_categorical_distance=_set_edge(
            tree.edge_categorical_distance,
            safe_parent,
            incoming_action,
            new_distance + jnp.asarray(1, dtype=jnp.int32),
            publish_edge,
        ),
    )


def backward(
    rng_key: chex.PRNGKey,
    tree: Tree,
    simulation: Simulation,
    step: base.RecurrentFnOutput,
    posterior_update: base.PosteriorUpdateFn,
    categorical_draw_rule: str,
) -> Tree:
    """Repair uncertain posteriors and propagate exact certificates upward."""

    batch = jnp.arange(tree.parents.shape[0])
    active = simulation.active
    node_index = jnp.where(active, simulation.parent_index, Tree.ROOT_INDEX)
    leaf_active = active

    def cond_fn(state) -> jax.Array:
        return jnp.any(state[3])

    def body_fn(state):
        key, tree, node_index, active, leaf_active = state
        key, update_key = jax.random.split(key)
        safe_node = jnp.where(active, node_index, Tree.ROOT_INDEX)
        node = _gather_node(tree, safe_node)
        children = _gather_children(tree, safe_node)
        context = PosteriorUpdateContext(
            node=node,
            children=children,
            leaf=LeafView(
                action=simulation.action,
                value_alpha=step.value,
                to_play=step.to_play,
                active=leaf_active,
            ),
            active=active,
            edge_categorical_outcome=tree.edge_categorical_outcome[
                batch,
                safe_node,
            ],
            edge_categorical_distance=tree.edge_categorical_distance[
                batch,
                safe_node,
            ],
        )
        repaired = posterior_update(update_key, context)
        tree = _set_node_posterior(tree, safe_node, repaired, active)
        tree = _categorize_node_and_publish(
            tree,
            safe_node,
            active,
            categorical_draw_rule,
        )
        at_root = active & (safe_node == Tree.ROOT_INDEX)
        parent = tree.parents[batch, safe_node]
        next_node = jnp.where(active & ~at_root, parent, Tree.ROOT_INDEX)
        return (
            key,
            tree,
            next_node,
            active & ~at_root,
            jnp.zeros_like(leaf_active),
        )

    _, tree, _, _, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (rng_key, tree, node_index, active, leaf_active),
    )
    return tree


def search(
    params: base.Params,
    rng_key: chex.PRNGKey,
    *,
    root: base.RootFnOutput,
    recurrent_fn: base.RecurrentFn,
    action_selection_fn: base.ActionSelectionFn,
    posterior_update: base.PosteriorUpdateFn,
    num_simulations: int,
    max_depth: int | None = None,
    invalid_actions: jax.Array | None = None,
    categorical_draw_rule: str = "policy_prior",
    loop_fn: base.LoopFn = jax.lax.fori_loop,
) -> Tree:
    """Run ``simulate -> expand -> bottom-up repair`` a fixed number of times."""

    if num_simulations < 0:
        raise ValueError(f"num_simulations must be >= 0, got {num_simulations}")
    if max_depth is None:
        max_depth = num_simulations
    if num_simulations > 0 and max_depth < 1:
        raise ValueError(
            "max_depth must be >= 1 when num_simulations is positive, "
            f"got {max_depth}"
        )
    max_depth = max(1, int(max_depth))
    if invalid_actions is None:
        invalid_actions = jnp.zeros_like(root.prior_logits, dtype=bool)
    categorical_draw_rule = action_selection.validate_categorical_draw_rule(
        categorical_draw_rule
    )

    tree = instantiate_tree_from_root(
        root,
        num_simulations,
        invalid_actions,
    )
    batch_size = root.prior_logits.shape[0]
    batch = jnp.arange(batch_size)

    def body_fn(simulation_index: int, state):
        key, tree = state
        key, simulate_key, expand_key, backward_key = jax.random.split(key, 4)
        simulation = simulate(
            jax.random.split(simulate_key, batch_size),
            tree,
            action_selection_fn,
            max_depth,
        )
        def run_active_simulation(tree: Tree) -> Tree:
            parent = jnp.where(simulation.active, simulation.parent_index, 0)
            action = jnp.where(simulation.active, simulation.action, 0)
            child = tree.children_index[batch, parent, action]
            is_new = simulation.active & (child == Tree.UNVISITED)
            new_node = jnp.asarray(simulation_index + 1, dtype=jnp.int32)
            tree, step = expand(
                params,
                expand_key,
                tree,
                recurrent_fn,
                simulation,
                new_node,
                is_new,
            )
            return backward(
                backward_key,
                tree,
                simulation,
                step,
                posterior_update,
                categorical_draw_rule,
            )

        tree = jax.lax.cond(
            jnp.any(simulation.active),
            run_active_simulation,
            lambda current_tree: current_tree,
            tree,
        )
        return key, tree

    _, tree = loop_fn(0, num_simulations, body_fn, (rng_key, tree))
    return tree
