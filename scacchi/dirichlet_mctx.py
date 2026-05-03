"""Dual-use vendored MCTS module.

A focused copy of select pieces of mctx (DeepMind's Monte Carlo Tree Search in JAX)
with surgical edits to support two value-shape modes:

- "scalar" (default): byte-identical to upstream mctx.gumbel_muzero_policy.
  Used today by Stage B (scalar value derived from U(mean(alpha_V))).
- "wdl": WDL-vector-valued tree. Each node tracks a [3] running mean (L, D, W)
  instead of a scalar; backup applies an explicit W↔L flip on perspective
  changes. Reserved for the upcoming Stage A switch in play.py.

Vendored from mctx==0.0.6. Sections kept byte-identical to upstream are marked
"verbatim from upstream"; mode-aware divergences are marked "# DUAL-USE EDIT".
We import the pieces we do NOT modify directly from mctx._src to keep the file
small and re-syncable.
"""
from __future__ import annotations

import functools
from typing import Any, ClassVar, Generic, Literal, Optional, Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp

from mctx import PolicyOutput, RecurrentFnOutput, RootFnOutput  # re-export upstream types
from mctx._src import action_selection as _action_selection
from mctx._src import qtransforms as _qtransforms

__all__ = [
    "PolicyOutput",
    "RecurrentFnOutput",
    "RootFnOutput",
    "Tree",
    "SearchSummary",
    "dirichlet_gumbel_policy",
    "gumbel_muzero_policy",
    "search",
    "qtransform_completed_by_mix_value",
    "flip_wdl",
]


T = TypeVar("T")
ValueMode = Literal["scalar", "wdl"]
WDL_DIM = 3
L_IDX = 0
W_IDX = 2


def flip_wdl(x: jax.Array) -> jax.Array:
    """Swap W and L along the last axis (D fixed). Identity on the other axes."""
    return x[..., ::-1]


def _wdl_to_scalar(x: jax.Array) -> jax.Array:
    """U(φ) = p_W − p_L along the last axis."""
    return x[..., W_IDX] - x[..., L_IDX]


# ----------------------------------------------------------------------------
# Tree dataclass — vendored from mctx._src.tree.Tree with generic value shape.
# ----------------------------------------------------------------------------


@chex.dataclass(frozen=True)
class Tree(Generic[T]):
    """State of a search tree.

    In scalar mode the value-bearing fields have shapes identical to upstream
    mctx.Tree:
        node_values:      [B, N]
        raw_values:       [B, N]
        children_values:  [B, N, A]
        children_rewards: [B, N, A]

    In WDL mode each value-bearing field gains a trailing 3-dim:
        node_values:      [B, N, 3]
        raw_values:       [B, N, 3]
        children_values:  [B, N, A, 3]
        children_rewards: [B, N, A, 3]

    Discount, visits, prior logits, and topology fields are unchanged in both modes.
    """
    node_visits: chex.Array
    raw_values: chex.Array
    node_values: chex.Array
    parents: chex.Array
    action_from_parent: chex.Array
    children_index: chex.Array
    children_prior_logits: chex.Array
    children_visits: chex.Array
    children_rewards: chex.Array
    children_discounts: chex.Array
    children_values: chex.Array
    embeddings: Any
    root_invalid_actions: chex.Array
    extra_data: T

    ROOT_INDEX: ClassVar[int] = 0
    NO_PARENT: ClassVar[int] = -1
    UNVISITED: ClassVar[int] = -1

    @property
    def num_actions(self):
        return self.children_index.shape[-1]

    @property
    def num_simulations(self):
        return self.node_visits.shape[-1] - 1

    def qvalues(self, indices):
        if jnp.asarray(indices).shape:
            return jax.vmap(_unbatched_qvalues)(self, indices)
        return _unbatched_qvalues(self, indices)

    def summary(self) -> "SearchSummary":
        chex.assert_rank(self.node_visits, 2)
        value = self.node_values[:, Tree.ROOT_INDEX]  # [B] or [B, 3]
        batch_size = value.shape[0]
        root_indices = jnp.full((batch_size,), Tree.ROOT_INDEX)
        qvalues = self.qvalues(root_indices)
        visit_counts = self.children_visits[:, Tree.ROOT_INDEX].astype(self.node_values.dtype)
        total_counts = jnp.sum(visit_counts, axis=-1, keepdims=True)
        visit_probs = visit_counts / jnp.maximum(total_counts, 1)
        visit_probs = jnp.where(total_counts > 0, visit_probs, 1 / self.num_actions)
        return SearchSummary(
            visit_counts=visit_counts,
            visit_probs=visit_probs,
            value=value,
            qvalues=qvalues,
        )


@chex.dataclass(frozen=True)
class SearchSummary:
    visit_counts: chex.Array
    visit_probs: chex.Array
    value: chex.Array
    qvalues: chex.Array


def _unbatched_qvalues(tree: Tree, index):
    # DUAL-USE EDIT: works for scalar children_values ([N, A]) and WDL ([N, A, 3]).
    cv = tree.children_values[index]
    cd = tree.children_discounts[index]
    cr = tree.children_rewards[index]
    if cv.ndim > cd.ndim:
        extra = cv.ndim - cd.ndim
        cd = cd.reshape(cd.shape + (1,) * extra)
    return cr + cd * cv


def _infer_batch_size(tree: Tree) -> int:
    if tree.node_visits.ndim != 2:
        raise ValueError("Input tree is not batched.")
    return tree.node_visits.shape[0]


# ----------------------------------------------------------------------------
# Tree mutation helpers — verbatim from upstream search.py.
# ----------------------------------------------------------------------------


def update(x, vals, *indices):
    return x.at[indices].set(vals)


batch_update = jax.vmap(update)


def _update_tree_node(
    tree: Tree[T],
    node_index: chex.Array,
    prior_logits: chex.Array,
    value: chex.Array,
    embedding: chex.Array,
) -> Tree[T]:
    """Updates the tree at the given node indices. Generic over value shape."""
    batch_size = _infer_batch_size(tree)
    batch_range = jnp.arange(batch_size)
    chex.assert_shape(prior_logits, (batch_size, tree.num_actions))

    new_visit = tree.node_visits[batch_range, node_index] + 1
    updates = dict(
        children_prior_logits=batch_update(
            tree.children_prior_logits, prior_logits, node_index),
        raw_values=batch_update(tree.raw_values, value, node_index),
        node_values=batch_update(tree.node_values, value, node_index),
        node_visits=batch_update(tree.node_visits, new_visit, node_index),
        embeddings=jax.tree.map(
            lambda t, s: batch_update(t, s, node_index),
            tree.embeddings, embedding),
    )
    return tree.replace(**updates)


# ----------------------------------------------------------------------------
# instantiate_tree_from_root — DUAL-USE EDIT: shape-aware allocation.
# ----------------------------------------------------------------------------


def _instantiate_tree_from_root(
    root: RootFnOutput,
    num_simulations: int,
    root_invalid_actions: chex.Array,
    extra_data: Any,
) -> Tree:
    chex.assert_rank(root.prior_logits, 2)
    batch_size, num_actions = root.prior_logits.shape
    # DUAL-USE EDIT: derive value shape from root.value (scalar or WDL).
    value_shape = root.value.shape[1:]  # () in scalar, (3,) in WDL
    chex.assert_shape(root.value, (batch_size,) + value_shape)
    num_nodes = num_simulations + 1
    data_dtype = root.value.dtype
    batch_node = (batch_size, num_nodes)
    batch_node_action = (batch_size, num_nodes, num_actions)

    def _zeros_emb(x):
        return jnp.zeros(batch_node + x.shape[1:], dtype=x.dtype)

    tree = Tree(
        node_visits=jnp.zeros(batch_node, dtype=jnp.int32),
        # DUAL-USE EDIT: extend value-bearing allocations with value_shape.
        raw_values=jnp.zeros(batch_node + value_shape, dtype=data_dtype),
        node_values=jnp.zeros(batch_node + value_shape, dtype=data_dtype),
        parents=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
        action_from_parent=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
        children_index=jnp.full(batch_node_action, Tree.UNVISITED, dtype=jnp.int32),
        children_prior_logits=jnp.zeros(batch_node_action, dtype=root.prior_logits.dtype),
        children_values=jnp.zeros(batch_node_action + value_shape, dtype=data_dtype),
        children_visits=jnp.zeros(batch_node_action, dtype=jnp.int32),
        children_rewards=jnp.zeros(batch_node_action + value_shape, dtype=data_dtype),
        children_discounts=jnp.zeros(batch_node_action, dtype=data_dtype),
        embeddings=jax.tree.map(_zeros_emb, root.embedding),
        root_invalid_actions=root_invalid_actions,
        extra_data=extra_data,
    )

    root_index = jnp.full([batch_size], Tree.ROOT_INDEX)
    tree = _update_tree_node(
        tree, root_index, root.prior_logits, root.value, root.embedding)
    return tree


# ----------------------------------------------------------------------------
# expand — DUAL-USE EDIT: relax shape asserts to allow trailing value dims.
# ----------------------------------------------------------------------------


def _expand(
    params: chex.Array,
    rng_key: chex.PRNGKey,
    tree: Tree[T],
    recurrent_fn,
    parent_index: chex.Array,
    action: chex.Array,
    next_node_index: chex.Array,
    *,
    value_shape: Tuple[int, ...],
) -> Tree[T]:
    batch_size = _infer_batch_size(tree)
    batch_range = jnp.arange(batch_size)
    chex.assert_shape([parent_index, action, next_node_index], (batch_size,))

    embedding = jax.tree.map(lambda x: x[batch_range, parent_index], tree.embeddings)
    step, embedding = recurrent_fn(params, rng_key, action, embedding)
    chex.assert_shape(step.prior_logits, [batch_size, tree.num_actions])
    # DUAL-USE EDIT: shape asserts honor value_shape.
    chex.assert_shape(step.reward, (batch_size,) + value_shape)
    chex.assert_shape(step.discount, [batch_size])
    chex.assert_shape(step.value, (batch_size,) + value_shape)
    tree = _update_tree_node(
        tree, next_node_index, step.prior_logits, step.value, embedding)

    return tree.replace(
        children_index=batch_update(
            tree.children_index, next_node_index, parent_index, action),
        children_rewards=batch_update(
            tree.children_rewards, step.reward, parent_index, action),
        children_discounts=batch_update(
            tree.children_discounts, step.discount, parent_index, action),
        parents=batch_update(tree.parents, parent_index, next_node_index),
        action_from_parent=batch_update(
            tree.action_from_parent, action, next_node_index),
    )


# ----------------------------------------------------------------------------
# backward — DUAL-USE EDIT: explicit W↔L flip in WDL mode, scalar otherwise.
# ----------------------------------------------------------------------------


def _make_backward(is_wdl: bool):
    @jax.vmap
    def backward(tree: Tree[T], leaf_index: chex.Numeric) -> Tree[T]:
        def cond_fun(loop_state):
            _, _, index = loop_state
            return index != Tree.ROOT_INDEX

        def body_fun(loop_state):
            tree, leaf_value, index = loop_state
            parent = tree.parents[index]
            count = tree.node_visits[parent]
            action = tree.action_from_parent[index]
            reward = tree.children_rewards[parent, action]
            discount = tree.children_discounts[parent, action]
            if is_wdl:
                # DUAL-USE EDIT: flip child-perspective WDL into parent perspective,
                # then apply terminal-cutoff via discount ∈ {0, 1}.
                flipped = flip_wdl(leaf_value)
                leaf_value = reward + discount[..., None] * flipped
            else:
                # Verbatim from upstream: discount = -1 implements scalar perspective flip.
                leaf_value = reward + discount * leaf_value
            parent_value = (
                tree.node_values[parent] * count + leaf_value) / (count + 1.0)
            children_values = tree.node_values[index]
            children_counts = tree.children_visits[parent, action] + 1

            tree = tree.replace(
                node_values=update(tree.node_values, parent_value, parent),
                node_visits=update(tree.node_visits, count + 1, parent),
                children_values=update(
                    tree.children_values, children_values, parent, action),
                children_visits=update(
                    tree.children_visits, children_counts, parent, action),
            )
            return tree, leaf_value, parent

        leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)
        loop_state = (tree, tree.node_values[leaf_index], leaf_index)
        tree, _, _ = jax.lax.while_loop(cond_fun, body_fun, loop_state)
        return tree

    return backward


# ----------------------------------------------------------------------------
# simulate — verbatim from upstream search.py.
# ----------------------------------------------------------------------------


class _SimulationState(Tuple):
    pass


@functools.partial(jax.vmap, in_axes=[0, 0, None, None], out_axes=0)
def _simulate(
    rng_key: chex.PRNGKey,
    tree: Tree,
    action_selection_fn,
    max_depth: int,
):
    def cond_fun(state):
        return state[5]  # is_continuing

    def body_fun(state):
        rng_key, node_index, action, next_node_index, depth, _ = state
        node_index = next_node_index
        rng_key, action_selection_key = jax.random.split(rng_key)
        action = action_selection_fn(action_selection_key, tree, node_index, depth)
        next_node_index = tree.children_index[node_index, action]
        depth = depth + 1
        is_before_depth_cutoff = depth < max_depth
        is_visited = next_node_index != Tree.UNVISITED
        is_continuing = jnp.logical_and(is_visited, is_before_depth_cutoff)
        return (rng_key, node_index, action, next_node_index, depth, is_continuing)

    node_index = jnp.array(Tree.ROOT_INDEX, dtype=jnp.int32)
    depth = jnp.zeros((), dtype=tree.children_prior_logits.dtype)
    initial_state = (
        rng_key,
        jnp.asarray(Tree.NO_PARENT, dtype=jnp.int32),
        jnp.asarray(Tree.NO_PARENT, dtype=jnp.int32),
        node_index,
        depth,
        jnp.array(True),
    )
    end_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)
    _, end_node_index, end_action, _, _, _ = end_state
    return end_node_index, end_action


# ----------------------------------------------------------------------------
# search — verbatim orchestrator from upstream, with mode threading.
# ----------------------------------------------------------------------------


def search(
    params,
    rng_key: chex.PRNGKey,
    *,
    root: RootFnOutput,
    recurrent_fn,
    root_action_selection_fn,
    interior_action_selection_fn,
    num_simulations: int,
    max_depth: Optional[int] = None,
    invalid_actions: Optional[chex.Array] = None,
    extra_data: Any = None,
    loop_fn=jax.lax.fori_loop,
    value_mode: ValueMode = "scalar",
) -> Tree:
    """DUAL-USE EDIT: adds value_mode dispatch over expand/backward."""
    is_wdl = value_mode == "wdl"
    value_shape: Tuple[int, ...] = (WDL_DIM,) if is_wdl else ()

    action_selection_fn = _action_selection.switching_action_selection_wrapper(
        root_action_selection_fn=root_action_selection_fn,
        interior_action_selection_fn=interior_action_selection_fn,
    )
    batch_size = root.value.shape[0]
    batch_range = jnp.arange(batch_size)
    if max_depth is None:
        max_depth = num_simulations
    if invalid_actions is None:
        invalid_actions = jnp.zeros_like(root.prior_logits)

    backward = _make_backward(is_wdl)

    def body_fun(sim, loop_state):
        rng_key, tree = loop_state
        rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3)
        simulate_keys = jax.random.split(simulate_key, batch_size)
        parent_index, action = _simulate(
            simulate_keys, tree, action_selection_fn, max_depth)
        next_node_index = tree.children_index[batch_range, parent_index, action]
        next_node_index = jnp.where(
            next_node_index == Tree.UNVISITED, sim + 1, next_node_index)
        tree = _expand(
            params, expand_key, tree, recurrent_fn, parent_index,
            action, next_node_index, value_shape=value_shape)
        tree = backward(tree, next_node_index)
        return rng_key, tree

    tree = _instantiate_tree_from_root(
        root, num_simulations,
        root_invalid_actions=invalid_actions,
        extra_data=extra_data)
    _, tree = loop_fn(0, num_simulations, body_fun, (rng_key, tree))
    return tree


# ----------------------------------------------------------------------------
# qtransform_completed_by_mix_value — DUAL-USE EDIT: WDL collapse hook.
# Verbatim from mctx._src.qtransforms.qtransform_completed_by_mix_value with
# one extra preamble that collapses WDL → scalar U if needed.
# ----------------------------------------------------------------------------


def qtransform_completed_by_mix_value(
    tree: Tree,
    node_index: chex.Numeric,
    *,
    value_scale: chex.Numeric = 0.1,
    maxvisit_init: chex.Numeric = 50.0,
    rescale_values: bool = True,
    use_mixed_value: bool = True,
    epsilon: chex.Numeric = 1e-8,
) -> chex.Array:
    chex.assert_shape(node_index, ())
    qvalues = tree.qvalues(node_index)
    raw_value = tree.raw_values[node_index]
    visit_counts = tree.children_visits[node_index]
    # DUAL-USE EDIT: collapse WDL to scalar U for selection-time scoring.
    if qvalues.ndim > visit_counts.ndim:
        qvalues = _wdl_to_scalar(qvalues)
        raw_value = _wdl_to_scalar(raw_value)

    prior_probs = jax.nn.softmax(tree.children_prior_logits[node_index])
    if use_mixed_value:
        value = _qtransforms._compute_mixed_value(
            raw_value, qvalues=qvalues, visit_counts=visit_counts,
            prior_probs=prior_probs)
    else:
        value = raw_value
    completed_qvalues = _qtransforms._complete_qvalues(
        qvalues, visit_counts=visit_counts, value=value)

    if rescale_values:
        completed_qvalues = _qtransforms._rescale_qvalues(completed_qvalues, epsilon)
    maxvisit = jnp.max(visit_counts, axis=-1)
    visit_scale = maxvisit_init + maxvisit
    return visit_scale * value_scale * completed_qvalues


# ----------------------------------------------------------------------------
# dirichlet_gumbel_policy — vendored gumbel_muzero_policy with value_mode.
# ----------------------------------------------------------------------------


def _mask_invalid_actions(logits, invalid_actions):
    if invalid_actions is None:
        return logits
    chex.assert_equal_shape([logits, invalid_actions])
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    min_logit = jnp.finfo(logits.dtype).min
    return jnp.where(invalid_actions, min_logit, logits)


def dirichlet_gumbel_policy(
    params,
    rng_key: chex.PRNGKey,
    root: RootFnOutput,
    recurrent_fn,
    num_simulations: int,
    invalid_actions: Optional[chex.Array] = None,
    max_depth: Optional[int] = None,
    loop_fn=jax.lax.fori_loop,
    *,
    qtransform=qtransform_completed_by_mix_value,
    max_num_considered_actions: int = 16,
    gumbel_scale: chex.Numeric = 1.0,
    value_mode: ValueMode = "scalar",
) -> PolicyOutput[_action_selection.GumbelMuZeroExtraData]:
    """Drop-in replacement for mctx.gumbel_muzero_policy with optional WDL mode.

    In `value_mode="scalar"` the behavior is byte-identical to upstream.
    In `value_mode="wdl"` the tree carries [3]-shaped values and the qtransform
    collapses to scalar U at action-selection time.
    """
    root = root.replace(
        prior_logits=_mask_invalid_actions(root.prior_logits, invalid_actions))

    rng_key, gumbel_rng = jax.random.split(rng_key)
    gumbel = gumbel_scale * jax.random.gumbel(
        gumbel_rng, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype)

    extra_data = _action_selection.GumbelMuZeroExtraData(root_gumbel=gumbel)
    search_tree = search(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        root_action_selection_fn=functools.partial(
            _action_selection.gumbel_muzero_root_action_selection,
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
            qtransform=qtransform,
        ),
        interior_action_selection_fn=functools.partial(
            _action_selection.gumbel_muzero_interior_action_selection,
            qtransform=qtransform,
        ),
        num_simulations=num_simulations,
        max_depth=max_depth,
        invalid_actions=invalid_actions,
        extra_data=extra_data,
        loop_fn=loop_fn,
        value_mode=value_mode,
    )
    summary = search_tree.summary()

    considered_visit = jnp.max(summary.visit_counts, axis=-1, keepdims=True)
    completed_qvalues = jax.vmap(qtransform, in_axes=[0, None])(
        search_tree, search_tree.ROOT_INDEX)
    from mctx._src import seq_halving as _seq_halving
    to_argmax = _seq_halving.score_considered(
        considered_visit, gumbel, root.prior_logits, completed_qvalues,
        summary.visit_counts)
    action = _action_selection.masked_argmax(to_argmax, invalid_actions)

    completed_search_logits = _mask_invalid_actions(
        root.prior_logits + completed_qvalues, invalid_actions)
    action_weights = jax.nn.softmax(completed_search_logits)
    return PolicyOutput(
        action=action,
        action_weights=action_weights,
        search_tree=search_tree,
    )


# Alias for drop-in compatibility with `import mctx` call sites.
gumbel_muzero_policy = dirichlet_gumbel_policy
