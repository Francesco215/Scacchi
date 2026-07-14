"""Fixed-capacity data for Dirichlet Thompson tree search."""

from __future__ import annotations

from typing import Any, ClassVar

import chex
import jax
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class NodePosterior:
    """Search state repaired for one node or stored for every node.

    ``action_alpha`` is the edge message ``B`` from the Tic-Tac-Toe model and
    contains the Q fallback until ``action_count`` becomes positive.
    ``action_count`` is the structural edge count ``R``; because every direct
    or child message has positive ``R``, it also represents the demo's ``m``
    bit without a duplicate boolean table. ``value_alpha`` is the cached state
    posterior. Leading dimensions are generic: the tree adds ``[B, N]`` and a
    gathered node adds only ``[B]``.
    """

    action_alpha: jax.Array
    action_count: jax.Array
    value_alpha: jax.Array


@chex.dataclass(frozen=True)
class NodeView:
    """The current node passed to a configurable posterior repair rule."""

    index: jax.Array
    embedding: Any
    value_prior: jax.Array
    to_play: jax.Array
    terminal: jax.Array
    invalid_actions: jax.Array
    posterior: NodePosterior


@chex.dataclass(frozen=True)
class ChildrenView:
    """Action-indexed child summaries needed to repair their parent.

    ``embedding_table`` is left ungathered so large game states are not copied
    across the full action axis. A custom rule that truly needs them can gather
    ``embedding_table[batch, index]`` using ``visited`` and safe indices.
    """

    index: jax.Array
    visited: jax.Array
    embedding_table: Any
    value_prior: jax.Array
    value_alpha: jax.Array
    count: jax.Array
    to_play: jax.Array
    terminal: jax.Array


@chex.dataclass(frozen=True)
class LeafView:
    """The selected leaf edge evaluated by the current simulation.

    ``active`` is true only for the deepest path node.  It is false for every
    ancestor, allowing the same posterior-update callback to own both the
    direct leaf message and all child-to-parent repairs.
    """

    action: jax.Array
    value_alpha: jax.Array
    to_play: jax.Array
    active: jax.Array


@chex.dataclass(frozen=True)
class PosteriorUpdateContext:
    """Per-node input to ``PosteriorUpdateFn`` during bottom-up backup.

    Every invocation has the same node, children, and selected-leaf contract.
    The callback is responsible for the deepest direct ``B/m/R`` write as well
    as child repairs, so replacing it replaces the complete backward rule.
    Ancestors observe the freshly repaired value posterior of every child.
    """

    node: NodeView
    children: ChildrenView
    leaf: LeafView
    active: jax.Array


@chex.dataclass(frozen=True)
class SearchSummary:
    visit_counts: jax.Array
    alpha: jax.Array
    value_alpha: jax.Array


@chex.dataclass(frozen=True)
class Tree:
    """A small JAX tree specialized for Dirichlet message passing.

    The topology is MCTX-like, but the statistics mirror the Tic-Tac-Toe
    implementation: every edge has ``(B, R)`` (with ``m == (R > 0)``) and every
    node caches a full value Dirichlet. ``node_n_down`` caches the edge-count
    reduction so gathering child summaries stays linear in the action count.
    The posterior rule repairs these data bottom-up.
    """

    parents: jax.Array  # [B, N]
    children_index: jax.Array  # [B, N, A]
    node_to_play: jax.Array  # [B, N]
    node_terminal: jax.Array  # [B, N]
    node_value_priors: jax.Array  # [B, N, O]
    node_n_down: jax.Array  # [B, N]
    invalid_actions: jax.Array  # [B, N, A]
    posterior: NodePosterior
    embeddings: Any  # pytree with leaves [B, N, ...]

    ROOT_INDEX: ClassVar[int] = 0
    NO_PARENT: ClassVar[int] = -1
    UNVISITED: ClassVar[int] = -1

    @property
    def num_actions(self) -> int:
        return self.children_index.shape[-1]

    @property
    def num_simulations(self) -> int:
        return self.children_index.shape[1] - 1

    @property
    def root_posterior(self) -> NodePosterior:
        return NodePosterior(
            action_alpha=self.posterior.action_alpha[:, self.ROOT_INDEX],
            action_count=self.posterior.action_count[:, self.ROOT_INDEX],
            value_alpha=self.posterior.value_alpha[:, self.ROOT_INDEX],
        )

    @property
    def root_invalid_actions(self) -> jax.Array:
        return self.invalid_actions[:, self.ROOT_INDEX]

    def summary(self) -> SearchSummary:
        posterior = self.root_posterior
        visit_counts = posterior.action_count.astype(posterior.action_alpha.dtype)
        child_index = self.children_index[:, self.ROOT_INDEX]
        visited = child_index != self.UNVISITED
        safe_child = jnp.where(visited, child_index, self.ROOT_INDEX)
        batch = jnp.arange(self.parents.shape[0])[:, None]
        child_prior = self.node_value_priors[batch, safe_child]
        child_player = self.node_to_play[batch, safe_child]
        root_player = self.node_to_play[:, self.ROOT_INDEX, None]
        child_prior = jnp.where(
            (child_player == root_player)[..., None],
            child_prior,
            child_prior[..., ::-1],
        )
        fallback = jnp.where(
            visited[..., None],
            child_prior,
            posterior.action_alpha,
        )
        alpha = jnp.where(
            (posterior.action_count > 0)[..., None],
            posterior.action_alpha,
            fallback,
        )
        return SearchSummary(
            visit_counts=visit_counts,
            alpha=alpha,
            value_alpha=posterior.value_alpha,
        )
