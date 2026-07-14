"""The minimal batched tree needed by Dirichlet Thompson search."""

from __future__ import annotations

from typing import Any, ClassVar

import chex
import jax
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class Posterior:
    """Action posterior split into its base prior and search evidence.

    The leading dimensions are deliberately generic.  Stored tree posteriors
    have shape ``[B, N, A, O]``; a node view has shape ``[B, A, O]``.
    """

    base: jax.Array
    evidence: jax.Array
    explored: jax.Array

    @property
    def alpha(self) -> jax.Array:
        return self.base + self.evidence


@chex.dataclass(frozen=True)
class SearchSummary:
    visit_counts: jax.Array
    visit_probs: jax.Array
    alpha: jax.Array
    evidence: jax.Array
    explored: jax.Array


@chex.dataclass(frozen=True)
class NodeView:
    """A gathered node, suitable for a configurable posterior update."""

    index: jax.Array
    embedding: Any
    value: jax.Array
    outcome: jax.Array
    evidence_weight: jax.Array
    to_play: jax.Array
    terminal: jax.Array
    invalid_actions: jax.Array
    visit_counts: jax.Array
    posterior: Posterior


@chex.dataclass(frozen=True)
class ChildrenView:
    """All action-indexed children of a node.

    Unvisited entries contain safe placeholder data and must be ignored using
    ``visited``.  Keeping the action axis intact makes update rules easy to
    express with ordinary JAX array operations.
    """

    nodes: NodeView
    visited: jax.Array


@chex.dataclass(frozen=True)
class PosteriorUpdateContext:
    """Everything a node-local posterior rule may inspect during backup.

    ``outcome`` is the current simulation's leaf outcome, already aligned to
    ``node.to_play``.  Children are presented after any deeper path node has
    been updated, so a rule may derive a parent from child posteriors.
    """

    node: NodeView
    children: ChildrenView
    action: jax.Array
    outcome: jax.Array
    evidence_weight: jax.Array
    is_leaf_edge: jax.Array
    active: jax.Array


@chex.dataclass(frozen=True)
class Tree:
    """Fixed-capacity state of a batch of Thompson-search trees.

    Unlike MCTX's scalar-value tree, this stores no rewards, discounts, or
    running scalar value means.  Each live node owns an action posterior, so
    the same Thompson rule can select actions at the root and in the interior.
    Node evaluation data is retained to make posterior backup replaceable.
    """

    parents: jax.Array  # [B, N]
    action_from_parent: jax.Array  # [B, N]
    children_index: jax.Array  # [B, N, A]
    children_visits: jax.Array  # [B, N, A]
    node_to_play: jax.Array  # [B, N]
    node_terminal: jax.Array  # [B, N]
    node_values: jax.Array  # [B, N, O]
    node_outcomes: jax.Array  # [B, N, O]
    node_evidence_weights: jax.Array  # [B, N]
    invalid_actions: jax.Array  # [B, N, A]
    action_posteriors: Posterior  # [B, N, A, O], [B, N, A]
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
    def root_posterior(self) -> Posterior:
        return Posterior(
            base=self.action_posteriors.base[..., self.ROOT_INDEX, :, :],
            evidence=self.action_posteriors.evidence[..., self.ROOT_INDEX, :, :],
            explored=self.action_posteriors.explored[..., self.ROOT_INDEX, :],
        )

    @property
    def root_invalid_actions(self) -> jax.Array:
        return self.invalid_actions[..., self.ROOT_INDEX, :]

    def summary(self) -> SearchSummary:
        posterior = self.root_posterior
        visit_counts = self.children_visits[:, self.ROOT_INDEX].astype(
            posterior.base.dtype
        )
        total = jnp.sum(visit_counts, axis=-1, keepdims=True)
        legal = ~self.root_invalid_actions
        legal_count = jnp.sum(legal, axis=-1, keepdims=True)
        fallback = legal.astype(visit_counts.dtype) / jnp.maximum(legal_count, 1)
        visit_probs = jnp.where(
            total > 0,
            visit_counts / jnp.maximum(total, 1),
            fallback,
        )
        return SearchSummary(
            visit_counts=visit_counts,
            visit_probs=visit_probs,
            alpha=posterior.alpha,
            evidence=posterior.evidence,
            explored=posterior.explored,
        )
