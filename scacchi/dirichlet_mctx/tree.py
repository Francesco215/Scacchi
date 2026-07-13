"""The minimal batched tree needed by Dirichlet Thompson search."""

from __future__ import annotations

from typing import Any, ClassVar

import chex
import jax
import jax.numpy as jnp


@chex.dataclass(frozen=True)
class Posterior:
    """Root-action posterior split into its base prior and search evidence."""

    base: jax.Array  # [B, A, O]
    evidence: jax.Array  # [B, A, O]
    explored: jax.Array  # [B, A]

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
class Tree:
    """Fixed-capacity state of a batch of Thompson-search trees.

    Unlike MCTX's scalar-value tree, this stores no rewards, discounts, raw
    values, running value means, or duplicate node visit counts.  Search only
    needs topology, policy priors, edge visits, state embeddings, player and
    terminal metadata, and the root Dirichlet posterior.
    """

    parents: jax.Array  # [B, N]
    action_from_parent: jax.Array  # [B, N]
    children_index: jax.Array  # [B, N, A]
    children_prior_logits: jax.Array  # [B, N, A]
    children_visits: jax.Array  # [B, N, A]
    node_to_play: jax.Array  # [B, N]
    node_terminal: jax.Array  # [B, N]
    embeddings: Any  # pytree with leaves [B, N, ...]
    root_invalid_actions: jax.Array  # [B, A]
    posterior: Posterior

    ROOT_INDEX: ClassVar[int] = 0
    NO_PARENT: ClassVar[int] = -1
    UNVISITED: ClassVar[int] = -1

    @property
    def num_actions(self) -> int:
        return self.children_index.shape[-1]

    @property
    def num_simulations(self) -> int:
        return self.children_index.shape[1] - 1

    def summary(self) -> SearchSummary:
        visit_counts = self.children_visits[:, self.ROOT_INDEX].astype(
            self.posterior.base.dtype
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
            alpha=self.posterior.alpha,
            evidence=self.posterior.evidence,
            explored=self.posterior.explored,
        )
