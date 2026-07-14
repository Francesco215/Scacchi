"""Core types for the lightweight Dirichlet tree-search backend."""

from __future__ import annotations

from typing import Any, Callable, Generic, TypeVar

import chex
import jax


Params = chex.ArrayTree
Action = jax.Array
RecurrentState = Any


@chex.dataclass(frozen=True)
class RecurrentFnOutput:
    """Evaluation returned after taking one tree edge.

    All outcome distributions use the perspective of ``to_play``.
    ``value`` is the child state's Dirichlet value prior and
    ``action_values`` contains its action-value priors. ``leaf_value`` is the
    full Dirichlet message used when this expansion ends a simulation: the
    value prior for a new non-terminal leaf, or a concentrated outcome for a
    terminal leaf. ``outcome`` and ``evidence_weight`` remain available to the
    Dirichlet-Gumbel adapter.
    """

    prior_logits: jax.Array
    value: jax.Array
    action_values: jax.Array
    leaf_value: jax.Array
    outcome: jax.Array
    evidence_weight: jax.Array
    terminal: jax.Array
    to_play: jax.Array


RecurrentFn = Callable[
    [Params, chex.PRNGKey, Action, RecurrentState],
    tuple[RecurrentFnOutput, RecurrentState],
]


@chex.dataclass(frozen=True)
class RootFnOutput:
    """Network output used to initialize a batch of Dirichlet search roots.

    ``value`` has shape ``[B, O]`` and ``action_values`` has shape
    ``[B, A, O]``.  They contain Dirichlet parameters, not scalar means.
    """

    prior_logits: jax.Array
    value: jax.Array
    action_values: jax.Array
    embedding: RecurrentState
    terminal: jax.Array
    to_play: jax.Array


T = TypeVar("T")


@chex.dataclass(frozen=True)
class PolicyOutput(Generic[T]):
    """The selected action, policy target, and completed search tree."""

    action: jax.Array
    action_weights: jax.Array
    search_tree: T


ActionSelectionFn = Callable[[chex.PRNGKey, Any, jax.Array], jax.Array]
PosteriorUpdateFn = Callable[[chex.PRNGKey, Any], Any]
LoopFn = Callable[[int, int, Callable[[Any, Any], Any], Any], Any]
