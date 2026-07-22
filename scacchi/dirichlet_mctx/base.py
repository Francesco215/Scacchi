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

    ``value`` is the child state's Dirichlet value prior and
    ``action_values`` contains its action-value priors. ``invalid_actions``
    explicitly identifies actions unavailable from the child state.
    ``terminal_outcome`` is an exact categorical index from the perspective
    of ``to_play`` for a terminal child and ``NO_OUTCOME`` otherwise.
    """

    value: jax.Array
    action_values: jax.Array
    invalid_actions: jax.Array
    terminal_outcome: jax.Array
    to_play: jax.Array


RecurrentFn = Callable[
    [Params, chex.PRNGKey, Action, RecurrentState],
    tuple[RecurrentFnOutput, RecurrentState],
]
NodeEvaluationFn = Callable[
    [Params, chex.PRNGKey, RecurrentState],
    jax.Array,
]


@chex.dataclass(frozen=True)
class RootFnOutput:
    """Network output used to initialize a batch of Dirichlet search roots.

    ``value`` has shape ``[B, O]`` and ``action_values`` has shape
    ``[B, A, O]``. They contain Dirichlet parameters, not scalar means.
    ``terminal_outcome`` is an exact outcome from the perspective of
    ``to_play`` for a terminal root and ``NO_OUTCOME`` otherwise.
    """

    prior_logits: jax.Array
    value: jax.Array
    action_values: jax.Array
    embedding: RecurrentState
    terminal_outcome: jax.Array
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
