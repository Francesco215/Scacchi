"""Core types for the lightweight Dirichlet tree-search backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generic, TypeVar

import chex
from jaxtyping import Array, Bool, Float, Int, Key, PRNGKeyArray, PyTree, Shaped, UInt32

if TYPE_CHECKING:
    from .tree import PosteriorUpdate, PosteriorUpdateContext, Tree


Params = PyTree[Shaped[Array, "..."]]
Action = Int[Array, "batch"]
RecurrentState = PyTree[Shaped[Array, "batch ..."]]
PRNGKey = PRNGKeyArray
BatchedPRNGKey = Key[Array, "batch"] | UInt32[Array, "batch 2"]


@chex.dataclass(frozen=True)
class RecurrentFnOutput:
    """Evaluation returned after taking one tree edge.

    ``value`` is the child state's Dirichlet value prior and
    ``action_values`` contains its action-value priors. ``invalid_actions``
    explicitly identifies actions unavailable from the child state.
    ``terminal_outcome`` is an exact categorical index from the perspective
    of ``to_play`` for a terminal child and ``NO_OUTCOME`` otherwise.
    """

    value: Float[Array, "batch outcome"]
    action_values: Float[Array, "batch action outcome"]
    invalid_actions: Bool[Array, "batch action"]
    terminal_outcome: Int[Array, "batch"]
    to_play: Int[Array, "batch"]


RecurrentFn = Callable[[Params, PRNGKey, Action, RecurrentState], tuple[RecurrentFnOutput, RecurrentState]]


@chex.dataclass(frozen=True)
class RootFnOutput:
    """Network output used to initialize a batch of Dirichlet search roots.

    ``value`` has shape ``[B, O]`` and ``action_values`` has shape
    ``[B, A, O]``. They contain Dirichlet parameters, not scalar means.
    ``terminal_outcome`` is an exact outcome from the perspective of
    ``to_play`` for a terminal root and ``NO_OUTCOME`` otherwise.
    """

    prior_logits: Float[Array, "batch action"]
    value: Float[Array, "batch outcome"]
    action_values: Float[Array, "batch action outcome"]
    embedding: RecurrentState
    terminal_outcome: Int[Array, "batch"]
    to_play: Int[Array, "batch"]


T = TypeVar("T")
LoopState = TypeVar("LoopState")


@chex.dataclass(frozen=True)
class PolicyOutput(Generic[T]):
    """The selected action, policy target, and completed search tree."""

    action: Int[Array, "batch"]
    action_weights: Float[Array, "batch action"]
    search_tree: T


ActionSelectionFn = Callable[[PRNGKey, "Tree", Int[Array, ""]], Int[Array, ""]]
PosteriorUpdateFn = Callable[[PRNGKey, "PosteriorUpdateContext"], "PosteriorUpdate"]
LoopFn = Callable[[int, int, Callable[[int, LoopState], LoopState], LoopState], LoopState]
