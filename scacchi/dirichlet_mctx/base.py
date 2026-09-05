"""Core types for the lightweight Dirichlet tree-search backend."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Generic, NamedTuple, TypeVar

import chex
from jaxtyping import Array, Bool, Float, Int, Int8, Int32, Key, PyTree, Shaped, UInt32

if TYPE_CHECKING:
    from .tree import PosteriorUpdate, PosteriorUpdateContext, Tree, UnbatchedTree


Params = PyTree[Shaped[Array, "*?param_axes"], "Params"]
Action = Int32[Array, "batch"]
RecurrentState = PyTree[Shaped[Array, "batch *?state_axes"], "State"]
StoredRecurrentState = PyTree[Shaped[Array, "batch node *?state_axes"], "State"]
UnbatchedRecurrentState = PyTree[Shaped[Array, "*?state_axes"], "State"]
PRNGKey = Key[Array, ""] | UInt32[Array, "2"]
BatchedPRNGKey = Key[Array, "batch"] | UInt32[Array, "batch 2"]


class Simulation(NamedTuple):
    """Selected edge for one lane or a batch of lanes under ``vmap``."""

    parent_index: Int32[Array, "*batch"]
    action: Int32[Array, "*batch"]
    active: Bool[Array, "*batch"]


class _SimulationState(NamedTuple):
    rng_key: PRNGKey
    node_index: Int32[Array, ""]
    action: Int32[Array, ""]
    next_node_index: Int32[Array, ""]
    depth: Int32[Array, ""]
    is_continuing: Bool[Array, ""]


type _BackwardState = tuple[PRNGKey, Tree, Int32[Array, "batch"], Bool[Array, "batch"], Bool[Array, "batch"]]
type _SearchState = tuple[PRNGKey, Tree]


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
    terminal_outcome: Int8[Array, "batch"]
    to_play: Int32[Array, "batch"]


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
    terminal_outcome: Int8[Array, "batch"]
    to_play: Int32[Array, "batch"]


T = TypeVar("T")
LoopState = TypeVar("LoopState")


@chex.dataclass(frozen=True)
class PolicyOutput(Generic[T]):
    """The selected action, policy target, and completed search tree."""

    action: Int32[Array, "batch"]
    action_weights: Float[Array, "batch action"]
    search_tree: T


ActionSelectionFn = Callable[[PRNGKey, "UnbatchedTree", Int32[Array, ""]], Int32[Array, ""]]
PosteriorUpdateFn = Callable[[PRNGKey, "PosteriorUpdateContext"], "PosteriorUpdate"]
LoopFn = Callable[[int, int, Callable[[Int[Array, ""], LoopState], LoopState], LoopState], LoopState]
