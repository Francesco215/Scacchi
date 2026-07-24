"""Shared outcome semantics for search, readout, and neural targets."""

from __future__ import annotations

import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Int8, Int32


NO_OUTCOME = -1
NO_DISTANCE = -1


def flip_outcome(outcome: Float[Array, "*batch outcome"]) -> Float[Array, "*batch outcome"]:
    return outcome[..., ::-1]


def align_outcome(outcome: Float[Array, "*batch outcome"], source_player: Int32[Array, "*batch"], target_player: Int32[Array, "*batch"]) -> Float[Array, "*batch outcome"]:
    return jnp.where((source_player == target_player)[..., None], outcome, flip_outcome(outcome))


def align_categorical_outcome(outcome: Int8[Array, "*batch"], source_player: Int32[Array, "*batch"], target_player: Int32[Array, "*batch"], num_outcomes: int) -> Int8[Array, "*batch"]:
    """Align exact outcome indices while preserving the unresolved sentinel."""

    outcome = jnp.asarray(outcome, dtype=jnp.int8)
    flipped = (int(num_outcomes) - 1 - outcome).astype(jnp.int8)
    aligned = jnp.where(source_player == target_player, outcome, flipped)
    return jnp.where(outcome == int(NO_OUTCOME), jnp.asarray(int(NO_OUTCOME), dtype=jnp.int8), aligned)


def outcome_mean(alpha: Float[Array, "*batch outcome"]) -> Float[Array, "*batch outcome"]:
    return alpha / jnp.sum(alpha, axis=-1, keepdims=True)


def outcome_utility(outcome: Float[Array, "*batch outcome"]) -> Float[Array, "*batch"]:
    return outcome[..., -1] - outcome[..., 0]


def categorical_utility(outcome: Int[Array, "*batch"], num_outcomes: int) -> Float[Array, "*batch"]:
    """Return exact scalar utility for a categorical outcome index."""

    outcome = jnp.asarray(outcome)
    dtype = jnp.result_type(outcome, jnp.float32)
    return jnp.where(outcome == int(num_outcomes) - 1, jnp.asarray(1.0, dtype=dtype), jnp.where(outcome == 0, jnp.asarray(-1.0, dtype=dtype), jnp.asarray(0.0, dtype=dtype)))


__all__ = [
    "NO_DISTANCE",
    "NO_OUTCOME",
    "align_categorical_outcome",
    "align_outcome",
    "categorical_utility",
    "flip_outcome",
    "outcome_mean",
    "outcome_utility",
]
