"""Shared typed containers."""

from __future__ import annotations

from typing import Any, NamedTuple

import optax
from flax import nnx
from jaxtyping import Array, Bool, Float


class SelfplayBatch(NamedTuple):
    observation: Float[Array, "time batch height width channels"]
    action_weights: Float[Array, "time batch action"]
    reward: Float[Array, "time batch"]
    discount: Float[Array, "time batch"]
    terminated: Bool[Array, "time batch"]


class TrainingBatch(NamedTuple):
    observation: Float[Array, "batch height width channels"]
    policy_target: Float[Array, "batch action"]
    value_target: Float[Array, "batch"]
    value_mask: Bool[Array, "batch"]


class TrainState(NamedTuple):
    params: nnx.State
    opt_state: optax.OptState


ModelGraphDef = Any
