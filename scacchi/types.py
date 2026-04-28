"""Shared typed containers."""

from __future__ import annotations

from typing import NamedTuple

import jax


class SelfplayBatch(NamedTuple):
    observation: jax.Array
    action_weights: jax.Array
    reward: jax.Array
    discount: jax.Array
    terminated: jax.Array


class TrainingBatch(NamedTuple):
    observation: jax.Array
    policy_target: jax.Array
    value_target: jax.Array
    value_mask: jax.Array
