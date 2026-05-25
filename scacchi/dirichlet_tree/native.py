from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import gammaln


TARGET_PAD = np.int8(0)
TARGET_DIRICHLET = np.int8(1)
TARGET_CATEGORICAL = np.int8(2)

OUTCOME_LOSS = np.int8(0)
OUTCOME_DRAW = np.int8(1)
OUTCOME_WIN = np.int8(2)
NO_OUTCOME = np.int8(-1)
NO_DISTANCE = np.int32(-1)
INF_DISTANCE = np.int32(2**30)


@dataclass(frozen=True, slots=True)
class NativeTarget:
    kind: int
    alpha: np.ndarray | None = None
    outcome: int = int(NO_OUTCOME)
    distance: int = int(NO_DISTANCE)
    weight: float = 1.0


def dirichlet_target(alpha: np.ndarray) -> NativeTarget:
    return NativeTarget(kind=int(TARGET_DIRICHLET), alpha=_positive(alpha))


def categorical_target(outcome: int, distance: int) -> NativeTarget:
    return NativeTarget(
        kind=int(TARGET_CATEGORICAL),
        alpha=None,
        outcome=int(outcome),
        distance=int(distance),
    )


def empty_q_native(
    beta_q: Any,
    *,
    kind: int = int(TARGET_DIRICHLET),
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    beta_q = jnp.asarray(beta_q)
    target_shape = beta_q.shape[:-1]
    target_kind = jnp.full(target_shape, int(kind), dtype=jnp.int8)
    target_weight = jnp.ones(target_shape, dtype=beta_q.dtype)
    target_outcome = jnp.full(target_shape, int(NO_OUTCOME), dtype=jnp.int8)
    target_distance = jnp.full(target_shape, int(NO_DISTANCE), dtype=jnp.int32)
    return target_kind, target_weight, target_outcome, target_distance


def empty_v_native(
    beta_v: Any,
    *,
    kind: int = int(TARGET_DIRICHLET),
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    beta_v = jnp.asarray(beta_v)
    target_shape = beta_v.shape[:-1]
    target_kind = jnp.full(target_shape, int(kind), dtype=jnp.int8)
    target_weight = jnp.ones(target_shape, dtype=beta_v.dtype)
    target_outcome = jnp.full(target_shape, int(NO_OUTCOME), dtype=jnp.int8)
    target_distance = jnp.full(target_shape, int(NO_DISTANCE), dtype=jnp.int32)
    return target_kind, target_weight, target_outcome, target_distance


def native_fields_from_beta(
    beta_q: Any,
    beta_v: Any,
) -> dict[str, jax.Array]:
    q_kind, q_weight, q_outcome, q_distance = empty_q_native(beta_q)
    v_kind, v_weight, v_outcome, v_distance = empty_v_native(beta_v)
    return {
        "q_target_kind": q_kind,
        "q_target_weight": q_weight,
        "q_target_outcome": q_outcome,
        "q_target_distance": q_distance,
        "v_target_kind": v_kind,
        "v_target_weight": v_weight,
        "v_target_outcome": v_outcome,
        "v_target_distance": v_distance,
    }


def align_outcome(outcome: int, source_player: int, target_player: int) -> int:
    outcome = int(outcome)
    if outcome < 0 or int(source_player) == int(target_player):
        return outcome
    if outcome == int(OUTCOME_LOSS):
        return int(OUTCOME_WIN)
    if outcome == int(OUTCOME_WIN):
        return int(OUTCOME_LOSS)
    return int(OUTCOME_DRAW)


def flip_outcome_index(outcome: int) -> int:
    return align_outcome(outcome, 0, 1)


def outcome_utility_index(outcome: int | np.ndarray) -> np.ndarray:
    outcome_arr = np.asarray(outcome)
    return np.where(
        outcome_arr == int(OUTCOME_WIN),
        1.0,
        np.where(outcome_arr == int(OUTCOME_LOSS), -1.0, 0.0),
    )


def categorical_utility_np(outcome: int | np.ndarray, num_outcomes: int) -> np.ndarray:
    outcome_arr = np.asarray(outcome)
    return np.where(
        outcome_arr == int(num_outcomes) - 1,
        1.0,
        np.where(outcome_arr == 0, -1.0, 0.0),
    )


def categorical_proxy_np(
    outcome: int,
    num_outcomes: int,
    *,
    epsilon: float = 1e-6,
) -> np.ndarray:
    proxy = np.full((int(num_outcomes),), float(epsilon), dtype=np.float32)
    if 0 <= int(outcome) < int(num_outcomes):
        proxy[int(outcome)] = np.float32(
            1.0 - (float(num_outcomes) - 1.0) * float(epsilon)
        )
    return _positive(proxy)


def native_policy_target_np(
    rng: np.random.Generator,
    alpha: np.ndarray,
    legal_action_mask: np.ndarray,
    target_kind: np.ndarray | None,
    target_outcome: np.ndarray | None,
    num_samples: int,
) -> np.ndarray:
    alpha = _positive(alpha)
    legal = np.asarray(legal_action_mask, dtype=bool)
    target = np.zeros((alpha.shape[-2],), dtype=np.float32)
    legal_actions = np.flatnonzero(legal)
    if legal_actions.size == 0:
        return target
    if int(num_samples) <= 0:
        raise ValueError(f"num_samples must be positive, got {num_samples}")

    if target_kind is None:
        target_kind = np.full((alpha.shape[-2],), int(TARGET_DIRICHLET), dtype=np.int8)
    if target_outcome is None:
        target_outcome = np.full((alpha.shape[-2],), int(NO_OUTCOME), dtype=np.int8)
    kind = np.asarray(target_kind)
    outcome = np.asarray(target_outcome)
    hits = np.zeros((legal_actions.size,), dtype=np.float32)
    for _ in range(int(num_samples)):
        utilities = np.empty((legal_actions.size,), dtype=np.float32)
        for ix, action in enumerate(legal_actions):
            if int(kind[int(action)]) == int(TARGET_CATEGORICAL):
                utilities[ix] = float(categorical_utility_np(int(outcome[int(action)]), alpha.shape[-1]))
            else:
                phi = rng.dirichlet(alpha[int(action)])
                utilities[ix] = float(phi[-1] - phi[0])
        hits[int(np.argmax(utilities))] += 1.0
    target[legal_actions] = hits / float(num_samples)
    target_sum = float(np.sum(target))
    if target_sum <= 0.0:
        target[legal_actions] = 1.0 / float(legal_actions.size)
    else:
        target /= target_sum
    return target


def sample_native_utility(
    rng: np.random.Generator,
    alpha: np.ndarray,
    kind: np.ndarray | int,
    outcome: np.ndarray | int,
) -> np.ndarray:
    alpha = _positive(alpha)
    kind_arr = np.asarray(kind)
    outcome_arr = np.asarray(outcome)
    if kind_arr.shape == ():
        if int(kind_arr) == int(TARGET_CATEGORICAL):
            return np.asarray(
                categorical_utility_np(int(outcome_arr), alpha.shape[-1]),
                dtype=np.float32,
            )
        phi = rng.dirichlet(alpha)
        return np.asarray(phi[-1] - phi[0], dtype=np.float32)

    flat_alpha = alpha.reshape((-1, alpha.shape[-1]))
    flat_kind = kind_arr.reshape((-1,))
    flat_outcome = outcome_arr.reshape((-1,))
    utility = np.empty((flat_kind.shape[0],), dtype=np.float32)
    for ix, target_kind in enumerate(flat_kind):
        if int(target_kind) == int(TARGET_CATEGORICAL):
            utility[ix] = float(categorical_utility_np(int(flat_outcome[ix]), alpha.shape[-1]))
        else:
            phi = rng.dirichlet(flat_alpha[ix])
            utility[ix] = float(phi[-1] - phi[0])
    return utility.reshape(kind_arr.shape)


def categorical_point(
    outcome: jax.Array,
    num_outcomes: int,
    epsilon: float,
    dtype: Any = jnp.float32,
) -> jax.Array:
    if int(num_outcomes) < 2:
        raise ValueError(f"unsupported outcome count: {num_outcomes}")
    eps = jnp.asarray(epsilon, dtype=dtype)
    outcome = jnp.asarray(outcome, dtype=jnp.int32)
    one_hot = jax.nn.one_hot(outcome, num_outcomes, dtype=dtype)
    peak = 1.0 - (float(num_outcomes) - 1.0) * eps
    return one_hot * peak + (1.0 - one_hot) * eps


def dirichlet_nll_at_categorical(
    alpha: jax.Array,
    outcome: jax.Array,
    epsilon: float,
) -> jax.Array:
    dtype = jnp.result_type(alpha, jnp.float32)
    eps = jnp.asarray(1e-6, dtype=dtype)
    alpha = jnp.maximum(alpha.astype(dtype), eps)
    point = jax.lax.stop_gradient(
        categorical_point(outcome, alpha.shape[-1], epsilon, dtype=dtype)
    )
    alpha_sum = jnp.sum(alpha, axis=-1)
    return (
        -gammaln(alpha_sum)
        + jnp.sum(gammaln(alpha), axis=-1)
        - jnp.sum((alpha - 1.0) * jnp.log(point), axis=-1)
    )


def _positive(alpha: np.ndarray) -> np.ndarray:
    return np.maximum(np.asarray(alpha, dtype=np.float32), np.float32(1e-6))
