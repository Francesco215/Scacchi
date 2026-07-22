"""Tagged categorical targets shared by search and neural training."""

from __future__ import annotations

from typing import TypedDict

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln
from jaxtyping import Array, DTypeLike, Float, Int, ScalarLike, Shaped


TARGET_PAD = 0
TARGET_DIRICHLET = 1
TARGET_CATEGORICAL = 2
NO_OUTCOME = -1
NO_DISTANCE = -1


class _NativeTargetFields(TypedDict):
    q_target_kind: Int[Array, "*batch action"]
    q_target_weight: Float[Array, "*batch action"]
    q_target_outcome: Int[Array, "*batch action"]
    q_target_distance: Int[Array, "*batch action"]
    v_target_kind: Int[Array, "*batch"]
    v_target_weight: Float[Array, "*batch"]
    v_target_outcome: Int[Array, "*batch"]
    v_target_distance: Int[Array, "*batch"]


_EmptyNative = tuple[Int[Array, "*target"], Float[Array, "*target"], Int[Array, "*target"], Int[Array, "*target"]]


def _full_like_input_sharding(source: Shaped[Array, "*batch"], value: ScalarLike, dtype: DTypeLike) -> Shaped[Array, "*batch"]:
    # Plain full_like/ones_like constants can be replicated under jit.
    # Keep a zero-valued dependency so defaults inherit source sharding.
    zero = jax.lax.stop_gradient(source - jax.lax.stop_gradient(source))
    return zero.astype(dtype) + jnp.asarray(value, dtype=dtype)


def _empty_native(beta: Float[Array, "*target outcome"], *, kind: int = int(TARGET_DIRICHLET)) -> _EmptyNative:
    beta = jnp.asarray(beta)
    target = beta[..., 0]
    target_kind = _full_like_input_sharding(target, int(kind), jnp.int8)
    target_weight = _full_like_input_sharding(target, 1.0, beta.dtype)
    target_outcome = _full_like_input_sharding(target, int(NO_OUTCOME), jnp.int8)
    target_distance = _full_like_input_sharding(target, int(NO_DISTANCE), jnp.int32)
    return target_kind, target_weight, target_outcome, target_distance


def native_fields_from_beta(beta_q: Float[Array, "*batch action outcome"], beta_v: Float[Array, "*batch outcome"]) -> _NativeTargetFields:
    """Create ordinary-Dirichlet sidecars matching Q and V target shapes."""

    q_kind, q_weight, q_outcome, q_distance = _empty_native(beta_q)
    v_kind, v_weight, v_outcome, v_distance = _empty_native(beta_v)
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


def categorical_point(outcome: Int[Array, "*batch"], num_outcomes: int, epsilon: float, dtype: DTypeLike = jnp.float32) -> Float[Array, "*batch outcome"]:
    """Return an epsilon-interior simplex point for a categorical outcome.

    Every non-target coordinate is ``epsilon`` and the target coordinate is
    ``1 - (num_outcomes - 1) * epsilon``. The helper enforces the categorical
    constraint ``0 < epsilon < 1 / num_outcomes``. Out-of-range outcome
    indices produce ``NaN`` rather than a non-normalized pseudo-target, so a
    malformed categorical certificate cannot silently train the wrong class.
    """

    num_outcomes = int(num_outcomes)
    if num_outcomes < 2:
        raise ValueError(f"unsupported outcome count: {num_outcomes}")
    epsilon = float(epsilon)
    if not 0.0 < epsilon < 1.0 / float(num_outcomes):
        raise ValueError(f"epsilon must be > 0 and < 1 / num_outcomes; got epsilon={epsilon}, num_outcomes={num_outcomes}")
    eps = jnp.asarray(epsilon, dtype=dtype)
    outcome = jnp.asarray(outcome, dtype=jnp.int32)
    one_hot = jax.nn.one_hot(outcome, num_outcomes, dtype=dtype)
    peak = 1.0 - (float(num_outcomes) - 1.0) * eps
    point = one_hot * peak + (1.0 - one_hot) * eps
    valid_outcome = (outcome >= 0) & (outcome < num_outcomes)
    return jnp.where(valid_outcome[..., None], point, jnp.full_like(point, jnp.nan))


def dirichlet_nll_at_categorical(alpha: Float[Array, "*batch outcome"], outcome: Int[Array, "*batch"], epsilon: float) -> Float[Array, "*batch"]:
    """Negative log Dirichlet density at an epsilon-smoothed category."""

    dtype = jnp.result_type(alpha, jnp.float32)
    alpha_epsilon = jnp.asarray(1e-6, dtype=dtype)
    alpha = jnp.maximum(alpha.astype(dtype), alpha_epsilon)
    point = jax.lax.stop_gradient(categorical_point(outcome, alpha.shape[-1], epsilon, dtype=dtype))
    alpha_sum = jnp.sum(alpha, axis=-1)
    return (
        -gammaln(alpha_sum)
        + jnp.sum(gammaln(alpha), axis=-1)
        - jnp.sum((alpha - 1.0) * jnp.log(point), axis=-1)
    )


__all__ = [
    "NO_DISTANCE",
    "NO_OUTCOME",
    "TARGET_CATEGORICAL",
    "TARGET_DIRICHLET",
    "TARGET_PAD",
    "categorical_point",
    "dirichlet_nll_at_categorical",
    "native_fields_from_beta",
]
