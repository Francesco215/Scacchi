"""Tagged native targets emitted by Dirichlet search."""

from __future__ import annotations

from typing import TypedDict

import jax
import jax.numpy as jnp
from jaxtyping import Array, DTypeLike, Float, Int8, Int32, ScalarLike, Shaped

from .outcomes import NO_DISTANCE, NO_OUTCOME


TARGET_PAD = 0
TARGET_DIRICHLET = 1
TARGET_CATEGORICAL = 2


class NativeTargetFields(TypedDict):
    q_target_kind: Int8[Array, "*batch action"]
    q_target_weight: Float[Array, "*batch action"]
    q_target_outcome: Int8[Array, "*batch action"]
    q_target_distance: Int32[Array, "*batch action"]
    v_target_kind: Int8[Array, "*batch"]
    v_target_weight: Float[Array, "*batch"]
    v_target_outcome: Int8[Array, "*batch"]
    v_target_distance: Int32[Array, "*batch"]


_EmptyNative = tuple[Int8[Array, "*target"], Float[Array, "*target"], Int8[Array, "*target"], Int32[Array, "*target"]]


def _full_like_input_sharding(source: Float[Array, "*batch"], value: ScalarLike, dtype: DTypeLike) -> Shaped[Array, "*batch"]:
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


def native_fields_from_beta(beta_q: Float[Array, "*batch action outcome"], beta_v: Float[Array, "*batch outcome"]) -> NativeTargetFields:
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


__all__ = [
    "NativeTargetFields",
    "TARGET_CATEGORICAL",
    "TARGET_DIRICHLET",
    "TARGET_PAD",
    "native_fields_from_beta",
]
