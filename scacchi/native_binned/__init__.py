"""CUDA kernels for binned Thompson tree search and inference."""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np


@lru_cache(maxsize=32)
def _quadrature(num_actions: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss((num_actions + 1) // 2)
    return ((nodes + 1) / 2).astype(np.float32), (weights / 2).astype(np.float32)


@lru_cache(maxsize=32)
def _fast_quadrature(num_actions: int) -> tuple[np.ndarray, np.ndarray]:
    q = min((num_actions + 1) // 2, 9)
    nodes, weights = np.polynomial.legendre.leggauss(q)
    return ((nodes + 1) / 2).astype(np.float32), (weights / 2).astype(np.float32)


def fused_backup(edges: jax.Array, invalid: jax.Array, points: jax.Array, repaired: jax.Array) -> jax.Array:
    """``[B, A, K]`` edge beliefs -> ``[B, K]`` node belief; zeros where not ``repaired``."""
    from ._library import TARGET, register

    register()
    batch, actions, atoms = edges.shape
    nodes, weights = _quadrature(actions)
    call = jax.ffi.ffi_call(TARGET, jax.ShapeDtypeStruct((batch, atoms), jnp.float32), vmap_method="sequential")
    return call(
        edges.astype(jnp.float32), invalid.astype(jnp.bool_), jnp.asarray(points, jnp.float32),
        repaired.astype(jnp.bool_), jnp.asarray(nodes), jnp.asarray(weights),
    )
