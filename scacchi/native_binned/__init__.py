"""Opt-in CUDA kernel for the weighted-sum backup.

``fused_backup(edges, invalid, points, repaired)`` computes, for the lanes in
``repaired`` and zeros elsewhere::

    binned.weighted_sum(edges, binned.best_policy(edges, invalid), points)

as one XLA FFI custom call on the GPU, callable inside ``jit``,
``lax.while_loop`` and ``shard_map``. The search uses it when
``search.binned_thompson.backup_kernel=cuda``; the default stays the JAX code.
Same sums in a different order, float32 throughout, no truncation.

Limits: CUDA GPUs only; at most 128 moves (Hex up to 11x11) and 2-129 atoms on
increasing points from 0 to 1. Large boards with many atoms can also exceed the
GPU's shared memory per block, which fails with an error rather than a result.
Nothing is compiled at import time; the first call builds ``backup.cu`` with
nvcc into ``outputs/native_binned/``.
"""

from __future__ import annotations

from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np


@lru_cache(maxsize=32)
def _quadrature(num_actions: int) -> tuple[np.ndarray, np.ndarray]:
    # Same nodes and weights as binned._tie_quadrature (ceil(A / 2) Gauss-Legendre
    # nodes on [0, 1]), recomputed here so this package does not touch binned.py.
    nodes, weights = np.polynomial.legendre.leggauss((num_actions + 1) // 2)
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
