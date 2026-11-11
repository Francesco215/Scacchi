"""The CUDA backup kernel computes the same node beliefs as the JAX backup."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scacchi import binned
from scacchi.binned_search import only_repaired
from scacchi.types import BinnedThompsonSearchConfig


def _kernel_available():
    if jax.default_backend() != "gpu":
        return False
    try:
        from scacchi.native_binned._library import find_nvcc

        find_nvcc()
    except RuntimeError:
        return False
    return True


needs_kernel = pytest.mark.skipif(not _kernel_available(), reason="needs a CUDA GPU and nvcc")


def _inputs(lanes, actions, atoms, seed):
    rng = np.random.default_rng(seed)
    logits = 2.0 * rng.normal(size=(lanes, actions, atoms))
    edges = np.exp(logits - logits.max(-1, keepdims=True))
    edges /= edges.sum(-1, keepdims=True)
    invalid = rng.uniform(size=(lanes, actions)) < 0.3
    invalid[:, 0] = False
    repaired = rng.uniform(size=lanes) < 0.6
    points = np.linspace(0.0, 1.0, atoms, dtype=np.float32)
    return jnp.asarray(edges, jnp.float32), jnp.asarray(invalid), jnp.asarray(points), jnp.asarray(repaired)


def _jax_backup(edges, invalid, points, repaired):
    return only_repaired(lambda e, i: binned.weighted_sum(e, binned.best_policy(e, i), points), repaired, edges, invalid)


# Moves of Hex 7x7 to 11x11 (cells plus the swap move).
@needs_kernel
@pytest.mark.parametrize("actions", [50, 65, 82, 101, 122])
@pytest.mark.parametrize("atoms", [9, 65])
def test_kernel_matches_jax_backup(actions, atoms):
    from scacchi.native_binned import fused_backup
    from scacchi.native_binned.reference import backup64

    edges, invalid, points, repaired = _inputs(64, actions, atoms, seed=actions + atoms)
    kernel = np.asarray(jax.jit(fused_backup)(edges, invalid, points, repaired))
    expected = np.asarray(jax.jit(_jax_backup)(edges, invalid, points, repaired))

    mask = np.asarray(repaired)
    np.testing.assert_array_equal(kernel[~mask], 0.0)
    np.testing.assert_allclose(kernel[mask], expected[mask], atol=2e-6)
    np.testing.assert_allclose(kernel[mask].sum(-1), 1.0, atol=1e-5)
    for lane in np.flatnonzero(mask)[:4]:
        exact = backup64(np.asarray(edges[lane], np.float64), np.asarray(invalid[lane]), np.asarray(points))
        np.testing.assert_allclose(kernel[lane], exact, atol=1e-6)


@needs_kernel
def test_kernel_inside_a_while_loop():
    from scacchi.native_binned import fused_backup

    edges, invalid, points, repaired = _inputs(32, 50, 9, seed=0)

    def run(backup):
        def body(state):
            step, total = state
            return step + 1, total + backup(edges, invalid, points, repaired & (jnp.arange(32) >= step))

        return jax.lax.while_loop(lambda s: s[0] < 3, body, (0, jnp.zeros((32, 9))))[1]

    np.testing.assert_allclose(jax.jit(lambda: run(fused_backup))(), jax.jit(lambda: run(_jax_backup))(), atol=6e-6)


def test_search_kernel_values():
    assert BinnedThompsonSearchConfig().search_kernel == "auto"
    assert BinnedThompsonSearchConfig(search_kernel="cuda").search_kernel == "cuda"
    with pytest.raises(ValueError, match="search_kernel"):
        BinnedThompsonSearchConfig(search_kernel="triton")


def test_auto_search_kernel(monkeypatch):
    import scacchi.binned_search as binned_search
    from scacchi.envs import make_env

    hex7, c4 = make_env("hex", 7), make_env("connect_four", None)
    monkeypatch.setattr(binned_search, "_kernels_available", lambda: True)
    assert binned_search.resolve_search_kernel("auto", hex7, 9) == "fused"
    assert binned_search.resolve_search_kernel("auto", c4, 9) == "cuda"
    assert binned_search.resolve_search_kernel("auto", make_env("hex", 13), 9) == "xla"  # 169 moves
    assert binned_search.resolve_search_kernel("xla", hex7, 9) == "xla"
    with pytest.raises(ValueError, match="Hex only"):
        binned_search.resolve_search_kernel("fused", c4, 9)
    monkeypatch.setattr(binned_search, "_kernels_available", lambda: False)
    assert binned_search.resolve_search_kernel("auto", hex7, 9) == "xla"
