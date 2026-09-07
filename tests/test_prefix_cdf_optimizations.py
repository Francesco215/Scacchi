"""Equivalence of the fused maximum-CDF stage to the original formulation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scacchi.dirichlet_mctx.prefix_cdf import _maximum_cdf


def _original_maximum_cdf(cdf, unresolved):
    cdf = jnp.clip(cdf, 0.0, 1.0)
    cdf = cdf.at[..., 0].set(0.0)
    cdf = cdf.at[..., -1].set(jnp.where(unresolved, 1.0, 0.0))
    log_joint = jnp.sum(
        jnp.where(
            unresolved[..., None],
            jnp.where(cdf > 0.0, jnp.log(cdf), -jnp.inf),
            0.0,
        ),
        axis=-2,
    )
    result = jnp.maximum.accumulate(jnp.exp(log_joint), axis=-1)
    result = result.at[..., 0].set(jnp.where(jnp.any(unresolved, axis=-1), 0.0, 1.0))
    return result.at[..., -1].set(1.0)


def _optimized_maximum_cdf(cdf, unresolved):
    positive = cdf > 0.0
    return _maximum_cdf(jnp.where(positive, jnp.log(cdf), 0.0), positive, unresolved)


@pytest.mark.parametrize("batch_shape", [(), (7,), (2, 3)])
@pytest.mark.parametrize("grid_points", [3, 21, 81])
def test_maximum_cdf_matches_original_with_clipping_and_zero_factors(batch_shape, grid_points):
    rng = np.random.default_rng(96)
    # Also exercise non-monotone intermediate inputs, so the prefix maximum
    # cannot silently be removed just because genuine CDFs should be monotone.
    shape = (*batch_shape, 5, grid_points)
    cdf = rng.uniform(-0.1, 1.1, shape).astype(np.float32)
    cdf[..., 0, 0] = 0.0
    cdf[..., 1, 1] = np.finfo(np.float32).tiny
    cdf[..., 2, -1] = np.inf
    cdf[..., 3, 1] = np.nan
    cdf[..., 0, 1] = np.inf
    cdf[..., 4, 1] = -np.inf
    unresolved = rng.random(shape[:-1]) > 0.3
    for mask in (unresolved, np.zeros_like(unresolved), np.ones_like(unresolved)):
        original = jax.jit(_original_maximum_cdf)(cdf, mask)
        optimized = jax.jit(_optimized_maximum_cdf)(cdf, mask)
        np.testing.assert_allclose(optimized, original, rtol=2e-6, atol=2e-7)
        np.testing.assert_array_equal(optimized[..., -1], 1.0)
        np.testing.assert_array_equal(optimized[..., 0], np.where(mask.any(axis=-1), 0.0, 1.0))
        assert bool(jnp.all(jnp.diff(optimized, axis=-1) >= 0.0))


def test_maximum_cdf_uses_parallel_prefix_primitive():
    cdf = jnp.linspace(0.0, 1.0, 21)[None, None, :]
    unresolved = jnp.ones((1, 1), dtype=jnp.bool_)
    jaxpr = jax.make_jaxpr(_optimized_maximum_cdf)(cdf, unresolved).jaxpr
    assert any(equation.primitive.name == "cummax" for equation in jaxpr.eqns)
    assert not any(equation.primitive.name == "scan" for equation in jaxpr.eqns)
