"""The GPU-friendly search must retain every histogram boundary exactly."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scacchi.loss import (
    CONCENTRATION_HISTOGRAM_NUM_BINS,
    _masked_concentration_histogram_counts,
    concentration_histogram_bin_edges,
)


def _original_scan_counts(concentration, mask):
    dtype = jnp.result_type(concentration.dtype, jnp.float32)
    concentration = concentration.astype(dtype)
    valid = mask & jnp.isfinite(concentration)
    indices = jnp.searchsorted(
        concentration_histogram_bin_edges(dtype)[1:-1],
        jnp.where(valid, concentration, 0.0),
        side="right",
        method="scan",
    )
    return jnp.bincount(
        indices.reshape(-1),
        weights=valid.astype(dtype).reshape(-1),
        length=CONCENTRATION_HISTOGRAM_NUM_BINS,
    )


@pytest.mark.parametrize("dtype", [jnp.float16, jnp.bfloat16, jnp.float32])
@pytest.mark.parametrize("rank", [1, 3])
@pytest.mark.parametrize("mask_mode", ["all", "alternating", "none"])
def test_unrolled_histogram_is_exact_at_edges_neighbors_and_nonfinite(dtype, rank, mask_mode):
    edges = concentration_histogram_bin_edges(dtype)
    values = jnp.concatenate((
        jnp.nextafter(edges, jnp.asarray(-jnp.inf, dtype)),
        edges,
        jnp.nextafter(edges, jnp.asarray(jnp.inf, dtype)),
        jnp.asarray([-jnp.inf, -3.0, -0.0, 0.0, 0.5, 2048.0, jnp.inf, jnp.nan, 1.0], dtype),
    ))
    if rank == 3:
        values = values.reshape(2, 3, -1)
    if mask_mode == "all":
        mask = jnp.ones(values.shape, dtype=jnp.bool_)
    elif mask_mode == "none":
        mask = jnp.zeros(values.shape, dtype=jnp.bool_)
    else:
        mask = (jnp.arange(values.size) % 2 == 0).reshape(values.shape)

    actual = jax.jit(_masked_concentration_histogram_counts)(values, mask)
    original = jax.jit(_original_scan_counts)(values, mask)
    np.testing.assert_array_equal(actual, original)

    # Both implementations also match a host oracle with the same promoted
    # float32 grid, open-ended overflow bins, and finite/masking convention.
    host_values = np.asarray(values.astype(jnp.float32)).reshape(-1)
    valid = np.asarray(mask).reshape(-1) & np.isfinite(host_values)
    indices = np.searchsorted(
        np.asarray(concentration_histogram_bin_edges())[1:-1],
        host_values[valid],
        side="right",
    )
    expected = np.bincount(indices, minlength=CONCENTRATION_HISTOGRAM_NUM_BINS)
    np.testing.assert_array_equal(actual, expected)
