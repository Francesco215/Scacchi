from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from scacchi.loss import (
    CONCENTRATION_HISTOGRAM_BIN_EDGES,
    CONCENTRATION_HISTOGRAM_NUM_BINS,
    CONCENTRATION_HISTOGRAM_SERIES,
)
from scacchi.train import _concentration_histograms_for_logging


def test_concentration_histograms_pool_updates_with_shared_edges() -> None:
    per_update = np.zeros(
        (2, len(CONCENTRATION_HISTOGRAM_SERIES), CONCENTRATION_HISTOGRAM_NUM_BINS),
        dtype=np.float32,
    )
    per_update[0, :, 3] = np.asarray([1, 2, 3, 4])
    per_update[1, :, 3] = np.asarray([10, 20, 30, 40])
    metrics = SimpleNamespace(
        dirichlet_concentration_histogram_counts=jnp.asarray(per_update)
    )

    histograms = _concentration_histograms_for_logging(metrics)

    assert tuple(histograms) == (
        "train/V_C_prior_hist_dir",
        "train/V_C_posterior_hist_dir",
        "train/Q_C_prior_hist_dir",
        "train/Q_C_posterior_hist_dir",
    )
    for index, histogram in enumerate(histograms.values()):
        expected = np.zeros(CONCENTRATION_HISTOGRAM_NUM_BINS, dtype=np.int64)
        expected[3] = 11 * (index + 1)
        np.testing.assert_array_equal(histogram.counts, expected)
        np.testing.assert_allclose(
            histogram.bin_edges,
            CONCENTRATION_HISTOGRAM_BIN_EDGES,
        )


def test_concentration_histograms_reject_wrong_metric_shape() -> None:
    metrics = SimpleNamespace(
        dirichlet_concentration_histogram_counts=jnp.zeros((4, 99))
    )

    with pytest.raises(ValueError, match="must end in"):
        _concentration_histograms_for_logging(metrics)
