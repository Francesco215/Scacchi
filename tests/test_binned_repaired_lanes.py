"""Repairing only the lanes that need it must not change any lane's result."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scacchi import binned
from scacchi.binned_search import only_repaired

LANES, ACTIONS, ATOMS = 64, 6, 9
POINTS = np.linspace(0.0, 1.0, ATOMS, dtype=np.float32)


def _inputs():
    rng = np.random.default_rng(0)
    masses = rng.uniform(0.05, 1.0, size=(LANES, ACTIONS, ATOMS))
    edges = jnp.asarray(masses / masses.sum(axis=-1, keepdims=True), dtype=jnp.float32)
    invalid = jnp.asarray(rng.uniform(size=(LANES, ACTIONS)) < 0.3).at[:, 0].set(False)
    return rng, edges, invalid


def _backup(edges, invalid):
    return binned.weighted_sum(edges, binned.best_policy(edges, invalid), POINTS)


@pytest.mark.parametrize("count", [0, 1, 3, 4, 5, 16, 17, 33, 63, 64])
def test_each_repaired_lane_gets_the_full_batch_result(count):
    rng, edges, invalid = _inputs()
    repaired = np.zeros(LANES, dtype=bool)
    repaired[rng.choice(LANES, size=count, replace=False)] = True

    value = jax.jit(lambda mask: only_repaired(_backup, mask, edges, invalid))(jnp.asarray(repaired))

    np.testing.assert_array_equal(value[~repaired], 0.0)
    np.testing.assert_allclose(value[repaired], jax.jit(_backup)(edges, invalid)[repaired], atol=1e-6)


def test_one_lane_is_computed_directly():
    _, edges, invalid = _inputs()

    value = only_repaired(_backup, jnp.array([True]), edges[:1], invalid[:1])

    np.testing.assert_allclose(value, _backup(edges[:1], invalid[:1]), atol=1e-6)
