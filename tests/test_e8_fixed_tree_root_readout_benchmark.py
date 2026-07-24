import jax.numpy as jnp
import numpy as np

from scripts.e8_fixed_tree_root_readout_benchmark import (
    _categorical_population,
)


def test_categorical_population_is_uniform_over_optimal_distance_ties():
    node_outcome = jnp.asarray([1, 0], dtype=jnp.int8)
    edge_outcome = jnp.asarray(
        [
            [1, 1, 1, -1],
            [0, 0, 0, -1],
        ],
        dtype=jnp.int8,
    )
    edge_distance = jnp.asarray(
        [
            [2, 2, 3, -1],
            [2, 4, 4, -1],
        ],
        dtype=jnp.int32,
    )
    invalid = jnp.asarray(
        [
            [False, False, False, True],
            [False, False, False, True],
        ]
    )

    policy = _categorical_population(
        node_outcome,
        edge_outcome,
        edge_distance,
        invalid,
        num_outcomes=2,
    )

    # Certified wins prefer the shortest edge; certified losses prefer the
    # longest.  Equal candidates receive the exact expectation of the
    # production selector's uniform random tie break.
    np.testing.assert_array_equal(
        policy,
        jnp.asarray(
            [
                [0.5, 0.5, 0.0, 0.0],
                [0.0, 0.5, 0.5, 0.0],
            ],
            dtype=jnp.float32,
        ),
    )

    permutation = jnp.asarray([2, 0, 3, 1])
    permuted = _categorical_population(
        node_outcome,
        edge_outcome[:, permutation],
        edge_distance[:, permutation],
        invalid[:, permutation],
        num_outcomes=2,
    )
    np.testing.assert_array_equal(permuted, policy[:, permutation])
