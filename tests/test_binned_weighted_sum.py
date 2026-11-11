"""Exact toy cases for the weighted-sum backup p_s = sum_a pi(a) X_a."""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scacchi import binned
from scacchi.binned_search import action_selection, update_posterior
from scacchi.dirichlet_mctx import RecurrentFnOutput, RootFnOutput
from scacchi.dirichlet_mctx.search import search
from scacchi.types import BinConfig

UNIFORM5 = np.linspace(0.0, 1.0, 5)


def _beliefs(seed, count, atoms):
    masses = np.random.default_rng(seed).uniform(0.05, 1.0, size=(count, atoms))
    return (masses / masses.sum(axis=-1, keepdims=True)).astype(np.float32)


def _exact(children, policy, points):
    """Brute-force law of the weighted sum, moved onto ``points`` by its mean."""
    out = np.zeros(len(points))
    for atoms in itertools.product(range(len(points)), repeat=len(children)):
        mass = np.prod([child[i] for child, i in zip(children, atoms)])
        value = float(np.dot(policy, points[list(atoms)]))
        upper = int(np.clip(np.searchsorted(points, value, side="right"), 1, len(points) - 1))
        fraction = np.clip((value - points[upper - 1]) / (points[upper] - points[upper - 1]), 0, 1)
        out[upper - 1] += mass * (1 - fraction)
        out[upper] += mass * fraction
    return out


def test_two_coin_flip_moves_put_half_the_mass_in_the_centre():
    coin = np.array([0.5, 0, 0, 0, 0.5], dtype=np.float32)

    value = binned.weighted_sum(jnp.asarray([coin, coin]), jnp.array([0.5, 0.5]), UNIFORM5)

    np.testing.assert_allclose(value, [0.25, 0, 0.5, 0, 0.25], atol=1e-6)


def test_a_certain_policy_returns_that_move_unchanged():
    children = _beliefs(0, 3, 5)

    value = binned.weighted_sum(jnp.asarray(children), jnp.array([0.0, 1.0, 0.0]), UNIFORM5)

    np.testing.assert_allclose(value, children[1], atol=1e-6)


def test_matches_brute_force_when_every_sum_lies_on_the_lattice():
    children = _beliefs(1, 3, 5)
    policy = np.array([0.5, 0.25, 0.25])

    value = binned.weighted_sum(jnp.asarray(children), jnp.asarray(policy), UNIFORM5)

    np.testing.assert_allclose(value, _exact(children, policy, UNIFORM5), atol=2e-6)


@pytest.mark.parametrize("count", [9, 33])
def test_mean_is_the_policy_weighted_mean(count):
    points = binned.support(BinConfig(count=count)).astype(np.float64)
    children = _beliefs(2, 6, count)
    policy = np.random.default_rng(3).dirichlet(np.ones(6))

    value = np.asarray(binned.weighted_sum(jnp.asarray(children), jnp.asarray(policy), points), dtype=np.float64)

    np.testing.assert_allclose(value.sum(), 1.0, atol=1e-5)
    np.testing.assert_allclose(value @ points, policy @ (children @ points), atol=2e-5)


def test_variance_shrinks_by_the_squared_policy_weights():
    points = np.linspace(0.0, 1.0, 129)
    children = _beliefs(4, 4, 129)
    policy = np.array([0.4, 0.3, 0.2, 0.1])

    value = np.asarray(binned.weighted_sum(jnp.asarray(children), jnp.asarray(policy), points), dtype=np.float64)

    def variance(masses):
        mean = masses @ points
        return masses @ (points - mean) ** 2

    expected = sum(w**2 * variance(child.astype(np.float64)) for w, child in zip(policy, children))
    # Interpolation onto the lattice can only add spread, never remove it.
    assert expected - 1e-6 <= variance(value) <= expected + 5 * (1 / 128) ** 2 / 4


def test_zero_weight_moves_do_not_matter():
    children = _beliefs(5, 3, 5)
    junk = np.array([0, 0, 0, 0, 1], dtype=np.float32)

    with_junk = binned.weighted_sum(jnp.asarray([children[0], junk, children[1]]), jnp.array([0.5, 0.0, 0.5]), UNIFORM5)
    without = binned.weighted_sum(jnp.asarray(children[:2]), jnp.array([0.5, 0.5]), UNIFORM5)

    np.testing.assert_allclose(with_junk, without, atol=1e-6)


def test_batched_inputs_keep_their_leading_shape():
    children = jnp.asarray(_beliefs(6, 12, 5)).reshape(2, 2, 3, 5)
    policy = jnp.full((2, 2, 3), 1 / 3)

    value = binned.weighted_sum(children, policy, UNIFORM5)

    assert value.shape == (2, 2, 5)
    np.testing.assert_allclose(value[1, 0], binned.weighted_sum(children[1, 0], policy[1, 0], UNIFORM5), atol=1e-6)


def test_search_root_is_the_weighted_sum_of_its_current_edges():
    """Root action 0 is an exact loss; action 1 leads down a one-move chain."""
    root_value, root_q0, root_q1, node_value, node_q = _beliefs(7, 5, 5)
    points = jnp.asarray(UNIFORM5, dtype=jnp.float32)
    root = RootFnOutput(
        prior_logits=jnp.zeros((1, 2)), value=jnp.asarray(root_value)[None],
        action_values=jnp.asarray([root_q0, root_q1])[None], embedding=jnp.array([[0, 0]]),
        terminal_outcome=jnp.array([-1], dtype=jnp.int8), to_play=jnp.array([0]),
    )

    def expand(params, key, action, embedding):
        kind, player = embedding[:, 0], embedding[:, 1]
        child_kind = jnp.where(kind == 0, jnp.where(action == 0, 1, 2), 2)
        step = RecurrentFnOutput(
            value=jnp.asarray(node_value)[None], action_values=jnp.asarray([node_q, node_q])[None],
            invalid_actions=jnp.array([[False, True]]),
            terminal_outcome=jnp.where(child_kind == 1, 1, -1).astype(jnp.int8), to_play=1 - player,
        )
        return step, jnp.stack([child_kind, 1 - player], axis=-1)

    tree = search(
        params=(), rng_key=jax.random.key(0), root=root, recurrent_fn=expand, action_selection_fn=action_selection,
        posterior_update=lambda key, ctx: update_posterior(key, ctx, points=points),
        num_simulations=8, num_outcomes=2,
    )
    summary = tree.summary()

    np.testing.assert_array_equal(summary.q_categorical_outcome, [[0, -1]])
    edges = binned.categorical_edges(summary.alpha[0], summary.q_categorical_outcome[0])
    policy = binned.best_policy(edges, jnp.zeros(2, dtype=bool))
    np.testing.assert_allclose(summary.value_alpha[0], binned.weighted_sum(edges, policy, points), atol=1e-6)
    assert np.abs(np.asarray(summary.value_alpha[0]) - root_value).max() > 1e-3
