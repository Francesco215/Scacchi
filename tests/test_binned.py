import itertools
from pathlib import Path

from flax import nnx
from hydra import compose, initialize_config_dir
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from scacchi import binned
from scacchi.binned_search import action_selection, update_posterior
from scacchi.dirichlet_mctx import RootFnOutput, RecurrentFnOutput
from scacchi.dirichlet_mctx.search import search
from scacchi.network import build_model
from scacchi.types import BinConfig, load_config
from scacchi.loss import _native_binned_loss
from scacchi.dirichlet_mctx.native_targets import TARGET_BINNED, TARGET_CATEGORICAL, TARGET_PAD


@pytest.mark.parametrize("count", [3, 9, 33, 129])
def test_grid_has_exact_endpoints_and_symmetric_cells(count):
    grid = binned.support(BinConfig(count=count))
    assert grid[0] == 0 and grid[-1] == 1 and grid[count // 2] == .5
    assert np.all(np.diff(grid) > 0)
    np.testing.assert_allclose(grid, 1 - grid[::-1], atol=6e-8)


def test_best_policy_and_uniform_ties_match_exhaustive_joint_distribution():
    probs = np.array([[.3, .2, .5], [.1, .8, .1], [.4, .1, .5]], dtype=np.float32)
    expected_policy = np.zeros(3)
    for atoms in itertools.product(range(3), repeat=3):
        mass = np.prod([probs[a, i] for a, i in enumerate(atoms)])
        winners = np.array(atoms) == max(atoms)
        expected_policy += mass * winners / winners.sum()
    invalid = jnp.zeros(3, dtype=bool)
    np.testing.assert_allclose(binned.best_policy(jnp.asarray(probs), invalid), expected_policy, atol=2e-7)
    masked = binned.best_policy(jnp.asarray(probs), jnp.array([False, True, True]))
    np.testing.assert_array_equal(masked, [1, 0, 0])


def test_identical_atoms_tie_uniformly_and_illegal_actions_never_win():
    probs = jnp.array([[0., 1., 0.]] * 4)
    invalid = jnp.array([False, True, False, False])
    np.testing.assert_allclose(binned.best_policy(probs, invalid), [1/3, 0, 1/3, 1/3], atol=1e-7)
    samples = jax.vmap(lambda key: binned.sample_action(key, probs, invalid))(jax.random.split(jax.random.key(0), 6000))
    counts = np.bincount(np.asarray(samples), minlength=4) / 6000
    np.testing.assert_allclose(counts, [1/3, 0, 1/3, 1/3], atol=.025)


@pytest.mark.parametrize("dtype", ["float32", "bfloat16"])
@pytest.mark.parametrize("count", [3, 9, 33, 129])
def test_binned_recipe_initializes_uniform_density_in_value_and_every_action(dtype, count):
    config_dir = Path(__file__).parents[1] / "scacchi" / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        recipe = compose(config_name="hex3", overrides=[
            "+algorithm=binned", "model.num_channels=8", "model.num_layers=1",
            f"model.compute_dtype={dtype}", f"model.bins.count={count}",
        ])
    observations = jax.random.normal(jax.random.key(42), (2, 3, 3, 4))
    model = build_model(load_config(recipe), num_actions=9, observation_shape=(3, 3, 4), rngs=nnx.Rngs(0))
    output = model(observations, train=False)
    assert isinstance(output, binned.BinnedOutput)
    assert output.action_probs.shape == (2, 9, count)
    points = np.asarray(output.support, dtype=np.float64)
    right_edges = np.r_[(points[:-1] + points[1:]) / 2, 1.]
    # A uniform density has F(x) = x: the masses must reproduce that CDF at
    # the cell boundaries for every V and Q belief.
    for probs in (output.value_probs, output.action_probs):
        np.testing.assert_allclose(np.cumsum(probs, axis=-1), np.broadcast_to(right_edges, probs.shape), atol=3e-7, rtol=0)
        np.testing.assert_allclose(binned.mean(probs, output.support), .5, atol=1e-7, rtol=0)


def test_proof_tags_remain_binary_with_more_than_128_atoms():
    count = 129
    points = jnp.linspace(0.0, 1.0, count)
    root = RootFnOutput(prior_logits=jnp.zeros((1, 1)), value=jnp.full((1, count), 1/count),
                        action_values=jnp.full((1, 1, count), 1/count), embedding=jnp.zeros((1, 1)),
                        terminal_outcome=jnp.array([-1], dtype=jnp.int8), to_play=jnp.array([0]))
    def recurrent(params, key, action, embedding):
        return RecurrentFnOutput(value=root.value, action_values=root.action_values,
                                 invalid_actions=jnp.ones((1, 1), dtype=bool),
                                 terminal_outcome=jnp.array([0], dtype=jnp.int8), to_play=jnp.array([1])), embedding
    tree = search(params=(), rng_key=jax.random.key(0), root=root, recurrent_fn=recurrent,
                  action_selection_fn=action_selection,
                  posterior_update=lambda key, ctx: update_posterior(key, ctx, points=points),
                  num_simulations=2, num_outcomes=2)
    summary = tree.summary()
    np.testing.assert_array_equal(summary.v_categorical_outcome, [1])
    np.testing.assert_array_equal(summary.q_categorical_outcome, [[1]])
    np.testing.assert_array_equal(summary.v_categorical_distance, [1])
    projected = binned.categorical_edges(root.value, jnp.array([1], dtype=jnp.int8))
    np.testing.assert_array_equal(projected[0, :-1], 0)
    assert projected[0, -1] == 1


def test_distribution_loss_distinguishes_equal_mean_beliefs():
    target = jnp.array([.45, .1, .45])
    diffuse = jnp.array([.1, .8, .1])
    def score(probs):
        return _native_binned_loss(target, probs, jnp.int8(TARGET_BINNED), -1, 1.)
    assert score(target) < score(diffuse)
    gradient = jax.grad(lambda logits: score(jax.nn.softmax(logits)))(jnp.log(diffuse))
    assert gradient[0] < 0 and gradient[1] > 0 and gradient[2] < 0


def test_exact_win_loss_uses_last_atom_and_padding_has_zero_gradient():
    target = jnp.zeros(129)
    def score(logits, kind):
        return _native_binned_loss(target, jax.nn.softmax(logits), jnp.asarray(kind), jnp.int8(1), 1.)
    gradient = jax.grad(score)(jnp.zeros(129), TARGET_CATEGORICAL)
    assert gradient[-1] < -.99
    assert jnp.all(gradient[:-1] > 0)
    np.testing.assert_array_equal(jax.grad(score)(jnp.zeros(129), TARGET_PAD), 0)


def test_unused_padding_payload_cannot_poison_loss_gradient():
    def score(logits):
        return _native_binned_loss(jnp.full(3, jnp.nan), jax.nn.softmax(logits),
                                   TARGET_PAD, -1, jnp.nan)
    assert score(jnp.zeros(3)) == 0
    np.testing.assert_array_equal(jax.grad(score)(jnp.zeros(3)), 0)
