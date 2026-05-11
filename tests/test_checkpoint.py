import jax
import jax.numpy as jnp
import optax
import pgx
import pytest
from flax import nnx

from scacchi.checkpoint import build_checkpoint_manager, from_pretrained, maybe_save, restore
from scacchi.network import build_model
from scacchi.train import Config


@pytest.fixture
def tiny_config():
    return Config(
        env_id="gardner_chess",
        num_channels=4,
        num_layers=1,
        max_num_iters=10,
        ckpt_max_to_keep=2,
        ckpt_save_interval_steps=1,
        wandb_enabled=False,
    )


def _make_model_and_optimizer(config, seed=0):
    env = pgx.make(config.env_id)
    h, w, _ = env.observation_shape
    obs_shape = (h, w, config.num_history_steps * 14 + 3)
    model = build_model(config, num_actions=env.num_actions, observation_shape=obs_shape, rngs=nnx.Rngs(seed))
    optimizer = nnx.Optimizer(model, optax.adam(1e-3), wrt=nnx.Param)
    return model, optimizer


def test_restore_from_empty_dir_is_noop(tiny_config, tmp_path):
    model, optimizer = _make_model_and_optimizer(tiny_config)
    rng_key = jax.random.PRNGKey(42)

    manager = build_checkpoint_manager(tiny_config, tmp_path / "ckpt")
    start_iter, restored_key, hours, frames = restore(manager, model, optimizer, rng_key)

    assert start_iter == 0
    assert jnp.array_equal(restored_key, rng_key)
    assert hours == 0.0
    assert frames == 0


def test_save_restore_roundtrip(tiny_config, tmp_path):
    model_a, optimizer_a = _make_model_and_optimizer(tiny_config, seed=0)
    rng_saved = jax.random.PRNGKey(7)

    ckpt_dir = tmp_path / "ckpt"
    manager = build_checkpoint_manager(tiny_config, ckpt_dir)
    maybe_save(manager, step=5, model=model_a, optimizer=optimizer_a,
               rng_key=rng_saved, config=tiny_config, hours=1.5, frames=1000)
    manager.wait_until_finished()

    # restore onto a model with different weights
    model_b, optimizer_b = _make_model_and_optimizer(tiny_config, seed=99)
    start_iter, restored_key, hours, frames = restore(manager, model_b, optimizer_b, jax.random.PRNGKey(0))

    assert start_iter == 6
    assert jnp.array_equal(restored_key, rng_saved)
    assert hours == pytest.approx(1.5)
    assert frames == 1000

    leaves_a = jax.tree_util.tree_leaves(nnx.state(model_a))
    leaves_b = jax.tree_util.tree_leaves(nnx.state(model_b))
    for a, b in zip(leaves_a, leaves_b):
        assert jnp.allclose(a, b), "restored model weights don't match saved weights"


def test_from_pretrained_loads_correct_architecture(tiny_config, tmp_path):
    model_orig, optimizer = _make_model_and_optimizer(tiny_config, seed=0)
    rng_key = jax.random.PRNGKey(0)

    ckpt_dir = tmp_path / "ckpt"
    manager = build_checkpoint_manager(tiny_config, ckpt_dir)
    maybe_save(manager, step=3, model=model_orig, optimizer=optimizer,
               rng_key=rng_key, config=tiny_config, hours=0.5, frames=500)
    manager.wait_until_finished()

    env = pgx.make(tiny_config.env_id)
    loaded_model = from_pretrained(str(ckpt_dir), env)

    leaves_orig = jax.tree_util.tree_leaves(nnx.state(model_orig))
    leaves_loaded = jax.tree_util.tree_leaves(nnx.state(loaded_model))
    assert len(leaves_orig) == len(leaves_loaded), "architecture mismatch after from_pretrained"
    for orig, loaded in zip(leaves_orig, leaves_loaded):
        assert jnp.allclose(orig, loaded), "from_pretrained weights don't match saved weights"
