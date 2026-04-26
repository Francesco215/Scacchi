from __future__ import annotations

import jax
import jax.numpy as jnp
import pgx
from flax import nnx
from omegaconf import OmegaConf

from scacchi.checkpoint import (
    build_checkpoint_manager,
    maybe_save_checkpoint,
    restore_checkpoint,
)
from scacchi.models import AlphaZeroResNet
from scacchi.optim import make_optimizer
from scacchi.training import init_optimizer


def _small_model_and_optimizer(seed: int = 0):
    env = pgx.make("chess")
    model = AlphaZeroResNet(
        OmegaConf.create(
            {
                "name": "resnet",
                "channels": 4,
                "blocks": 1,
                "policy_channels": 1,
                "value_channels": 1,
                "value_hidden": 4,
            }
        ),
        observation_shape=tuple(env.observation_shape),
        num_actions=env.num_actions,
        seed=seed,
    )
    tx = make_optimizer(
        OmegaConf.create({"name": "adamw", "learning_rate": 1.0e-3, "weight_decay": 0.0})
    )
    return model, init_optimizer(model, tx)


def test_orbax_checkpoint_round_trip(tmp_path):
    model, optimizer = _small_model_and_optimizer(seed=0)
    rng_key = jax.random.key(3)
    cfg = OmegaConf.create(
        {
            "max_to_keep": 2,
            "save_interval_steps": 1,
            "resume": True,
        }
    )

    with build_checkpoint_manager(cfg, tmp_path, max_steps=2) as manager:
        saved = maybe_save_checkpoint(
            manager,
            0,
            cfg={"example": True},
            model=model,
            optimizer=optimizer,
            rng_key=rng_key,
            metadata={"elo": 12.5},
        )
        manager.wait_until_finished()
        assert saved
        assert manager.latest_step() == 0

        restored_model, restored_optimizer = _small_model_and_optimizer(seed=1)
        restored = restore_checkpoint(
            manager,
            model=restored_model,
            optimizer=restored_optimizer,
            rng_key=jax.random.key(0),
        )
        assert restored.start_step == 1
        assert restored.meta["metadata"]["elo"] == 12.5
        assert jnp.array_equal(restored.rng_key, rng_key)

        original_leaves = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
        restored_leaves = jax.tree_util.tree_leaves(nnx.state(restored_model, nnx.Param))
        assert len(original_leaves) == len(restored_leaves)
        for original, restored_leaf in zip(original_leaves, restored_leaves, strict=True):
            assert jnp.array_equal(original, restored_leaf)
        assert int(restored_optimizer.step[...]) == int(optimizer.step[...])
