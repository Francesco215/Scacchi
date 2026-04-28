from __future__ import annotations

import jax
import jax.numpy as jnp
import pgx
from flax import nnx
from omegaconf import OmegaConf

from scacchi.config import SearchConfig, TrainConfig
from scacchi.models import AlphaZeroResNet
from scacchi.optim import make_optimizer
from scacchi.runtime import batch_sharding, create_mesh, replicated_sharding, validate_batch_size
from scacchi.search import run_search
from scacchi.selfplay import run_selfplay
from scacchi.training import compute_training_batch, init_optimizer, make_iteration_step
from scacchi.types import SelfplayBatch


def small_model_cfg():
    return OmegaConf.create(
        {
            "name": "resnet",
            "channels": 4,
            "blocks": 1,
            "policy_channels": 1,
            "value_channels": 1,
            "value_hidden": 4,
        }
    )


def make_small_state():
    env = pgx.make("chess")
    model = AlphaZeroResNet(
        small_model_cfg(),
        observation_shape=tuple(env.observation_shape),
        num_actions=env.num_actions,
        seed=0,
    )
    tx = make_optimizer(
        OmegaConf.create({"name": "adamw", "learning_rate": 1.0e-3, "weight_decay": 0.0})
    )
    optimizer = init_optimizer(model, tx)
    return env, model, optimizer, tx


def test_pgx_chess_and_mesh_contracts():
    env = pgx.make("chess")
    assert tuple(env.observation_shape) == (8, 8, 119)
    assert env.num_actions == 4672

    mesh = create_mesh(OmegaConf.create({"name": "single_mesh", "axis_name": "data", "num_devices": 1}))
    validate_batch_size("batch", 2, mesh)
    assert batch_sharding(mesh, 2).spec == jax.sharding.PartitionSpec("data", None)
    assert replicated_sharding(mesh).spec == jax.sharding.PartitionSpec()


def test_model_and_optimizer_factories():
    env, _, optimizer, _ = make_small_state()
    state = jax.vmap(env.init)(jax.random.split(jax.random.key(0), 3))
    model = AlphaZeroResNet(
        small_model_cfg(),
        observation_shape=tuple(env.observation_shape),
        num_actions=env.num_actions,
        seed=1,
    )
    logits, value = model(state.observation, train=False)
    assert logits.shape == (3, env.num_actions)
    assert value.shape == (3,)
    assert jnp.isfinite(logits).all()
    assert jnp.isfinite(value).all()
    assert optimizer.opt_state is not None

    muon = make_optimizer(
        OmegaConf.create(
            {
                "name": "muon",
                "learning_rate": 1.0e-3,
                "weight_decay": 0.0,
                "adam_weight_decay": 0.0,
            }
        )
    )
    assert muon is not None
    assert make_optimizer(OmegaConf.create({"name": "adam", "learning_rate": 1.0e-3})) is not None


def test_mctx_search_returns_legal_actions():
    env, model, _, _ = make_small_state()
    state = jax.vmap(env.init)(jax.random.split(jax.random.key(0), 2))
    policy_output = run_search(
        env=env,
        model=model,
        rng_key=jax.random.key(1),
        state=state,
        cfg=SearchConfig(
            name="gumbel",
            num_simulations=2,
            max_num_considered_actions=4,
            max_depth=2,
            gumbel_scale=1.0,
        ),
    )
    assert policy_output.action.shape == (2,)
    legal = state.legal_action_mask[jnp.arange(2), policy_output.action]
    assert bool(jnp.all(legal))


def test_tiny_selfplay_and_iteration_step():
    env, model, optimizer, _ = make_small_state()
    cfg = TrainConfig(
        seed=0,
        num_iters=1,
        selfplay_batch_size=2,
        max_num_steps=2,
        search=SearchConfig(
            name="gumbel",
            num_simulations=2,
            max_num_considered_actions=4,
            max_depth=2,
            gumbel_scale=1.0,
        ),
        batch_size=4,
        log_interval=1,
    )
    selfplay_fn = nnx.jit(
        lambda model, key: run_selfplay(env=env, model=model, rng_key=key, cfg=cfg)
    )
    data = selfplay_fn(model, jax.random.key(2))
    assert data.observation.shape == (2, 2, *env.observation_shape)
    assert data.action_weights.shape == (2, 2, env.num_actions)
    assert jnp.isfinite(data.action_weights).all()

    before = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    metrics = make_iteration_step(env, cfg)(model, optimizer, jax.random.key(3))
    after = jax.tree_util.tree_leaves(nnx.state(model, nnx.Param))
    assert jnp.isfinite(metrics.loss)
    assert jnp.isfinite(metrics.policy_loss)
    assert jnp.isfinite(metrics.value_loss)
    assert int(metrics.samples) == 4
    assert int(metrics.num_updates) == 1
    changed = [jnp.any(a != b) for a, b in zip(before, after, strict=True)]
    assert bool(jnp.any(jnp.asarray(changed)))


def test_training_targets_keep_drawn_terminal_values():
    data = SelfplayBatch(
        observation=jnp.zeros((3, 1, 1, 1, 1), dtype=jnp.float32),
        action_weights=jnp.zeros((3, 1, 2), dtype=jnp.float32),
        reward=jnp.zeros((3, 1), dtype=jnp.float32),
        discount=jnp.asarray([[-1.0], [-1.0], [0.0]], dtype=jnp.float32),
        terminated=jnp.asarray([[False], [False], [True]]),
    )
    batch = compute_training_batch(data)

    assert bool(jnp.all(batch.value_mask))
    assert bool(jnp.all(batch.value_target == 0.0))
