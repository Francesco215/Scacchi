from __future__ import annotations

import jax
import jax.numpy as jnp
import pgx
from omegaconf import OmegaConf

from scacchi.models import AlphaZeroResNet
from scacchi.optim import make_optimizer
from scacchi.runtime import batch_sharding, create_mesh, replicated_sharding, validate_batch_size
from scacchi.search import run_search
from scacchi.selfplay import compute_training_batch, run_selfplay
from scacchi.training import init_train_state, make_train_step, take_batch


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
    graphdef, train_state = init_train_state(model, tx)
    return env, graphdef, train_state, tx


def test_pgx_chess_and_mesh_contracts():
    env = pgx.make("chess")
    assert tuple(env.observation_shape) == (8, 8, 119)
    assert env.num_actions == 4672

    mesh = create_mesh(OmegaConf.create({"name": "single_mesh", "axis_name": "data", "num_devices": 1}))
    validate_batch_size(4, mesh)
    assert batch_sharding(mesh, 2).spec == jax.sharding.PartitionSpec("data", None)
    assert replicated_sharding(mesh).spec == jax.sharding.PartitionSpec()


def test_model_and_optimizer_factories():
    env, graphdef, train_state, _ = make_small_state()
    del graphdef
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
    assert train_state.opt_state is not None

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


def test_mctx_search_returns_legal_actions():
    env, graphdef, train_state, _ = make_small_state()
    state = jax.vmap(env.init)(jax.random.split(jax.random.key(0), 2))
    policy_output = run_search(
        env=env,
        graphdef=graphdef,
        params=train_state.params,
        rng_key=jax.random.key(1),
        state=state,
        num_simulations=2,
        max_num_considered_actions=4,
        max_depth=2,
        gumbel_scale=1.0,
    )
    assert policy_output.action.shape == (2,)
    legal = state.legal_action_mask[jnp.arange(2), policy_output.action]
    assert bool(jnp.all(legal))


def test_tiny_selfplay_and_train_step():
    env, graphdef, train_state, tx = make_small_state()
    selfplay_fn = jax.jit(
        lambda params, key: run_selfplay(
            env=env,
            graphdef=graphdef,
            params=params,
            rng_key=key,
            batch_size=2,
            max_num_steps=2,
            num_simulations=2,
            max_num_considered_actions=4,
            max_depth=2,
            gumbel_scale=1.0,
        )
    )
    data = selfplay_fn(train_state.params, jax.random.key(2))
    assert data.observation.shape == (2, 2, *env.observation_shape)
    assert data.action_weights.shape == (2, 2, env.num_actions)
    assert jnp.isfinite(data.action_weights).all()

    batch = take_batch(compute_training_batch(data), 4)
    before = jax.tree_util.tree_leaves(train_state.params)
    train_state, metrics = make_train_step(graphdef, tx)(train_state, batch)
    after = jax.tree_util.tree_leaves(train_state.params)
    assert jnp.isfinite(metrics.loss)
    assert jnp.isfinite(metrics.policy_loss)
    assert jnp.isfinite(metrics.value_loss)
    changed = [jnp.any(a != b) for a, b in zip(before, after, strict=True)]
    assert bool(jnp.any(jnp.asarray(changed)))
