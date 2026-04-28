"""Self-play data generation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pgx
from flax import nnx
from pgx.experimental import auto_reset

from scacchi.config import TrainConfig
from scacchi.search import run_search
from scacchi.types import SelfplayBatch


def run_selfplay(
    *,
    env: pgx.Env,
    model: nnx.Module,
    rng_key,
    cfg: TrainConfig,
) -> SelfplayBatch:
    """Generate a fixed-length batch of self-play trajectories."""

    batch_size = cfg.selfplay_batch_size
    rng_key, init_key = jax.random.split(rng_key)
    state = jax.vmap(env.init)(jax.random.split(init_key, batch_size))
    step = jax.vmap(auto_reset(env.step, env.init))

    def step_fn(state, key):
        search_key, reset_key = jax.random.split(key)
        obs = state.observation
        actor = state.current_player
        policy = run_search(env=env, model=model, rng_key=search_key, state=state, cfg=cfg.search)
        state = step(state, policy.action, jax.random.split(reset_key, batch_size))
        reward = state.rewards[jnp.arange(batch_size), actor]
        return state, SelfplayBatch(
            observation=obs,
            action_weights=policy.action_weights,
            reward=reward,
            discount=jnp.where(state.terminated, 0.0, -jnp.ones_like(reward)),
            terminated=state.terminated,
        )

    _, data = jax.lax.scan(step_fn, state, jax.random.split(rng_key, cfg.max_num_steps))
    return data
