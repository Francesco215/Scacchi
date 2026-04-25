"""Self-play data generation."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pgx
from flax import nnx
from jaxtyping import PRNGKeyArray
from pgx.experimental import auto_reset

from scacchi.search import run_search
from scacchi.types import ModelGraphDef, SelfplayBatch, TrainingBatch


def run_selfplay(
    *,
    env: pgx.Env,
    graphdef: ModelGraphDef,
    params: nnx.State,
    rng_key: PRNGKeyArray,
    batch_size: int,
    max_num_steps: int,
    num_simulations: int,
    max_num_considered_actions: int,
    max_depth: int | None,
    gumbel_scale: float,
) -> SelfplayBatch:
    """Generate a fixed-length batch of self-play trajectories."""

    init = jax.vmap(env.init)
    step = jax.vmap(auto_reset(env.step, env.init))
    rng_key, init_key, scan_key = jax.random.split(rng_key, 3)
    state = init(jax.random.split(init_key, batch_size))

    def step_fn(state: pgx.State, key: PRNGKeyArray) -> tuple[pgx.State, SelfplayBatch]:
        search_key, reset_key = jax.random.split(key)
        observation = state.observation
        actor = state.current_player
        policy_output = run_search(
            env=env,
            graphdef=graphdef,
            params=params,
            rng_key=search_key,
            state=state,
            num_simulations=num_simulations,
            max_num_considered_actions=max_num_considered_actions,
            max_depth=max_depth,
            gumbel_scale=gumbel_scale,
        )
        next_state = step(state, policy_output.action, jax.random.split(reset_key, batch_size))
        reward = next_state.rewards[jnp.arange(batch_size), actor]
        discount = jnp.where(next_state.terminated, 0.0, -jnp.ones_like(reward))
        output = SelfplayBatch(
            observation=observation,
            action_weights=jnp.asarray(policy_output.action_weights),
            reward=reward,
            discount=discount,
            terminated=next_state.terminated,
        )
        return next_state, output

    keys = jax.random.split(scan_key, max_num_steps)
    _, data = jax.lax.scan(step_fn, state, keys)
    return data


def compute_training_batch(data: SelfplayBatch) -> TrainingBatch:
    """Build flat policy/value targets from self-play trajectories."""

    max_num_steps, batch_size = data.reward.shape
    value_mask = jnp.cumsum(data.terminated[::-1], axis=0)[::-1] >= 1

    def body_fn(carry, i):
        ix = max_num_steps - i - 1
        value = data.reward[ix] + data.discount[ix] * carry
        return value, value

    _, value_target = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size, dtype=data.reward.dtype),
        jnp.arange(max_num_steps),
    )
    value_target = value_target[::-1]

    return TrainingBatch(
        observation=data.observation.reshape((-1, *data.observation.shape[2:])),
        policy_target=data.action_weights.reshape((-1, data.action_weights.shape[-1])),
        value_target=value_target.reshape((-1,)),
        value_mask=value_mask.reshape((-1,)),
    )
