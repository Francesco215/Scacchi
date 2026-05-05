from typing import NamedTuple

import chex
from flax import nnx
import jax
import jax.numpy as jnp
import mctx
import pgx
from pgx.experimental import auto_reset


class SelfplayOutput(NamedTuple):
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    action_weights: chex.Array
    discount: jax.Array


def slice_obs(obs: jax.Array, num_history_steps: int) -> jax.Array:
    return jnp.concatenate([obs[..., : num_history_steps * 14], obs[..., 112:]], axis=-1)


def make_recurrent_fn(env, predict_fn):
    def recurrent_fn(
        _,
        rng_key: chex.PRNGKey,
        action: chex.Array,
        env_state: pgx.State,
    ):
        del rng_key

        current_player = env_state.current_player
        env_state = jax.vmap(env.step)(env_state, action)
        logits, value = predict_fn(env_state.observation)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(
            env_state.legal_action_mask,
            logits,
            jnp.finfo(logits.dtype).min,
        )

        reward = env_state.rewards[
            jnp.arange(env_state.rewards.shape[0]),
            current_player,
        ]
        value = jnp.where(env_state.terminated, 0.0, value)
        discount = -jnp.ones_like(value)
        discount = jnp.where(env_state.terminated, 0.0, discount)

        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=logits,
                value=value,
            ),
            env_state,
        )

    return recurrent_fn


def make_selfplay(env, config):
    nhs = config.num_history_steps

    def selfplay(model: nnx.Module, rng_key: jax.Array) -> SelfplayOutput:
        recurrent_fn = make_recurrent_fn(
            env, lambda obs: model(slice_obs(obs, nhs), train=False)
        )

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def step_fn(
            env_state: pgx.State,
            key: jax.Array,
        ) -> tuple[pgx.State, SelfplayOutput]:
            search_key, reset_key = jax.random.split(key)
            observation = slice_obs(env_state.observation, nhs)
            logits, value = model(observation, train=False)
            root = mctx.RootFnOutput(
                prior_logits=logits,
                value=value,
                embedding=env_state,
            )

            policy_output = mctx.gumbel_muzero_policy(
                params=(),
                rng_key=search_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=config.num_simulations,
                invalid_actions=~env_state.legal_action_mask,
                qtransform=mctx.qtransform_completed_by_mix_value,
                gumbel_scale=1.0,
            )
            actor = env_state.current_player
            reset_keys = jax.random.split(reset_key, config.selfplay_batch_size)
            env_state = jax.vmap(auto_reset(env.step, env.init))(
                env_state,
                policy_output.action,
                reset_keys,
            )
            discount = -jnp.ones_like(value)
            discount = jnp.where(env_state.terminated, 0.0, discount)
            return env_state, SelfplayOutput(
                obs=observation,  # already sliced above
                action_weights=policy_output.action_weights,
                reward=env_state.rewards[jnp.arange(env_state.rewards.shape[0]), actor],
                terminated=env_state.terminated,
                discount=discount,
            )

        rng_key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, config.selfplay_batch_size)
        env_state = jax.vmap(env.init)(init_keys)
        step_keys = jax.random.split(rng_key, config.max_num_steps)
        _, data = step_fn(env_state, step_keys)
        return data

    return selfplay
