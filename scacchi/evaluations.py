from typing import NamedTuple

import chex
from flax import nnx
import jax
import jax.numpy as jnp
import mctx
import optax

from .network import AZNet
from .play import make_recurrent_fn, slice_obs

def make_evaluate(env, baseline, config):
    nhs = config.num_history_steps

    @nnx.jit
    def evaluate(rng_key: jax.Array, model: AZNet):
        """A simplified evaluation by sampling. Only for debugging.
        Please use MCTS and run tournaments for serious evaluation."""
        my_player = 0

        key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, config.selfplay_batch_size)
        env_state = jax.vmap(env.init)(init_keys)

        def body_fn(val):
            key, env_state, returns = val
            my_logits, _ = model(slice_obs(env_state.observation, nhs), train=False)
            opp_logits, _ = baseline(env_state.observation)
            is_my_turn = (env_state.current_player == my_player).reshape((-1, 1))
            logits = jnp.where(is_my_turn, my_logits, opp_logits)
            key, action_key = jax.random.split(key)
            action = jax.random.categorical(action_key, logits, axis=-1)
            env_state = jax.vmap(env.step)(env_state, action)
            returns = returns + env_state.rewards[
                jnp.arange(config.selfplay_batch_size),
                my_player,
            ]
            return key, env_state, returns

        _, _, returns = nnx.while_loop(
            lambda x: ~(x[1].terminated.all()),
            body_fn,
            (key, env_state, jnp.zeros(config.selfplay_batch_size)),
        )
        return returns

    return evaluate


def make_mcts_evaluate(env, baseline, config):
    # WARNING: THIS HAS NOT BEEN TESTED
    nhs = config.num_history_steps

    @nnx.jit
    def evaluate(rng_key: jax.Array, model: AZNet):
        """MCTS evaluation: model vs baseline. Both sides search with
        gumbel_muzero_policy using their own network."""
        my_player = 0
        my_predict = lambda obs: model(slice_obs(obs, nhs), train=False)
        my_recurrent_fn = make_recurrent_fn(env, my_predict)
        opp_recurrent_fn = make_recurrent_fn(env, baseline)

        key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, config.selfplay_batch_size)
        env_state = jax.vmap(env.init)(init_keys)

        def body_fn(val):
            key, env_state, returns = val
            observation = env_state.observation

            my_logits, my_value = my_predict(observation)  # slice_obs applied inside my_predict
            opp_logits, opp_value = baseline(observation)

            key, my_key, opp_key = jax.random.split(key, 3)

            my_root = mctx.RootFnOutput(
                prior_logits=my_logits,
                value=my_value,
                embedding=env_state,
            )
            my_policy = mctx.gumbel_muzero_policy(
                params=(),
                rng_key=my_key,
                root=my_root,
                recurrent_fn=my_recurrent_fn,
                num_simulations=config.num_simulations,
                invalid_actions=~env_state.legal_action_mask,
                qtransform=mctx.qtransform_completed_by_mix_value,
                gumbel_scale=1.0,
            )

            opp_root = mctx.RootFnOutput(
                prior_logits=opp_logits,
                value=opp_value,
                embedding=env_state,
            )
            opp_policy = mctx.gumbel_muzero_policy(
                params=(),
                rng_key=opp_key,
                root=opp_root,
                recurrent_fn=opp_recurrent_fn,
                num_simulations=config.num_simulations,
                invalid_actions=~env_state.legal_action_mask,
                qtransform=mctx.qtransform_completed_by_mix_value,
                gumbel_scale=1.0,
            )

            is_my_turn = env_state.current_player == my_player
            action = jnp.where(is_my_turn, my_policy.action, opp_policy.action)

            env_state = jax.vmap(env.step)(env_state, action)
            returns = returns + env_state.rewards[
                jnp.arange(config.selfplay_batch_size),
                my_player,
            ]
            return key, env_state, returns

        _, _, returns = nnx.while_loop(
            lambda x: ~(x[1].terminated.all()),
            body_fn,
            (key, env_state, jnp.zeros(config.selfplay_batch_size)),
        )
        return returns

    return evaluate
