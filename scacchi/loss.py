from typing import NamedTuple

import chex
from flax import nnx
import jax
import jax.numpy as jnp
import optax

from .network import AZNet
from .play import SelfplayOutput


class Sample(NamedTuple):
    obs: jax.Array
    policy_tgt: chex.Array
    value_tgt: jax.Array
    mask: jax.Array


def make_compute_loss_input(config):
    def compute_loss_input(data: SelfplayOutput) -> Sample:
        value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def body_fn(carry: jax.Array, i: jax.Array) -> tuple[jax.Array, jax.Array]:
            ix = config.max_num_steps - i - 1
            value = data.reward[ix] + data.discount[ix] * carry
            return value, value

        _, value_tgt = body_fn(
            jnp.zeros(config.selfplay_batch_size, dtype=data.reward.dtype),
            jnp.arange(config.max_num_steps),
        )
        value_tgt = value_tgt[::-1, :]

        return Sample(
            obs=data.obs,
            policy_tgt=data.action_weights,
            value_tgt=value_tgt,
            mask=value_mask,
        )

    return compute_loss_input


def train(model: AZNet, optimizer: nnx.Optimizer, data: Sample):
    def loss_fn(model: AZNet):
        logits, value = model(data.obs, train=True)
        policy_loss = optax.softmax_cross_entropy(logits, data.policy_tgt)
        policy_loss = jnp.mean(policy_loss)
        value_loss = optax.l2_loss(value, data.value_tgt)
        value_loss = jnp.mean(value_loss * data.mask)
        return policy_loss + value_loss, (policy_loss, value_loss)

    (_, (policy_loss, value_loss)), grads = nnx.value_and_grad(
        loss_fn,
        has_aux=True,
    )(model)
    optimizer.update(model, grads)
    return policy_loss, value_loss


def make_evaluate(env, baseline, config):
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
            my_logits, _ = model(env_state.observation, train=False)
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
