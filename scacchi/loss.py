from typing import NamedTuple

import chex
from flax import nnx
import jax
import jax.numpy as jnp
import optax

from .play import SelfplayOutput


class Sample(NamedTuple):
    obs: jax.Array
    policy_tgt: chex.Array
    value_tgt: jax.Array
    policy_mask: jax.Array
    value_mask: jax.Array


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

        return Sample(obs=data.obs, policy_tgt=data.action_weights, value_tgt=value_tgt, policy_mask=data.legal_action_mask, value_mask=value_mask)

    return compute_loss_input


def _masked_mean(loss: jax.Array, mask: jax.Array) -> jax.Array:
    mask = mask.astype(loss.dtype)
    return jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1)


def _compute_losses(logits: jax.Array, value: jax.Array, data: Sample) -> tuple[jax.Array, jax.Array]:
    policy_loss = optax.softmax_cross_entropy(logits, data.policy_tgt, where=data.policy_mask)
    policy_loss = _masked_mean(policy_loss, data.value_mask)
    value_loss = optax.l2_loss(value, data.value_tgt)
    value_loss = _masked_mean(value_loss, data.value_mask)
    return policy_loss, value_loss


def train(model: nnx.Module, optimizer: nnx.Optimizer, data: Sample):
    def loss_fn(model: nnx.Module):
        logits, value = model(data.obs, train=True)
        policy_loss, value_loss = _compute_losses(logits, value, data)
        return policy_loss + value_loss, (policy_loss, value_loss)

    (_, (policy_loss, value_loss)), grads = nnx.value_and_grad(
        loss_fn,
        has_aux=True,
    )(model)
    optimizer.update(model, grads)
    return policy_loss, value_loss
