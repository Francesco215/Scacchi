from typing import NamedTuple

import chex
from flax import nnx
import jax
import jax.numpy as jnp
import optax

from .network import outcome_mean
from .play import SelfplayOutput


class Sample(NamedTuple):
    obs: jax.Array
    policy_tgt: chex.Array
    value_tgt: jax.Array
    played_action: jax.Array
    policy_mask: jax.Array
    value_mask: jax.Array


class TrainMetrics(NamedTuple):
    policy_loss: jax.Array
    value_loss: jax.Array
    value_outcome_loss: jax.Array
    q_outcome_loss: jax.Array
    alpha_V_concentration: jax.Array
    alpha_Q_concentration: jax.Array


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
            played_action=data.played_action,
            policy_mask=data.legal_action_mask,
            value_mask=value_mask,
        )

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


def _outcome_target(value_tgt: jax.Array, num_outcomes: int) -> jax.Array:
    outcome_idx = (jnp.round(value_tgt).astype(jnp.int32) + 1) // 2
    return jax.nn.one_hot(outcome_idx, num_outcomes)


def _categorical_ce_from_probs(probs: jax.Array, target: jax.Array) -> jax.Array:
    log_probs = jnp.log(jnp.clip(probs, jnp.finfo(probs.dtype).tiny, 1.0))
    return -jnp.sum(target * log_probs, axis=-1)


def _gather_played_action(alpha_q: jax.Array, played_action: jax.Array) -> jax.Array:
    gather_ix = jnp.broadcast_to(
        played_action[..., None, None],
        (*played_action.shape, 1, alpha_q.shape[-1]),
    )
    return jnp.take_along_axis(alpha_q, gather_ix, axis=-2).squeeze(axis=-2)


def _compute_dirichlet_losses(
    logits: jax.Array,
    alpha_v: jax.Array,
    alpha_q: jax.Array,
    data: Sample,
    config,
) -> tuple[jax.Array, TrainMetrics]:
    policy_loss = optax.softmax_cross_entropy(logits, data.policy_tgt, where=data.policy_mask)
    policy_loss = _masked_mean(policy_loss, data.value_mask)

    outcome_tgt = _outcome_target(data.value_tgt, alpha_v.shape[-1])
    value_outcome_loss = _categorical_ce_from_probs(outcome_mean(alpha_v), outcome_tgt)
    value_outcome_loss = _masked_mean(value_outcome_loss, data.value_mask)

    played_alpha_q = _gather_played_action(alpha_q, data.played_action)
    q_outcome_loss = _categorical_ce_from_probs(outcome_mean(played_alpha_q), outcome_tgt)
    q_outcome_loss = _masked_mean(q_outcome_loss, data.value_mask)

    alpha_v_concentration = _masked_mean(jnp.sum(alpha_v, axis=-1), data.value_mask)
    alpha_q_concentration = _masked_mean(
        jnp.sum(alpha_q, axis=-1),
        data.policy_mask & data.value_mask[..., None],
    )

    total_loss = (
        config.policy_loss_weight * policy_loss
        + config.value_outcome_weight * value_outcome_loss
        + config.q_outcome_weight * q_outcome_loss
    )
    metrics = TrainMetrics(
        policy_loss=policy_loss,
        value_loss=value_outcome_loss,
        value_outcome_loss=value_outcome_loss,
        q_outcome_loss=q_outcome_loss,
        alpha_V_concentration=alpha_v_concentration,
        alpha_Q_concentration=alpha_q_concentration,
    )
    return total_loss, metrics


def train(model: nnx.Module, optimizer: nnx.Optimizer, data: Sample, config):
    def loss_fn(model: nnx.Module):
        output = model(data.obs, train=True)
        if len(output) == 2:
            logits, value = output
            policy_loss, value_loss = _compute_losses(logits, value, data)
            metrics = TrainMetrics(
                policy_loss=policy_loss,
                value_loss=value_loss,
                value_outcome_loss=jnp.zeros_like(value_loss),
                q_outcome_loss=jnp.zeros_like(value_loss),
                alpha_V_concentration=jnp.zeros_like(value_loss),
                alpha_Q_concentration=jnp.zeros_like(value_loss),
            )
            return policy_loss + value_loss, metrics

        logits, alpha_v, alpha_q = output
        return _compute_dirichlet_losses(logits, alpha_v, alpha_q, data, config)

    (_, metrics), grads = nnx.value_and_grad(
        loss_fn,
        has_aux=True,
    )(model)
    optimizer.update(model, grads)
    return metrics
