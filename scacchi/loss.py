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
    wdl_tgt: jax.Array
    played_action: jax.Array
    mask: jax.Array


class LossOutputs(NamedTuple):
    policy_loss: jax.Array
    value_outcome_loss: jax.Array
    q_outcome_loss: jax.Array


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
        # value_tgt ∈ {-1, 0, +1} from player-to-move perspective. Convert to WDL one-hot
        # indexed [L, D, W] = [0, 1, 2].
        wdl_tgt = jax.nn.one_hot(jnp.round(value_tgt).astype(jnp.int32) + 1, 3)

        return Sample(
            obs=data.obs,
            policy_tgt=data.policy_target,
            wdl_tgt=wdl_tgt,
            played_action=data.played_action,
            mask=value_mask,
        )

    return compute_loss_input


def _wdl_mean(alpha: jax.Array) -> jax.Array:
    return alpha / jnp.sum(alpha, axis=-1, keepdims=True)


def _xent_against_wdl(target_one_hot: jax.Array, pred_mean: jax.Array) -> jax.Array:
    log_pred = jnp.log(jnp.clip(pred_mean, 1e-8, 1.0))
    return -(target_one_hot * log_pred).sum(axis=-1)


def make_train_step(config):
    def train(model: AZNet, optimizer: nnx.Optimizer, data: Sample) -> LossOutputs:
        def loss_fn(model: AZNet):
            logits, alpha_V, alpha_Q = model(data.obs, train=True)
            mask_f = data.mask.astype(logits.dtype)
            mask_sum = jnp.maximum(mask_f.sum(), 1.0)

            policy_loss = optax.softmax_cross_entropy(logits, data.policy_tgt)
            policy_loss = policy_loss.mean()

            v_mean = _wdl_mean(alpha_V)
            value_outcome_loss = _xent_against_wdl(data.wdl_tgt, v_mean)
            value_outcome_loss = (value_outcome_loss * mask_f).sum() / mask_sum

            batch_idx = jnp.arange(alpha_Q.shape[0])
            played_alpha_Q = alpha_Q[batch_idx, data.played_action]  # [B, 3]
            q_mean = _wdl_mean(played_alpha_Q)
            q_outcome_loss = _xent_against_wdl(data.wdl_tgt, q_mean)
            q_outcome_loss = (q_outcome_loss * mask_f).sum() / mask_sum

            total = (
                config.policy_loss_weight * policy_loss
                + config.value_outcome_weight * value_outcome_loss
                + config.q_outcome_weight * q_outcome_loss
            )
            return total, LossOutputs(
                policy_loss=policy_loss,
                value_outcome_loss=value_outcome_loss,
                q_outcome_loss=q_outcome_loss,
            )

        (_, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return losses

    return train
