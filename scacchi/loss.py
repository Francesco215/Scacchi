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
    q_evidence_sum: chex.Array  # [B, A, 3] Σ_n c_n · y_n^aligned per root action


class LossOutputs(NamedTuple):
    policy_loss: jax.Array
    value_outcome_loss: jax.Array
    q_outcome_loss: jax.Array
    value_search_mean_loss: jax.Array
    q_search_mean_loss: jax.Array
    value_dir_kl_loss: jax.Array
    q_dir_kl_loss: jax.Array


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
            q_evidence_sum=data.q_evidence_sum,
        )

    return compute_loss_input


def _wdl_mean(alpha: jax.Array) -> jax.Array:
    return alpha / jnp.sum(alpha, axis=-1, keepdims=True)


def _xent_against_wdl(target_one_hot: jax.Array, pred_mean: jax.Array) -> jax.Array:
    log_pred = jnp.log(jnp.clip(pred_mean, 1e-8, 1.0))
    return -(target_one_hot * log_pred).sum(axis=-1)


def _dirichlet_kl(beta: jax.Array, alpha: jax.Array) -> jax.Array:
    """KL(Dir(beta) || Dir(alpha)) per math.md Appendix A. Inputs [..., K]; output [...]."""
    beta_0 = beta.sum(-1)
    alpha_0 = alpha.sum(-1)
    digamma_diff = jax.lax.digamma(beta) - jax.lax.digamma(beta_0)[..., None]
    return (
        jax.scipy.special.gammaln(beta_0)
        - jax.scipy.special.gammaln(beta).sum(-1)
        - jax.scipy.special.gammaln(alpha_0)
        + jax.scipy.special.gammaln(alpha).sum(-1)
        + ((beta - alpha) * digamma_diff).sum(-1)
    )


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

            # Search-based mean and Dirichlet-KL losses (math.md §8.3, §8.4, Appendix A).
            q_evidence_sum = data.q_evidence_sum.astype(alpha_Q.dtype)        # [B, A, 3]
            q_evidence_w = q_evidence_sum.sum(-1)                              # [B, A]
            q_search_mask_f = (q_evidence_w > 0).astype(alpha_Q.dtype)         # [B, A]
            eps = jnp.asarray(1e-8, dtype=alpha_Q.dtype)
            q_search_mean = q_evidence_sum / jnp.maximum(q_evidence_w[..., None], eps)
            beta_Q = 1.0 + q_evidence_sum

            policy_tgt_sg = jax.lax.stop_gradient(data.policy_tgt).astype(alpha_Q.dtype)
            v_evidence_sum = (policy_tgt_sg[..., None] * q_evidence_sum).sum(-2)  # [B, 3]
            v_evidence_w = v_evidence_sum.sum(-1)                                  # [B]
            v_mask_f = (v_evidence_w > 0).astype(alpha_V.dtype)
            v_search_mean = v_evidence_sum / jnp.maximum(v_evidence_w[..., None], eps)
            beta_V = 1.0 + v_evidence_sum

            v_search_xe = _xent_against_wdl(v_search_mean, v_mean)                 # [B]
            value_search_mean_loss = (v_search_xe * v_mask_f).sum() / jnp.maximum(v_mask_f.sum(), 1.0)

            q_pred_mean = _wdl_mean(alpha_Q)                                       # [B, A, 3]
            q_search_xe = _xent_against_wdl(q_search_mean, q_pred_mean)            # [B, A]
            q_search_mean_loss = (q_search_xe * q_search_mask_f).sum() / jnp.maximum(q_search_mask_f.sum(), 1.0)

            v_dir_kl = _dirichlet_kl(jax.lax.stop_gradient(beta_V), alpha_V)       # [B]
            value_dir_kl_loss = (v_dir_kl * v_mask_f).sum() / jnp.maximum(v_mask_f.sum(), 1.0)
            q_dir_kl = _dirichlet_kl(jax.lax.stop_gradient(beta_Q), alpha_Q)       # [B, A]
            q_dir_kl_loss = (q_dir_kl * q_search_mask_f).sum() / jnp.maximum(q_search_mask_f.sum(), 1.0)

            total = (
                config.policy_loss_weight * policy_loss
                + config.value_outcome_weight * value_outcome_loss
                + config.q_outcome_weight * q_outcome_loss
                + config.value_search_weight * value_search_mean_loss
                + config.q_search_weight * q_search_mean_loss
                + config.dir_kl_weight * (value_dir_kl_loss + q_dir_kl_loss)
            )
            return total, LossOutputs(
                policy_loss=policy_loss,
                value_outcome_loss=value_outcome_loss,
                q_outcome_loss=q_outcome_loss,
                value_search_mean_loss=value_search_mean_loss,
                q_search_mean_loss=q_search_mean_loss,
                value_dir_kl_loss=value_dir_kl_loss,
                q_dir_kl_loss=q_dir_kl_loss,
            )

        (_, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return losses

    return train
