from typing import NamedTuple

import chex
from flax import nnx
import jax
import jax.numpy as jnp

from .network import AZNet
from .play import SelfplayOutput


class Sample(NamedTuple):
    obs: jax.Array
    policy_tgt: chex.Array
    mask: jax.Array
    beta_V_tgt: jax.Array
    value_tgt_mask: jax.Array
    beta_Q_tgt: jax.Array
    q_tgt_mask: jax.Array


class LossOutputs(NamedTuple):
    policy_nll_loss: jax.Array
    policy_kl_hat: jax.Array
    value_dir_kl_loss: jax.Array
    q_dir_kl_loss: jax.Array


def make_compute_loss_input(config):
    def compute_loss_input(data: SelfplayOutput) -> Sample:
        value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

        return Sample(
            obs=data.obs,
            policy_tgt=data.policy_target,
            mask=value_mask,
            beta_V_tgt=data.beta_V_target,
            value_tgt_mask=data.value_target_mask,
            beta_Q_tgt=data.beta_Q_target,
            q_tgt_mask=data.q_target_mask,
        )

    return compute_loss_input


def dirichlet_cross_entropy(beta: jax.Array, alpha: jax.Array) -> jax.Array:
    """H(Dir(beta), Dir(alpha)) = -E_{Dir(beta)}[log Dir(alpha)]. Inputs [..., K]; output [...]."""
    beta_0 = beta.sum(-1)
    alpha_0 = alpha.sum(-1)
    expected_log_x = jax.lax.digamma(beta) - jax.lax.digamma(beta_0)[..., None]
    return (
        -jax.scipy.special.gammaln(alpha_0)
        + jax.scipy.special.gammaln(alpha).sum(-1)
        - ((alpha - 1.0) * expected_log_x).sum(-1)
    )


def dirichlet_entropy(beta: jax.Array) -> jax.Array:
    """H(Dir(beta)). Inputs [..., K]; output [...]."""
    beta_0 = beta.sum(-1)
    expected_log_x = jax.lax.digamma(beta) - jax.lax.digamma(beta_0)[..., None]
    return (
        jax.scipy.special.gammaln(beta).sum(-1)
        - jax.scipy.special.gammaln(beta_0)
        - ((beta - 1.0) * expected_log_x).sum(-1)
    )


def dirichlet_kl(beta: jax.Array, alpha: jax.Array) -> jax.Array:
    """KL(Dir(beta) || Dir(alpha)) per math.md Appendix A. Inputs [..., K]; output [...]."""
    return dirichlet_cross_entropy(beta, alpha) - dirichlet_entropy(beta)


def make_train_step(config):
    def train(model: AZNet, optimizer: nnx.Optimizer, data: Sample) -> LossOutputs:
        def loss_fn(model: AZNet):
            logits, alpha_V, alpha_Q = model(data.obs, train=True)
            mask_f = data.mask.astype(logits.dtype)

            policy_log_probs = jax.nn.log_softmax(logits)
            policy_tgt_sg = jax.lax.stop_gradient(data.policy_tgt).astype(logits.dtype)
            policy_nll = -(policy_tgt_sg * policy_log_probs).sum(axis=-1)
            policy_nll_loss = policy_nll.mean()
            policy_tgt_log = jnp.log(jnp.clip(policy_tgt_sg, 1e-8, 1.0))
            policy_kl_hat = (policy_tgt_sg * (policy_tgt_log - policy_log_probs)).sum(axis=-1)
            policy_kl_hat = policy_kl_hat.mean()

            beta_V = jax.lax.stop_gradient(data.beta_V_tgt).astype(alpha_V.dtype)
            beta_Q = jax.lax.stop_gradient(data.beta_Q_tgt).astype(alpha_Q.dtype)
            v_mask_f = data.value_tgt_mask.astype(alpha_V.dtype) * mask_f
            q_mask_f = data.q_tgt_mask.astype(alpha_Q.dtype)

            v_dir_kl = dirichlet_kl(beta_V, alpha_V)
            value_dir_kl_loss = (v_dir_kl * v_mask_f).sum() / jnp.maximum(v_mask_f.sum(), 1.0)
            q_dir_kl = dirichlet_kl(beta_Q, alpha_Q)
            q_mask_f = q_mask_f * mask_f[..., None]
            q_dir_kl_loss = (q_dir_kl * q_mask_f).sum() / jnp.maximum(q_mask_f.sum(), 1.0)

            total = (
                config.policy_loss_weight * policy_nll_loss
                + config.value_loss_weight * value_dir_kl_loss
                + config.q_loss_weight * q_dir_kl_loss
            )
            return total, LossOutputs(
                policy_nll_loss=policy_nll_loss,
                policy_kl_hat=policy_kl_hat,
                value_dir_kl_loss=value_dir_kl_loss,
                q_dir_kl_loss=q_dir_kl_loss,
            )

        (_, losses), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model)
        optimizer.update(model, grads)
        return losses

    return train
