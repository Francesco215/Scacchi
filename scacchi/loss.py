from typing import NamedTuple

import chex
from flax import nnx
import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln
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
    beta_Q_target: jax.Array
    beta_V_target: jax.Array
    q_loss_weight: jax.Array
    policy_loss_mask: jax.Array | None = None
    value_loss_mask: jax.Array | None = None
    search_loss_mask: jax.Array | None = None
    outcome_mask: jax.Array | None = None

    @property
    def q_evidence_mass(self) -> jax.Array:
        return self.q_loss_weight


class TrainMetrics(NamedTuple):
    policy_loss: jax.Array
    value_loss: jax.Array
    policy_nll_loss: jax.Array
    policy_kl_hat: jax.Array
    policy_target_entropy: jax.Array
    value_dir_kl_loss: jax.Array
    q_dir_kl_loss: jax.Array
    value_outcome_loss: jax.Array
    q_outcome_loss: jax.Array
    alpha_V_concentration: jax.Array
    alpha_Q_concentration: jax.Array
    q_loss_weight_mean: jax.Array

    @property
    def q_evidence_mass_mean(self) -> jax.Array:
        return self.q_loss_weight_mean


def make_compute_loss_input(config):
    def compute_loss_input(data: SelfplayOutput) -> Sample:
        value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1
        legal_policy_mask = jnp.any(data.legal_action_mask, axis=-1)
        policy_target_mask = jnp.sum(data.action_weights, axis=-1) > 0
        search_loss_mask = (
            data.search_loss_mask
            if data.search_loss_mask is not None
            else policy_target_mask
        )

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

        sample = Sample(
            obs=data.obs,
            policy_tgt=data.action_weights,
            value_tgt=value_tgt,
            played_action=data.played_action,
            policy_mask=data.legal_action_mask,
            value_mask=value_mask,
            beta_Q_target=data.beta_Q_target,
            beta_V_target=data.beta_V_target,
            q_loss_weight=data.q_loss_weight,
            policy_loss_mask=legal_policy_mask & search_loss_mask,
            value_loss_mask=search_loss_mask,
            search_loss_mask=search_loss_mask,
            outcome_mask=value_mask,
        )
        if data.tree_data is None:
            return sample

        tree = data.tree_data

        def flatten_root(x: jax.Array) -> jax.Array:
            return x.reshape((-1, *x.shape[2:]))

        def wrap_rows(x: jax.Array) -> jax.Array:
            return x[None, ...]

        root_obs = flatten_root(sample.obs)
        tree_obs = flatten_root(tree.obs)
        root_policy_tgt = flatten_root(sample.policy_tgt)
        tree_policy_tgt = flatten_root(tree.action_weights)
        root_value_tgt = flatten_root(sample.value_tgt)
        tree_value_tgt = flatten_root(tree.value_tgt)
        root_played_action = flatten_root(sample.played_action)
        tree_played_action = flatten_root(tree.played_action)
        root_policy_mask = flatten_root(sample.policy_mask)
        tree_policy_mask = flatten_root(tree.legal_action_mask)
        root_beta_q = flatten_root(sample.beta_Q_target)
        tree_beta_q = flatten_root(tree.beta_Q_target)
        root_beta_v = flatten_root(sample.beta_V_target)
        tree_beta_v = flatten_root(tree.beta_V_target)
        root_q_weight = flatten_root(sample.q_loss_weight)
        tree_q_weight = flatten_root(tree.q_loss_weight)
        root_policy_loss_mask = flatten_root(sample.policy_loss_mask)
        tree_policy_loss_mask = flatten_root(tree.policy_loss_mask)
        root_value_loss_mask = flatten_root(sample.value_loss_mask)
        tree_value_loss_mask = flatten_root(tree.value_loss_mask)
        root_search_loss_mask = flatten_root(sample.search_loss_mask)
        tree_search_loss_mask = flatten_root(tree.search_loss_mask)
        root_outcome_mask = flatten_root(sample.outcome_mask)
        tree_outcome_mask = flatten_root(tree.outcome_mask)

        return Sample(
            obs=wrap_rows(jnp.concatenate([root_obs, tree_obs], axis=0)),
            policy_tgt=wrap_rows(jnp.concatenate([root_policy_tgt, tree_policy_tgt], axis=0)),
            value_tgt=wrap_rows(jnp.concatenate([root_value_tgt, tree_value_tgt], axis=0)),
            played_action=wrap_rows(jnp.concatenate([root_played_action, tree_played_action], axis=0)),
            policy_mask=wrap_rows(jnp.concatenate([root_policy_mask, tree_policy_mask], axis=0)),
            value_mask=wrap_rows(jnp.concatenate([root_value_loss_mask, tree_value_loss_mask], axis=0)),
            beta_Q_target=wrap_rows(jnp.concatenate([root_beta_q, tree_beta_q], axis=0)),
            beta_V_target=wrap_rows(jnp.concatenate([root_beta_v, tree_beta_v], axis=0)),
            q_loss_weight=wrap_rows(jnp.concatenate([root_q_weight, tree_q_weight], axis=0)),
            policy_loss_mask=wrap_rows(
                jnp.concatenate([root_policy_loss_mask, tree_policy_loss_mask], axis=0)
            ),
            value_loss_mask=wrap_rows(
                jnp.concatenate([root_value_loss_mask, tree_value_loss_mask], axis=0)
            ),
            search_loss_mask=wrap_rows(
                jnp.concatenate([root_search_loss_mask, tree_search_loss_mask], axis=0)
            ),
            outcome_mask=wrap_rows(jnp.concatenate([root_outcome_mask, tree_outcome_mask], axis=0)),
        )

    return compute_loss_input


def _masked_mean(loss: jax.Array, mask: jax.Array) -> jax.Array:
    mask = mask.astype(loss.dtype)
    return jnp.sum(loss * mask) / jnp.maximum(jnp.sum(mask), 1)


def _mask_or(mask: jax.Array | None, fallback: jax.Array) -> jax.Array:
    return fallback if mask is None else mask


def _compute_losses(logits: jax.Array, value: jax.Array, data: Sample) -> tuple[jax.Array, jax.Array]:
    policy_loss_mask = _mask_or(data.policy_loss_mask, data.value_mask)
    value_loss_mask = _mask_or(data.value_loss_mask, data.value_mask)
    policy_loss = optax.softmax_cross_entropy(logits, data.policy_tgt, where=data.policy_mask)
    policy_loss = _masked_mean(policy_loss, policy_loss_mask)
    value_loss = optax.l2_loss(value, data.value_tgt)
    value_loss = _masked_mean(value_loss, value_loss_mask)
    return policy_loss, value_loss


def _outcome_target(value_tgt: jax.Array, num_outcomes: int) -> jax.Array:
    rounded = jnp.round(value_tgt).astype(jnp.int32)
    if num_outcomes == 2:
        outcome_idx = (rounded + 1) // 2
    elif num_outcomes == 3:
        outcome_idx = rounded + 1
    else:
        raise ValueError(f"unsupported outcome count: {num_outcomes}")
    return jax.nn.one_hot(outcome_idx, num_outcomes)


def _categorical_ce_from_probs(probs: jax.Array, target: jax.Array) -> jax.Array:
    log_probs = jnp.log(jnp.clip(probs, jnp.finfo(probs.dtype).tiny, 1.0))
    return -jnp.sum(target * log_probs, axis=-1)


def _categorical_entropy_from_probs(probs: jax.Array, mask: jax.Array) -> jax.Array:
    log_probs = jnp.log(jnp.clip(probs, jnp.finfo(probs.dtype).tiny, 1.0))
    entropy_terms = jnp.where(mask, probs * log_probs, 0.0)
    return -jnp.sum(entropy_terms, axis=-1)


def _dirichlet_kl(beta: jax.Array, alpha: jax.Array) -> jax.Array:
    dtype = jnp.result_type(beta, alpha)
    eps = jnp.asarray(1e-6, dtype=dtype)
    beta = jax.lax.stop_gradient(jnp.maximum(beta.astype(dtype), eps))
    alpha = jnp.maximum(alpha.astype(dtype), eps)

    beta_sum = jnp.sum(beta, axis=-1)
    alpha_sum = jnp.sum(alpha, axis=-1)
    return (
        gammaln(beta_sum)
        - gammaln(alpha_sum)
        + jnp.sum(gammaln(alpha) - gammaln(beta), axis=-1)
        + jnp.sum(
            (beta - alpha) * (digamma(beta) - digamma(beta_sum)[..., None]),
            axis=-1,
        )
    )


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
    policy_loss_mask = _mask_or(data.policy_loss_mask, data.value_mask)
    value_loss_mask = _mask_or(data.value_loss_mask, data.value_mask)
    search_loss_mask = _mask_or(data.search_loss_mask, policy_loss_mask)
    outcome_mask = _mask_or(data.outcome_mask, data.value_mask)
    policy_loss = optax.softmax_cross_entropy(logits, data.policy_tgt, where=data.policy_mask)
    policy_loss = _masked_mean(policy_loss, policy_loss_mask)
    policy_target_entropy = _categorical_entropy_from_probs(data.policy_tgt, data.policy_mask)
    policy_target_entropy = _masked_mean(policy_target_entropy, policy_loss_mask)
    policy_kl_hat = jax.lax.stop_gradient(policy_loss - policy_target_entropy)

    value_dir_kl = _dirichlet_kl(data.beta_V_target, alpha_v)
    value_dir_kl_loss = _masked_mean(value_dir_kl, value_loss_mask)

    q_dir_kl = _dirichlet_kl(data.beta_Q_target, alpha_q)
    q_weights = jnp.where(
        data.policy_mask & search_loss_mask[..., None],
        data.q_loss_weight,
        0.0,
    )
    q_eps = jnp.asarray(jnp.finfo(q_dir_kl.dtype).eps, dtype=q_dir_kl.dtype)
    q_dir_kl_loss = jnp.sum(q_weights * q_dir_kl) / jnp.maximum(jnp.sum(q_weights), q_eps)

    outcome_tgt = _outcome_target(data.value_tgt, alpha_v.shape[-1])
    value_outcome_loss = _categorical_ce_from_probs(outcome_mean(alpha_v), outcome_tgt)
    value_outcome_loss = _masked_mean(value_outcome_loss, outcome_mask)

    played_alpha_q = _gather_played_action(alpha_q, data.played_action)
    q_outcome_loss = _categorical_ce_from_probs(outcome_mean(played_alpha_q), outcome_tgt)
    q_outcome_loss = _masked_mean(q_outcome_loss, outcome_mask)

    alpha_v_concentration = _masked_mean(jnp.sum(alpha_v, axis=-1), value_loss_mask)
    alpha_q_concentration = _masked_mean(
        jnp.sum(alpha_q, axis=-1),
        q_weights > 0,
    )
    q_loss_weight_mean = _masked_mean(data.q_loss_weight, q_weights > 0)

    total_loss = (
        config.policy_loss_weight * policy_loss
        + config.value_dir_kl_weight * value_dir_kl_loss
        + config.q_dir_kl_weight * q_dir_kl_loss
        + config.value_outcome_weight * value_outcome_loss
        + config.q_outcome_weight * q_outcome_loss
    )
    metrics = TrainMetrics(
        policy_loss=policy_loss,
        value_loss=value_dir_kl_loss,
        policy_nll_loss=policy_loss,
        policy_kl_hat=policy_kl_hat,
        policy_target_entropy=policy_target_entropy,
        value_dir_kl_loss=value_dir_kl_loss,
        q_dir_kl_loss=q_dir_kl_loss,
        value_outcome_loss=value_outcome_loss,
        q_outcome_loss=q_outcome_loss,
        alpha_V_concentration=alpha_v_concentration,
        alpha_Q_concentration=alpha_q_concentration,
        q_loss_weight_mean=q_loss_weight_mean,
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
                policy_nll_loss=policy_loss,
                policy_kl_hat=jnp.zeros_like(policy_loss),
                policy_target_entropy=jnp.zeros_like(policy_loss),
                value_dir_kl_loss=jnp.zeros_like(value_loss),
                q_dir_kl_loss=jnp.zeros_like(value_loss),
                value_outcome_loss=jnp.zeros_like(value_loss),
                q_outcome_loss=jnp.zeros_like(value_loss),
                alpha_V_concentration=jnp.zeros_like(value_loss),
                alpha_Q_concentration=jnp.zeros_like(value_loss),
                q_loss_weight_mean=jnp.zeros_like(value_loss),
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
