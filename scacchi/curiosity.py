"""Self-play that is curious about what the network has not learned.

A row's bits are the KL divergence from the network's prediction to what
search concluded about the position, summed over the policy, V and supervised
Q targets with the loss weights: the information the search added.

A head predicts, for every move, the bits of the position that move leads to.
It is trained by regression on what self-play then measured, one ply ahead:
no reward is propagated through time. Self-play can tilt its move choice
towards moves whose next position is expected to carry more bits. The search
target of a position does not depend on how the game reached it, so the tilt
changes which positions are seen and leaves every target as it was.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import optax

from . import binned
from .dirichlet_mctx.native_targets import TARGET_BINNED
from .loss import (
    Sample,
    _categorical_entropy_from_probs,
    _mask_or,
    _native_binned_loss,
    _native_target_fields,
)

_LN2 = math.log(2.0)


def row_bits(output: binned.BinnedOutput, data: Sample, config) -> tuple[jax.Array, jax.Array]:
    """Per row: KL(search targets || prediction) in bits, and whether it is trained on.

    The masks, weights and the Q reduction are those of the training loss, so
    the mean of ``bits * ln 2`` over the valid rows is the search loss minus
    the entropy of the targets.
    """
    losses = config.training.losses
    fields = _native_target_fields(data, data.beta_Q_target, data.beta_V_target)
    policy_mask = _mask_or(data.policy_loss_mask, data.value_mask)
    value_mask = _mask_or(data.value_loss_mask, data.value_mask)
    search_mask = _mask_or(data.search_loss_mask, policy_mask)

    def target_entropy(target, kind, weight):
        safe = jnp.maximum(target, jnp.finfo(target.dtype).tiny)
        entropy = -jnp.sum(jnp.where(target > 0, target * jnp.log(safe), 0.0), axis=-1)
        return jnp.where(jnp.asarray(kind) == TARGET_BINNED, entropy * weight, 0.0)  # a proven outcome has none

    policy_kl = optax.softmax_cross_entropy(output.logits, data.policy_tgt, where=data.policy_mask) - _categorical_entropy_from_probs(
        data.policy_tgt, data.policy_mask
    )
    value_kl = _native_binned_loss(
        data.beta_V_target, output.value_probs, fields["v_target_kind"], fields["v_target_outcome"], fields["v_target_weight"]
    ) - target_entropy(data.beta_V_target, fields["v_target_kind"], fields["v_target_weight"])
    q_kl = _native_binned_loss(
        data.beta_Q_target, output.action_probs, fields["q_target_kind"], fields["q_target_outcome"], fields["q_target_weight"]
    ) - target_entropy(data.beta_Q_target, fields["q_target_kind"], fields["q_target_weight"])
    q_row_mask = value_mask if losses.loss_mask_mode == "value" else search_mask
    pair_mask = data.q_supervised_pair_mask & data.policy_mask & q_row_mask[..., None] & jnp.isfinite(q_kl)
    valid = policy_mask | value_mask | q_row_mask
    # The loss averages Q over pairs; a row's share is its pair sum over the mean pair count.
    pairs_per_row = jnp.maximum(jnp.sum(pair_mask) / jnp.maximum(jnp.sum(valid), 1), 1e-6)
    nats = (
        losses.policy_weight * jnp.where(policy_mask, policy_kl, 0.0)
        + losses.value_dir_kl_weight * jnp.where(value_mask & jnp.isfinite(value_kl), value_kl, 0.0)
        + losses.q_dir_kl_weight * jnp.sum(jnp.where(pair_mask, q_kl, 0.0), axis=-1) / pairs_per_row
    )
    return jnp.where(valid, jnp.nan_to_num(nats) / _LN2, 0.0).astype(jnp.float32), valid


def next_bits_targets(bits: jax.Array, valid: jax.Array, terminated: jax.Array) -> tuple[jax.Array, jax.Array]:
    """For rows ``[game lane, step]``: the bits of the position the played move led to.

    There is a target where the move did not end the game and both positions
    are trained on; a move that ends the game leads to no position to search.
    """
    following = jnp.concatenate((bits[:, 1:], jnp.zeros_like(bits[:, :1])), axis=1)
    following_valid = jnp.concatenate((valid[:, 1:], jnp.zeros_like(valid[:, :1])), axis=1)
    mask = valid & following_valid & ~terminated
    return jnp.where(mask, following, 0.0), mask


def next_bits_nll(log_predicted: jax.Array, played_action: jax.Array, target: jax.Array, mask: jax.Array) -> dict[str, jax.Array]:
    """NLL of the measured next-position bits under an exponential law.

    Bits are non-negative with an unknown scale; an exponential distribution
    with mean ``mu`` has NLL ``log mu + x / mu``, minimised at ``mu = E[x]``.
    The head outputs ``log mu`` for every move; only the played one has a target.
    """
    log_mu = jnp.take_along_axis(log_predicted, played_action[..., None].astype(jnp.int32), axis=-1)[..., 0].astype(jnp.float32)
    weight = mask.astype(jnp.float32)
    count = jnp.maximum(jnp.sum(weight), 1.0)
    mean = lambda x: jnp.sum(x * weight) / count
    predicted = jnp.exp(log_mu)
    nll = mean(log_mu + target * jnp.exp(-log_mu))
    centred_p, centred_t = predicted - mean(predicted), target - mean(target)
    correlation = mean(centred_p * centred_t) / jnp.sqrt(jnp.maximum(mean(centred_p**2) * mean(centred_t**2), 1e-12))
    return {"nll": nll, "predicted": mean(predicted), "realised": mean(target), "correlation": correlation}


def curious_action(key: jax.Array, policy: jax.Array, legal: jax.Array, next_bits: jax.Array, temperature: float) -> jax.Array:
    """A move drawn with probability proportional to ``policy * exp(next_bits / T)``.

    ``policy`` is the search's probability that each move is best; moves it
    gives no probability stay unplayed. ``temperature`` is in bits.
    """
    logits = jnp.where(legal & (policy > 0), jnp.log(jnp.maximum(policy, 1e-30)) + next_bits.astype(jnp.float32) / temperature, -jnp.inf)
    return jax.random.categorical(key, logits, axis=-1)
