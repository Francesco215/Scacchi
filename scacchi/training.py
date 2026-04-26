"""Training utilities."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jaxtyping import Array, Float, PRNGKeyArray

from scacchi.types import TrainingBatch


class LossMetrics(NamedTuple):
    loss: Float[Array, ""]
    policy_loss: Float[Array, ""]
    value_loss: Float[Array, ""]


def init_optimizer(model: nnx.Module, tx: optax.GradientTransformation) -> nnx.Optimizer:
    """Initialize an NNX optimizer for model parameters."""

    return nnx.Optimizer(model, tx, wrt=nnx.Param)


def loss_fn(
    model: nnx.Module,
    batch: TrainingBatch,
) -> tuple[Float[Array, ""], LossMetrics]:
    """Policy cross-entropy plus masked value MSE."""

    logits, value = model(batch.observation, train=True)
    policy_loss = optax.softmax_cross_entropy(logits, batch.policy_target).mean()
    value_loss = optax.l2_loss(value, batch.value_target)
    mask = batch.value_mask.astype(value_loss.dtype)
    value_loss = jnp.sum(value_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    loss = policy_loss + value_loss
    return loss, LossMetrics(loss=loss, policy_loss=policy_loss, value_loss=value_loss)


def make_train_step():
    """Create a jitted train step for an NNX model and optimizer."""

    @nnx.jit
    def train_step(
        model: nnx.Module, optimizer: nnx.Optimizer, batch: TrainingBatch
    ) -> LossMetrics:
        (loss, metrics), grads = nnx.value_and_grad(
            loss_fn,
            argnums=nnx.DiffState(0, nnx.Param),
            has_aux=True,
        )(model, batch)
        del loss
        optimizer.update(model, grads)
        return metrics

    return train_step


def shuffle_batch(batch: TrainingBatch, rng_key: PRNGKeyArray) -> TrainingBatch:
    """Shuffle a flat training batch along the leading axis."""

    order = jax.random.permutation(rng_key, batch.observation.shape[0])
    return TrainingBatch(
        observation=batch.observation[order],
        policy_target=batch.policy_target[order],
        value_target=batch.value_target[order],
        value_mask=batch.value_mask[order],
    )


def take_batch(batch: TrainingBatch, batch_size: int) -> TrainingBatch:
    """Take the first minibatch, clamped to the available sample count."""

    n = min(int(batch_size), int(batch.observation.shape[0]))
    return TrainingBatch(
        observation=batch.observation[:n],
        policy_target=batch.policy_target[:n],
        value_target=batch.value_target[:n],
        value_mask=batch.value_mask[:n],
    )
