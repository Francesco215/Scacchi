"""Training utilities."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from scacchi.types import TrainingBatch


class LossMetrics(NamedTuple):
    loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array


def init_optimizer(model: nnx.Module, tx: optax.GradientTransformation) -> nnx.Optimizer:
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


def loss_fn(model: nnx.Module, batch: TrainingBatch):
    logits, value = model(batch.observation, train=True)
    policy_loss = optax.softmax_cross_entropy(logits, batch.policy_target).mean()
    value_loss = jnp.mean(optax.l2_loss(value, batch.value_target) * batch.value_mask)
    loss = policy_loss + value_loss
    return loss, LossMetrics(loss, policy_loss, value_loss)


def make_train_step():
    @nnx.jit
    def train_step(model: nnx.Module, optimizer: nnx.Optimizer, batch: TrainingBatch):
        (_, metrics), grads = nnx.value_and_grad(
            loss_fn,
            argnums=nnx.DiffState(0, nnx.Param),
            has_aux=True,
        )(model, batch)
        optimizer.update(model, grads)
        return metrics

    return train_step


def make_minibatches(batch: TrainingBatch, batch_size: int, rng_key) -> TrainingBatch:
    order = jax.random.permutation(rng_key, batch.observation.shape[0])
    size = min(int(batch_size), int(batch.observation.shape[0]))
    num_updates = max(1, int(batch.observation.shape[0]) // size)

    def reshape(x):
        x = x[order][: num_updates * size]
        return x.reshape((num_updates, size, *x.shape[1:]))

    return TrainingBatch(
        observation=reshape(batch.observation),
        policy_target=reshape(batch.policy_target),
        value_target=reshape(batch.value_target),
        value_mask=reshape(batch.value_mask),
    )
