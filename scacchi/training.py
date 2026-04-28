"""Training utilities."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax import nnx

from scacchi.config import TrainConfig
from scacchi.selfplay import run_selfplay
from scacchi.types import SelfplayBatch, TrainingBatch


class LossMetrics(NamedTuple):
    loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array


class UpdateMetrics(NamedTuple):
    loss: jax.Array
    policy_loss: jax.Array
    value_loss: jax.Array
    per_update: LossMetrics
    samples: jax.Array
    num_updates: jax.Array
    value_mask_frac: jax.Array
    mean_value_target: jax.Array
    mean_abs_value_target: jax.Array


def init_optimizer(model: nnx.Module, tx: optax.GradientTransformation) -> nnx.Optimizer:
    return nnx.Optimizer(model, tx, wrt=nnx.Param)


def compute_training_batch(data: SelfplayBatch) -> TrainingBatch:
    max_num_steps, batch_size = data.reward.shape
    value_mask = jnp.cumsum(data.terminated[::-1], axis=0)[::-1] >= 1

    def body_fn(carry, i):
        ix = max_num_steps - i - 1
        value = data.reward[ix] + data.discount[ix] * carry
        return value, value

    _, value_target = jax.lax.scan(
        body_fn,
        jnp.zeros(batch_size, dtype=data.reward.dtype),
        jnp.arange(max_num_steps),
    )

    return TrainingBatch(
        observation=data.observation.reshape((-1, *data.observation.shape[2:])),
        policy_target=data.action_weights.reshape((-1, data.action_weights.shape[-1])),
        value_target=value_target[::-1].reshape((-1,)),
        value_mask=value_mask.reshape((-1,)),
    )


def _minibatches(batch: TrainingBatch, batch_size: int, minibatch_size: int, rng_key) -> TrainingBatch:
    total = int(batch.observation.shape[0])
    num_accum = batch_size // minibatch_size
    num_updates = max(1, total // batch_size)
    order = jax.random.permutation(rng_key, total)

    def reshape(x):
        x = x[order][: num_updates * batch_size]
        return x.reshape((num_updates, num_accum, minibatch_size, *x.shape[1:]))

    return TrainingBatch(
        observation=reshape(batch.observation),
        policy_target=reshape(batch.policy_target),
        value_target=reshape(batch.value_target),
        value_mask=reshape(batch.value_mask),
    )


def loss_fn(model: nnx.Module, batch: TrainingBatch):
    logits, value = model(batch.observation, train=True)
    policy_loss = optax.softmax_cross_entropy(logits, batch.policy_target).mean()
    value_loss = jnp.mean(optax.l2_loss(value, batch.value_target) * batch.value_mask)
    loss = policy_loss + value_loss
    return loss, LossMetrics(loss, policy_loss, value_loss)


def _update(model: nnx.Module, optimizer: nnx.Optimizer, data: SelfplayBatch,
            batch_size: int, minibatch_size: int, rng_key):
    minibatches = _minibatches(compute_training_batch(data), batch_size, minibatch_size, rng_key)
    grad_fn = nnx.value_and_grad(loss_fn, argnums=nnx.DiffState(0, nnx.Param), has_aux=True)
    num_accum = minibatches.observation.shape[1]

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0), unroll=1)
    def train_batch(carry, batch):
        model, optimizer = carry
        zero_grads = jax.tree.map(jnp.zeros_like, nnx.state(model, nnx.Param))

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0), unroll=1)
        def accum_step(carry, minibatch):
            model, accum_grads = carry
            (_, metric), grads = grad_fn(model, minibatch)
            accum_grads = jax.tree.map(lambda a, g: a + g, accum_grads, grads)
            return (model, accum_grads), metric

        (model, accum_grads), metrics = accum_step((model, zero_grads), batch)
        avg_grads = jax.tree.map(lambda g: g / num_accum, accum_grads)
        optimizer.update(model, avg_grads)

        mean_metric = jax.tree.map(lambda x: jnp.mean(x, axis=0), metrics)
        return (model, optimizer), mean_metric

    (_, _), per_update = train_batch((model, optimizer), minibatches)

    value_mask = minibatches.value_mask.reshape((-1,)).astype(jnp.float32)
    value_target = minibatches.value_target.reshape((-1,))

    return UpdateMetrics(
        loss=jnp.mean(per_update.loss),
        policy_loss=jnp.mean(per_update.policy_loss),
        value_loss=jnp.mean(per_update.value_loss),
        per_update=per_update,
        samples=jnp.asarray(value_mask.shape[0]),
        num_updates=jnp.asarray(minibatches.observation.shape[0]),
        value_mask_frac=jnp.mean(value_mask),
        mean_value_target=jnp.mean(value_target),
        mean_abs_value_target=jnp.sum(jnp.abs(value_target) * value_mask) / jnp.maximum(jnp.sum(value_mask), 1.0),
    )


def make_iteration_step(env, cfg: TrainConfig):
    @nnx.jit
    def iteration_step(model: nnx.Module, optimizer: nnx.Optimizer, rng_key):
        selfplay_key, shuffle_key = jax.random.split(rng_key)
        data = run_selfplay(env=env, model=model, rng_key=selfplay_key, cfg=cfg)
        return _update(model, optimizer, data, cfg.batch_size, cfg.minibatch_size, shuffle_key)

    return iteration_step
