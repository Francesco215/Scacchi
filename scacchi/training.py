"""Training utilities."""

from __future__ import annotations

from typing import NamedTuple, cast

import jax
import jax.numpy as jnp
import optax
from flax import nnx
from jaxtyping import Array, Float, PRNGKeyArray

from scacchi.search import predict
from scacchi.types import ModelGraphDef, TrainState, TrainingBatch


class LossMetrics(NamedTuple):
    loss: Float[Array, ""]
    policy_loss: Float[Array, ""]
    value_loss: Float[Array, ""]


def init_train_state(model: nnx.Module, tx: optax.GradientTransformation) -> tuple[
    ModelGraphDef, TrainState
]:
    """Split an NNX model into a pure graph/state pair and initialize Optax."""

    graphdef, params = nnx.split(model)
    return graphdef, TrainState(params=params, opt_state=tx.init(params))


def loss_fn(
    graphdef: ModelGraphDef,
    params: nnx.State,
    batch: TrainingBatch,
) -> tuple[Float[Array, ""], LossMetrics]:
    """Policy cross-entropy plus masked value MSE."""

    logits, value = predict(graphdef, params, batch.observation, train=True)
    policy_loss = optax.softmax_cross_entropy(logits, batch.policy_target).mean()
    value_loss = optax.l2_loss(value, batch.value_target)
    mask = batch.value_mask.astype(value_loss.dtype)
    value_loss = jnp.sum(value_loss * mask) / jnp.maximum(jnp.sum(mask), 1.0)
    loss = policy_loss + value_loss
    return loss, LossMetrics(loss=loss, policy_loss=policy_loss, value_loss=value_loss)


def make_train_step(graphdef: ModelGraphDef, tx: optax.GradientTransformation):
    """Create a jitted train step closing over static model graph and optimizer."""

    @jax.jit
    def train_step(state: TrainState, batch: TrainingBatch) -> tuple[TrainState, LossMetrics]:
        (loss, metrics), grads = jax.value_and_grad(loss_fn, argnums=1, has_aux=True)(
            graphdef, state.params, batch
        )
        del loss
        updates, opt_state = tx.update(grads, state.opt_state, state.params)
        params = cast(nnx.State, optax.apply_updates(state.params, updates))
        return TrainState(params=params, opt_state=opt_state), metrics

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
