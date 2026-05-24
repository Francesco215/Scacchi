from flax import nnx
import jax
import jax.numpy as jnp

from .loss import Sample, TrainMetrics, make_compute_loss_input, train
from .play import make_selfplay
from .posterior_tree import is_posterior_tree_policy


def make_minibatches(
    samples: Sample,
    rng_key: jax.Array,
    training_batch_size: int,
) -> Sample:
    samples = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, *x.shape[2:])),
        samples,
    )
    num_rows = samples.obs.shape[0]
    num_updates = num_rows // training_batch_size
    num_train_samples = num_updates * training_batch_size
    active_mask = _active_sample_rows(samples)
    active_indices = jnp.nonzero(active_mask, size=num_rows, fill_value=0)[0]
    active_count = jnp.sum(active_mask.astype(jnp.int32))
    safe_active_count = jnp.maximum(active_count, 1)
    draw_key, fallback_key = jax.random.split(rng_key)
    raw_draws = jax.random.randint(
        draw_key,
        (num_train_samples,),
        minval=0,
        maxval=max(num_rows, 1),
        dtype=jnp.int32,
    )
    active_ixs = active_indices[raw_draws % safe_active_count]
    fallback_ixs = jax.random.permutation(fallback_key, jnp.arange(num_rows))[
        :num_train_samples
    ]
    ixs = jnp.where(active_count > 0, active_ixs, fallback_ixs)
    samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)
    minibatches = jax.tree_util.tree_map(
        lambda x: x.reshape((num_updates, training_batch_size) + x.shape[1:]),
        samples,
    )
    return minibatches


def _active_sample_rows(samples: Sample) -> jax.Array:
    policy_mask = (
        samples.value_mask if samples.policy_loss_mask is None else samples.policy_loss_mask
    )
    value_mask = (
        samples.value_mask if samples.value_loss_mask is None else samples.value_loss_mask
    )
    outcome_mask = samples.value_mask if samples.outcome_mask is None else samples.outcome_mask
    return policy_mask | value_mask | outcome_mask


def train_minibatches(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    minibatches: Sample,
    config,
) -> TrainMetrics:
    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_step(state, minibatch):
        model, optimizer = state
        metrics = train(model, optimizer, minibatch, config)
        return (model, optimizer), metrics

    _, metrics = scan_step((model, optimizer), minibatches)
    return metrics


def make_training_iteration(env, config):
    selfplay = make_selfplay(env, config)
    compute_loss_input = make_compute_loss_input(config)

    if is_posterior_tree_policy(config.search_policy):
        @nnx.jit
        def train_from_selfplay_data(
            model: nnx.Module,
            optimizer: nnx.Optimizer,
            data,
            perm_key: jax.Array,
        ) -> TrainMetrics:
            samples = compute_loss_input(data)
            minibatches = make_minibatches(samples, perm_key, config.training_batch_size)
            return train_minibatches(model, optimizer, minibatches, config)

        def training_iteration(
            model: nnx.Module,
            optimizer: nnx.Optimizer,
            rng_key: jax.Array,
        ) -> TrainMetrics:
            selfplay_key, perm_key = jax.random.split(rng_key)
            data = selfplay(model, selfplay_key)
            return train_from_selfplay_data(model, optimizer, data, perm_key)

        return training_iteration

    @nnx.jit
    def training_iteration(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        rng_key: jax.Array,
    ) -> TrainMetrics:
        selfplay_key, perm_key = jax.random.split(rng_key)
        data = selfplay(model, selfplay_key)
        samples = compute_loss_input(data)
        minibatches = make_minibatches(samples, perm_key, config.training_batch_size)
        return train_minibatches(model, optimizer, minibatches, config)

    return training_iteration
