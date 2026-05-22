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
    ixs = jax.random.permutation(rng_key, jnp.arange(samples.obs.shape[0]))
    samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)

    num_updates = samples.obs.shape[0] // training_batch_size
    num_train_samples = num_updates * training_batch_size
    samples = jax.tree_util.tree_map(lambda x: x[:num_train_samples], samples)
    minibatches = jax.tree_util.tree_map(
        lambda x: x.reshape((num_updates, training_batch_size) + x.shape[1:]),
        samples,
    )
    return minibatches


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
