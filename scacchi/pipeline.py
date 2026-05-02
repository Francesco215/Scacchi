from flax import nnx
import jax
import jax.numpy as jnp

from .loss import Sample, make_compute_loss_input, train
from .play import make_selfplay


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
) -> tuple[jax.Array, jax.Array]:
    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_step(state, minibatch):
        model, optimizer = state
        policy_loss, value_loss = train(model, optimizer, minibatch)
        return (model, optimizer), (policy_loss, value_loss)

    _, (policy_losses, value_losses) = scan_step((model, optimizer), minibatches)
    return policy_losses, value_losses


def make_training_iteration(env, config):
    selfplay = make_selfplay(env, config)
    compute_loss_input = make_compute_loss_input(config)

    @nnx.jit
    def training_iteration(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        rng_key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        selfplay_key, perm_key = jax.random.split(rng_key)
        data = selfplay(model, selfplay_key)
        samples = compute_loss_input(data)
        minibatches = make_minibatches(samples, perm_key, config.training_batch_size)
        return train_minibatches(model, optimizer, minibatches)

    return training_iteration
