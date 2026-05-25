from flax import nnx
import jax
import jax.numpy as jnp

from .loss import Sample, TrainMetrics, make_compute_loss_input, train
from .play import make_selfplay
from .play import SelfplayOutput
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


def _concat_selfplay_outputs(outputs: list[SelfplayOutput]) -> SelfplayOutput:
    if len(outputs) == 1:
        return outputs[0]

    def concat_batch(*xs):
        return jnp.concatenate(xs, axis=1)

    tree_data = None
    if outputs[0].tree_data is not None:
        tree_data = jax.tree_util.tree_map(
            concat_batch,
            *(output.tree_data for output in outputs),
        )

    search_loss_mask = None
    if outputs[0].search_loss_mask is not None:
        search_loss_mask = concat_batch(
            *(output.search_loss_mask for output in outputs),
        )

    return SelfplayOutput(
        obs=concat_batch(*(output.obs for output in outputs)),
        reward=concat_batch(*(output.reward for output in outputs)),
        terminated=concat_batch(*(output.terminated for output in outputs)),
        action_weights=concat_batch(*(output.action_weights for output in outputs)),
        played_action=concat_batch(*(output.played_action for output in outputs)),
        legal_action_mask=concat_batch(*(output.legal_action_mask for output in outputs)),
        beta_Q_target=concat_batch(*(output.beta_Q_target for output in outputs)),
        beta_V_target=concat_batch(*(output.beta_V_target for output in outputs)),
        q_loss_weight=concat_batch(*(output.q_loss_weight for output in outputs)),
        discount=concat_batch(*(output.discount for output in outputs)),
        tree_data=tree_data,
        search_loss_mask=search_loss_mask,
    )


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
    replay_buffer_size = int(getattr(config, "replay_buffer_size", 1))

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

        replay_buffer: list[SelfplayOutput] = []

        def training_iteration(
            model: nnx.Module,
            optimizer: nnx.Optimizer,
            rng_key: jax.Array,
        ) -> TrainMetrics:
            selfplay_key, perm_key = jax.random.split(rng_key)
            data = selfplay(model, selfplay_key)
            replay_buffer.append(data)
            del replay_buffer[:-replay_buffer_size]
            replay_data = _concat_selfplay_outputs(replay_buffer)
            return train_from_selfplay_data(model, optimizer, replay_data, perm_key)

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
