from flax import nnx
import jax
import jax.numpy as jnp
from einops import rearrange

from .loss import Sample, TrainMetrics, make_compute_input_for_lossfn, train
from .play import make_selfplay
from .play import TrainingSamples
from .distributed import DISABLED_BATCH_PARALLEL, BatchParallel, assert_batch_axis_sharded


def make_minibatches(
    samples: Sample,
    rng_key: jax.Array,
    training_batch_size: int,
    max_updates_per_iter: int | None = None,
    parallel: BatchParallel | None = None,
) -> Sample:
    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel
    if samples.obs.ndim < 2:
        raise ValueError("minibatching requires samples shaped [batch, time, ...].")

    samples = assert_batch_axis_sharded(samples, parallel, batch_axis=0, label="minibatch samples")
    batch_size = samples.obs.shape[0]
    num_steps = samples.obs.shape[1]
    device_count = parallel.device_count if parallel.enabled else 1
    assert batch_size % device_count == 0, f"batch_size={batch_size} must be divisible by device_count={device_count}."
    assert training_batch_size % device_count == 0, f"training_batch_size={training_batch_size} must be divisible by device_count={device_count}."

    local_batch_size = batch_size // device_count
    local_training_batch_size = training_batch_size // device_count
    local_rows = num_steps * local_batch_size
    num_updates = local_rows // local_training_batch_size
    if max_updates_per_iter is not None:
        num_updates = min(num_updates, max_updates_per_iter)
    local_train_rows = num_updates * local_training_batch_size

    local_keys = parallel.split(rng_key, device_count)
    local_row_ixs = jax.vmap(lambda key: jax.random.permutation(key, jnp.arange(local_rows))[:local_train_rows])(local_keys)

    def local_shuffle(x: jax.Array) -> jax.Array:
        x = rearrange(x, "(d b) t ... -> d (b t) ...", d=device_count, b=local_batch_size)
        x = jax.vmap(lambda local_x, ixs: local_x[ixs])(x, local_row_ixs)
        x = rearrange(x, "d (u b) ... -> u (d b) ...", u=num_updates, b=local_training_batch_size)
        return x
    minibatches = jax.tree_util.tree_map(local_shuffle, samples)
    return minibatches


def _mean_or_zero(value: jax.Array | None, dtype) -> jax.Array:
    if value is None:
        return jnp.asarray(0.0, dtype=dtype)
    return jnp.asarray(jnp.mean(value), dtype=dtype)


def _with_search_diagnostics(
    metrics: TrainMetrics,
    data: TrainingSamples,
) -> TrainMetrics:
    diagnostics = data.search_diagnostics
    if diagnostics is None:
        return metrics
    dtype = metrics.policy_loss.dtype
    return metrics._replace(
        search_path_depth_mean=_mean_or_zero(diagnostics.path_depth_mean, dtype),
        search_path_depth_p50=_mean_or_zero(diagnostics.path_depth_p50, dtype),
        search_path_depth_p90=_mean_or_zero(diagnostics.path_depth_p90, dtype),
        search_path_depth_max=_mean_or_zero(diagnostics.path_depth_max, dtype),
        search_expanded_nodes=_mean_or_zero(diagnostics.expanded_nodes, dtype),
        search_terminal_fraction=_mean_or_zero(diagnostics.terminal_fraction, dtype),
        search_root_policy_entropy=_mean_or_zero(diagnostics.root_policy_entropy, dtype),
        search_root_gamma=_mean_or_zero(diagnostics.root_gamma, dtype),
        search_root_downstream_eval_count=_mean_or_zero(
            diagnostics.root_downstream_eval_count,
            dtype,
        ),
        search_root_q_concentration=_mean_or_zero(diagnostics.root_q_concentration, dtype),
    )


def train_minibatches(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    minibatches: Sample,
    config,
    parallel: BatchParallel | None = None,
) -> TrainMetrics:
    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel
    minibatches = assert_batch_axis_sharded(minibatches, parallel, batch_axis=1, label="train minibatches")

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_step(state, minibatch):
        model, optimizer = state
        minibatch = assert_batch_axis_sharded(minibatch, parallel, batch_axis=0, label="train minibatch")
        metrics = train(model, optimizer, minibatch, config)
        return (model, optimizer), metrics

    _, metrics = scan_step((model, optimizer), minibatches)
    return metrics


def _with_data_stats(
    metrics: TrainMetrics,
    data: TrainingSamples,
    num_actions: int,
) -> TrainMetrics:
    terminated = data.terminated
    value_mask = jnp.cumsum(terminated[:, ::-1], axis=1)[:, ::-1] >= 1
    dtype = metrics.policy_loss.dtype
    pass_action = num_actions - 1
    num_terminations = jnp.sum(terminated.astype(dtype))
    psk_terminated = (
        jnp.zeros_like(terminated)
        if data.psk_terminated is None
        else data.psk_terminated
    )
    psk_fraction = jnp.sum(psk_terminated.astype(dtype)) / jnp.maximum(
        num_terminations, 1.0
    )
    return metrics._replace(
        data_value_mask_fraction=jnp.mean(value_mask.astype(dtype)),
        data_pass_fraction=jnp.mean((data.played_action == pass_action).astype(dtype)),
        data_terminations_per_row=jnp.mean(
            jnp.sum(terminated.astype(dtype), axis=1)
        ),
        data_psk_termination_fraction=psk_fraction,
    )


def make_training_iteration(env, config, parallel: BatchParallel | None = None):
    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel
    selfplay = make_selfplay(env, config, parallel=parallel)
    num_actions = int(env.num_actions)
    compute_input_for_lossfn = make_compute_input_for_lossfn(config, parallel=parallel)

    @nnx.jit
    def train_from_selfplay_data(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        data: TrainingSamples,
        perm_key: jax.Array,
    ) -> TrainMetrics:
        samples = compute_input_for_lossfn(data)
        minibatches = make_minibatches(
            samples,
            perm_key,
            config.training.batch_size,
            config.training.max_updates_per_iter,
            parallel,
        )
        metrics = train_minibatches(model, optimizer, minibatches, config, parallel)
        metrics = _with_search_diagnostics(metrics, data)
        return _with_data_stats(metrics, data, num_actions)

    def training_iteration(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        rng_key: jax.Array,
    ) -> TrainMetrics:
        selfplay_key, perm_key = jax.random.split(rng_key)
        data = selfplay(model, selfplay_key)
        return train_from_selfplay_data(model, optimizer, data, perm_key)

    return training_iteration
