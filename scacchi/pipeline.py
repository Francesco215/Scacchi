from flax import nnx
import jax
import jax.numpy as jnp
from einops import rearrange

from .loss import Sample, TrainMetrics, make_compute_input_for_lossfn, train
from .play import make_selfplay
from .play import SelfplayOutput
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


def _concat_selfplay_outputs(outputs: list[SelfplayOutput]) -> SelfplayOutput:
    if len(outputs) == 1:
        return outputs[0]

    def concat_batch(*xs):
        return jnp.concatenate(xs, axis=0)

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

    search_diagnostics = None
    if outputs[0].search_diagnostics is not None:
        search_diagnostics = jax.tree_util.tree_map(
            concat_batch,
            *(output.search_diagnostics for output in outputs),
        )

    def concat_optional(name: str):
        first = getattr(outputs[0], name)
        if first is None:
            return None
        return concat_batch(*(getattr(output, name) for output in outputs))

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
        search_diagnostics=search_diagnostics,
        q_target_kind=concat_optional("q_target_kind"),
        q_target_weight=concat_optional("q_target_weight"),
        q_target_outcome=concat_optional("q_target_outcome"),
        q_target_distance=concat_optional("q_target_distance"),
        v_target_kind=concat_optional("v_target_kind"),
        v_target_weight=concat_optional("v_target_weight"),
        v_target_outcome=concat_optional("v_target_outcome"),
        v_target_distance=concat_optional("v_target_distance"),
    )


def _fixed_replay_window(
    outputs: list[SelfplayOutput],
    replay_buffer_size: int,
) -> list[SelfplayOutput]:
    if not outputs:
        raise ValueError("replay buffer is empty")
    if replay_buffer_size <= 1:
        return [outputs[-1]]
    window = outputs[-replay_buffer_size:]
    if len(window) == replay_buffer_size:
        return window
    return [window[0]] * (replay_buffer_size - len(window)) + window


def _mean_or_zero(value: jax.Array | None, dtype) -> jax.Array:
    if value is None:
        return jnp.asarray(0.0, dtype=dtype)
    return jnp.asarray(jnp.mean(value), dtype=dtype)


def _with_search_diagnostics(
    metrics: TrainMetrics,
    data: SelfplayOutput,
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


def make_training_iteration(env, config, parallel: BatchParallel | None = None):
    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel
    selfplay = make_selfplay(env, config, parallel=parallel)
    compute_input_for_lossfn = make_compute_input_for_lossfn(config, parallel=parallel)

    @nnx.jit
    def train_from_selfplay_data(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        data: SelfplayOutput,
        perm_key: jax.Array,
    ) -> TrainMetrics:
        samples = compute_input_for_lossfn(data) # it digests the data in such a way that it prepares the input for the loss function
        minibatches = make_minibatches(
            samples,
            perm_key,
            config.training.batch_size,
            config.training.max_updates_per_iter,
            parallel,
        )
        return train_minibatches(model, optimizer, minibatches, config, parallel)

    def training_iteration(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        rng_key: jax.Array,
    ) -> TrainMetrics:
        selfplay_key, perm_key = jax.random.split(rng_key)
        data = selfplay(model, selfplay_key)
        return train_from_selfplay_data(model, optimizer, data, perm_key)

    return training_iteration
