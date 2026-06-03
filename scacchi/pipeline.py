from flax import nnx
import jax
import jax.numpy as jnp

from .loss import Sample, TrainMetrics, make_compute_loss_input, train
from .play import make_selfplay
from .play import SelfplayOutput
from .posterior_tree import is_posterior_tree_policy
from .distributed import DISABLED_BATCH_PARALLEL, BatchParallel, constrain_batch_axis


def make_minibatches(
    samples: Sample,
    rng_key: jax.Array,
    training_batch_size: int,
    max_updates_per_iter: int | None = None,
    sampling: str = "active_with_replacement",
) -> Sample:
    samples = jax.tree_util.tree_map(
        lambda x: x.reshape((-1, *x.shape[2:])),
        samples,
    )
    num_rows = samples.obs.shape[0]
    num_updates = num_rows // training_batch_size
    if max_updates_per_iter is not None:
        num_updates = min(num_updates, max_updates_per_iter)
    num_train_samples = num_updates * training_batch_size
    if sampling in {"permutation", "as_is"}:
        if sampling == "permutation":
            ixs = jax.random.permutation(rng_key, jnp.arange(num_rows))
            samples = jax.tree_util.tree_map(lambda x: x[ixs[:num_train_samples]], samples)
        else:
            samples = jax.tree_util.tree_map(lambda x: x[:num_train_samples], samples)
        return jax.tree_util.tree_map(
            lambda x: x.reshape((num_updates, training_batch_size) + x.shape[1:]),
            samples,
        )
    if sampling != "active_with_replacement":
        raise ValueError(f"unknown minibatch sampling mode: {sampling!r}")
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
    return policy_mask | value_mask


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
    minibatches = constrain_batch_axis(minibatches, parallel, batch_axis=1)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_step(state, minibatch):
        model, optimizer = state
        minibatch = constrain_batch_axis(minibatch, parallel, batch_axis=0)
        metrics = train(model, optimizer, minibatch, config)
        return (model, optimizer), metrics

    _, metrics = scan_step((model, optimizer), minibatches)
    return metrics


def make_training_iteration(env, config, parallel: BatchParallel | None = None):
    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel
    selfplay = make_selfplay(env, config, parallel=parallel)
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
            samples = constrain_batch_axis(samples, parallel, batch_axis=1)
            minibatches = make_minibatches(
                samples,
                perm_key,
                config.training_batch_size,
                getattr(config, "max_updates_per_iter", None),
                getattr(config, "minibatch_sampling", "active_with_replacement"),
            )
            return train_minibatches(model, optimizer, minibatches, config, parallel)

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
            replay_data = _concat_selfplay_outputs(
                _fixed_replay_window(replay_buffer, replay_buffer_size)
            )
            metrics = train_from_selfplay_data(model, optimizer, replay_data, perm_key)
            return _with_search_diagnostics(metrics, data)

        return training_iteration

    @nnx.jit
    def train_from_selfplay_data(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        data: SelfplayOutput,
        perm_key: jax.Array,
    ) -> TrainMetrics:
        samples = compute_loss_input(data)
        samples = constrain_batch_axis(samples, parallel, batch_axis=1)
        minibatches = make_minibatches(
            samples,
            perm_key,
            config.training_batch_size,
            getattr(config, "max_updates_per_iter", None),
            getattr(config, "minibatch_sampling", "active_with_replacement"),
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
