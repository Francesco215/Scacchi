from flax import nnx
import jax
import jax.numpy as jnp
from einops import rearrange

from .loss import (
    CaptureLifecycleDiagnostics,
    Sample,
    TrainMetrics,
    evaluate_distillation_discrepancy,
    make_compute_input_for_lossfn,
    train,
)
from .play import make_selfplay
from .play import TrainingSamples
from .search_diagnostics import DistillationDiscrepancy, SearchDiagnostics
from .distributed import DISABLED_BATCH_PARALLEL, BatchParallel, assert_batch_axis_sharded


TRAJECTORY_DIVERSITY_EARLY_PLIES = 6


def optimizer_updates_per_iteration(config) -> int:
    """Return the exact number of optimizer minibatches in one iteration."""

    rows = int(config.selfplay.batch_size) * int(config.selfplay.max_num_steps)
    updates = rows // int(config.training.batch_size)
    if config.training.max_updates_per_iter is not None:
        updates = min(updates, int(config.training.max_updates_per_iter))
    return updates


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
    device_count = parallel.device_count
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
        x = rearrange(x, "(d b) t ... -> d (t b) ...", d=device_count, b=local_batch_size)
        x = jax.vmap(lambda local_x, ixs: local_x[ixs])(x, local_row_ixs)
        x = rearrange(x, "d (u b) ... -> u (d b) ...", u=num_updates, b=local_training_batch_size)
        return x
    minibatches = jax.tree_util.tree_map(local_shuffle, samples)
    return minibatches


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


def _zero_distillation_discrepancy_like(
    reference: jax.Array,
) -> DistillationDiscrepancy:
    """Return a scalar zero with the fixed-probe discrepancy pytree shape."""

    zero = jnp.zeros((), dtype=reference.dtype)
    return DistillationDiscrepancy(
        **{
            field: zero
            for field in DistillationDiscrepancy._fields
        }
    )


def train_minibatches_with_probe(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    minibatches: Sample,
    probe: Sample,
    probe_index: int,
    config,
    parallel: BatchParallel | None = None,
) -> tuple[
    TrainMetrics,
    DistillationDiscrepancy,
    DistillationDiscrepancy,
]:
    """Train unchanged minibatches while observing one update boundary.

    The scan performs the same optimizer calls, in the same order, as
    :func:`train_minibatches`.  Two inference-only discrepancy evaluations
    execute only when the scan reaches ``probe_index``: one immediately
    before and one immediately after that minibatch's existing update.
    """

    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel
    minibatches = assert_batch_axis_sharded(
        minibatches,
        parallel,
        batch_axis=1,
        label="train minibatches",
    )
    if not 0 <= probe_index < minibatches.obs.shape[0]:
        raise ValueError(
            "probe_index must select an optimizer minibatch; "
            f"got {probe_index} for {minibatches.obs.shape[0]} updates."
        )
    update_indices = jnp.arange(minibatches.obs.shape[0], dtype=jnp.int32)

    @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
    def scan_step(state, indexed_minibatch):
        model, optimizer = state
        update_index, minibatch = indexed_minibatch
        minibatch = assert_batch_axis_sharded(
            minibatch,
            parallel,
            batch_axis=0,
            label="train minibatch",
        )
        zero = _zero_distillation_discrepancy_like(minibatch.policy_tgt)
        selected = update_index == jnp.asarray(
            probe_index,
            dtype=update_index.dtype,
        )

        def evaluate_probe(_):
            return evaluate_distillation_discrepancy(
                model,
                probe,
                config,
            )

        discrepancy_before = jax.lax.cond(
            selected,
            evaluate_probe,
            lambda _: zero,
            operand=None,
        )
        metrics = train(model, optimizer, minibatch, config)
        discrepancy_after = jax.lax.cond(
            selected,
            evaluate_probe,
            lambda _: zero,
            operand=None,
        )
        return (model, optimizer), (
            metrics,
            discrepancy_before,
            discrepancy_after,
        )

    _, (metrics, before_by_update, after_by_update) = scan_step(
        (model, optimizer),
        (update_indices, minibatches),
    )

    def select_nonzero(value: jax.Array) -> jax.Array:
        # Exactly one scan lane is populated. Summing avoids a dynamic gather
        # and preserves each discrepancy's scalar shape.
        return jnp.sum(value, axis=0)

    return (
        metrics,
        jax.tree_util.tree_map(select_nonzero, before_by_update),
        jax.tree_util.tree_map(select_nonzero, after_by_update),
    )


def _with_data_stats(
    metrics: TrainMetrics,
    data: TrainingSamples,
    num_actions: int,
) -> TrainMetrics:
    """Attach additive and histogram summaries of generated trajectories.

    Only the compact ``[time + 1]`` game-length histogram and
    ``[6, num_actions]`` early-ply action counts leave the device.  Entropies
    and quantiles are derived by the logger, so no observations, trajectories,
    or replay-sized copies are transferred to the host.
    """

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
    diagnostics = data.posterior.diagnostics
    metrics = _with_search_diagnostics(metrics, diagnostics)

    num_steps = terminated.shape[1]
    time_index = jnp.arange(num_steps, dtype=jnp.int32)[None, :]
    terminal_index = jnp.where(terminated, time_index, -1)
    inclusive_last_terminal = jax.lax.associative_scan(
        jnp.maximum,
        terminal_index,
        axis=1,
    )
    previous_terminal = jnp.concatenate(
        (
            -jnp.ones_like(inclusive_last_terminal[:, :1]),
            inclusive_last_terminal[:, :-1],
        ),
        axis=1,
    )
    episode_ply = time_index - previous_terminal - 1
    game_length = episode_ply + 1
    terminal_weight = terminated.astype(dtype)
    game_length_histogram = jnp.bincount(
        jnp.where(terminated, game_length, 0).reshape(-1),
        weights=terminal_weight.reshape(-1),
        length=num_steps + 1,
    )

    actions = data.played_action.astype(jnp.int32)
    valid_action = (actions >= 0) & (actions < num_actions)
    safe_actions = jnp.clip(actions, 0, num_actions - 1)
    early_ply = jnp.clip(
        episode_ply,
        0,
        TRAJECTORY_DIVERSITY_EARLY_PLIES - 1,
    )
    early_action_bin = early_ply * num_actions + safe_actions
    early_action_weight = (
        (episode_ply < TRAJECTORY_DIVERSITY_EARLY_PLIES) & valid_action
    ).astype(dtype)
    early_ply_action_counts = jnp.bincount(
        early_action_bin.reshape(-1),
        weights=early_action_weight.reshape(-1),
        length=TRAJECTORY_DIVERSITY_EARLY_PLIES * num_actions,
    ).reshape(TRAJECTORY_DIVERSITY_EARLY_PLIES, num_actions)

    # PGX self-play stores reward from the pre-transition actor's viewpoint.
    # Converting it to player-0's viewpoint makes seat attribution exact even
    # when an environment does not alternate players on every transition.
    actor = episode_ply % 2 if data.actor is None else data.actor
    known_actor = (actor == 0) | (actor == 1)
    finite_reward = jnp.isfinite(data.reward)
    valid_outcome = terminated & known_actor & finite_reward
    reward = jnp.nan_to_num(data.reward, nan=0.0, posinf=0.0, neginf=0.0)
    first_player_return = jnp.where(actor == 0, reward, -reward)
    return metrics._replace(
        data_value_mask_fraction=jnp.mean(value_mask.astype(dtype)),
        data_frame_count=jnp.asarray(terminated.size, dtype=dtype),
        data_termination_count=num_terminations,
        data_pass_fraction=jnp.mean((data.played_action == pass_action).astype(dtype)),
        data_terminations_per_row=jnp.mean(
            jnp.sum(terminated.astype(dtype), axis=1)
        ),
        data_psk_termination_fraction=psk_fraction,
        data_first_player_win_count=jnp.sum(
            (valid_outcome & (first_player_return > 0)).astype(dtype)
        ),
        data_second_player_win_count=jnp.sum(
            (valid_outcome & (first_player_return < 0)).astype(dtype)
        ),
        data_draw_count=jnp.sum(
            (valid_outcome & (first_player_return == 0)).astype(dtype)
        ),
        data_game_length_histogram_counts=game_length_histogram,
        data_early_ply_action_counts=early_ply_action_counts,
    )


def _with_search_diagnostics(
    metrics: TrainMetrics,
    diagnostics: SearchDiagnostics | None,
) -> TrainMetrics:
    """Pool additive generation metrics before any learner minibatch averaging."""

    if diagnostics is None:
        return metrics
    pooled = {
        field: jnp.sum(value)
        for field, value in diagnostics._asdict().items()
    }
    return metrics._replace(**pooled)


def _with_capture_diagnostics(
    metrics: TrainMetrics,
    before: DistillationDiscrepancy,
    after: DistillationDiscrepancy,
    *,
    update_before: DistillationDiscrepancy | None = None,
    update_after: DistillationDiscrepancy | None = None,
    age1_before: DistillationDiscrepancy | None = None,
    age1_after: DistillationDiscrepancy | None = None,
    age1_valid: bool = False,
) -> TrainMetrics:
    """Attach raw fixed-probe gaps at scan and retention boundaries.

    ``before``/``after`` preserve the original scan-start/scan-end metric.
    The optional lifecycle values decompose that net change into the probe's
    own update, later within-scan erosion, and the following fresh scan's
    effect on the immutable age-1 target.
    """

    populations = {
        "policy": ("policy_kl", "count"),
        "v_semantic": ("v_semantic_kl", "count"),
        "v_dirichlet": ("v_dirichlet_kl", "count"),
        "q_semantic": ("q_semantic_kl", "count"),
        "q_dirichlet": ("q_dirichlet_kl", "count"),
        "q_weighted_semantic": ("q_weighted_semantic_kl", "weight"),
        "q_weighted_dirichlet": ("q_weighted_dirichlet_kl", "weight"),
    }
    values: dict[str, jax.Array] = {}
    for destination, (source, denominator) in populations.items():
        values[f"capture_{destination}_before_sum"] = getattr(
            before,
            f"{source}_sum",
        )
        values[f"capture_{destination}_before_{denominator}"] = getattr(
            before,
            f"{source}_{denominator}",
        )
        values[f"capture_{destination}_after_sum"] = getattr(
            after,
            f"{source}_sum",
        )
        values[f"capture_{destination}_after_{denominator}"] = getattr(
            after,
            f"{source}_{denominator}",
        )
    if update_before is None:
        update_before = _zero_distillation_discrepancy_like(
            metrics.policy_loss
        )
    if update_after is None:
        update_after = _zero_distillation_discrepancy_like(
            metrics.policy_loss
        )
    if age1_before is None:
        age1_before = _zero_distillation_discrepancy_like(
            metrics.policy_loss
        )
    if age1_after is None:
        age1_after = _zero_distillation_discrepancy_like(
            metrics.policy_loss
        )
    lifecycle = CaptureLifecycleDiagnostics(
        update_before=update_before,
        update_after=update_after,
        age1_before=age1_before,
        age1_after=age1_after,
        age1_valid=jnp.asarray(
            age1_valid,
            dtype=metrics.policy_loss.dtype,
        ),
    )
    return metrics._replace(
        **values,
        capture_lifecycle=lifecycle,
    )


def _with_age1_capture_diagnostics(
    metrics: TrainMetrics,
    before: DistillationDiscrepancy | None,
    after: DistillationDiscrepancy | None,
) -> TrainMetrics:
    """Attach the prior iteration's probe around the current fresh scan."""

    valid = before is not None and after is not None
    if before is None:
        before = _zero_distillation_discrepancy_like(metrics.policy_loss)
    if after is None:
        after = _zero_distillation_discrepancy_like(metrics.policy_loss)
    return metrics._replace(
        capture_lifecycle=metrics.capture_lifecycle._replace(
            age1_before=before,
            age1_after=after,
            age1_valid=jnp.asarray(
                valid,
                dtype=metrics.policy_loss.dtype,
            ),
        )
    )


def make_training_iteration(env, config, parallel: BatchParallel | None = None):
    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel
    selfplay = make_selfplay(env, config, parallel=parallel)
    num_actions = int(env.num_actions)
    compute_input_for_lossfn = make_compute_input_for_lossfn(config, parallel=parallel)

    @nnx.jit
    def evaluate_probe(
        model: nnx.Module,
        probe: Sample,
    ) -> DistillationDiscrepancy:
        return evaluate_distillation_discrepancy(model, probe, config)

    @nnx.jit
    def train_from_selfplay_data(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        data: TrainingSamples,
        perm_key: jax.Array,
    ) -> tuple[TrainMetrics, Sample]:
        samples = compute_input_for_lossfn(data)
        minibatches = make_minibatches(
            samples,
            perm_key,
            config.training.batch_size,
            config.training.max_updates_per_iter,
            parallel,
        )
        # This is deliberately an in-sample train probe: it measures how much
        # of one fixed target minibatch survives the complete in-iteration
        # optimizer scan, not held-out generalization.  The midpoint reduces
        # the strong recency/forgetting bias of choosing either scan endpoint.
        probe_index = minibatches.obs.shape[0] // 2
        probe = jax.tree_util.tree_map(
            lambda leaf: leaf[probe_index],
            minibatches,
        )
        discrepancy_before = evaluate_distillation_discrepancy(
            model,
            probe,
            config,
        )
        (
            metrics,
            discrepancy_update_before,
            discrepancy_update_after,
        ) = train_minibatches_with_probe(
            model,
            optimizer,
            minibatches,
            probe,
            probe_index,
            config,
            parallel,
        )
        discrepancy_after = evaluate_distillation_discrepancy(
            model,
            probe,
            config,
        )
        metrics = _with_capture_diagnostics(
            metrics,
            discrepancy_before,
            discrepancy_after,
            update_before=discrepancy_update_before,
            update_after=discrepancy_update_after,
        )
        return _with_data_stats(metrics, data, num_actions), probe

    lagged_probe: Sample | None = None

    def training_iteration(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        rng_key: jax.Array,
    ) -> TrainMetrics:
        nonlocal lagged_probe
        previous_probe = lagged_probe
        age1_before = (
            None
            if previous_probe is None
            else evaluate_probe(model, previous_probe)
        )
        selfplay_key, perm_key = jax.random.split(rng_key)
        data = selfplay(model, selfplay_key)
        metrics, current_probe = train_from_selfplay_data(
            model,
            optimizer,
            data,
            perm_key,
        )
        age1_after = (
            None
            if previous_probe is None
            else evaluate_probe(model, previous_probe)
        )
        lagged_probe = current_probe
        return _with_age1_capture_diagnostics(
            metrics,
            age1_before,
            age1_after,
        )

    return training_iteration
