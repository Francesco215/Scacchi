from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
from einops import rearrange
import optax

from .loss import Sample, TrainMetrics, make_compute_input_for_lossfn, train, train_binned_fast
from .network import AZBinnedNet
from .play import make_selfplay
from .play import TrainingSamples
from .distributed import DISABLED_BATCH_PARALLEL, BatchParallel, assert_batch_axis_sharded
from .types import Config


def _muon_weight_dimension_numbers(params: Any) -> Any:
    """Use Muon for hidden block kernels and auxiliary Adam elsewhere."""

    def dimension_numbers(path: tuple[Any, ...], value: Any):
        parts = tuple(getattr(key, "key", getattr(key, "idx", getattr(key, "name", None))) for key in path)
        is_kernel = parts[-1:] == ("kernel",) or parts[-2:] == (
            "kernel",
            "value",
        )
        if parts[:1] != ("blocks",) or not is_kernel or value.ndim < 2:
            return None
        return optax.contrib.MuonDimensionNumbers(
            reduction_axis=tuple(range(value.ndim - 1)),
            output_axis=value.ndim - 1,
        )

    return jax.tree_util.tree_map_with_path(dimension_numbers, params)


def _batched_scale_by_muon(
    *,
    weight_dimension_numbers: Any,
    ns_steps: int = 5,
    beta: float = 0.95,
    eps: float = 1e-8,
    nesterov: bool = True,
) -> optax.GradientTransformation:
    """Drop-in replacement for optax.contrib.scale_by_muon with shape-grouped vmap Newton-Schulz.

    Produces the exact same MuonState and orthogonalized updates (checkpoint-compatible with
    optax.contrib.muon), while collapsing 12-32 sequential NS loops into 1-4 batched GPU calls.
    """
    from optax._src import numerics as _optax_numerics
    from optax.contrib import _muon
    from optax.transforms import _masking as _optax_masking

    base_tx = _muon.scale_by_muon(
        ns_steps=ns_steps,
        beta=beta,
        eps=eps,
        nesterov=nesterov,
        weight_dimension_numbers=weight_dimension_numbers,
    )

    def update_fn(updates, state, params=None):
        del params
        resolved = (
            weight_dimension_numbers(updates)
            if callable(weight_dimension_numbers)
            else weight_dimension_numbers
        )
        mu = optax.tree.update_moment(updates, state.mu, beta, 1)
        count_inc = _optax_numerics.safe_increment(state.count)
        if nesterov:
            mu_hat = jax.tree.map(
                lambda m, g: beta * m + (1 - beta) * g,
                optax.tree.bias_correction(mu, beta, _optax_numerics.safe_increment(count_inc)),
                optax.tree.bias_correction(updates, beta, count_inc),
            )
        else:
            mu_hat = optax.tree.bias_correction(mu, beta, count_inc)

        leaves, treedef = jax.tree_util.tree_flatten(
            mu_hat,
            is_leaf=lambda x: isinstance(x, _optax_masking.MaskedNode),
        )
        dim_leaves, _ = jax.tree_util.tree_flatten(
            resolved,
            is_leaf=lambda x: x is None or _muon._is_weight_dim_nums(x) or isinstance(x, _optax_masking.MaskedNode),
        )
        assert len(leaves) == len(dim_leaves), f"muon tree mismatch: {len(leaves)} vs {len(dim_leaves)}"
        out_leaves: list[Any] = list(leaves)
        groups: dict[tuple[Any, ...], list[int]] = {}
        for idx, (leaf, dn) in enumerate(zip(leaves, dim_leaves)):
            if isinstance(leaf, _optax_masking.MaskedNode) or dn is None or isinstance(dn, _optax_masking.MaskedNode):
                continue
            key = (leaf.shape, leaf.dtype, dn.reduction_axis, dn.output_axis)
            groups.setdefault(key, []).append(idx)

        for idxs in groups.values():
            dn = dim_leaves[idxs[0]]
            if len(idxs) == 1:
                i = idxs[0]
                out_leaves[i] = _muon.orthogonalize_via_newton_schulz(
                    leaves[i], state.ns_coeffs, ns_steps, "frobenius", eps, dn
                )
            else:
                stacked = jnp.stack([leaves[i] for i in idxs], axis=0)
                ortho = jax.vmap(
                    lambda x: _muon.orthogonalize_via_newton_schulz(
                        x, state.ns_coeffs, ns_steps, "frobenius", eps, dn
                    )
                )(stacked)
                for k, i in enumerate(idxs):
                    out_leaves[i] = ortho[k]

        out_updates = jax.tree_util.tree_unflatten(treedef, out_leaves)
        return out_updates, _muon.MuonState(count=count_inc, mu=mu, ns_coeffs=state.ns_coeffs)

    return optax.GradientTransformation(base_tx.init, update_fn)


def _batched_muon(
    learning_rate: float | optax.Schedule,
    *,
    weight_decay: float,
    adam_learning_rate: float | optax.Schedule,
    muon_weight_dimension_numbers: Any,
    consistent_rms: float = 0.2,
) -> optax.GradientTransformation:
    """optax.contrib.muon with shape-batched Newton-Schulz (exact state & math parity)."""
    from optax.contrib import _muon
    from optax.transforms import _masking as _optax_masking

    def param_labels(params):
        dim_nums = (
            muon_weight_dimension_numbers(params)
            if callable(muon_weight_dimension_numbers)
            else muon_weight_dimension_numbers
        )
        return jax.tree.map(
            lambda dim_num, x: jax.tree.map(lambda _: "muon" if dim_num is not None else "adam", x),
            dim_nums,
            params,
            is_leaf=lambda x: x is None or _muon._is_weight_dim_nums(x),
        )

    def muon_dim_nums_fn(params):
        dim_nums = (
            muon_weight_dimension_numbers(params)
            if callable(muon_weight_dimension_numbers)
            else muon_weight_dimension_numbers
        )
        mask = jax.tree.map(lambda label: label == "muon", param_labels(params))
        is_leaf = lambda x: (
            x is None or _muon._is_weight_dim_nums(x) or isinstance(x, _optax_masking.MaskedNode)
        )
        return jax.tree.map(
            lambda dn, submask: jax.tree.map(lambda m: dn if m else _optax_masking.MaskedNode(), submask),
            dim_nums,
            mask,
            is_leaf=is_leaf,
        )

    return optax.partition(
        transforms={
            "muon": optax.chain(
                _batched_scale_by_muon(weight_dimension_numbers=muon_dim_nums_fn),
                _muon.scale_by_shape(
                    weight_dimension_numbers=muon_dim_nums_fn,
                    consistent_rms=consistent_rms,
                ),
                optax.add_decayed_weights(weight_decay),
                optax.scale_by_learning_rate(learning_rate),
            ),
            "adam": optax.adamw(
                learning_rate=adam_learning_rate,
                weight_decay=0.0,
                nesterov=True,
            ),
        },
        param_labels=param_labels,
    )


def build_optimizer(model: nnx.Module, config: Config) -> nnx.Optimizer:
    transforms: list[optax.GradientTransformation] = []
    if config.training.grad_clip_norm is not None:
        transforms.append(optax.clip_by_global_norm(config.training.grad_clip_norm))

    learning_rate: float | optax.Schedule = config.training.learning_rate
    muon_lr: float | optax.Schedule = 5e-3
    if config.training.lr_decay_after_iters is not None and config.training.lr_decay_factor != 1.0:
        rows_per_iteration = max(1, config.selfplay.batch_size * config.selfplay.max_num_steps)
        updates_per_iteration = max(1, rows_per_iteration // config.training.batch_size)
        if getattr(config.training, "symmetry_rot180", False):
            updates_per_iteration *= max(1, int(getattr(config.training, "symmetry_epochs", 1)))
        if config.training.max_updates_per_iter is not None:
            updates_per_iteration = min(updates_per_iteration, config.training.max_updates_per_iter)
        boundary = config.training.lr_decay_after_iters * updates_per_iteration
        learning_rate = optax.piecewise_constant_schedule(config.training.learning_rate, {boundary: config.training.lr_decay_factor})
        muon_lr = optax.piecewise_constant_schedule(5e-3, {boundary: config.training.lr_decay_factor})

    transforms.append(
        _batched_muon(
            muon_lr,
            weight_decay=float(getattr(config.training, "muon_weight_decay", 0.0)),
            adam_learning_rate=learning_rate,
            muon_weight_dimension_numbers=_muon_weight_dimension_numbers,
            consistent_rms=0.2,
        )
    )
    return nnx.Optimizer(model, optax.chain(*transforms), wrt=nnx.Param)


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


def _apply_hex_rot180(minibatch: Sample, flip: jax.Array, cells: int) -> Sample:
    """Apply exact Z_2 180-degree board & action rotation on selected rows of a Hex minibatch."""

    def flip_action_vec(arr: jax.Array | None) -> jax.Array | None:
        if arr is None:
            return None
        flipped = jnp.concatenate([jnp.flip(arr[:, :cells], axis=1), arr[:, cells:]], axis=1)
        return jnp.where(flip[:, None], flipped, arr)

    obs_flipped = jnp.where(flip[:, None, None, None], minibatch.obs[:, ::-1, ::-1, :], minibatch.obs)
    bq = minibatch.beta_Q_target
    bq_flipped = jnp.where(
        flip[:, None, None],
        jnp.concatenate([jnp.flip(bq[:, :cells, :], axis=1), bq[:, cells:, :]], axis=1),
        bq,
    )
    pa = minibatch.played_action
    pa_flipped = jnp.where(flip & (pa < cells), (cells - 1) - pa, pa)
    return minibatch._replace(
        obs=obs_flipped,
        policy_tgt=flip_action_vec(minibatch.policy_tgt),
        played_action=pa_flipped,
        policy_mask=flip_action_vec(minibatch.policy_mask),
        beta_Q_target=bq_flipped,
        q_supervised_pair_mask=flip_action_vec(minibatch.q_supervised_pair_mask),
        q_pair_weight=flip_action_vec(minibatch.q_pair_weight),
        q_positive_evidence_action_mask=flip_action_vec(minibatch.q_positive_evidence_action_mask),
        q_positive_policy_action_mask=flip_action_vec(minibatch.q_positive_policy_action_mask),
        q_target_kind=flip_action_vec(minibatch.q_target_kind),
        q_target_weight=flip_action_vec(minibatch.q_target_weight),
        q_target_outcome=flip_action_vec(minibatch.q_target_outcome),
        q_target_distance=flip_action_vec(minibatch.q_target_distance),
    )


def _train_minibatches_fast(
    model: nnx.Module,
    optimizer: nnx.Optimizer,
    samples: Sample,
    rng_key: jax.Array,
    config: Config,
    num_actions: int,
) -> TrainMetrics:
    """Lazy per-step gather + optional complementary Z_2 rot180 epochs + last-step-only diagnostics."""
    batch_size, num_steps = samples.obs.shape[:2]
    total_rows = batch_size * num_steps
    train_bs = int(config.training.batch_size)
    use_sym = bool(getattr(config.training, "symmetry_rot180", False)) and getattr(config.env, "id", None) == "hex"
    epochs = max(1, int(getattr(config.training, "symmetry_epochs", 1))) if use_sym else 1
    updates_per_epoch = total_rows // train_bs
    if updates_per_epoch < 1:
        raise ValueError(f"total_rows={total_rows} < training.batch_size={train_bs}")

    flat_samples = jax.tree.map(
        lambda x: x.reshape((total_rows, *x.shape[2:])) if isinstance(x, jax.Array) else x,
        samples,
    )

    flip_key, perm_root = jax.random.split(rng_key)
    base_flip = jax.random.bernoulli(flip_key, 0.5, (total_rows,))
    epoch_keys = jax.random.split(perm_root, epochs)
    ix_list = []
    fl_list = []
    rows_used = updates_per_epoch * train_bs
    for ep in range(epochs):
        perm = jax.random.permutation(epoch_keys[ep], total_rows)[:rows_used]
        ep_ix = perm.reshape((updates_per_epoch, train_bs))
        # Alternate base_flip and ~base_flip across consecutive epochs so 2 epochs visit
        # every searched position once in each exact 180-degree orientation!
        ep_fl = (base_flip[perm] if (ep % 2 == 0) else ~base_flip[perm]).reshape((updates_per_epoch, train_bs))
        ix_list.append(ep_ix)
        fl_list.append(ep_fl)

    all_ixs = jnp.concatenate(ix_list, axis=0)
    all_flips = jnp.concatenate(fl_list, axis=0)
    if config.training.max_updates_per_iter is not None:
        cap = min(all_ixs.shape[0], int(config.training.max_updates_per_iter))
        all_ixs = all_ixs[:cap]
        all_flips = all_flips[:cap]
    num_updates = int(all_ixs.shape[0])
    cells = num_actions - 1

    def gather_batch(ixs: jax.Array, flips: jax.Array) -> Sample:
        mb = jax.tree.map(lambda x: x[ixs] if isinstance(x, jax.Array) else x, flat_samples)
        if use_sym:
            mb = _apply_hex_rot180(mb, flips, cells)
        return mb

    last_only = (
        bool(getattr(config.training, "last_step_diagnostics_only", False))
        and getattr(model, "bins", None) is not None
        and num_updates > 1
    )
    if not last_only:
        @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
        def full_scan_step(state, ixs, flips):
            m, opt = state
            mb = gather_batch(ixs, flips)
            met = train(m, opt, mb, config)
            return (m, opt), met

        _, metrics = full_scan_step((model, optimizer), all_ixs, all_flips)
        return metrics

    @nnx.scan(in_axes=(nnx.Carry, 0, 0), out_axes=(nnx.Carry, 0))
    def fast_scan_step(state, ixs, flips):
        m, opt = state
        mb = gather_batch(ixs, flips)
        losses = train_binned_fast(m, opt, mb, config)
        return (m, opt), losses

    _, (p_losses, v_losses, q_losses) = fast_scan_step(
        (model, optimizer), all_ixs[:-1], all_flips[:-1]
    )
    last_mb = gather_batch(all_ixs[-1], all_flips[-1])
    last_m = train(model, optimizer, last_mb, config)

    u_f = jnp.asarray(float(num_updates), dtype=last_m.policy_loss.dtype)
    avg_p = (jnp.sum(p_losses) + last_m.policy_loss) / u_f
    avg_v = (jnp.sum(v_losses) + last_m.value_binned_loss) / u_f
    avg_q = (jnp.sum(q_losses) + last_m.q_binned_loss) / u_f
    return last_m._replace(
        policy_loss=avg_p,
        value_loss=avg_v,
        policy_nll_loss=avg_p,
        policy_kl_hat=jax.lax.stop_gradient(avg_p - last_m.policy_target_entropy),
        value_binned_loss=avg_v,
        q_binned_loss=avg_q,
        v_categorical_target_count=last_m.v_categorical_target_count * u_f,
        v_native_target_count=last_m.v_native_target_count * u_f,
        q_categorical_target_count=last_m.q_categorical_target_count * u_f,
        q_native_target_count=last_m.q_native_target_count * u_f,
        q_positive_evidence_action_count=last_m.q_positive_evidence_action_count * u_f,
        q_positive_policy_action_count=last_m.q_positive_policy_action_count * u_f,
        q_solved_action_count=last_m.q_solved_action_count * u_f,
        q_supervised_action_count=last_m.q_supervised_action_count * u_f,
    )


def make_training_iteration(env, config, parallel: BatchParallel | None = None):
    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel
    selfplay = make_selfplay(env, config, parallel=parallel)
    num_actions = int(env.num_actions)
    compute_input_for_lossfn = make_compute_input_for_lossfn(config, parallel=parallel)
    use_fast_train = (
        parallel.device_count == 1
        and (
            bool(getattr(config.training, "lazy_minibatches", False))
            or bool(getattr(config.training, "symmetry_rot180", False))
            or bool(getattr(config.training, "last_step_diagnostics_only", False))
        )
    )

    @nnx.jit
    def train_from_selfplay_data(
        model: nnx.Module,
        optimizer: nnx.Optimizer,
        data: TrainingSamples,
        perm_key: jax.Array,
    ) -> TrainMetrics:
        samples = compute_input_for_lossfn(data)
        if use_fast_train:
            metrics = _train_minibatches_fast(model, optimizer, samples, perm_key, config, num_actions)
            return _with_data_stats(metrics, data, num_actions)
        minibatches = make_minibatches(
            samples,
            perm_key,
            config.training.batch_size,
            config.training.max_updates_per_iter,
            parallel,
        )
        metrics = train_minibatches(model, optimizer, minibatches, config, parallel)
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
