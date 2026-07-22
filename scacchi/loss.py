from collections.abc import Mapping
from typing import Any, NamedTuple, cast

from flax import nnx
import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln
from jaxtyping import Array, Bool, Float, Int
import optax

from .dirichlet_mctx.categorical import (
    TARGET_PAD,
    TARGET_CATEGORICAL,
    TARGET_DIRICHLET,
    dirichlet_nll_at_categorical,
    native_fields_from_beta,
)
from .distributed import DISABLED_BATCH_PARALLEL, BatchParallel, assert_batch_axis_sharded
from .play import TrainingSamples


DIRICHLET_KL_LOSS_CUTOFF = 1000.0

# A fixed, config-independent grid makes concentration histograms directly
# comparable across runs.  The first interval covers concentrations close to
# zero; the remaining intervals are approximately uniform in log2 space.  All
# current model caps fit comfortably below the final edge (the largest shipped
# cap is 300).  `_masked_concentration_histogram_counts` deliberately folds
# larger finite values into the final bin instead of dropping observations.
CONCENTRATION_HISTOGRAM_NUM_BINS = 100
CONCENTRATION_HISTOGRAM_BIN_EDGES: tuple[float, ...] = (
    0.0,
    *(
        2.0
        ** (
            -10.0
            + 20.0 * index / (CONCENTRATION_HISTOGRAM_NUM_BINS - 1)
        )
        for index in range(CONCENTRATION_HISTOGRAM_NUM_BINS)
    ),
)
CONCENTRATION_HISTOGRAM_SERIES = (
    "V_prior",
    "V_posterior",
    "Q_prior",
    "Q_posterior",
)


class Sample(NamedTuple):
    obs: Float[Array, "..."]
    policy_tgt: Float[Array, "*batch action"]
    value_tgt: Float[Array, "*batch"]
    played_action: Int[Array, "*batch"]
    policy_mask: Bool[Array, "*batch action"]
    value_mask: Bool[Array, "*batch"]
    beta_Q_target: Float[Array, "*batch action outcome"]
    beta_V_target: Float[Array, "*batch outcome"]
    q_loss_weight: Float[Array, "*batch action"]
    policy_loss_mask: Bool[Array, "*batch"] | None = None
    value_loss_mask: Bool[Array, "*batch"] | None = None
    search_loss_mask: Bool[Array, "*batch"] | None = None
    outcome_mask: Bool[Array, "*batch"] | None = None
    q_target_kind: Int[Array, "*batch action"] | None = None
    q_target_weight: Float[Array, "*batch action"] | None = None
    q_target_outcome: Int[Array, "*batch action"] | None = None
    q_target_distance: Int[Array, "*batch action"] | None = None
    v_target_kind: Int[Array, "*batch"] | None = None
    v_target_weight: Float[Array, "*batch"] | None = None
    v_target_outcome: Int[Array, "*batch"] | None = None
    v_target_distance: Int[Array, "*batch"] | None = None


class _NativeTargetFields(NamedTuple):
    q_target_kind: Int[Array, "*batch action"]
    q_target_weight: Float[Array, "*batch action"]
    q_target_outcome: Int[Array, "*batch action"]
    q_target_distance: Int[Array, "*batch action"]
    v_target_kind: Int[Array, "*batch"]
    v_target_weight: Float[Array, "*batch"]
    v_target_outcome: Int[Array, "*batch"]
    v_target_distance: Int[Array, "*batch"]


_NATIVE_TARGET_FIELD_NAMES = _NativeTargetFields._fields

class TrainMetrics(NamedTuple):
    policy_loss: Float[Array, "*batch"]
    value_loss: Float[Array, "*batch"]
    policy_nll_loss: Float[Array, "*batch"]
    policy_kl_hat: Float[Array, "*batch"]
    policy_target_entropy: Float[Array, "*batch"]
    value_dir_kl_loss: Float[Array, "*batch"]
    q_dir_kl_loss: Float[Array, "*batch"]
    value_outcome_loss: Float[Array, "*batch"]
    q_outcome_loss: Float[Array, "*batch"]
    dirichlet_concentration_histogram_counts: Float[
        Array,
        "4 concentration_bin",
    ]
    alpha_V_concentration: Float[Array, "*batch"]
    alpha_Q_concentration: Float[Array, "*batch"]
    alpha_V_concentration_std: Float[Array, "*batch"]
    alpha_Q_concentration_std: Float[Array, "*batch"]
    alpha_V_dirichlet_concentration: Float[Array, "*batch"]
    alpha_Q_dirichlet_concentration: Float[Array, "*batch"]
    alpha_V_dirichlet_concentration_std: Float[Array, "*batch"]
    alpha_Q_dirichlet_concentration_std: Float[Array, "*batch"]
    alpha_V_categorical_concentration: Float[Array, "*batch"]
    alpha_Q_categorical_concentration: Float[Array, "*batch"]
    beta_V_concentration: Float[Array, "*batch"]
    beta_Q_concentration: Float[Array, "*batch"]
    beta_V_concentration_std: Float[Array, "*batch"]
    beta_Q_concentration_std: Float[Array, "*batch"]
    v_dirichlet_log_concentration_mae: Float[Array, "*batch"]
    q_dirichlet_log_concentration_mae: Float[Array, "*batch"]
    v_categorical_target_fraction: Float[Array, "*batch"]
    q_categorical_target_fraction: Float[Array, "*batch"]
    alpha_V_dirichlet_concentration_floor_fraction: Float[Array, "*batch"]
    alpha_Q_dirichlet_concentration_floor_fraction: Float[Array, "*batch"]
    alpha_V_dirichlet_concentration_clip_fraction: Float[Array, "*batch"]
    alpha_Q_dirichlet_concentration_clip_fraction: Float[Array, "*batch"]
    alpha_V_concentration_clip_fraction: Float[Array, "*batch"]
    alpha_Q_concentration_clip_fraction: Float[Array, "*batch"]
    alpha_V_categorical_concentration_clip_fraction: Float[Array, "*batch"]
    alpha_Q_categorical_concentration_clip_fraction: Float[Array, "*batch"]
    v_dirichlet_target_count: Float[Array, "*batch"]
    q_dirichlet_target_count: Float[Array, "*batch"]
    v_categorical_target_count: Float[Array, "*batch"]
    q_categorical_target_count: Float[Array, "*batch"]
    v_native_target_count: Float[Array, "*batch"]
    q_native_target_count: Float[Array, "*batch"]
    q_loss_weight_mean: Float[Array, "*batch"]
    data_value_mask_fraction: Float[Array, "*batch"]
    data_pass_fraction: Float[Array, "*batch"]
    data_terminations_per_row: Float[Array, "*batch"]
    data_psk_termination_fraction: Float[Array, "*batch"]

def _num_outcomes_for_config(config: Any) -> int:
    num_outcomes = config.env.num_outcomes
    if num_outcomes is None:
        return 2 if config.env.id == "hex" else 3
    return int(num_outcomes)


def _empty_posterior_targets(
    policy_target: jax.Array,
    num_outcomes: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    q_loss_weight = policy_target * jnp.asarray(0.0, dtype=policy_target.dtype)
    beta_q = jnp.broadcast_to(
        q_loss_weight[..., None],
        policy_target.shape + (num_outcomes,),
    )
    beta_v_seed = jnp.sum(q_loss_weight, axis=-1, keepdims=True)
    beta_v = jnp.broadcast_to(
        beta_v_seed,
        policy_target.shape[:-1] + (num_outcomes,),
    )
    return beta_q, beta_v, q_loss_weight


def make_compute_input_for_lossfn(
    config,
    parallel: BatchParallel | None = None,
):
    parallel = DISABLED_BATCH_PARALLEL if parallel is None else parallel

    def compute_loss_input(source: TrainingSamples) -> Sample:
        source = assert_batch_axis_sharded(
            source,
            parallel,
            batch_axis=0,
            label="loss input data",
        )
        prediction = source.posterior.prediction
        metadata = source.posterior.metadata

        def metadata_value(field: str) -> Any | None:
            return None if metadata is None else getattr(metadata, field)

        policy = prediction.policy
        beta_q_target = prediction.alpha_q
        beta_v_target = prediction.alpha_v
        q_loss_weight = metadata_value("q_weight")
        if beta_q_target is None or beta_v_target is None or q_loss_weight is None:
            beta_q_target, beta_v_target, q_loss_weight = _empty_posterior_targets(
                policy,
                _num_outcomes_for_config(config),
            )

        def assert_native_targets(
            label: str,
            native_fields: _NativeTargetFields,
        ) -> _NativeTargetFields:
            return assert_batch_axis_sharded(native_fields, parallel, batch_axis=0, label=label)

        value_mask = jnp.cumsum(source.terminated[:, ::-1], axis=1)[:, ::-1] >= 1
        legal_policy_mask = jnp.any(source.legal_action_mask, axis=-1)
        policy_target_mask = jnp.sum(policy, axis=-1) > 0
        metadata_mask = metadata_value("mask")
        search_loss_mask = (
            metadata_mask
            if metadata_mask is not None
            else policy_target_mask
        )

        def trajectory_value_targets(
            reward: jax.Array,
            discount: jax.Array,
        ) -> jax.Array:
            def body_fn(carry: jax.Array, inputs) -> tuple[jax.Array, jax.Array]:
                step_reward, step_discount = inputs
                value = step_reward + step_discount * carry
                return value, value

            _, values = jax.lax.scan(
                body_fn,
                jnp.zeros((), dtype=reward.dtype),
                (reward[::-1], discount[::-1]),
            )
            return values[::-1]

        value_tgt = jax.vmap(trajectory_value_targets)(source.reward, source.discount)
        policy_tgt = jnp.asarray(policy)
        policy_loss_mask = legal_policy_mask & search_loss_mask
        value_loss_mask = search_loss_mask
        loss_mask_mode = config.training.losses.loss_mask_mode
        if loss_mask_mode == "value":
            policy_loss_mask = value_mask
            value_loss_mask = value_mask
        elif loss_mask_mode == "pgx":
            value_loss_mask = value_mask
        elif loss_mask_mode != "search":
            raise ValueError(f"unknown loss_mask_mode: {loss_mask_mode!r}")
        if config.training.losses.policy_target_mode == "winner_action":
            policy_tgt = jax.nn.one_hot(
                source.played_action,
                policy.shape[-1],
                dtype=policy.dtype,
            )
            policy_loss_mask = legal_policy_mask & value_mask & (value_tgt > 0)

        native_target_values = {
            field: metadata_value(field) for field in _NATIVE_TARGET_FIELD_NAMES
        }

        terminal_edge_targets = config.training.losses.terminal_edge_targets
        terminal_parent_targets = config.training.losses.terminal_parent_targets
        if terminal_edge_targets or terminal_parent_targets:
            native_defaults = native_fields_from_beta(beta_q_target, beta_v_target)
            native_defaults = assert_batch_axis_sharded(native_defaults, parallel, batch_axis=0, label="loss native_defaults")
            native_targets = _native_fields_from_values(
                native_target_values,
                native_defaults,
            )
            num_outcomes = beta_v_target.shape[-1]
            rounded_reward = jnp.round(source.reward).astype(jnp.int32)
            if num_outcomes == 2:
                outcome_index = (rounded_reward + 1) // 2
            elif num_outcomes == 3:
                outcome_index = rounded_reward + 1
            else:
                raise ValueError(f"unsupported outcome count: {num_outcomes}")
            played_action_mask = jax.nn.one_hot(
                source.played_action,
                beta_q_target.shape[-2],
                dtype=bool,
            )
            terminal_action_mask = source.terminated[..., None] & played_action_mask
            native_targets = assert_native_targets(
                "loss native_targets after defaults",
                native_targets,
            )

        if terminal_edge_targets:
            native_targets = native_targets._replace(
                q_target_kind=jnp.where(
                    terminal_action_mask,
                    jnp.asarray(
                        int(TARGET_CATEGORICAL),
                        dtype=native_targets.q_target_kind.dtype,
                    ),
                    native_targets.q_target_kind,
                ),
                q_target_weight=jnp.where(
                    terminal_action_mask,
                    jnp.ones((), dtype=native_targets.q_target_weight.dtype),
                    native_targets.q_target_weight,
                ),
                q_target_outcome=jnp.where(
                    terminal_action_mask,
                    outcome_index[..., None].astype(
                        native_targets.q_target_outcome.dtype,
                    ),
                    native_targets.q_target_outcome,
                ),
                q_target_distance=jnp.where(
                    terminal_action_mask,
                    jnp.ones((), dtype=native_targets.q_target_distance.dtype),
                    native_targets.q_target_distance,
                ),
            )
            q_loss_weight = jnp.where(
                terminal_action_mask,
                jnp.maximum(q_loss_weight, jnp.ones((), dtype=q_loss_weight.dtype)),
                q_loss_weight,
            )
            native_targets = assert_native_targets(
                "loss native_targets after terminal_edge_targets",
                native_targets,
            )

        if terminal_parent_targets:
            terminal_win_mask = source.terminated & (source.reward > 0)
            policy_tgt = jnp.where(
                terminal_win_mask[..., None],
                played_action_mask.astype(policy_tgt.dtype),
                policy_tgt,
            )
            policy_loss_mask = policy_loss_mask | (terminal_win_mask & legal_policy_mask)
            value_loss_mask = value_loss_mask | terminal_win_mask
            native_targets = native_targets._replace(
                v_target_kind=jnp.where(
                    terminal_win_mask,
                    jnp.asarray(
                        int(TARGET_CATEGORICAL),
                        dtype=native_targets.v_target_kind.dtype,
                    ),
                    native_targets.v_target_kind,
                ),
                v_target_weight=jnp.where(
                    terminal_win_mask,
                    jnp.ones((), dtype=native_targets.v_target_weight.dtype),
                    native_targets.v_target_weight,
                ),
                v_target_outcome=jnp.where(
                    terminal_win_mask,
                    outcome_index.astype(native_targets.v_target_outcome.dtype),
                    native_targets.v_target_outcome,
                ),
                v_target_distance=jnp.where(
                    terminal_win_mask,
                    jnp.ones((), dtype=native_targets.v_target_distance.dtype),
                    native_targets.v_target_distance,
                ),
            )
            native_targets = assert_native_targets(
                "loss native_targets after terminal_parent_targets",
                native_targets,
            )

        if terminal_edge_targets or terminal_parent_targets:
            native_target_values = native_targets._asdict()

        sample = Sample(
            obs=source.obs,
            policy_tgt=policy_tgt,
            value_tgt=value_tgt,
            played_action=source.played_action,
            policy_mask=source.legal_action_mask,
            value_mask=value_mask,
            beta_Q_target=beta_q_target,
            beta_V_target=beta_v_target,
            q_loss_weight=q_loss_weight,
            policy_loss_mask=policy_loss_mask,
            value_loss_mask=value_loss_mask,
            search_loss_mask=search_loss_mask,
            outcome_mask=value_mask,
            **native_target_values,
        )
        sample = assert_batch_axis_sharded(sample, parallel, batch_axis=0, label="loss sample before native defaults")
        native_fields = _native_target_fields(sample)
        native_fields = assert_batch_axis_sharded(native_fields, parallel, batch_axis=0, label="loss native_fields final")
        sample = _with_native_defaults(sample, native_fields)
        return assert_batch_axis_sharded(sample, parallel, batch_axis=0, label="loss sample after native defaults")

    return compute_loss_input


def _masked_mean(loss: jax.Array, mask: jax.Array) -> jax.Array:
    mask_bool = mask.astype(jnp.bool_)
    mask_float = mask_bool.astype(loss.dtype)
    safe_loss = jnp.where(mask_bool, loss, jnp.zeros_like(loss))
    return jnp.sum(safe_loss) / jnp.maximum(jnp.sum(mask_float), 1)


def _masked_std(value: jax.Array, mask: jax.Array) -> jax.Array:
    """Population standard deviation over active finite-shape entries."""

    mean = _masked_mean(value, mask)
    variance = _masked_mean(jnp.square(value - mean), mask)
    return jnp.sqrt(jnp.maximum(variance, 0.0))


def _bounded_loss_mask(
    loss: jax.Array,
    mask: jax.Array,
    *,
    cutoff: float = DIRICHLET_KL_LOSS_CUTOFF,
) -> jax.Array:
    return mask.astype(jnp.bool_) & jnp.isfinite(loss) & (loss <= cutoff)


def _native_loss_mask(
    loss: jax.Array,
    mask: jax.Array,
    target_kind: jax.Array,
    *,
    cutoff: float = DIRICHLET_KL_LOSS_CUTOFF,
) -> jax.Array:
    """Bound posterior KLs while retaining every finite categorical NLL."""

    target_kind = jnp.asarray(target_kind)
    categorical = target_kind == int(TARGET_CATEGORICAL)
    within_bound = categorical | (loss <= cutoff)
    return mask.astype(jnp.bool_) & jnp.isfinite(loss) & within_bound


def _bounded_masked_mean(
    loss: jax.Array,
    mask: jax.Array,
    *,
    cutoff: float = DIRICHLET_KL_LOSS_CUTOFF,
) -> jax.Array:
    active_mask = mask.astype(jnp.bool_)
    bounded_mask = _bounded_loss_mask(loss, active_mask, cutoff=cutoff)
    safe_loss = jnp.where(
        bounded_mask,
        jnp.nan_to_num(loss),
        jnp.zeros_like(loss),
    )
    mean = _masked_mean(safe_loss, bounded_mask)
    active_count = jnp.sum(active_mask.astype(loss.dtype))
    kept_count = jnp.sum(bounded_mask.astype(loss.dtype))
    return jnp.where(
        (active_count > 0) & (kept_count == 0),
        jnp.asarray(jnp.nan, dtype=loss.dtype),
        mean,
    )


def _native_masked_mean(
    loss: jax.Array,
    mask: jax.Array,
    target_kind: jax.Array,
) -> jax.Array:
    active_mask = mask.astype(jnp.bool_)
    kept_mask = _native_loss_mask(loss, active_mask, target_kind)
    safe_loss = jnp.where(
        kept_mask,
        jnp.nan_to_num(loss),
        jnp.zeros_like(loss),
    )
    mean = _masked_mean(safe_loss, kept_mask)
    active_count = jnp.sum(active_mask.astype(loss.dtype))
    kept_count = jnp.sum(kept_mask.astype(loss.dtype))
    return jnp.where(
        (active_count > 0) & (kept_count == 0),
        jnp.asarray(jnp.nan, dtype=loss.dtype),
        mean,
    )


def _mask_or(mask: jax.Array | None, fallback: jax.Array) -> jax.Array:
    return fallback if mask is None else mask


def _native_target_values(source: Any) -> dict[str, jax.Array | None]:
    return {field: getattr(source, field) for field in _NATIVE_TARGET_FIELD_NAMES}


def _native_fields_from_values(
    values: dict[str, jax.Array | None],
    defaults: Mapping[str, object],
) -> _NativeTargetFields:
    def value_or_default(field: str) -> jax.Array:
        value = values[field]
        return cast(jax.Array, defaults[field]) if value is None else value

    return _NativeTargetFields(
        *(value_or_default(field) for field in _NATIVE_TARGET_FIELD_NAMES)
    )


def _native_target_fields(sample: Sample) -> _NativeTargetFields:
    defaults = native_fields_from_beta(sample.beta_Q_target, sample.beta_V_target)
    return _native_fields_from_values(_native_target_values(sample), defaults)


def _with_native_defaults(
    sample: Sample,
    native_fields: _NativeTargetFields | None = None,
) -> Sample:
    if native_fields is None:
        native_fields = _native_target_fields(sample)
    return sample._replace(**native_fields._asdict())


def _zero_train_metrics_like(reference: jax.Array, **values: jax.Array) -> TrainMetrics:
    fields = {field: jnp.zeros_like(reference) for field in TrainMetrics._fields}
    fields["dirichlet_concentration_histogram_counts"] = jnp.zeros(
        (len(CONCENTRATION_HISTOGRAM_SERIES), CONCENTRATION_HISTOGRAM_NUM_BINS),
        dtype=reference.dtype,
    )
    fields.update(values)
    return TrainMetrics(**fields)


def concentration_histogram_bin_edges(
    dtype: jnp.dtype = jnp.float32,
) -> jax.Array:
    """Return the shared concentration histogram edges on a JAX device.

    `CONCENTRATION_HISTOGRAM_BIN_EDGES` is the host-friendly float tuple for
    constructing an equivalent pre-binned W&B histogram.
    """

    return jnp.asarray(CONCENTRATION_HISTOGRAM_BIN_EDGES, dtype=dtype)


def _masked_concentration_histogram_counts(
    concentration: jax.Array,
    mask: jax.Array,
) -> jax.Array:
    """Count finite, selected concentrations on the shared fixed grid."""

    histogram_dtype = jnp.result_type(concentration.dtype, jnp.float32)
    concentration = concentration.astype(histogram_dtype)
    edges = concentration_histogram_bin_edges(histogram_dtype)
    valid = mask & jnp.isfinite(concentration)
    safe_concentration = jnp.where(valid, concentration, 0.0)
    # Searching only the internal edges makes the first and final bins catch
    # underflow and overflow respectively.  `side="right"` matches NumPy's
    # [left, right) convention at exact internal boundaries.
    bin_index = jnp.searchsorted(
        edges[1:-1],
        safe_concentration,
        side="right",
    )
    return jnp.bincount(
        bin_index.reshape((-1,)),
        weights=valid.astype(histogram_dtype).reshape((-1,)),
        length=CONCENTRATION_HISTOGRAM_NUM_BINS,
    )


def _compute_losses(
    logits: jax.Array,
    value: jax.Array,
    data: Sample,
    config=None,
) -> tuple[jax.Array, jax.Array]:
    loss_mask_mode = (
        None if config is None else config.training.losses.loss_mask_mode
    )
    if loss_mask_mode == "pgx":
        # Exact replica of pgx examples/alphazero/train.py loss_fn: policy CE
        # over the full softmax (illegal actions included) on every frame;
        # value l2 multiplied by the completed-game mask, averaged over all
        # frames (not a masked mean).
        policy_loss = optax.softmax_cross_entropy(logits, data.policy_tgt)
        policy_loss = jnp.mean(policy_loss)
        value_loss = optax.l2_loss(value, data.value_tgt)
        value_loss = jnp.mean(value_loss * data.value_mask)
        return policy_loss, value_loss
    policy_loss_mask = _mask_or(data.policy_loss_mask, data.value_mask)
    value_loss_mask = _mask_or(data.value_loss_mask, data.value_mask)
    policy_loss = optax.softmax_cross_entropy(logits, data.policy_tgt, where=data.policy_mask)
    policy_loss = _masked_mean(policy_loss, policy_loss_mask)
    value_loss = optax.l2_loss(value, data.value_tgt)
    value_loss = _masked_mean(value_loss, value_loss_mask)
    return policy_loss, value_loss


def _categorical_entropy_from_probs(probs: jax.Array, mask: jax.Array) -> jax.Array:
    log_probs = jnp.log(jnp.clip(probs, jnp.finfo(probs.dtype).tiny, 1.0))
    entropy_terms = jnp.where(mask, probs * log_probs, 0.0)
    return -jnp.sum(entropy_terms, axis=-1)


def _outcome_index(value_tgt: jax.Array, num_outcomes: int) -> jax.Array:
    rounded = jnp.round(value_tgt).astype(jnp.int32)
    if num_outcomes == 2:
        return (rounded + 1) // 2
    if num_outcomes == 3:
        return rounded + 1
    raise ValueError(f"unsupported outcome count: {num_outcomes}")


def _gather_played_action(alpha_q: jax.Array, played_action: jax.Array) -> jax.Array:
    gather_ix = jnp.broadcast_to(
        played_action[..., None, None],
        (*played_action.shape, 1, alpha_q.shape[-1]),
    )
    return jnp.take_along_axis(alpha_q, gather_ix, axis=-2).squeeze(axis=-2)


def _gather_played_action_field(
    action_field: jax.Array,
    played_action: jax.Array,
) -> jax.Array:
    return jnp.take_along_axis(
        action_field,
        played_action[..., None],
        axis=-1,
    ).squeeze(axis=-1)


def _dirichlet_mean_categorical_nll(alpha: jax.Array, outcome: jax.Array) -> jax.Array:
    dtype = jnp.result_type(alpha, jnp.float32)
    eps = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    alpha = jnp.maximum(alpha.astype(dtype), eps)
    probs = alpha / jnp.sum(alpha, axis=-1, keepdims=True)
    clipped_outcome = jnp.clip(jnp.asarray(outcome, dtype=jnp.int32), 0, alpha.shape[-1] - 1)
    outcome_prob = jnp.take_along_axis(
        probs,
        clipped_outcome[..., None],
        axis=-1,
    ).squeeze(axis=-1)
    return -jnp.log(jnp.maximum(outcome_prob, eps))


def _dirichlet_kl(beta: jax.Array, alpha: jax.Array) -> jax.Array:
    dtype = jnp.result_type(beta, alpha)
    eps = jnp.asarray(1e-6, dtype=dtype)
    beta = jax.lax.stop_gradient(jnp.maximum(beta.astype(dtype), eps))
    alpha = jnp.maximum(alpha.astype(dtype), eps)

    beta_sum = jnp.sum(beta, axis=-1)
    alpha_sum = jnp.sum(alpha, axis=-1)
    return (
        gammaln(beta_sum)
        - gammaln(alpha_sum)
        + jnp.sum(gammaln(alpha) - gammaln(beta), axis=-1)
        + jnp.sum(
            (beta - alpha) * (digamma(beta) - digamma(beta_sum)[..., None]),
            axis=-1,
        )
    )


def _dirichlet_mean_kl(beta: jax.Array, alpha: jax.Array) -> jax.Array:
    """Categorical KL between two Dirichlet means, independent of mass."""

    dtype = jnp.result_type(beta, alpha, jnp.float32)
    eps = jnp.asarray(jnp.finfo(dtype).tiny, dtype=dtype)
    beta = jnp.maximum(beta.astype(dtype), eps)
    alpha = jnp.maximum(alpha.astype(dtype), eps)
    target = jax.lax.stop_gradient(beta / jnp.sum(beta, axis=-1, keepdims=True))
    prediction = alpha / jnp.sum(alpha, axis=-1, keepdims=True)
    return jnp.sum(
        target
        * (
            jnp.log(jnp.maximum(target, eps))
            - jnp.log(jnp.maximum(prediction, eps))
        ),
        axis=-1,
    )


def _native_dirichlet_loss(
    beta: jax.Array,
    alpha: jax.Array,
    target_kind: jax.Array,
    target_outcome: jax.Array,
    target_weight: jax.Array,
    categorical_epsilon: float,
    loss_mode: str,
) -> jax.Array:
    target_kind = jnp.asarray(target_kind)
    target_outcome = jnp.asarray(target_outcome)
    target_weight = jnp.asarray(target_weight, dtype=alpha.dtype)
    dirichlet_target = target_kind == int(TARGET_DIRICHLET)
    categorical_target = target_kind == int(TARGET_CATEGORICAL)
    valid_categorical_outcome = (
        (target_outcome >= 0) & (target_outcome < alpha.shape[-1])
    )
    safe_beta = jnp.where(
        dirichlet_target[..., None],
        beta,
        jnp.ones_like(beta),
    )
    if loss_mode == "full":
        dir_loss = _dirichlet_kl(safe_beta, alpha)
    elif loss_mode == "mean":
        dir_loss = _dirichlet_mean_kl(safe_beta, alpha)
    else:
        raise ValueError(f"unknown dirichlet_loss_mode: {loss_mode!r}")
    cat_loss = dirichlet_nll_at_categorical(
        alpha,
        jnp.where(valid_categorical_outcome, target_outcome, 0),
        categorical_epsilon,
    )
    cat_loss = jnp.where(
        ~categorical_target | valid_categorical_outcome,
        cat_loss,
        jnp.asarray(jnp.nan, dtype=cat_loss.dtype),
    )
    loss = jnp.where(categorical_target, cat_loss, dir_loss)
    loss = jnp.where(target_kind == int(TARGET_PAD), 0.0, loss)
    loss = jnp.where(
        (target_kind == int(TARGET_DIRICHLET)) | (target_kind == int(TARGET_CATEGORICAL)),
        loss,
        0.0,
    )
    return target_weight * loss


def _weighted_loss_term(weight: float, loss: jax.Array) -> jax.Array:
    """Apply a configured loss weight without allowing ``0 * NaN``."""

    weight = float(weight)
    if weight == 0.0:
        return jnp.zeros_like(loss)
    return jnp.asarray(weight, dtype=loss.dtype) * loss


def _compute_dirichlet_losses(
    logits: jax.Array,
    alpha_v: jax.Array,
    alpha_q: jax.Array,
    data: Sample,
    config,
) -> tuple[jax.Array, TrainMetrics]:
    native_fields = _native_target_fields(data)
    data = _with_native_defaults(data, native_fields)
    policy_loss_mask = _mask_or(data.policy_loss_mask, data.value_mask)
    value_loss_mask = _mask_or(data.value_loss_mask, data.value_mask)
    search_loss_mask = _mask_or(data.search_loss_mask, policy_loss_mask)
    policy_loss = optax.softmax_cross_entropy(logits, data.policy_tgt, where=data.policy_mask)
    policy_loss = _masked_mean(policy_loss, policy_loss_mask)
    policy_target_entropy = _categorical_entropy_from_probs(data.policy_tgt, data.policy_mask)
    policy_target_entropy = _masked_mean(policy_target_entropy, policy_loss_mask)
    policy_kl_hat = jax.lax.stop_gradient(policy_loss - policy_target_entropy)

    categorical_epsilon = float(config.training.losses.categorical_epsilon)
    dirichlet_loss_mode = str(config.training.losses.dirichlet_loss_mode)
    value_dir_kl = _native_dirichlet_loss(
        data.beta_V_target,
        alpha_v,
        native_fields.v_target_kind,
        native_fields.v_target_outcome,
        native_fields.v_target_weight,
        categorical_epsilon,
        dirichlet_loss_mode,
    )
    value_dir_kl_loss = _native_masked_mean(
        value_dir_kl,
        value_loss_mask,
        native_fields.v_target_kind,
    )

    q_dir_kl = _native_dirichlet_loss(
        data.beta_Q_target,
        alpha_q,
        native_fields.q_target_kind,
        native_fields.q_target_outcome,
        native_fields.q_target_weight,
        categorical_epsilon,
        dirichlet_loss_mode,
    )
    q_row_mask = (
        value_loss_mask
        if config.training.losses.loss_mask_mode == "value"
        else search_loss_mask
    )
    q_weights = jnp.where(
        data.policy_mask & q_row_mask[..., None],
        data.q_loss_weight,
        0.0,
    )
    q_metric_mask = q_weights > 0
    q_dir_kl_mask = _native_loss_mask(
        q_dir_kl,
        q_metric_mask,
        native_fields.q_target_kind,
    )
    q_dir_kl_reduction = config.training.losses.q_dir_kl_reduction
    if q_dir_kl_reduction == "masked_mean":
        q_dir_kl_loss = _native_masked_mean(
            q_dir_kl,
            q_metric_mask,
            native_fields.q_target_kind,
        )
    elif q_dir_kl_reduction == "weighted":
        q_weights = jnp.where(q_dir_kl_mask, q_weights, 0.0)
        q_dir_kl = jnp.where(
            q_dir_kl_mask,
            jnp.nan_to_num(q_dir_kl),
            jnp.zeros_like(q_dir_kl),
        )
        q_eps = jnp.asarray(jnp.finfo(q_dir_kl.dtype).eps, dtype=q_dir_kl.dtype)
        q_dir_kl_loss = jnp.sum(q_weights * q_dir_kl) / jnp.maximum(
            jnp.sum(q_weights),
            q_eps,
        )
        q_dir_kl_loss = jnp.where(
            jnp.any(q_metric_mask) & ~jnp.any(q_dir_kl_mask),
            jnp.asarray(jnp.nan, dtype=q_dir_kl_loss.dtype),
            q_dir_kl_loss,
        )
    else:
        raise ValueError(f"unknown q_dir_kl_reduction: {q_dir_kl_reduction!r}")

    outcome_mask = _mask_or(data.outcome_mask, data.value_mask)
    outcome_index = _outcome_index(data.value_tgt, alpha_v.shape[-1])
    value_outcome = _dirichlet_mean_categorical_nll(alpha_v, outcome_index)
    value_outcome_mask = outcome_mask & (
        native_fields.v_target_kind != int(TARGET_CATEGORICAL)
    )
    value_outcome_loss = _masked_mean(value_outcome, value_outcome_mask)
    played_alpha_q = _gather_played_action(alpha_q, data.played_action)
    q_outcome = _dirichlet_mean_categorical_nll(played_alpha_q, outcome_index)
    played_q_target_kind = _gather_played_action_field(
        native_fields.q_target_kind,
        data.played_action,
    )
    q_outcome_mask = outcome_mask & (
        played_q_target_kind != int(TARGET_CATEGORICAL)
    )
    q_outcome_loss = _masked_mean(q_outcome, q_outcome_mask)

    alpha_v_mass = jnp.sum(alpha_v, axis=-1)
    alpha_q_mass = jnp.sum(alpha_q, axis=-1)
    beta_v_mass = jnp.sum(data.beta_V_target, axis=-1)
    beta_q_mass = jnp.sum(data.beta_Q_target, axis=-1)
    v_dirichlet_mask = value_loss_mask & (
        native_fields.v_target_kind == int(TARGET_DIRICHLET)
    )
    q_dirichlet_mask = q_metric_mask & (
        native_fields.q_target_kind == int(TARGET_DIRICHLET)
    )
    v_categorical_mask = value_loss_mask & (
        native_fields.v_target_kind == int(TARGET_CATEGORICAL)
    )
    q_categorical_mask = q_metric_mask & (
        native_fields.q_target_kind == int(TARGET_CATEGORICAL)
    )
    # Keep each prior/posterior pair on exactly the same population.  This is
    # normally identical to the unresolved mask, while also ensuring that a
    # corrupted non-finite member cannot make the two histogram totals differ.
    v_concentration_histogram_mask = (
        v_dirichlet_mask
        & jnp.isfinite(alpha_v_mass)
        & jnp.isfinite(beta_v_mass)
    )
    q_concentration_histogram_mask = (
        q_dirichlet_mask
        & jnp.isfinite(alpha_q_mass)
        & jnp.isfinite(beta_q_mass)
    )
    dirichlet_concentration_histogram_counts = jnp.stack(
        (
            _masked_concentration_histogram_counts(
                alpha_v_mass,
                v_concentration_histogram_mask,
            ),
            _masked_concentration_histogram_counts(
                beta_v_mass,
                v_concentration_histogram_mask,
            ),
            _masked_concentration_histogram_counts(
                alpha_q_mass,
                q_concentration_histogram_mask,
            ),
            _masked_concentration_histogram_counts(
                beta_q_mass,
                q_concentration_histogram_mask,
            ),
        ),
        axis=0,
    )
    alpha_v_concentration = _masked_mean(alpha_v_mass, value_loss_mask)
    alpha_q_concentration = _masked_mean(alpha_q_mass, q_metric_mask)
    alpha_v_concentration_std = _masked_std(alpha_v_mass, value_loss_mask)
    alpha_q_concentration_std = _masked_std(alpha_q_mass, q_metric_mask)
    alpha_v_dirichlet_concentration = _masked_mean(
        alpha_v_mass,
        v_dirichlet_mask,
    )
    alpha_q_dirichlet_concentration = _masked_mean(
        alpha_q_mass,
        q_dirichlet_mask,
    )
    alpha_v_dirichlet_concentration_std = _masked_std(
        alpha_v_mass,
        v_dirichlet_mask,
    )
    alpha_q_dirichlet_concentration_std = _masked_std(
        alpha_q_mass,
        q_dirichlet_mask,
    )
    alpha_v_categorical_concentration = _masked_mean(
        alpha_v_mass,
        v_categorical_mask,
    )
    alpha_q_categorical_concentration = _masked_mean(
        alpha_q_mass,
        q_categorical_mask,
    )
    beta_v_concentration = _masked_mean(beta_v_mass, v_dirichlet_mask)
    beta_q_concentration = _masked_mean(beta_q_mass, q_dirichlet_mask)
    beta_v_concentration_std = _masked_std(beta_v_mass, v_dirichlet_mask)
    beta_q_concentration_std = _masked_std(beta_q_mass, q_dirichlet_mask)
    tiny = jnp.asarray(jnp.finfo(alpha_v_mass.dtype).tiny, alpha_v_mass.dtype)
    v_log_concentration_error = jnp.abs(
        jnp.log(jnp.maximum(alpha_v_mass, tiny))
        - jnp.log(jnp.maximum(beta_v_mass, tiny))
    )
    q_log_concentration_error = jnp.abs(
        jnp.log(jnp.maximum(alpha_q_mass, tiny))
        - jnp.log(jnp.maximum(beta_q_mass, tiny))
    )
    v_dirichlet_log_concentration_mae = _masked_mean(
        v_log_concentration_error,
        v_dirichlet_mask,
    )
    q_dirichlet_log_concentration_mae = _masked_mean(
        q_log_concentration_error,
        q_dirichlet_mask,
    )
    v_native_mask = value_loss_mask & (
        native_fields.v_target_kind != int(TARGET_PAD)
    )
    q_native_mask = q_metric_mask & (
        native_fields.q_target_kind != int(TARGET_PAD)
    )
    count_dtype = alpha_v_mass.dtype
    v_dirichlet_target_count = jnp.sum(
        v_dirichlet_mask.astype(count_dtype)
    )
    q_dirichlet_target_count = jnp.sum(
        q_dirichlet_mask.astype(count_dtype)
    )
    v_categorical_target_count = jnp.sum(
        v_categorical_mask.astype(count_dtype)
    )
    q_categorical_target_count = jnp.sum(
        q_categorical_mask.astype(count_dtype)
    )
    v_native_target_count = jnp.sum(v_native_mask.astype(count_dtype))
    q_native_target_count = jnp.sum(q_native_mask.astype(count_dtype))
    v_categorical_target_fraction = _masked_mean(
        v_categorical_mask.astype(alpha_v.dtype),
        v_native_mask,
    )
    q_categorical_target_fraction = _masked_mean(
        q_categorical_mask.astype(alpha_q.dtype),
        q_native_mask,
    )
    concentration_clip = config.training.regularization.dirichlet_concentration_clip
    concentration_floor = config.model.dirichlet_concentration_floor
    if concentration_floor is None:
        alpha_v_dirichlet_concentration_floor_fraction = jnp.zeros_like(
            alpha_v_concentration
        )
        alpha_q_dirichlet_concentration_floor_fraction = jnp.zeros_like(
            alpha_q_concentration
        )
    else:
        floor_tolerance = max(1e-3, 0.01 * float(concentration_floor))
        floor_threshold = jnp.asarray(
            float(concentration_floor) + floor_tolerance,
            dtype=alpha_v_mass.dtype,
        )
        alpha_v_dirichlet_concentration_floor_fraction = _masked_mean(
            (alpha_v_mass <= floor_threshold).astype(alpha_v_mass.dtype),
            v_dirichlet_mask,
        )
        alpha_q_dirichlet_concentration_floor_fraction = _masked_mean(
            (alpha_q_mass <= floor_threshold).astype(alpha_q_mass.dtype),
            q_dirichlet_mask,
        )
    if concentration_clip is None:
        alpha_v_concentration_clip_fraction = jnp.zeros_like(
            alpha_v_concentration
        )
        alpha_q_concentration_clip_fraction = jnp.zeros_like(
            alpha_q_concentration
        )
        alpha_v_categorical_concentration_clip_fraction = jnp.zeros_like(
            alpha_v_concentration
        )
        alpha_q_categorical_concentration_clip_fraction = jnp.zeros_like(
            alpha_q_concentration
        )
        alpha_v_dirichlet_concentration_clip_fraction = jnp.zeros_like(
            alpha_v_concentration
        )
        alpha_q_dirichlet_concentration_clip_fraction = jnp.zeros_like(
            alpha_q_concentration
        )
    else:
        if concentration_floor is None:
            clip_tolerance = 0.01 * float(concentration_clip)
        else:
            clip_tolerance = 0.01 * (
                float(concentration_clip) - float(concentration_floor)
            )
        clip_threshold = jnp.asarray(
            float(concentration_clip) - clip_tolerance,
            dtype=alpha_v_mass.dtype,
        )
        alpha_v_concentration_clip_fraction = _masked_mean(
            (alpha_v_mass >= clip_threshold).astype(alpha_v_mass.dtype),
            value_loss_mask,
        )
        alpha_q_concentration_clip_fraction = _masked_mean(
            (alpha_q_mass >= clip_threshold).astype(alpha_q_mass.dtype),
            q_metric_mask,
        )
        alpha_v_dirichlet_concentration_clip_fraction = _masked_mean(
            (alpha_v_mass >= clip_threshold).astype(alpha_v_mass.dtype),
            v_dirichlet_mask,
        )
        alpha_q_dirichlet_concentration_clip_fraction = _masked_mean(
            (alpha_q_mass >= clip_threshold).astype(alpha_q_mass.dtype),
            q_dirichlet_mask,
        )
        alpha_v_categorical_concentration_clip_fraction = _masked_mean(
            (alpha_v_mass >= clip_threshold).astype(alpha_v_mass.dtype),
            v_categorical_mask,
        )
        alpha_q_categorical_concentration_clip_fraction = _masked_mean(
            (alpha_q_mass >= clip_threshold).astype(alpha_q_mass.dtype),
            q_categorical_mask,
        )
    q_loss_weight_mean = _masked_mean(data.q_loss_weight, q_metric_mask)

    total_loss = sum(
        (
            _weighted_loss_term(config.training.losses.policy_weight, policy_loss),
            _weighted_loss_term(
                config.training.losses.value_dir_kl_weight,
                value_dir_kl_loss,
            ),
            _weighted_loss_term(
                config.training.losses.q_dir_kl_weight,
                q_dir_kl_loss,
            ),
            _weighted_loss_term(
                config.training.losses.value_outcome_weight,
                value_outcome_loss,
            ),
            _weighted_loss_term(
                config.training.losses.q_outcome_weight,
                q_outcome_loss,
            ),
        ),
        start=jnp.zeros_like(policy_loss),
    )
    metrics = _zero_train_metrics_like(
        policy_loss,
        policy_loss=policy_loss,
        value_loss=value_dir_kl_loss,
        policy_nll_loss=policy_loss,
        policy_kl_hat=policy_kl_hat,
        policy_target_entropy=policy_target_entropy,
        value_dir_kl_loss=value_dir_kl_loss,
        q_dir_kl_loss=q_dir_kl_loss,
        value_outcome_loss=value_outcome_loss,
        q_outcome_loss=q_outcome_loss,
        dirichlet_concentration_histogram_counts=(
            dirichlet_concentration_histogram_counts
        ),
        alpha_V_concentration=alpha_v_concentration,
        alpha_Q_concentration=alpha_q_concentration,
        alpha_V_concentration_std=alpha_v_concentration_std,
        alpha_Q_concentration_std=alpha_q_concentration_std,
        alpha_V_dirichlet_concentration=alpha_v_dirichlet_concentration,
        alpha_Q_dirichlet_concentration=alpha_q_dirichlet_concentration,
        alpha_V_dirichlet_concentration_std=(
            alpha_v_dirichlet_concentration_std
        ),
        alpha_Q_dirichlet_concentration_std=(
            alpha_q_dirichlet_concentration_std
        ),
        alpha_V_categorical_concentration=alpha_v_categorical_concentration,
        alpha_Q_categorical_concentration=alpha_q_categorical_concentration,
        beta_V_concentration=beta_v_concentration,
        beta_Q_concentration=beta_q_concentration,
        beta_V_concentration_std=beta_v_concentration_std,
        beta_Q_concentration_std=beta_q_concentration_std,
        v_dirichlet_log_concentration_mae=(
            v_dirichlet_log_concentration_mae
        ),
        q_dirichlet_log_concentration_mae=(
            q_dirichlet_log_concentration_mae
        ),
        v_categorical_target_fraction=v_categorical_target_fraction,
        q_categorical_target_fraction=q_categorical_target_fraction,
        alpha_V_dirichlet_concentration_floor_fraction=(
            alpha_v_dirichlet_concentration_floor_fraction
        ),
        alpha_Q_dirichlet_concentration_floor_fraction=(
            alpha_q_dirichlet_concentration_floor_fraction
        ),
        alpha_V_dirichlet_concentration_clip_fraction=(
            alpha_v_dirichlet_concentration_clip_fraction
        ),
        alpha_Q_dirichlet_concentration_clip_fraction=(
            alpha_q_dirichlet_concentration_clip_fraction
        ),
        alpha_V_concentration_clip_fraction=(
            alpha_v_concentration_clip_fraction
        ),
        alpha_Q_concentration_clip_fraction=(
            alpha_q_concentration_clip_fraction
        ),
        alpha_V_categorical_concentration_clip_fraction=(
            alpha_v_categorical_concentration_clip_fraction
        ),
        alpha_Q_categorical_concentration_clip_fraction=(
            alpha_q_categorical_concentration_clip_fraction
        ),
        v_dirichlet_target_count=v_dirichlet_target_count,
        q_dirichlet_target_count=q_dirichlet_target_count,
        v_categorical_target_count=v_categorical_target_count,
        q_categorical_target_count=q_categorical_target_count,
        v_native_target_count=v_native_target_count,
        q_native_target_count=q_native_target_count,
        q_loss_weight_mean=q_loss_weight_mean,
    )
    return total_loss, metrics


def train(model: Any, optimizer: nnx.Optimizer, data: Sample, config):
    def loss_fn(model: Any):
        output = model(data.obs, train=True)
        if len(output) == 2:
            logits, value = output
            policy_loss, value_loss = _compute_losses(logits, value, data, config)
            metrics = _zero_train_metrics_like(
                policy_loss,
                policy_loss=policy_loss,
                value_loss=value_loss,
                policy_nll_loss=policy_loss,
            )
            return policy_loss + value_loss, metrics

        logits, alpha_v, alpha_q = output
        return _compute_dirichlet_losses(logits, alpha_v, alpha_q, data, config)

    (_, metrics), grads = nnx.value_and_grad(
        loss_fn,
        has_aux=True,
    )(model)
    optimizer.update(model, grads)
    return metrics
