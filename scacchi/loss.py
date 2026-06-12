from typing import Any, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln
from jaxtyping import Array, Bool, Float, Int
import optax

from .dirichlet_tree.native import (
    TARGET_PAD,
    TARGET_CATEGORICAL,
    TARGET_DIRICHLET,
    dirichlet_nll_at_categorical,
    native_fields_from_beta,
)
from .distributed import DISABLED_BATCH_PARALLEL, BatchParallel, assert_batch_axis_sharded
from .play import TrainingSamples


DIRICHLET_KL_LOSS_CUTOFF = 1000.0


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

_TREE_TO_SAMPLE_FIELDS = {
    "obs": "obs",
    "policy_tgt": "action_weights",
    "value_tgt": "value_tgt",
    "played_action": "played_action",
    "policy_mask": "legal_action_mask",
    "value_mask": "value_loss_mask",
    "beta_Q_target": "beta_Q_target",
    "beta_V_target": "beta_V_target",
    "q_loss_weight": "q_loss_weight",
    "policy_loss_mask": "policy_loss_mask",
    "value_loss_mask": "value_loss_mask",
    "search_loss_mask": "search_loss_mask",
    "outcome_mask": "outcome_mask",
}


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
    alpha_V_concentration: Float[Array, "*batch"]
    alpha_Q_concentration: Float[Array, "*batch"]
    q_loss_weight_mean: Float[Array, "*batch"]
    search_path_depth_mean: Float[Array, "*batch"]
    search_path_depth_p50: Float[Array, "*batch"]
    search_path_depth_p90: Float[Array, "*batch"]
    search_path_depth_max: Float[Array, "*batch"]
    search_expanded_nodes: Float[Array, "*batch"]
    search_terminal_fraction: Float[Array, "*batch"]
    search_root_policy_entropy: Float[Array, "*batch"]
    search_root_gamma: Float[Array, "*batch"]
    search_root_downstream_eval_count: Float[Array, "*batch"]
    search_root_q_concentration: Float[Array, "*batch"]
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
        sample = assert_batch_axis_sharded(sample, parallel, batch_axis=0, label="loss sample after native defaults")
        tree_data = metadata_value("tree_data")
        if tree_data is None:
            return sample

        tree = tree_data
        tree_native_defaults = native_fields_from_beta(
            tree.beta_Q_target,
            tree.beta_V_target,
        )
        tree_native_defaults = assert_batch_axis_sharded(tree_native_defaults, parallel, batch_axis=0, label="loss tree native_defaults")
        sample = _concat_sample_rows(
            sample._replace(value_mask=sample.value_loss_mask),
            _tree_as_sample(tree, tree_native_defaults),
        )
        return assert_batch_axis_sharded(sample, parallel, batch_axis=0, label="loss sample after tree concat")

    return compute_loss_input


def _masked_mean(loss: jax.Array, mask: jax.Array) -> jax.Array:
    mask_bool = mask.astype(jnp.bool_)
    mask_float = mask_bool.astype(loss.dtype)
    safe_loss = jnp.where(mask_bool, loss, jnp.zeros_like(loss))
    return jnp.sum(safe_loss) / jnp.maximum(jnp.sum(mask_float), 1)


def _bounded_loss_mask(
    loss: jax.Array,
    mask: jax.Array,
    *,
    cutoff: float = DIRICHLET_KL_LOSS_CUTOFF,
) -> jax.Array:
    return mask.astype(jnp.bool_) & jnp.isfinite(loss) & (loss <= cutoff)


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


def _mask_or(mask: jax.Array | None, fallback: jax.Array) -> jax.Array:
    return fallback if mask is None else mask


def _native_target_values(source: Any) -> dict[str, jax.Array | None]:
    return {field: getattr(source, field) for field in _NATIVE_TARGET_FIELD_NAMES}


def _native_fields_from_values(
    values: dict[str, jax.Array | None],
    defaults: dict[str, jax.Array],
) -> _NativeTargetFields:
    def value_or_default(field: str) -> jax.Array:
        value = values[field]
        return defaults[field] if value is None else value

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


def _tree_as_sample(tree: Any, native_defaults: dict[str, jax.Array]) -> Sample:
    fields = {
        sample_field: getattr(tree, tree_field)
        for sample_field, tree_field in _TREE_TO_SAMPLE_FIELDS.items()
    }
    fields.update(
        _native_fields_from_values(
            _native_target_values(tree),
            native_defaults,
        )._asdict()
    )
    return Sample(**fields)


def _flatten_rows_like(reference: jax.Array, value: jax.Array) -> jax.Array:
    keep_feature_ndim = reference.ndim - 2
    feature_shape = value.shape[-keep_feature_ndim:] if keep_feature_ndim else ()
    return value.reshape((value.shape[0], -1, *feature_shape))


def _concat_sample_rows(root: Sample, tree: Sample) -> Sample:
    return jax.tree_util.tree_map(
        lambda root_leaf, tree_leaf: jnp.concatenate(
            [
                _flatten_rows_like(root_leaf, root_leaf),
                _flatten_rows_like(root_leaf, tree_leaf),
            ],
            axis=1,
        ),
        root,
        tree,
    )


def _zero_train_metrics_like(reference: jax.Array, **values: jax.Array) -> TrainMetrics:
    fields = {field: jnp.zeros_like(reference) for field in TrainMetrics._fields}
    fields.update(values)
    return TrainMetrics(**fields)


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


def _native_dirichlet_loss(
    beta: jax.Array,
    alpha: jax.Array,
    target_kind: jax.Array,
    target_outcome: jax.Array,
    target_weight: jax.Array,
    categorical_epsilon: float,
) -> jax.Array:
    target_kind = jnp.asarray(target_kind)
    target_outcome = jnp.asarray(target_outcome)
    target_weight = jnp.asarray(target_weight, dtype=alpha.dtype)
    clipped_outcome = jnp.clip(target_outcome, 0, alpha.shape[-1] - 1)
    dir_loss = _dirichlet_kl(beta, alpha)
    cat_loss = dirichlet_nll_at_categorical(alpha, clipped_outcome, categorical_epsilon)
    loss = jnp.where(target_kind == int(TARGET_CATEGORICAL), cat_loss, dir_loss)
    loss = jnp.where(target_kind == int(TARGET_PAD), 0.0, loss)
    loss = jnp.where(
        (target_kind == int(TARGET_DIRICHLET)) | (target_kind == int(TARGET_CATEGORICAL)),
        loss,
        0.0,
    )
    return target_weight * loss


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

    categorical_epsilon = float(
        config.selfplay.search.active_constants().categorical_epsilon
    )
    value_dir_kl = _native_dirichlet_loss(
        data.beta_V_target,
        alpha_v,
        native_fields.v_target_kind,
        native_fields.v_target_outcome,
        native_fields.v_target_weight,
        categorical_epsilon,
    )
    value_dir_kl_loss = _bounded_masked_mean(value_dir_kl, value_loss_mask)

    q_dir_kl = _native_dirichlet_loss(
        data.beta_Q_target,
        alpha_q,
        native_fields.q_target_kind,
        native_fields.q_target_outcome,
        native_fields.q_target_weight,
        categorical_epsilon,
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
    q_dir_kl_mask = _bounded_loss_mask(q_dir_kl, q_metric_mask)
    q_dir_kl_reduction = config.training.losses.q_dir_kl_reduction
    if q_dir_kl_reduction == "masked_mean":
        q_dir_kl_loss = _bounded_masked_mean(q_dir_kl, q_metric_mask)
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
    value_outcome_loss = _masked_mean(value_outcome, outcome_mask)
    played_alpha_q = _gather_played_action(alpha_q, data.played_action)
    q_outcome = _dirichlet_mean_categorical_nll(played_alpha_q, outcome_index)
    q_outcome_loss = _masked_mean(q_outcome, outcome_mask)

    alpha_v_concentration = _masked_mean(jnp.sum(alpha_v, axis=-1), value_loss_mask)
    alpha_q_concentration = _masked_mean(
        jnp.sum(alpha_q, axis=-1),
        q_metric_mask,
    )
    q_loss_weight_mean = _masked_mean(data.q_loss_weight, q_metric_mask)

    total_loss = (
        config.training.losses.policy_weight * policy_loss
        + config.training.losses.value_dir_kl_weight * value_dir_kl_loss
        + config.training.losses.q_dir_kl_weight * q_dir_kl_loss
        + config.training.losses.value_outcome_weight * value_outcome_loss
        + config.training.losses.q_outcome_weight * q_outcome_loss
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
        alpha_V_concentration=alpha_v_concentration,
        alpha_Q_concentration=alpha_q_concentration,
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
