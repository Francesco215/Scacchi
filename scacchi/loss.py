from typing import Any, NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
from jax.scipy.special import digamma, gammaln
import optax

from .dirichlet_tree.native import (
    TARGET_PAD,
    TARGET_CATEGORICAL,
    TARGET_DIRICHLET,
    dirichlet_nll_at_categorical,
    native_fields_from_beta,
)
from .play import SelfplayOutput


DIRICHLET_KL_LOSS_CUTOFF = 1000.0


class Sample(NamedTuple):
    obs: jax.Array
    policy_tgt: jax.Array
    value_tgt: jax.Array
    played_action: jax.Array
    policy_mask: jax.Array
    value_mask: jax.Array
    beta_Q_target: jax.Array
    beta_V_target: jax.Array
    q_loss_weight: jax.Array
    policy_loss_mask: jax.Array | None = None
    value_loss_mask: jax.Array | None = None
    search_loss_mask: jax.Array | None = None
    outcome_mask: jax.Array | None = None
    q_target_kind: jax.Array | None = None
    q_target_weight: jax.Array | None = None
    q_target_outcome: jax.Array | None = None
    q_target_distance: jax.Array | None = None
    v_target_kind: jax.Array | None = None
    v_target_weight: jax.Array | None = None
    v_target_outcome: jax.Array | None = None
    v_target_distance: jax.Array | None = None

    @property
    def q_evidence_mass(self) -> jax.Array:
        return self.q_loss_weight


class _NativeTargetFields(NamedTuple):
    q_target_kind: jax.Array
    q_target_weight: jax.Array
    q_target_outcome: jax.Array
    q_target_distance: jax.Array
    v_target_kind: jax.Array
    v_target_weight: jax.Array
    v_target_outcome: jax.Array
    v_target_distance: jax.Array


class TrainMetrics(NamedTuple):
    policy_loss: jax.Array
    value_loss: jax.Array
    policy_nll_loss: jax.Array
    policy_kl_hat: jax.Array
    policy_target_entropy: jax.Array
    value_dir_kl_loss: jax.Array
    q_dir_kl_loss: jax.Array
    value_outcome_loss: jax.Array
    q_outcome_loss: jax.Array
    alpha_V_concentration: jax.Array
    alpha_Q_concentration: jax.Array
    q_loss_weight_mean: jax.Array
    search_path_depth_mean: jax.Array
    search_path_depth_p50: jax.Array
    search_path_depth_p90: jax.Array
    search_path_depth_max: jax.Array
    search_expanded_nodes: jax.Array
    search_terminal_fraction: jax.Array
    search_root_policy_entropy: jax.Array
    search_root_gamma: jax.Array
    search_root_downstream_eval_count: jax.Array
    search_root_q_concentration: jax.Array

    @property
    def q_evidence_mass_mean(self) -> jax.Array:
        return self.q_loss_weight_mean


def make_compute_input_for_lossfn(config):
    def compute_loss_input(data: SelfplayOutput) -> Sample:
        value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1
        legal_policy_mask = jnp.any(data.legal_action_mask, axis=-1)
        policy_target_mask = jnp.sum(data.action_weights, axis=-1) > 0
        search_loss_mask = (
            data.search_loss_mask
            if data.search_loss_mask is not None
            else policy_target_mask
        )

        def body_fn(carry: jax.Array, i: jax.Array) -> tuple[jax.Array, jax.Array]:
            ix = config.selfplay.max_num_steps - i - 1
            value = data.reward[ix] + data.discount[ix] * carry
            return value, value

        _, value_tgt = jax.lax.scan(
            body_fn,
            jnp.zeros(data.reward.shape[1], dtype=data.reward.dtype),
            jnp.arange(config.selfplay.max_num_steps),
        )
        value_tgt = value_tgt[::-1, :]
        policy_tgt = jnp.asarray(data.action_weights)
        policy_loss_mask = legal_policy_mask & search_loss_mask
        value_loss_mask = search_loss_mask
        loss_mask_mode = config.training.losses.loss_mask_mode
        if loss_mask_mode == "value":
            policy_loss_mask = value_mask
            value_loss_mask = value_mask
        elif loss_mask_mode != "search":
            raise ValueError(f"unknown loss_mask_mode: {loss_mask_mode!r}")
        if config.training.losses.policy_target_mode == "winner_action":
            policy_tgt = jax.nn.one_hot(
                data.played_action,
                data.action_weights.shape[-1],
                dtype=data.action_weights.dtype,
            )
            policy_loss_mask = legal_policy_mask & value_mask & (value_tgt > 0)

        beta_q_target = data.beta_Q_target
        beta_v_target = data.beta_V_target
        q_loss_weight = data.q_loss_weight
        q_target_kind = data.q_target_kind
        q_target_weight = data.q_target_weight
        q_target_outcome = data.q_target_outcome
        q_target_distance = data.q_target_distance
        v_target_kind = data.v_target_kind
        v_target_weight = data.v_target_weight
        v_target_outcome = data.v_target_outcome
        v_target_distance = data.v_target_distance

        terminal_edge_targets = config.training.losses.terminal_edge_targets
        terminal_parent_targets = config.training.losses.terminal_parent_targets
        if terminal_edge_targets or terminal_parent_targets:
            native_defaults = native_fields_from_beta(beta_q_target, beta_v_target)
            q_target_kind = (
                native_defaults["q_target_kind"] if q_target_kind is None else q_target_kind
            )
            q_target_weight = (
                native_defaults["q_target_weight"]
                if q_target_weight is None
                else q_target_weight
            )
            q_target_outcome = (
                native_defaults["q_target_outcome"]
                if q_target_outcome is None
                else q_target_outcome
            )
            q_target_distance = (
                native_defaults["q_target_distance"]
                if q_target_distance is None
                else q_target_distance
            )
            v_target_kind = (
                native_defaults["v_target_kind"] if v_target_kind is None else v_target_kind
            )
            v_target_weight = (
                native_defaults["v_target_weight"]
                if v_target_weight is None
                else v_target_weight
            )
            v_target_outcome = (
                native_defaults["v_target_outcome"]
                if v_target_outcome is None
                else v_target_outcome
            )
            v_target_distance = (
                native_defaults["v_target_distance"]
                if v_target_distance is None
                else v_target_distance
            )
            num_outcomes = beta_v_target.shape[-1]
            rounded_reward = jnp.round(data.reward).astype(jnp.int32)
            if num_outcomes == 2:
                outcome_index = (rounded_reward + 1) // 2
            elif num_outcomes == 3:
                outcome_index = rounded_reward + 1
            else:
                raise ValueError(f"unsupported outcome count: {num_outcomes}")
            played_action_mask = jax.nn.one_hot(
                data.played_action,
                beta_q_target.shape[-2],
                dtype=bool,
            )
            terminal_action_mask = data.terminated[..., None] & played_action_mask

        if terminal_edge_targets:
            q_target_kind = jnp.where(
                terminal_action_mask,
                jnp.asarray(int(TARGET_CATEGORICAL), dtype=q_target_kind.dtype),
                q_target_kind,
            )
            q_target_weight = jnp.where(
                terminal_action_mask,
                jnp.ones((), dtype=q_target_weight.dtype),
                q_target_weight,
            )
            q_target_outcome = jnp.where(
                terminal_action_mask,
                outcome_index[..., None].astype(q_target_outcome.dtype),
                q_target_outcome,
            )
            q_target_distance = jnp.where(
                terminal_action_mask,
                jnp.ones((), dtype=q_target_distance.dtype),
                q_target_distance,
            )
            q_loss_weight = jnp.where(
                terminal_action_mask,
                jnp.maximum(q_loss_weight, jnp.ones((), dtype=q_loss_weight.dtype)),
                q_loss_weight,
            )

        if terminal_parent_targets:
            terminal_win_mask = data.terminated & (data.reward > 0)
            policy_tgt = jnp.where(
                terminal_win_mask[..., None],
                played_action_mask.astype(policy_tgt.dtype),
                policy_tgt,
            )
            policy_loss_mask = policy_loss_mask | (terminal_win_mask & legal_policy_mask)
            value_loss_mask = value_loss_mask | terminal_win_mask
            v_target_kind = jnp.where(
                terminal_win_mask,
                jnp.asarray(int(TARGET_CATEGORICAL), dtype=v_target_kind.dtype),
                v_target_kind,
            )
            v_target_weight = jnp.where(
                terminal_win_mask,
                jnp.ones((), dtype=v_target_weight.dtype),
                v_target_weight,
            )
            v_target_outcome = jnp.where(
                terminal_win_mask,
                outcome_index.astype(v_target_outcome.dtype),
                v_target_outcome,
            )
            v_target_distance = jnp.where(
                terminal_win_mask,
                jnp.ones((), dtype=v_target_distance.dtype),
                v_target_distance,
            )

        sample = Sample(
            obs=data.obs,
            policy_tgt=policy_tgt,
            value_tgt=value_tgt,
            played_action=data.played_action,
            policy_mask=data.legal_action_mask,
            value_mask=value_mask,
            beta_Q_target=beta_q_target,
            beta_V_target=beta_v_target,
            q_loss_weight=q_loss_weight,
            policy_loss_mask=policy_loss_mask,
            value_loss_mask=value_loss_mask,
            search_loss_mask=search_loss_mask,
            outcome_mask=value_mask,
            q_target_kind=q_target_kind,
            q_target_weight=q_target_weight,
            q_target_outcome=q_target_outcome,
            q_target_distance=q_target_distance,
            v_target_kind=v_target_kind,
            v_target_weight=v_target_weight,
            v_target_outcome=v_target_outcome,
            v_target_distance=v_target_distance,
        )
        native_fields = _native_target_fields(sample)
        sample = _with_native_defaults(sample, native_fields)
        if data.tree_data is None:
            return sample

        tree = data.tree_data

        def flatten_root(x: jax.Array) -> jax.Array:
            return x.reshape((-1, *x.shape[2:]))

        def wrap_rows(x: jax.Array) -> jax.Array:
            return x[None, ...]

        root_obs = flatten_root(sample.obs)
        tree_obs = flatten_root(tree.obs)
        root_policy_tgt = flatten_root(sample.policy_tgt)
        tree_policy_tgt = flatten_root(tree.action_weights)
        root_value_tgt = flatten_root(sample.value_tgt)
        tree_value_tgt = flatten_root(tree.value_tgt)
        root_played_action = flatten_root(sample.played_action)
        tree_played_action = flatten_root(tree.played_action)
        root_policy_mask = flatten_root(sample.policy_mask)
        tree_policy_mask = flatten_root(tree.legal_action_mask)
        root_beta_q = flatten_root(sample.beta_Q_target)
        tree_beta_q = flatten_root(tree.beta_Q_target)
        root_beta_v = flatten_root(sample.beta_V_target)
        tree_beta_v = flatten_root(tree.beta_V_target)
        root_q_weight = flatten_root(sample.q_loss_weight)
        tree_q_weight = flatten_root(tree.q_loss_weight)
        tree_q_defaults = _tree_native_defaults(tree.beta_Q_target, tree.beta_V_target)
        root_q_kind = flatten_root(native_fields.q_target_kind)
        tree_q_kind = flatten_root(_tree_field_or_default(tree, tree_q_defaults, "q_target_kind"))
        root_q_target_weight = flatten_root(native_fields.q_target_weight)
        tree_q_target_weight = flatten_root(
            _tree_field_or_default(tree, tree_q_defaults, "q_target_weight")
        )
        root_q_outcome = flatten_root(native_fields.q_target_outcome)
        tree_q_outcome = flatten_root(
            _tree_field_or_default(tree, tree_q_defaults, "q_target_outcome")
        )
        root_q_distance = flatten_root(native_fields.q_target_distance)
        tree_q_distance = flatten_root(
            _tree_field_or_default(tree, tree_q_defaults, "q_target_distance")
        )
        root_v_kind = flatten_root(native_fields.v_target_kind)
        tree_v_kind = flatten_root(_tree_field_or_default(tree, tree_q_defaults, "v_target_kind"))
        root_v_target_weight = flatten_root(native_fields.v_target_weight)
        tree_v_target_weight = flatten_root(
            _tree_field_or_default(tree, tree_q_defaults, "v_target_weight")
        )
        root_v_outcome = flatten_root(native_fields.v_target_outcome)
        tree_v_outcome = flatten_root(
            _tree_field_or_default(tree, tree_q_defaults, "v_target_outcome")
        )
        root_v_distance = flatten_root(native_fields.v_target_distance)
        tree_v_distance = flatten_root(
            _tree_field_or_default(tree, tree_q_defaults, "v_target_distance")
        )
        root_policy_loss_mask = flatten_root(policy_loss_mask)
        tree_policy_loss_mask = flatten_root(tree.policy_loss_mask)
        root_value_loss_mask = flatten_root(sample.value_loss_mask)
        tree_value_loss_mask = flatten_root(tree.value_loss_mask)
        root_search_loss_mask = flatten_root(search_loss_mask)
        tree_search_loss_mask = flatten_root(tree.search_loss_mask)
        root_outcome_mask = flatten_root(value_mask)
        tree_outcome_mask = flatten_root(tree.outcome_mask)

        return Sample(
            obs=wrap_rows(jnp.concatenate([root_obs, tree_obs], axis=0)),
            policy_tgt=wrap_rows(jnp.concatenate([root_policy_tgt, tree_policy_tgt], axis=0)),
            value_tgt=wrap_rows(jnp.concatenate([root_value_tgt, tree_value_tgt], axis=0)),
            played_action=wrap_rows(jnp.concatenate([root_played_action, tree_played_action], axis=0)),
            policy_mask=wrap_rows(jnp.concatenate([root_policy_mask, tree_policy_mask], axis=0)),
            value_mask=wrap_rows(jnp.concatenate([root_value_loss_mask, tree_value_loss_mask], axis=0)),
            beta_Q_target=wrap_rows(jnp.concatenate([root_beta_q, tree_beta_q], axis=0)),
            beta_V_target=wrap_rows(jnp.concatenate([root_beta_v, tree_beta_v], axis=0)),
            q_loss_weight=wrap_rows(jnp.concatenate([root_q_weight, tree_q_weight], axis=0)),
            policy_loss_mask=wrap_rows(
                jnp.concatenate([root_policy_loss_mask, tree_policy_loss_mask], axis=0)
            ),
            value_loss_mask=wrap_rows(
                jnp.concatenate([root_value_loss_mask, tree_value_loss_mask], axis=0)
            ),
            search_loss_mask=wrap_rows(
                jnp.concatenate([root_search_loss_mask, tree_search_loss_mask], axis=0)
            ),
            outcome_mask=wrap_rows(jnp.concatenate([root_outcome_mask, tree_outcome_mask], axis=0)),
            q_target_kind=wrap_rows(jnp.concatenate([root_q_kind, tree_q_kind], axis=0)),
            q_target_weight=wrap_rows(
                jnp.concatenate([root_q_target_weight, tree_q_target_weight], axis=0)
            ),
            q_target_outcome=wrap_rows(
                jnp.concatenate([root_q_outcome, tree_q_outcome], axis=0)
            ),
            q_target_distance=wrap_rows(
                jnp.concatenate([root_q_distance, tree_q_distance], axis=0)
            ),
            v_target_kind=wrap_rows(jnp.concatenate([root_v_kind, tree_v_kind], axis=0)),
            v_target_weight=wrap_rows(
                jnp.concatenate([root_v_target_weight, tree_v_target_weight], axis=0)
            ),
            v_target_outcome=wrap_rows(
                jnp.concatenate([root_v_outcome, tree_v_outcome], axis=0)
            ),
            v_target_distance=wrap_rows(
                jnp.concatenate([root_v_distance, tree_v_distance], axis=0)
            ),
        )

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


def _native_target_fields(sample: Sample) -> _NativeTargetFields:
    defaults = native_fields_from_beta(sample.beta_Q_target, sample.beta_V_target)
    return _NativeTargetFields(
        q_target_kind=(
            defaults["q_target_kind"] if sample.q_target_kind is None else sample.q_target_kind
        ),
        q_target_weight=(
            defaults["q_target_weight"]
            if sample.q_target_weight is None
            else sample.q_target_weight
        ),
        q_target_outcome=(
            defaults["q_target_outcome"]
            if sample.q_target_outcome is None
            else sample.q_target_outcome
        ),
        q_target_distance=(
            defaults["q_target_distance"]
            if sample.q_target_distance is None
            else sample.q_target_distance
        ),
        v_target_kind=(
            defaults["v_target_kind"] if sample.v_target_kind is None else sample.v_target_kind
        ),
        v_target_weight=(
            defaults["v_target_weight"]
            if sample.v_target_weight is None
            else sample.v_target_weight
        ),
        v_target_outcome=(
            defaults["v_target_outcome"]
            if sample.v_target_outcome is None
            else sample.v_target_outcome
        ),
        v_target_distance=(
            defaults["v_target_distance"]
            if sample.v_target_distance is None
            else sample.v_target_distance
        ),
    )


def _with_native_defaults(
    sample: Sample,
    native_fields: _NativeTargetFields | None = None,
) -> Sample:
    if native_fields is None:
        native_fields = _native_target_fields(sample)
    return sample._replace(
        q_target_kind=native_fields.q_target_kind,
        q_target_weight=native_fields.q_target_weight,
        q_target_outcome=native_fields.q_target_outcome,
        q_target_distance=native_fields.q_target_distance,
        v_target_kind=native_fields.v_target_kind,
        v_target_weight=native_fields.v_target_weight,
        v_target_outcome=native_fields.v_target_outcome,
        v_target_distance=native_fields.v_target_distance,
    )


def _tree_native_defaults(beta_q: jax.Array, beta_v: jax.Array) -> dict[str, jax.Array]:
    return native_fields_from_beta(beta_q, beta_v)


def _tree_field_or_default(tree, defaults: dict[str, jax.Array], field: str) -> jax.Array:
    value = getattr(tree, field)
    return defaults[field] if value is None else value


def _compute_losses(logits: jax.Array, value: jax.Array, data: Sample) -> tuple[jax.Array, jax.Array]:
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

    categorical_epsilon = float(config.search.active_constants().categorical_epsilon)
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
    metrics = TrainMetrics(
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
        search_path_depth_mean=jnp.zeros_like(policy_loss),
        search_path_depth_p50=jnp.zeros_like(policy_loss),
        search_path_depth_p90=jnp.zeros_like(policy_loss),
        search_path_depth_max=jnp.zeros_like(policy_loss),
        search_expanded_nodes=jnp.zeros_like(policy_loss),
        search_terminal_fraction=jnp.zeros_like(policy_loss),
        search_root_policy_entropy=jnp.zeros_like(policy_loss),
        search_root_gamma=jnp.zeros_like(policy_loss),
        search_root_downstream_eval_count=jnp.zeros_like(policy_loss),
        search_root_q_concentration=jnp.zeros_like(policy_loss),
    )
    return total_loss, metrics


def train(model: Any, optimizer: nnx.Optimizer, data: Sample, config):
    def loss_fn(model: Any):
        output = model(data.obs, train=True)
        if len(output) == 2:
            logits, value = output
            policy_loss, value_loss = _compute_losses(logits, value, data)
            metrics = TrainMetrics(
                policy_loss=policy_loss,
                value_loss=value_loss,
                policy_nll_loss=policy_loss,
                policy_kl_hat=jnp.zeros_like(policy_loss),
                policy_target_entropy=jnp.zeros_like(policy_loss),
                value_dir_kl_loss=jnp.zeros_like(value_loss),
                q_dir_kl_loss=jnp.zeros_like(value_loss),
                value_outcome_loss=jnp.zeros_like(value_loss),
                q_outcome_loss=jnp.zeros_like(value_loss),
                alpha_V_concentration=jnp.zeros_like(value_loss),
                alpha_Q_concentration=jnp.zeros_like(value_loss),
                q_loss_weight_mean=jnp.zeros_like(value_loss),
                search_path_depth_mean=jnp.zeros_like(value_loss),
                search_path_depth_p50=jnp.zeros_like(value_loss),
                search_path_depth_p90=jnp.zeros_like(value_loss),
                search_path_depth_max=jnp.zeros_like(value_loss),
                search_expanded_nodes=jnp.zeros_like(value_loss),
                search_terminal_fraction=jnp.zeros_like(value_loss),
                search_root_policy_entropy=jnp.zeros_like(value_loss),
                search_root_gamma=jnp.zeros_like(value_loss),
                search_root_downstream_eval_count=jnp.zeros_like(value_loss),
                search_root_q_concentration=jnp.zeros_like(value_loss),
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
