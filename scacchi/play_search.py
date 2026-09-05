import functools
from typing import Any, Callable, NamedTuple, cast

import chex
from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int
import mctx
import pgx

from . import dirichlet_mctx
from .dirichlet_mctx.action_selection import (
    categorical_action_population,
    posterior_best_policy,
)
from .dirichlet_mctx.native_targets import (
    TARGET_CATEGORICAL,
    TARGET_DIRICHLET,
    TARGET_PAD,
)
from .dirichlet_mctx.outcomes import NO_OUTCOME, outcome_mean, outcome_utility
from .dirichlet_mctx.posterior_updates import (
    DEFAULT_POLICY_SAMPLE_CHUNK_SIZE,
    DEFAULT_PREFIX_DENSITY_LOG_INTEGRAL_TOLERANCE,
)
from .dirichlet_q_search import (
    QSupervision,
    build_q_supervision,
    make_dirichlet_expand_fn,
    posterior_best_action,
    posterior_sample_action,
    terminal_outcome_from_reward,
)
from .network import policy_value_from_output
from .types import (
    ActionCommitmentConfig,
    ActionCommitmentType,
    DirichletThompsonSearchConfig,
    GumbelSearchConfig,
    MonteCarloPosteriorUpdateConfig,
    NumericalPosteriorUpdateConfig,
    PolicySearchConfig,
    PosteriorUpdateKind,
    QSupervisionConfig,
    RootPolicySupport,
    SearchConfig,
    SearchKind,
)


class EvaluatorOutput(NamedTuple):
    logits: Float[Array, "*batch action"]
    value: Float[Array, "*batch"] | None = None
    alpha_v: Float[Array, "*batch outcome"] | None = None
    alpha_q: Float[Array, "*batch action outcome"] | None = None
Evaluator = Callable[[jax.Array], EvaluatorOutput]

class PosteriorPrediction(NamedTuple):
    policy: Float[Array, "*batch action"]
    value: Float[Array, "*batch"] | None = None
    alpha_v: Float[Array, "*batch outcome"] | None = None
    alpha_q: Float[Array, "*batch action outcome"] | None = None


class TargetMetadata(NamedTuple):
    mask: Bool[Array, "*batch"] | None = None
    q_supervision: QSupervision | None = None
    q_positive_evidence_action: Bool[Array, "*batch action"] | None = None
    q_positive_policy_action: Bool[Array, "*batch action"] | None = None
    search_action: Int[Array, "*batch"] | None = None
    q_target_kind: Int[Array, "*batch action"] | None = None
    q_target_weight: Float[Array, "*batch action"] | None = None
    q_target_outcome: Int[Array, "*batch action"] | None = None
    q_target_distance: Int[Array, "*batch action"] | None = None
    v_target_kind: Int[Array, "*batch"] | None = None
    v_target_weight: Float[Array, "*batch"] | None = None
    v_target_outcome: Int[Array, "*batch"] | None = None
    v_target_distance: Int[Array, "*batch"] | None = None


class PosteriorTargets(NamedTuple):
    prediction: PosteriorPrediction
    metadata: TargetMetadata | None = None


Search = Callable[[pgx.State, chex.PRNGKey], PosteriorTargets]


class PlayerOutput(NamedTuple):
    action: Int[Array, "*batch"]
    posterior: PosteriorTargets | None = None


def evaluator_output_from_model_output(model_output: Any) -> EvaluatorOutput:
    if isinstance(model_output, EvaluatorOutput):
        return model_output
    if isinstance(model_output, jax.Array):
        return EvaluatorOutput(logits=model_output)
    if len(model_output) == 2:
        logits, value = policy_value_from_output(model_output)
        return EvaluatorOutput(logits=logits, value=value)
    logits, alpha_v, alpha_q = model_output
    return EvaluatorOutput(
        logits=logits,
        value=outcome_utility(outcome_mean(alpha_v)),
        alpha_v=alpha_v,
        alpha_q=alpha_q,
    )


def make_evaluator(model: Any) -> Evaluator:
    def evaluator(obs: jax.Array) -> EvaluatorOutput:
        if isinstance(model, nnx.Module):
            return evaluator_output_from_model_output(model(obs, train=False))
        return evaluator_output_from_model_output(model(obs))

    return evaluator


def _required_output(value: jax.Array | None, name: str) -> jax.Array:
    if value is None:
        raise ValueError(f"evaluator output is missing {name}")
    return value


def mask_logits(logits: jax.Array, legal_action_mask: jax.Array) -> jax.Array:
    return jnp.where(legal_action_mask, logits, jnp.finfo(logits.dtype).min)


def _masked_policy(logits: jax.Array, legal_action_mask: jax.Array, *, temperature: float) -> jax.Array:
    policy = jax.nn.softmax(mask_logits(logits, legal_action_mask) / temperature, axis=-1)
    return jnp.where(jnp.any(legal_action_mask, axis=-1, keepdims=True), policy, 0.0)


def make_gumbel_expand_fn(env, evaluator: Evaluator):
    def expand_fn(_, rng_key: jax.Array, action: jax.Array, env_state: pgx.State):
        del rng_key

        current_player = env_state.current_player
        env_state = jax.vmap(env.step)(env_state, action)
        prediction = evaluator(env_state.observation)
        logits = mask_logits(prediction.logits, env_state.legal_action_mask)
        value = _required_output(prediction.value, "value")

        reward = env_state.rewards[jnp.arange(env_state.rewards.shape[0]), current_player]
        value = jnp.where(env_state.terminated, 0.0, value)
        discount = -jnp.ones_like(value)
        discount = jnp.where(env_state.terminated, 0.0, discount)

        fn_output = mctx.RecurrentFnOutput(reward=reward,discount=discount,prior_logits=logits,value=value)
        return fn_output, env_state

    return expand_fn


def _search_loss_mask(action_weights: jax.Array) -> jax.Array:
    return jnp.sum(action_weights, axis=-1) > 0


def _numerical_policy_readout(
    native_policy: jax.Array,
    *,
    alpha: jax.Array,
    q_categorical_outcome: jax.Array,
    v_categorical_outcome: jax.Array,
    legal_action_mask: jax.Array,
    config: NumericalPosteriorUpdateConfig,
) -> jax.Array:
    invalid_actions = ~legal_action_mask
    estimate = (
        dirichlet_mctx.binary_posterior_best_policy_prefix_quadrature(
            alpha,
            invalid_actions,
            q_categorical_outcome,
            half_width=int(config.half_width),
            tail_scale=float(config.tail_scale),
            min_half_range=float(config.min_half_range),
            max_half_range=float(config.max_half_range),
        )
    )
    density_error = jnp.max(
        jnp.abs(estimate.density_log_integral),
        axis=-1,
    )
    unsafe = (
        estimate.tail_range_clipped
        | ~estimate.finite
        | (
            density_error
            > DEFAULT_PREFIX_DENSITY_LOG_INTEGRAL_TOLERANCE
        )
    )

    has_legal_action = jnp.any(legal_action_mask, axis=-1)
    root_is_categorical = v_categorical_outcome != int(NO_OUTCOME)
    unresolved_root = has_legal_action & ~root_is_categorical
    accepted = unresolved_root & ~unsafe

    return jnp.where(
        accepted[:, None],
        estimate.policy,
        native_policy,
    )


def _root_policy_support(
    legal_action_mask: jax.Array,
    positive_search_evidence: jax.Array,
    categorical_outcome: jax.Array,
    support_mode: RootPolicySupport,
) -> jax.Array:
    """Choose a stable Q21 population without discarding legal fallback rows."""

    legal_action_mask = jnp.asarray(legal_action_mask, dtype=jnp.bool_)
    if support_mode == RootPolicySupport.all_legal:
        return legal_action_mask
    if support_mode != RootPolicySupport.search_evidence:
        raise ValueError(f"unknown root_policy_support: {support_mode!r}")
    candidate = legal_action_mask & (
        jnp.asarray(positive_search_evidence, dtype=jnp.bool_)
        | (jnp.asarray(categorical_outcome) != int(NO_OUTCOME))
    )
    return jnp.where(
        jnp.any(candidate, axis=-1, keepdims=True),
        candidate,
        legal_action_mask,
    )


def _normalize_policy_on_support(
    policy: jax.Array,
    support: jax.Array,
    *,
    temperature: float = 1.0,
) -> jax.Array:
    """Project and power-normalize a policy while preserving exact zeros."""

    policy = jnp.asarray(policy)
    support = jnp.asarray(support, dtype=jnp.bool_)
    positive = support & (policy > 0.0)
    log_policy = jnp.log(
        jnp.clip(
            policy,
            jnp.finfo(policy.dtype).tiny,
            1.0,
        )
    )
    masked_log_policy = jnp.where(
        positive,
        log_policy,
        jnp.finfo(policy.dtype).min,
    )
    centered_log_policy = masked_log_policy - jnp.max(
        masked_log_policy,
        axis=-1,
        keepdims=True,
    )
    scaled_logits = jnp.where(
        positive,
        centered_log_policy
        / jnp.asarray(float(temperature), dtype=policy.dtype),
        jnp.finfo(policy.dtype).min,
    )
    normalized = jax.nn.softmax(scaled_logits, axis=-1)
    normalized = jnp.where(positive, normalized, 0.0)
    has_positive = jnp.any(positive, axis=-1, keepdims=True)
    support_count = jnp.sum(support, axis=-1, keepdims=True)
    fallback = support.astype(policy.dtype) / jnp.maximum(support_count, 1)
    return jnp.where(has_positive, normalized, fallback)


def _dirichlet_root_policy_readout(
    native_policy: jax.Array,
    *,
    summary: Any,
    legal_action_mask: jax.Array,
    search_cfg: DirichletThompsonSearchConfig,
) -> jax.Array:
    """Build the replay readout with the search posterior updater."""

    if search_cfg.posterior_update.kind != PosteriorUpdateKind.numerical:
        return native_policy

    visit_counts = getattr(summary, "visit_counts", None)
    if visit_counts is None:
        positive_search_evidence = jnp.zeros_like(
            legal_action_mask,
            dtype=jnp.bool_,
        )
    else:
        positive_search_evidence = jnp.asarray(visit_counts) > 0
    readout_support = _root_policy_support(
        legal_action_mask,
        positive_search_evidence,
        summary.q_categorical_outcome,
        search_cfg.root_policy_support,
    )
    native_policy = _normalize_policy_on_support(
        native_policy,
        readout_support,
    )
    invalid_actions = ~readout_support
    target_policy = _numerical_policy_readout(
        native_policy,
        alpha=summary.alpha,
        q_categorical_outcome=summary.q_categorical_outcome,
        v_categorical_outcome=summary.v_categorical_outcome,
        legal_action_mask=readout_support,
        config=search_cfg.posterior_update.numerical,
    )
    categorical_policy = categorical_action_population(
        summary.v_categorical_outcome,
        summary.q_categorical_outcome,
        summary.q_categorical_distance,
        invalid_actions,
        num_outcomes=summary.alpha.shape[-1],
        dtype=native_policy.dtype,
    )
    has_categorical_candidate = jnp.sum(categorical_policy, axis=-1) > 0
    has_legal_action = jnp.any(legal_action_mask, axis=-1)
    root_is_categorical = (
        summary.v_categorical_outcome != int(NO_OUTCOME)
    )
    use_categorical_population = (
        root_is_categorical
        & has_legal_action
        & has_categorical_candidate
    )
    target_policy = jnp.where(
        use_categorical_population[:, None],
        categorical_policy,
        target_policy,
    )
    return _normalize_policy_on_support(
        target_policy,
        readout_support,
        temperature=search_cfg.policy_target_temperature,
    )


def _run_scalar_gumbel_search(env_state: pgx.State, prediction: EvaluatorOutput, expand_fn, rng_key: jax.Array, search_cfg: GumbelSearchConfig, q_supervision_config: QSupervisionConfig) -> PosteriorTargets:
    del q_supervision_config
    value = _required_output(prediction.value, "value")
    root = mctx.RootFnOutput(prior_logits=prediction.logits, value=value, embedding=env_state)
    policy_output = mctx.gumbel_muzero_policy(
        params=(),
        rng_key=rng_key,
        root=root,
        recurrent_fn=expand_fn,
        num_simulations=int(search_cfg.num_simulations),
        invalid_actions=~env_state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=float(search_cfg.gumbel_scale),
    )
    policy_target = cast(jax.Array, policy_output.action_weights)
    search_action = cast(jax.Array, policy_output.action)
    posterior_prediction = PosteriorPrediction(policy=policy_target, value=value)
    metadata = TargetMetadata(mask=_search_loss_mask(policy_target), search_action=search_action)
    return PosteriorTargets(prediction=posterior_prediction, metadata=metadata)


def _posterior_policy_sampling_budget(
    config: MonteCarloPosteriorUpdateConfig | NumericalPosteriorUpdateConfig,
) -> tuple[int, int]:
    if isinstance(config, NumericalPosteriorUpdateConfig):
        samples = config.fallback_policy_samples
        chunk_size = config.fallback_policy_sample_chunk_size
    else:
        samples = config.policy_samples
        chunk_size = config.policy_sample_chunk_size
    return (
        int(samples),
        (
            max(1, int(chunk_size))
            if chunk_size is not None
            else DEFAULT_POLICY_SAMPLE_CHUNK_SIZE
        ),
    )


def _make_posterior_update(
    config: MonteCarloPosteriorUpdateConfig | NumericalPosteriorUpdateConfig,
) -> Callable[
    [jax.Array, dirichlet_mctx.PosteriorUpdateContext],
    dirichlet_mctx.PosteriorUpdate,
]:
    policy_samples, policy_sample_chunk_size = (
        _posterior_policy_sampling_budget(config)
    )
    if isinstance(config, NumericalPosteriorUpdateConfig):
        return functools.partial(
            dirichlet_mctx.update_posterior_prefix_cdf,
            kappa=float(config.kappa),
            half_width=int(config.half_width),
            tail_scale=float(config.tail_scale),
            min_half_range=float(config.min_half_range),
            max_half_range=float(config.max_half_range),
            fallback_policy_samples=policy_samples,
            fallback_policy_sample_chunk_size=policy_sample_chunk_size,
        )
    return functools.partial(
        dirichlet_mctx.update_posterior,
        kappa=float(config.kappa),
        policy_samples=policy_samples,
        policy_sample_chunk_size=policy_sample_chunk_size,
    )


def _run_dirichlet_thompson_search(
    env_state: pgx.State,
    prediction: EvaluatorOutput,
    expand_fn,
    rng_key: jax.Array,
    search_cfg: DirichletThompsonSearchConfig,
    q_supervision_config: QSupervisionConfig,
) -> PosteriorTargets:
    """Run the MCTX-shaped Dirichlet Thompson backend."""

    alpha_v = _required_output(prediction.alpha_v, "alpha_v")
    alpha_q = _required_output(prediction.alpha_q, "alpha_q")
    root_reward = env_state.rewards[
        jnp.arange(env_state.rewards.shape[0]),
        env_state.current_player,
    ]
    root_terminal_outcome = jnp.where(
        env_state.terminated,
        terminal_outcome_from_reward(root_reward, alpha_v.shape[-1]),
        jnp.asarray(int(NO_OUTCOME), dtype=jnp.int8),
    )
    root = dirichlet_mctx.RootFnOutput(
        prior_logits=prediction.logits,
        value=alpha_v,
        action_values=alpha_q,
        embedding=env_state,
        terminal_outcome=root_terminal_outcome,
        to_play=env_state.current_player,
    )
    posterior_update_config = search_cfg.posterior_update.active()
    policy_samples, policy_sample_chunk_size = (
        _posterior_policy_sampling_budget(posterior_update_config)
    )
    policy_output = dirichlet_mctx.dirichlet_thompson_policy(
        params=(),
        rng_key=rng_key,
        root=root,
        recurrent_fn=expand_fn,
        num_simulations=int(search_cfg.num_simulations),
        invalid_actions=~env_state.legal_action_mask,
        posterior_update=_make_posterior_update(posterior_update_config),
        max_depth=search_cfg.max_depth,
        policy_samples=policy_samples,
        policy_sample_chunk_size=policy_sample_chunk_size,
    )
    native_policy = policy_output.action_weights
    search_action = policy_output.action
    tree = policy_output.search_tree
    summary = tree.summary()
    policy_target = _dirichlet_root_policy_readout(
        native_policy,
        summary=summary,
        legal_action_mask=env_state.legal_action_mask,
        search_cfg=search_cfg,
    )
    # These are the actual replacement-style B/cache posteriors exposed by
    # search. Unresolved R remains structural metadata; categorical payloads
    # hold distance. Neither changes Dirichlet concentration.
    beta_Q_target = summary.alpha
    beta_V_target = summary.value_alpha
    q_search_count = summary.visit_counts[..., None]
    q_is_categorical = summary.q_categorical_outcome != int(NO_OUTCOME)
    v_is_categorical = summary.v_categorical_outcome != int(NO_OUTCOME)
    legal = env_state.legal_action_mask
    q_target_kind = jnp.where(
        legal,
        jnp.where(
            q_is_categorical,
            jnp.asarray(int(TARGET_CATEGORICAL), dtype=jnp.int8),
            jnp.asarray(int(TARGET_DIRICHLET), dtype=jnp.int8),
        ),
        jnp.asarray(int(TARGET_PAD), dtype=jnp.int8),
    )
    v_target_kind = jnp.where(
        v_is_categorical,
        jnp.asarray(int(TARGET_CATEGORICAL), dtype=jnp.int8),
        jnp.asarray(int(TARGET_DIRICHLET), dtype=jnp.int8),
    )
    q_supervision = build_q_supervision(
        q_supervision_config.action_set,
        q_supervision_config.reduction,
        q_search_count,
        policy_target,
        solved_action=q_is_categorical,
        legal=legal,
    )
    positive_evidence_action = legal & (
        jnp.sum(q_search_count, axis=-1) > 0
    )
    positive_policy_action = legal & (policy_target > 0)
    posterior_prediction = PosteriorPrediction(
        policy=policy_target,
        alpha_v=beta_V_target,
        alpha_q=beta_Q_target,
    )
    metadata = TargetMetadata(
        mask=_search_loss_mask(policy_target),
        q_supervision=q_supervision,
        q_positive_evidence_action=positive_evidence_action,
        q_positive_policy_action=positive_policy_action,
        search_action=search_action,
        q_target_kind=q_target_kind,
        q_target_weight=legal.astype(beta_Q_target.dtype),
        q_target_outcome=summary.q_categorical_outcome,
        q_target_distance=summary.q_categorical_distance,
        v_target_kind=v_target_kind,
        v_target_weight=jnp.ones_like(
            summary.v_categorical_distance,
            dtype=beta_V_target.dtype,
        ),
        v_target_outcome=summary.v_categorical_outcome,
        v_target_distance=summary.v_categorical_distance,
    )
    return PosteriorTargets(prediction=posterior_prediction, metadata=metadata)


def _make_dirichlet_thompson_search(env, evaluator: Evaluator, search_cfg: DirichletThompsonSearchConfig, q_supervision_config: QSupervisionConfig) -> Search:
    expand_fn = make_dirichlet_expand_fn(env, evaluator)

    def search(root_state: pgx.State, rng_key: chex.PRNGKey) -> PosteriorTargets:
        prediction = evaluator(root_state.observation)
        return _run_dirichlet_thompson_search(
            root_state,
            prediction,
            expand_fn,
            rng_key,
            search_cfg,
            q_supervision_config,
        )

    return search


def _make_policy_search(env, evaluator: Evaluator, search_cfg: PolicySearchConfig, *args, **kwargs) -> Search:
    def search(root_state: pgx.State, rng_key: chex.PRNGKey) -> PosteriorTargets:
        prediction = evaluator(root_state.observation)
        policy = _masked_policy(prediction.logits, root_state.legal_action_mask, temperature=float(search_cfg.temperature))
        search_action = posterior_sample_action(
            rng_key,
            policy,
            root_state.legal_action_mask,
        )
        prediction = PosteriorPrediction(policy, prediction.value, prediction.alpha_v, prediction.alpha_q)
        metadata = TargetMetadata(
            mask=_search_loss_mask(policy),
            search_action=search_action,
        )
        return PosteriorTargets(prediction=prediction, metadata=metadata)

    return search


def _make_gumbel_search(env, evaluator: Evaluator, search_cfg: GumbelSearchConfig, q_supervision_config: QSupervisionConfig) -> Search:
    scalar_expand_fn = make_gumbel_expand_fn(env, evaluator)

    def search(root_state: pgx.State, rng_key: chex.PRNGKey) -> PosteriorTargets:
        prediction = evaluator(root_state.observation)
        if prediction.alpha_q is not None:
            raise ValueError(
                "Gumbel search supports scalar policy/value models only; "
                "use dirichlet_thompson for a Dirichlet-output model."
            )
        return _run_scalar_gumbel_search(
            root_state,
            prediction,
            scalar_expand_fn,
            rng_key,
            search_cfg,
            q_supervision_config,
        )

    return search

    
def make_search(
    env,
    evaluator: Evaluator,
    search_cfg: SearchConfig,
    q_supervision_config: QSupervisionConfig | None = None,
) -> Search:
    if q_supervision_config is None:
        q_supervision_config = QSupervisionConfig()
    
    active_search_cfg, _make_search_function = {
        SearchKind.policy: (search_cfg.policy, _make_policy_search),
        SearchKind.gumbel: (search_cfg.gumbel, _make_gumbel_search),
        SearchKind.dirichlet_thompson: (search_cfg.dirichlet_thompson, _make_dirichlet_thompson_search),
    }[search_cfg.kind]
    
    return _make_search_function(env, evaluator, active_search_cfg, q_supervision_config)  # ty:ignore[invalid-argument-type]


def _dirichlet_commitment_policy(
    posterior: PosteriorTargets,
    legal_action_mask: jax.Array,
    rng_key: jax.Array,
    search_config: DirichletThompsonSearchConfig,
    selected_update: PosteriorUpdateKind | None,
) -> jax.Array:
    prediction = posterior.prediction
    metadata = posterior.metadata
    alpha = _required_output(prediction.alpha_q, "alpha_q")
    if (
        metadata is None
        or metadata.search_action is None
        or metadata.q_target_outcome is None
        or metadata.v_target_outcome is None
    ):
        raise ValueError(
            "Dirichlet action commitment requires root posterior metadata."
        )

    update_config = search_config.posterior_update.select(selected_update)
    policy_samples, chunk_size = _posterior_policy_sampling_budget(
        update_config
    )
    positive_search_evidence = (
        jnp.zeros_like(legal_action_mask, dtype=jnp.bool_)
        if metadata.q_positive_evidence_action is None
        else metadata.q_positive_evidence_action
    )
    commitment_support = _root_policy_support(
        legal_action_mask,
        positive_search_evidence,
        metadata.q_target_outcome,
        search_config.root_policy_support,
    )
    invalid_actions = ~commitment_support
    native_policy = posterior_best_policy(
        rng_key,
        alpha,
        invalid_actions,
        policy_samples,
        chunk_size=chunk_size,
        categorical_outcome=metadata.q_target_outcome,
    )
    root_is_categorical = (
        metadata.v_target_outcome != int(NO_OUTCOME)
    )
    solved_policy = jax.nn.one_hot(
        metadata.search_action,
        alpha.shape[-2],
        dtype=alpha.dtype,
    )
    native_policy = jnp.where(
        root_is_categorical[:, None],
        solved_policy,
        native_policy,
    )
    native_policy = _normalize_policy_on_support(
        native_policy,
        commitment_support,
    )
    if isinstance(update_config, NumericalPosteriorUpdateConfig):
        return _numerical_policy_readout(
            native_policy,
            alpha=alpha,
            q_categorical_outcome=metadata.q_target_outcome,
            v_categorical_outcome=metadata.v_target_outcome,
            legal_action_mask=commitment_support,
            config=update_config,
        )
    return native_policy


def make_action_committer(
    config: ActionCommitmentConfig,
    dirichlet_search_config: DirichletThompsonSearchConfig | None = None,
):
    def action_committer(
        posterior: PosteriorTargets,
        legal_action_mask: jax.Array,
        rng_key: jax.Array,
    ) -> jax.Array:
        metadata = posterior.metadata
        search_action = None if metadata is None else metadata.search_action
        policy = posterior.prediction.policy
        action_key = rng_key
        if (
            dirichlet_search_config is not None
            and config.kind != ActionCommitmentType.search_action
        ):
            policy_key, action_key = jax.random.split(rng_key)
            policy = _dirichlet_commitment_policy(
                posterior,
                legal_action_mask,
                policy_key,
                dirichlet_search_config,
                config.posterior_update,
            )

        if config.kind == ActionCommitmentType.posterior_argmax:
            action = posterior_best_action(policy, legal_action_mask)
        elif config.kind == ActionCommitmentType.posterior_sample:
            action = posterior_sample_action(
                action_key,
                policy,
                legal_action_mask,
                temperature=config.posterior_sample_temperature,
            )
        elif config.kind == ActionCommitmentType.search_action:
            if search_action is None:
                raise ValueError("search_action commitment requires a backend action")
            return search_action
        else:
            raise ValueError(
                "unknown action_commitment_type: "
                f"{config.kind!r}"
            )
        return action

    return action_committer


def commit_action(
    action_commitment_type: str,
    rng_key: jax.Array,
    policy: jax.Array,
    legal_action_mask: jax.Array,
    search_action: jax.Array | None = None,
    posterior_sample_temperature: float = 1.0,
) -> jax.Array:
    """Compatibility boundary for committing an already-computed policy."""

    try:
        commitment_kind = ActionCommitmentType(action_commitment_type)
    except ValueError as error:
        raise ValueError(
            f"unknown action_commitment_type: {action_commitment_type!r}"
        ) from error
    posterior = PosteriorTargets(
        prediction=PosteriorPrediction(policy=policy),
        metadata=TargetMetadata(search_action=search_action),
    )
    return make_action_committer(
        ActionCommitmentConfig(
            kind=commitment_kind,
            posterior_sample_temperature=posterior_sample_temperature,
        ),
    )(
        posterior,
        legal_action_mask,
        rng_key,
    )


def make_search_player(
    env,
    model: Any,
    search_cfg: SearchConfig,
    action_commitment: ActionCommitmentConfig,
    q_supervision_config: QSupervisionConfig | None = None,
):
    search = make_search(
        env,
        make_evaluator(model),
        search_cfg,
        q_supervision_config,
    )
    action_committer = make_action_committer(
        action_commitment,
        (
            search_cfg.dirichlet_thompson
            if search_cfg.kind == SearchKind.dirichlet_thompson
            else None
        ),
    )

    def player(env_state: pgx.State, rng_key: jax.Array) -> PlayerOutput:
        search_key, action_key = jax.random.split(rng_key)
        posterior = search(env_state, search_key)
        action = action_committer(
            posterior,
            env_state.legal_action_mask,
            action_key,
        )
        return PlayerOutput(action=action, posterior=posterior)

    return player
