import functools
import math
from typing import Any, Callable, NamedTuple, cast

import chex
from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int
import mctx
import pgx

from . import dirichlet_mctx
from .dirichlet_mctx.action_selection import categorical_action_population
from .dirichlet_mctx.estimator_diagnostics import (
    binary_posterior_best_policy_prefix_quadrature,
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
    PosteriorPluralityResult,
    make_dirichlet_expand_fn,
    posterior_best_action,
    posterior_plurality_action,
    posterior_plurality_result,
    posterior_sample_action,
    q_loss_weight_from_mode,
    terminal_outcome_from_reward,
)
from .network import policy_value_from_output
from .search_diagnostics import (
    RootPolicyTargetDiagnostics,
    SearchDiagnostics,
    root_search_diagnostics,
)
from .types import (
    ActionCommitmentType,
    DirichletThompsonSearchConfig,
    GumbelSearchConfig,
    PolicySearchConfig,
    PosteriorPolicyEstimator,
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
    q_weight: Float[Array, "*batch action"] | None = None
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
    diagnostics: SearchDiagnostics | None = None


class SearchOutput(NamedTuple):
    posterior: PosteriorTargets
    # Ephemeral policy used only while committing an action.  Keeping this
    # outside PosteriorTargets prevents a second [batch,time,action] policy
    # tensor from being stored in replay.
    commitment_policy: Float[Array, "*batch action"] | None = None
    # Per-root native commitments must bypass any second stochastic readout.
    # This covers unsafe prefix fallbacks and solved native one-hot roots.
    commitment_resampling_bypass: Bool[Array, "*batch"] | None = None
Search = Callable[[pgx.State, chex.PRNGKey], SearchOutput]


class PlayerOutput(NamedTuple):
    action: Int[Array, "*batch"]
    posterior: PosteriorTargets | None = None


class _RootPolicyReadout(NamedTuple):
    policy: Float[Array, "batch action"]
    commitment_policy: Float[Array, "batch action"] | None
    commitment_resampling_bypass: Bool[Array, "batch"] | None
    diagnostics: RootPolicyTargetDiagnostics | None


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


def _dirichlet_root_policy_readouts(
    native_policy: jax.Array,
    *,
    summary,
    legal_action_mask: jax.Array,
    search_cfg: DirichletThompsonSearchConfig,
) -> _RootPolicyReadout:
    """Build optional deterministic target and commitment readouts.

    The completed tree and native winner-MC readout already exist before this
    function runs. If either readout requests prefix-CDF, quadrature is
    evaluated exactly once. Target-only mode leaves commitment native; action
    mode changes only unresolved posterior_argmax/plurality/sample
    commitment. Solved target roots use the exact uniform distance-optimal
    categorical population, while solved commitment remains the native
    one-hot draw.
    """

    target_estimator = search_cfg.root_policy_target_estimator
    action_estimator = search_cfg.root_action_estimator
    valid_estimators = {
        PosteriorPolicyEstimator.winner_mc,
        PosteriorPolicyEstimator.prefix_cdf,
    }
    if target_estimator not in valid_estimators:
        raise ValueError(
            "unknown root_policy_target_estimator: "
            f"{target_estimator!r}"
        )
    if action_estimator not in valid_estimators:
        raise ValueError(
            f"unknown root_action_estimator: {action_estimator!r}"
        )
    target_prefix = target_estimator == PosteriorPolicyEstimator.prefix_cdf
    action_prefix = action_estimator == PosteriorPolicyEstimator.prefix_cdf
    if not target_prefix and not action_prefix:
        return _RootPolicyReadout(
            policy=native_policy,
            commitment_policy=None,
            commitment_resampling_bypass=None,
            diagnostics=None,
        )

    invalid_actions = ~legal_action_mask
    estimate = binary_posterior_best_policy_prefix_quadrature(
        summary.alpha,
        invalid_actions,
        summary.q_categorical_outcome,
        half_width=int(search_cfg.prefix_cdf_half_width),
        adaptive_range=True,
        tail_scale=float(search_cfg.prefix_cdf_tail_scale),
        min_half_range=float(search_cfg.prefix_cdf_min_half_range),
        max_half_range=float(search_cfg.prefix_cdf_max_half_range),
        mass_conserving=True,
    )
    density_abs = jnp.max(
        jnp.abs(estimate.density_log_integral),
        axis=-1,
    )
    density_guard = (
        density_abs
        > DEFAULT_PREFIX_DENSITY_LOG_INTEGRAL_TOLERANCE
    )
    nonfinite = ~estimate.finite
    unsafe = estimate.tail_range_clipped | density_guard | nonfinite

    root_is_categorical = (
        summary.v_categorical_outcome != int(NO_OUTCOME)
    )
    has_legal_action = jnp.any(legal_action_mask, axis=-1)
    unresolved_root = ~root_is_categorical & has_legal_action
    prefix_eligible = unresolved_root & target_prefix
    prefix_accepted = prefix_eligible & ~unsafe
    prefix_fallback = prefix_eligible & unsafe
    action_prefix_eligible = unresolved_root & action_prefix
    action_prefix_accepted = action_prefix_eligible & ~unsafe
    action_prefix_fallback = action_prefix_eligible & unsafe
    solved_native_commitment = (
        action_prefix & root_is_categorical & has_legal_action
    )
    commitment_resampling_bypass = (
        action_prefix_fallback | solved_native_commitment
    )

    categorical_policy = categorical_action_population(
        summary.v_categorical_outcome,
        summary.q_categorical_outcome,
        summary.q_categorical_distance,
        invalid_actions,
        num_outcomes=summary.alpha.shape[-1],
        dtype=native_policy.dtype,
    )
    has_categorical_candidate = (
        jnp.sum(categorical_policy, axis=-1) > 0
    )
    categorical_population = (
        target_prefix
        & root_is_categorical
        & has_legal_action
        & has_categorical_candidate
    )

    target_policy = jnp.where(
        prefix_accepted[:, None],
        estimate.policy,
        native_policy,
    )
    target_policy = jnp.where(
        categorical_population[:, None],
        categorical_policy,
        target_policy,
    )
    target_enabled = has_legal_action & target_prefix
    action_policy = jnp.where(
        action_prefix_accepted[:, None],
        estimate.policy,
        native_policy,
    )
    action_enabled = has_legal_action & action_prefix
    native_l1 = jnp.sum(
        jnp.abs(target_policy - native_policy),
        axis=-1,
    )
    native_l2_sq = jnp.sum(
        jnp.square(target_policy - native_policy),
        axis=-1,
    )
    native_top1_agreement = (
        jnp.argmax(target_policy, axis=-1)
        == jnp.argmax(native_policy, axis=-1)
    )
    action_native_l1 = jnp.sum(
        jnp.abs(action_policy - native_policy),
        axis=-1,
    )
    action_native_l2_sq = jnp.sum(
        jnp.square(action_policy - native_policy),
        axis=-1,
    )
    action_native_top1_agreement = (
        jnp.argmax(action_policy, axis=-1)
        == jnp.argmax(native_policy, axis=-1)
    )
    diagnostics = RootPolicyTargetDiagnostics(
        target_enabled=target_enabled,
        categorical_population=categorical_population,
        prefix_eligible=prefix_eligible,
        prefix_accepted=prefix_accepted,
        prefix_fallback=prefix_fallback,
        prefix_tail_clipped=estimate.tail_range_clipped,
        prefix_density_guard=density_guard,
        prefix_nonfinite=nonfinite,
        prefix_density_abs=density_abs,
        native_l1=native_l1,
        native_l2_sq=native_l2_sq,
        native_top1_agreement=native_top1_agreement,
        action_enabled=action_enabled,
        action_prefix_eligible=action_prefix_eligible,
        action_prefix_accepted=action_prefix_accepted,
        action_prefix_fallback=action_prefix_fallback,
        action_native_l1=action_native_l1,
        action_native_l2_sq=action_native_l2_sq,
        action_native_top1_agreement=action_native_top1_agreement,
    )
    return _RootPolicyReadout(
        policy=target_policy,
        commitment_policy=action_policy,
        commitment_resampling_bypass=commitment_resampling_bypass,
        diagnostics=diagnostics,
    )


def _run_scalar_gumbel_search(env_state: pgx.State, prediction: EvaluatorOutput, expand_fn, rng_key: jax.Array, search_cfg: GumbelSearchConfig, q_loss_weight_mode: str) -> SearchOutput:
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
    return SearchOutput(PosteriorTargets(prediction=posterior_prediction, metadata=metadata))


def _run_dirichlet_thompson_search(
    env_state: pgx.State,
    prediction: EvaluatorOutput,
    expand_fn,
    rng_key: jax.Array,
    search_cfg: DirichletThompsonSearchConfig,
    q_loss_weight_mode: str,
) -> SearchOutput:
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
    # The public root policy and each repaired node estimate the same
    # posterior-best population.  Their Monte Carlo budgets may differ: one
    # internal draw is still an unbiased app.js pi_search estimate, while the
    # larger public population gives a lower-variance training/readout target.
    # Binding the internal budget into the callback keeps the external search
    # API lightweight and the complete backward rule replaceable.
    posterior_policy_samples = (
        int(search_cfg.policy_samples)
        if search_cfg.posterior_policy_samples is None
        else int(search_cfg.posterior_policy_samples)
    )
    posterior_chunk_size = (
        max(1, int(search_cfg.policy_sample_chunk_size))
        if search_cfg.policy_sample_chunk_size is not None
        else DEFAULT_POLICY_SAMPLE_CHUNK_SIZE
    )
    match search_cfg.posterior_policy_estimator:
        case PosteriorPolicyEstimator.winner_mc:
            posterior_update = functools.partial(
                dirichlet_mctx.update_posterior,
                kappa=float(search_cfg.kappa),
                policy_samples=max(1, posterior_policy_samples),
                policy_sample_chunk_size=posterior_chunk_size,
            )
        case PosteriorPolicyEstimator.prefix_cdf:
            posterior_update = functools.partial(
                dirichlet_mctx.update_posterior_prefix_cdf,
                kappa=float(search_cfg.kappa),
                half_width=int(search_cfg.prefix_cdf_half_width),
                tail_scale=float(search_cfg.prefix_cdf_tail_scale),
                min_half_range=float(
                    search_cfg.prefix_cdf_min_half_range
                ),
                max_half_range=float(
                    search_cfg.prefix_cdf_max_half_range
                ),
                fallback_policy_samples=max(
                    1,
                    posterior_policy_samples,
                ),
                fallback_policy_sample_chunk_size=posterior_chunk_size,
            )
    policy_output = dirichlet_mctx.dirichlet_thompson_policy(
        params=(),
        rng_key=rng_key,
        root=root,
        recurrent_fn=expand_fn,
        num_simulations=int(search_cfg.num_simulations),
        invalid_actions=~env_state.legal_action_mask,
        posterior_update=posterior_update,
        max_depth=search_cfg.max_depth,
        policy_samples=int(search_cfg.policy_samples),
        policy_sample_chunk_size=search_cfg.policy_sample_chunk_size,
    )
    native_policy = policy_output.action_weights
    search_action = policy_output.action
    tree = policy_output.search_tree
    summary = tree.summary()
    root_policy_readout = _dirichlet_root_policy_readouts(
        native_policy,
        summary=summary,
        legal_action_mask=env_state.legal_action_mask,
        search_cfg=search_cfg,
    )
    policy_target = root_policy_readout.policy
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
    q_loss_weight = q_loss_weight_from_mode(
        q_loss_weight_mode,
        q_search_count,
        native_policy,
    )
    # Every exact legal action target is useful supervision, even when a
    # one-hot solved policy or a sampled posterior gives that action no mass.
    q_loss_weight = jnp.where(
        legal & q_is_categorical,
        jnp.maximum(q_loss_weight, jnp.ones_like(q_loss_weight)),
        q_loss_weight,
    )
    posterior_prediction = PosteriorPrediction(
        policy=policy_target,
        alpha_v=beta_V_target,
        alpha_q=beta_Q_target,
    )
    metadata = TargetMetadata(
        mask=_search_loss_mask(policy_target),
        q_weight=q_loss_weight,
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
    diagnostics = root_search_diagnostics(
        prior_logits=prediction.logits,
        prior_alpha_v=alpha_v,
        prior_alpha_q=alpha_q,
        target_policy=policy_target,
        target_alpha_v=beta_V_target,
        target_alpha_q=beta_Q_target,
        q_target_kind=q_target_kind,
        q_target_outcome=summary.q_categorical_outcome,
        v_target_kind=v_target_kind,
        v_target_outcome=summary.v_categorical_outcome,
        legal_action_mask=legal,
        tree=tree,
        summary=summary,
        root_policy_target_diagnostics=root_policy_readout.diagnostics,
    )
    return SearchOutput(
        PosteriorTargets(
            prediction=posterior_prediction,
            metadata=metadata,
            diagnostics=diagnostics,
        ),
        commitment_policy=root_policy_readout.commitment_policy,
        commitment_resampling_bypass=(
            root_policy_readout.commitment_resampling_bypass
        ),
    )


def _make_dirichlet_thompson_search(env, evaluator: Evaluator, search_cfg: DirichletThompsonSearchConfig, q_loss_weight_mode: str) -> Search:
    expand_fn = make_dirichlet_expand_fn(env, evaluator)

    def search(root_state: pgx.State, rng_key: chex.PRNGKey) -> SearchOutput:
        prediction = evaluator(root_state.observation)
        return _run_dirichlet_thompson_search(
            root_state,
            prediction,
            expand_fn,
            rng_key,
            search_cfg,
            q_loss_weight_mode,
        )

    return search


def _make_policy_search(env, evaluator: Evaluator, search_cfg: PolicySearchConfig, *args, **kwargs) -> Search:
    def search(root_state: pgx.State, rng_key: chex.PRNGKey) -> SearchOutput:
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
        return SearchOutput(PosteriorTargets(prediction=prediction, metadata=metadata))

    return search


def _make_gumbel_search(env, evaluator: Evaluator, search_cfg: GumbelSearchConfig, q_loss_weight_mode: str) -> Search:
    scalar_expand_fn = make_gumbel_expand_fn(env, evaluator)

    def search(root_state: pgx.State, rng_key: chex.PRNGKey) -> SearchOutput:
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
            q_loss_weight_mode,
        )

    return search

    
def make_search(env, evaluator: Evaluator, search_cfg: SearchConfig, q_loss_weight_mode: str = "policy") -> Search:
    
    active_search_cfg, _make_search_function = {
        SearchKind.policy: (search_cfg.policy, _make_policy_search),
        SearchKind.gumbel: (search_cfg.gumbel, _make_gumbel_search),
        SearchKind.dirichlet_thompson: (search_cfg.dirichlet_thompson, _make_dirichlet_thompson_search),
    }[search_cfg.kind]
    
    return _make_search_function(env, evaluator, active_search_cfg, q_loss_weight_mode)  # ty:ignore[invalid-argument-type]


def make_action_committer(
    action_commitment_type: str,
    posterior_plurality_samples: int = 32,
    posterior_sample_temperature: float = 1.0,
):
    if posterior_plurality_samples < 1:
        raise ValueError(
            "posterior_plurality_samples must be >= 1; "
            f"got {posterior_plurality_samples}."
        )
    if (
        not math.isfinite(posterior_sample_temperature)
        or posterior_sample_temperature <= 0.0
    ):
        raise ValueError(
            "posterior_sample_temperature must be finite and > 0; "
            f"got {posterior_sample_temperature}."
        )

    def action_committer(
        posterior: PosteriorTargets,
        legal_action_mask: jax.Array,
        rng_key: jax.Array,
        commitment_policy: jax.Array | None = None,
        commitment_resampling_bypass: jax.Array | None = None,
    ) -> jax.Array:
        metadata = posterior.metadata
        search_action = None if metadata is None else metadata.search_action
        policy = (
            posterior.prediction.policy
            if commitment_policy is None
            else commitment_policy
        )

        def preserve_native_commitment(action: jax.Array) -> jax.Array:
            if commitment_resampling_bypass is None:
                return action
            native_action = posterior_best_action(policy, legal_action_mask)
            return jnp.where(
                commitment_resampling_bypass,
                native_action,
                action,
            )

        if action_commitment_type == "posterior_argmax":
            return posterior_best_action(policy, legal_action_mask)
        elif action_commitment_type in {
            "posterior_plurality",
            "posterior_plurality_uniform_ties",
        }:
            return preserve_native_commitment(
                posterior_plurality_action(
                    rng_key,
                    policy,
                    legal_action_mask,
                    num_samples=posterior_plurality_samples,
                    tie_break=(
                        "uniform"
                        if action_commitment_type
                        == "posterior_plurality_uniform_ties"
                        else "lowest"
                    ),
                )
            )
        elif action_commitment_type == "posterior_sample":
            return preserve_native_commitment(
                posterior_sample_action(
                    rng_key,
                    policy,
                    legal_action_mask,
                    temperature=posterior_sample_temperature,
                )
            )
        elif action_commitment_type == "search_action":
            if search_action is None:
                raise ValueError("search_action commitment requires a backend action")
            return search_action
        raise ValueError(f"unknown action_commitment_type: {action_commitment_type!r}")

    return action_committer


def _with_root_policy_top2_margin_diagnostics(
    search_output: SearchOutput,
    legal_action_mask: jax.Array,
    *,
    action_commitment_type: str,
    margin_reference_scale: float,
) -> SearchOutput:
    """Attach descriptive top-two metrics for the selected root policy.

    ``margin_reference_scale`` is only a reporting comparator.  In
    particular, ``1 / M`` describes the resolution of an M-sample empirical
    policy; it is not a decision threshold for stochastic commitment.
    """

    diagnostics = search_output.posterior.diagnostics
    policy_driven = action_commitment_type in {
        "posterior_argmax",
        "posterior_plurality",
        "posterior_plurality_uniform_ties",
        "posterior_sample",
    }
    if (
        diagnostics is None
        or not policy_driven
        or legal_action_mask.shape[-1] < 2
    ):
        return search_output

    policy = (
        search_output.posterior.prediction.policy
        if search_output.commitment_policy is None
        else search_output.commitment_policy
    )
    legal = jnp.asarray(legal_action_mask)
    finite_policy = jnp.all(~legal | jnp.isfinite(policy), axis=-1)
    probability = jnp.where(
        legal & jnp.isfinite(policy) & (policy > 0),
        policy,
        jnp.zeros_like(policy),
    )
    mass = jnp.sum(probability, axis=-1, keepdims=True)
    probability = probability / jnp.where(
        mass > 0,
        mass,
        jnp.ones_like(mass),
    )
    masked_probability = jnp.where(legal, probability, -jnp.inf)
    top_two, _ = jax.lax.top_k(masked_probability, 2)
    margin = top_two[..., 0] - top_two[..., 1]
    legal_count = jnp.sum(legal, axis=-1)
    solved = diagnostics.search_solved_root_count > 0
    valid = (
        ~solved
        & (legal_count >= 2)
        & finite_policy
        & (mass[..., 0] > 0)
        & jnp.isfinite(margin)
    )
    dtype = diagnostics.search_root_count.dtype
    valid_count = valid.astype(dtype)
    safe_margin = jnp.where(valid, margin, jnp.zeros_like(margin))
    reference_scale = jnp.asarray(
        margin_reference_scale,
        dtype=margin.dtype,
    )
    updated = diagnostics._replace(
        search_root_policy_top2_margin_sum=safe_margin,
        search_root_policy_top2_margin_count=valid_count,
        search_root_policy_top2_margin_tie_count=(
            valid & (margin == 0)
        ).astype(dtype),
        search_root_policy_top2_margin_below_reference_count=(
            valid & (margin <= reference_scale)
        ).astype(dtype),
        search_root_policy_top2_margin_reference_scale_sum=(
            valid_count * reference_scale
        ),
    )
    return search_output._replace(
        posterior=search_output.posterior._replace(diagnostics=updated)
    )


def _with_root_plurality_commitment_diagnostics(
    search_output: SearchOutput,
    result: PosteriorPluralityResult,
) -> SearchOutput:
    """Attach paired tie-rule diagnostics from one realized vote histogram."""

    diagnostics = search_output.posterior.diagnostics
    if diagnostics is None:
        return search_output

    eligible = result.resampling_eligible
    bypass = search_output.commitment_resampling_bypass
    if bypass is not None:
        eligible = eligible & ~bypass
    multiplicity = result.max_count_tie_multiplicity
    tied = eligible & (multiplicity > 1)
    disagreement = (
        eligible
        & (result.lowest_index_action != result.uniform_tie_action)
    )
    dtype = diagnostics.search_root_count.dtype
    eligible_count = eligible.astype(dtype)
    tied_count = tied.astype(dtype)
    multiplicity_float = multiplicity.astype(dtype)
    expected_disagreement = jnp.where(
        tied,
        1.0 - 1.0 / multiplicity_float,
        0.0,
    ).astype(dtype)
    updated = diagnostics._replace(
        search_root_plurality_commitment_count=eligible_count,
        search_root_plurality_max_count_tie_count=tied_count,
        search_root_plurality_tie_multiplicity_sum=jnp.where(
            tied,
            multiplicity_float,
            jnp.zeros_like(multiplicity_float),
        ),
        search_root_plurality_lowest_uniform_disagreement_count=(
            disagreement.astype(dtype)
        ),
        search_root_plurality_expected_disagreement_sum=expected_disagreement,
    )
    return search_output._replace(
        posterior=search_output.posterior._replace(diagnostics=updated)
    )


def commit_action(
    action_commitment_type: str,
    rng_key: jax.Array,
    policy: jax.Array,
    legal_action_mask: jax.Array,
    search_action: jax.Array | None = None,
    posterior_plurality_samples: int = 32,
    posterior_sample_temperature: float = 1.0,
    commitment_resampling_bypass: jax.Array | None = None,
) -> jax.Array:
    """Compatibility boundary for committing an already-computed policy."""

    posterior = PosteriorTargets(
        prediction=PosteriorPrediction(policy=policy),
        metadata=TargetMetadata(search_action=search_action),
    )
    return make_action_committer(
        action_commitment_type,
        posterior_plurality_samples,
        posterior_sample_temperature,
    )(
        posterior,
        legal_action_mask,
        rng_key,
        commitment_resampling_bypass=commitment_resampling_bypass,
    )


def make_search_player(env, model: Any, search_cfg: SearchConfig, action_commitment_type: ActionCommitmentType, q_loss_weight_mode: str = "policy"):
    search = make_search(env, make_evaluator(model), search_cfg, q_loss_weight_mode)
    action_commitment_name = str(action_commitment_type)
    action_committer = make_action_committer(
        action_commitment_name,
        int(search_cfg.posterior_plurality_samples),
        float(search_cfg.posterior_sample_temperature),
    )
    active_search = search_cfg.active()
    native_policy_samples = max(
        1,
        int(getattr(active_search, "policy_samples", 32)),
    )
    reference_sample_count = (
        int(search_cfg.posterior_plurality_samples)
        if action_commitment_name
        in {"posterior_plurality", "posterior_plurality_uniform_ties"}
        else native_policy_samples
    )
    # One empirical vote is a useful scale for describing the policy margin,
    # not a decision threshold for argmax, plurality, or posterior sampling.
    top2_margin_reference_scale = 1.0 / max(1, reference_sample_count)

    def player(env_state: pgx.State, rng_key: jax.Array) -> PlayerOutput:
        search_key, action_key = jax.random.split(rng_key)
        search_output = search(env_state, search_key)
        search_output = _with_root_policy_top2_margin_diagnostics(
            search_output,
            env_state.legal_action_mask,
            action_commitment_type=action_commitment_name,
            margin_reference_scale=top2_margin_reference_scale,
        )
        if action_commitment_name in {
            "posterior_plurality",
            "posterior_plurality_uniform_ties",
        }:
            policy = (
                search_output.posterior.prediction.policy
                if search_output.commitment_policy is None
                else search_output.commitment_policy
            )
            plurality_result = posterior_plurality_result(
                action_key,
                policy,
                env_state.legal_action_mask,
                num_samples=int(search_cfg.posterior_plurality_samples),
                tie_break=(
                    "uniform"
                    if action_commitment_name
                    == "posterior_plurality_uniform_ties"
                    else "lowest"
                ),
            )
            action = plurality_result.action
            bypass = search_output.commitment_resampling_bypass
            if bypass is not None:
                action = jnp.where(
                    bypass,
                    posterior_best_action(
                        policy,
                        env_state.legal_action_mask,
                    ),
                    action,
                )
            search_output = _with_root_plurality_commitment_diagnostics(
                search_output,
                plurality_result,
            )
        else:
            action = action_committer(
                search_output.posterior,
                env_state.legal_action_mask,
                action_key,
                search_output.commitment_policy,
                search_output.commitment_resampling_bypass,
            )
        return PlayerOutput(action=action, posterior=search_output.posterior)

    return player
