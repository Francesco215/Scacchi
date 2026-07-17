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
from .dirichlet_mctx.categorical import (
    NO_OUTCOME,
    TARGET_CATEGORICAL,
    TARGET_DIRICHLET,
    TARGET_PAD,
)
from .dirichlet_q_search import (
    make_dirichlet_expand_fn,
    outcome_mean,
    outcome_utility,
    posterior_best_action,
    posterior_sample_action,
    q_loss_weight_from_mode,
)
from .network import policy_value_from_output
from .types import (
    ActionCommitmentType,
    DirichletThompsonSearchConfig,
    GumbelSearchConfig,
    PolicySearchConfig,
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


class SearchOutput(NamedTuple):
    posterior: PosteriorTargets
Search = Callable[[pgx.State, chex.PRNGKey], SearchOutput]


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
    root = dirichlet_mctx.RootFnOutput(
        prior_logits=prediction.logits,
        value=alpha_v,
        action_values=alpha_q,
        embedding=env_state,
        terminal=env_state.terminated,
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
    posterior_update = functools.partial(
        dirichlet_mctx.update_posterior,
        kappa=float(search_cfg.kappa),
        policy_samples=max(1, posterior_policy_samples),
        policy_sample_chunk_size=(
            max(1, int(search_cfg.policy_sample_chunk_size))
            if search_cfg.policy_sample_chunk_size is not None
            else dirichlet_mctx.DEFAULT_POLICY_SAMPLE_CHUNK_SIZE
        ),
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
        categorical_draw_rule=str(search_cfg.categorical_draw_rule),
    )
    policy_target = policy_output.action_weights
    search_action = policy_output.action
    tree = policy_output.search_tree
    summary = tree.summary()
    # These are the actual replacement-style B/cache posteriors exposed by
    # the north-star implementation.  R remains structural metadata; changing
    # concentration here would train a different posterior than search uses.
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
        policy_target,
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
    return SearchOutput(
        PosteriorTargets(prediction=posterior_prediction, metadata=metadata)
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


def make_action_committer(action_commitment_type: str):
    def action_committer(posterior: PosteriorTargets, legal_action_mask: jax.Array, rng_key: jax.Array) -> jax.Array:
        metadata = posterior.metadata
        search_action = None if metadata is None else metadata.search_action
        policy = posterior.prediction.policy
        
        if action_commitment_type == "posterior_argmax":
            return posterior_best_action(policy, legal_action_mask)
        elif action_commitment_type == "posterior_sample":
            return posterior_sample_action(rng_key, policy, legal_action_mask)
        elif action_commitment_type == "search_action":
            if search_action is None:
                raise ValueError("search_action commitment requires a backend action")
            return search_action
        raise ValueError(f"unknown action_commitment_type: {action_commitment_type!r}")

    return action_committer


def commit_action(
    action_commitment_type: str,
    rng_key: jax.Array,
    policy: jax.Array,
    legal_action_mask: jax.Array,
    search_action: jax.Array | None = None,
) -> jax.Array:
    """Compatibility boundary for committing an already-computed policy."""

    posterior = PosteriorTargets(
        prediction=PosteriorPrediction(policy=policy),
        metadata=TargetMetadata(search_action=search_action),
    )
    return make_action_committer(action_commitment_type)(
        posterior,
        legal_action_mask,
        rng_key,
    )


def make_search_player(env, model: Any, search_cfg: SearchConfig, action_commitment_type: ActionCommitmentType, q_loss_weight_mode: str = "policy"):
    search = make_search(env, make_evaluator(model), search_cfg, q_loss_weight_mode)
    action_committer = make_action_committer(str(action_commitment_type))

    def player(env_state: pgx.State, rng_key: jax.Array) -> PlayerOutput:
        search_key, action_key = jax.random.split(rng_key)
        search_output = search(env_state, search_key)
        action = action_committer(search_output.posterior, env_state.legal_action_mask, action_key)
        return PlayerOutput(action=action, posterior=search_output.posterior)

    return player
