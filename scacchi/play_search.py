from functools import partial
from typing import Any, Callable, NamedTuple, cast

from flax import nnx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int
import mctx
import pgx

from .dirichlet_q_search import (
    NO_PARENT,
    NodeEmbedding,
    dirichlet_q_policy,
    flip_outcome,
    outcome_mean,
    outcome_utility,
    posterior_best_action,
    posterior_best_policy_target,
    posterior_sample_action,
    posterior_targets,
    q_evidence_sum_from_tree,
    root_action_value_priors_from_tree,
    terminal_outcome_from_reward,
)
from .network import policy_value_from_output
from .types import (
    ActionCommitmentType,
    DirichletThompsonSearchConfig,
    GumbelSearchConfig,
    PolicySearchConfig,
    SearchConfig,
    SearchConstantsConfig,
    SearchKind,
)


_POSTERIOR_POLICY_TARGET_SAMPLES = 32


class EvaluatorOutput(NamedTuple):
    logits: Float[Array, "*batch action"]
    value: Float[Array, "*batch"] | None = None
    alpha_v: Float[Array, "*batch outcome"] | None = None
    alpha_q: Float[Array, "*batch action outcome"] | None = None


class PosteriorPrediction(NamedTuple):
    policy: Float[Array, "*batch action"]
    value: Float[Array, "*batch"] | None = None
    alpha_v: Float[Array, "*batch outcome"] | None = None
    alpha_q: Float[Array, "*batch action outcome"] | None = None


class TargetMetadata(NamedTuple):
    mask: Bool[Array, "*batch"] | None = None
    q_weight: Float[Array, "*batch action"] | None = None
    search_action: Int[Array, "*batch"] | None = None


class PosteriorTargets(NamedTuple):
    prediction: PosteriorPrediction
    metadata: TargetMetadata | None = None


class SearchOutput(NamedTuple):
    posterior: PosteriorTargets


class PlayerOutput(NamedTuple):
    action: Int[Array, "*batch"]
    posterior: PosteriorTargets | None = None


class _DirichletRootContext(NamedTuple):
    alpha_v: jax.Array
    root: mctx.RootFnOutput
    action_value_prior: jax.Array


class _DirichletSearchBackendOutput(NamedTuple):
    action_weights: jax.Array
    action: jax.Array
    q_evidence_sum: jax.Array
    action_alpha_post: jax.Array
    action_value_target_prior: jax.Array


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


def make_evaluator(model: Any) -> Callable[[jax.Array], EvaluatorOutput]:
    def evaluator(obs: jax.Array) -> EvaluatorOutput:
        if isinstance(model, nnx.Module):
            return evaluator_output_from_model_output(model(obs, train=False))
        return evaluator_output_from_model_output(model(obs))

    return evaluator


def _required_output(value: jax.Array | None, name: str) -> jax.Array:
    if value is None:
        raise ValueError(f"evaluator output is missing {name}")
    return value


def _masked_logits(logits: jax.Array, legal_action_mask: jax.Array) -> jax.Array:
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    return jnp.where(legal_action_mask, logits, jnp.finfo(logits.dtype).min)


def _masked_policy(
    logits: jax.Array,
    legal_action_mask: jax.Array,
    *,
    temperature: float,
) -> jax.Array:
    masked_logits = _masked_logits(logits, legal_action_mask)
    policy = jax.nn.softmax(masked_logits / float(temperature), axis=-1)
    has_legal_action = jnp.any(legal_action_mask, axis=-1, keepdims=True)
    return jnp.where(has_legal_action, policy, jnp.zeros_like(policy))


def make_expand_fn(env, evaluator: Callable[[jax.Array], EvaluatorOutput]):
    def expand_fn(_, rng_key: jax.Array, action: jax.Array, env_state: pgx.State):
        del rng_key

        current_player = env_state.current_player
        env_state = jax.vmap(env.step)(env_state, action)
        prediction = evaluator(env_state.observation)
        logits = _masked_logits(prediction.logits, env_state.legal_action_mask)
        value = _required_output(prediction.value, "value")

        reward = env_state.rewards[jnp.arange(env_state.rewards.shape[0]), current_player]
        value = jnp.where(env_state.terminated, 0.0, value)
        discount = -jnp.ones_like(value)
        discount = jnp.where(env_state.terminated, 0.0, discount)

        fn_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return fn_output, env_state

    return expand_fn


def make_dirichlet_expand_fn_from_constants(
    env,
    evaluator: Callable[[jax.Array], EvaluatorOutput],
    constants: SearchConstantsConfig,
):
    kappa_terminal = float(constants.kappa_terminal)
    kappa_leaf = float(constants.kappa_leaf)

    def expand_fn(_, rng_key: jax.Array, action: jax.Array, embedding: NodeEmbedding):
        del rng_key

        current_player = embedding.state.current_player
        env_state = jax.vmap(env.step)(embedding.state, action)
        prediction = evaluator(env_state.observation)
        logits = _masked_logits(prediction.logits, env_state.legal_action_mask)
        alpha_v = _required_output(prediction.alpha_v, "alpha_v")
        alpha_q = _required_output(prediction.alpha_q, "alpha_q")

        reward = env_state.rewards[jnp.arange(env_state.rewards.shape[0]), current_player]
        nonterminal_outcome = outcome_mean(alpha_v)
        terminal_parent_outcome = terminal_outcome_from_reward(reward, alpha_v.shape[-1])
        terminal_child_outcome = flip_outcome(terminal_parent_outcome)
        outcome_dist = jnp.where(
            env_state.terminated[..., None],
            terminal_child_outcome,
            nonterminal_outcome,
        )
        evidence_weight = jnp.where(
            env_state.terminated,
            jnp.asarray(kappa_terminal, dtype=outcome_dist.dtype),
            jnp.asarray(kappa_leaf, dtype=outcome_dist.dtype),
        )
        root_action = jnp.where(embedding.root_action == NO_PARENT, action, embedding.root_action)
        depth_parity = 1 - embedding.depth_parity

        value = outcome_utility(outcome_dist)
        value = jnp.where(env_state.terminated, 0.0, value)
        discount = -jnp.ones_like(value)
        discount = jnp.where(env_state.terminated, 0.0, discount)

        next_embedding = NodeEmbedding(
            state=env_state,
            outcome_dist=outcome_dist,
            alpha_V_prior=alpha_v,
            evidence_weight=evidence_weight,
            root_action=root_action,
            depth_parity=depth_parity,
            alpha_Q_prior=alpha_q,
        )
        fn_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return fn_output, next_embedding

    return expand_fn


def _search_loss_mask(action_weights: jax.Array) -> jax.Array:
    return jnp.sum(action_weights, axis=-1) > 0


def _gumbel_qtransform(search_cfg: GumbelSearchConfig):
    return partial(
        mctx.qtransform_completed_by_mix_value,
        value_scale=float(search_cfg.completed_q_value_scale),
        rescale_values=bool(search_cfg.completed_q_rescale_values),
    )


def commit_action(
    action_commitment_type: str,
    rng_key: jax.Array,
    policy: jax.Array,
    legal_action_mask: jax.Array,
    search_action: jax.Array | None = None,
) -> jax.Array:
    if action_commitment_type == "posterior_argmax":
        selected = posterior_best_action(policy, legal_action_mask)
    elif action_commitment_type == "posterior_sample":
        selected = posterior_sample_action(rng_key, policy, legal_action_mask)
    elif action_commitment_type == "search_action":
        if search_action is None:
            raise ValueError("search_action commitment requires a backend action")
        selected = search_action
    else:
        raise ValueError(
            f"unknown action_commitment_type: {action_commitment_type!r}"
        )
    return legalize_action(selected, legal_action_mask)


def legalize_action(
    action: jax.Array,
    legal_action_mask: jax.Array,
) -> jax.Array:
    #TODO: remove this useless function. the rest of the code already avoids taking invalid actions.
    """Return `action` when legal, otherwise the first legal action in each row."""

    num_actions = legal_action_mask.shape[-1]
    first_legal = jnp.argmax(legal_action_mask, axis=-1).astype(jnp.int32)
    has_legal_action = jnp.any(legal_action_mask, axis=-1)
    action = jnp.asarray(action, dtype=jnp.int32)
    in_bounds = (0 <= action) & (action < num_actions)
    safe_action = jnp.clip(action, 0, num_actions - 1)
    selected_is_legal = jnp.take_along_axis(
        legal_action_mask,
        safe_action[..., None],
        axis=-1,
    )[..., 0]
    selected = jnp.where(in_bounds & selected_is_legal, action, first_legal)
    return jnp.where(has_legal_action, selected, jnp.zeros_like(selected))


def make_action_committer(action_commitment_type: str):
    def action_committer(
        posterior: PosteriorTargets,
        legal_action_mask: jax.Array,
        rng_key: jax.Array,
    ) -> jax.Array:
        metadata = posterior.metadata
        search_action = None if metadata is None else metadata.search_action
        return commit_action(
            action_commitment_type,
            rng_key,
            posterior.prediction.policy,
            legal_action_mask,
            search_action,
        )

    return action_committer


def make_player(search, action_committer):
    def player(env_state: pgx.State, rng_key: jax.Array) -> PlayerOutput:
        search_key = rng_key
        _, action_key = jax.random.split(rng_key)
        search_output = search(root_state=env_state, rng_key=search_key)
        action = action_committer(search_output.posterior, env_state.legal_action_mask, action_key)
        return PlayerOutput(
            action=action,
            posterior=search_output.posterior,
        )

    return player


def make_search_player(
    env,
    model: Any,
    search_cfg: SearchConfig,
    action_commitment_type: ActionCommitmentType,
    *,
    q_loss_weight_mode: str = "policy",
):
    return make_player(
        make_search(env, make_evaluator(model), search_cfg, q_loss_weight_mode=q_loss_weight_mode),
        make_action_committer(str(action_commitment_type)),
    )


def _make_dirichlet_root(
    env_state: pgx.State,
    logits: jax.Array,
    alpha_v: jax.Array,
    alpha_q: jax.Array,
) -> mctx.RootFnOutput:
    root_outcome = outcome_mean(alpha_v)
    value = outcome_utility(root_outcome)
    root_embedding = NodeEmbedding(
        state=env_state,
        outcome_dist=root_outcome,
        alpha_V_prior=alpha_v,
        evidence_weight=jnp.zeros_like(value),
        root_action=jnp.full_like(env_state.current_player, NO_PARENT),
        depth_parity=jnp.zeros_like(env_state.current_player),
        alpha_Q_prior=alpha_q,
    )
    return mctx.RootFnOutput(
        prior_logits=logits,
        value=value,
        embedding=root_embedding,
    )


def _q_loss_weight_from_mode(
    mode: str,
    q_evidence_sum: jax.Array,
    posterior_policy_target: jax.Array,
) -> jax.Array:
    if mode == "evidence_mass":
        return jnp.sum(q_evidence_sum, axis=-1) + jnp.zeros_like(posterior_policy_target)
    if mode == "policy":
        return posterior_policy_target
    raise ValueError(f"unknown q_loss_weight_mode: {mode!r}")


def _make_dirichlet_root_context(
    env_state: pgx.State,
    prediction: EvaluatorOutput,
) -> _DirichletRootContext:
    alpha_v = _required_output(prediction.alpha_v, "alpha_v")
    alpha_q = _required_output(prediction.alpha_q, "alpha_q")
    root = _make_dirichlet_root(env_state, prediction.logits, alpha_v, alpha_q)
    return _DirichletRootContext(
        alpha_v=alpha_v,
        root=root,
        action_value_prior=alpha_q,
    )


def _run_dirichlet_thompson_backend(
    *,
    env_state: pgx.State,
    root: mctx.RootFnOutput,
    expand_fn,
    search_key: jax.Array,
    search_cfg: DirichletThompsonSearchConfig,
    action_value_prior: jax.Array,
) -> _DirichletSearchBackendOutput:
    policy_output = dirichlet_q_policy(
        params=(),
        rng_key=search_key,
        root=root,
        expand_fn=expand_fn,
        action_value_prior=action_value_prior,
        num_simulations=int(search_cfg.num_simulations),
        max_depth=int(search_cfg.max_depth),
        invalid_actions=~env_state.legal_action_mask,
        num_search_blocks=int(search_cfg.num_blocks),
    )
    q_evidence_sum = policy_output.q_evidence_sum
    action_alpha_post = policy_output.alpha_search
    return _DirichletSearchBackendOutput(
        action_weights=cast(jax.Array, policy_output.action_weights),
        action=cast(jax.Array, policy_output.action),
        q_evidence_sum=q_evidence_sum,
        action_alpha_post=action_alpha_post,
        action_value_target_prior=action_alpha_post - q_evidence_sum,
    )


def _run_dirichlet_gumbel_backend(
    *,
    env_state: pgx.State,
    root: mctx.RootFnOutput,
    expand_fn,
    search_key: jax.Array,
    search_cfg: GumbelSearchConfig,
    action_value_prior: jax.Array,
) -> _DirichletSearchBackendOutput:
    policy_output = mctx.gumbel_muzero_policy(
        params=(),
        rng_key=search_key,
        root=root,
        recurrent_fn=expand_fn,
        num_simulations=int(search_cfg.num_simulations),
        invalid_actions=~env_state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=float(search_cfg.gumbel_scale),
    )
    q_evidence_sum = q_evidence_sum_from_tree(policy_output.search_tree)
    action_value_target_prior = root_action_value_priors_from_tree(
        policy_output.search_tree,
        action_value_prior,
    )
    return _DirichletSearchBackendOutput(
        action_weights=cast(jax.Array, policy_output.action_weights),
        action=cast(jax.Array, policy_output.action),
        q_evidence_sum=q_evidence_sum,
        action_alpha_post=action_value_target_prior + q_evidence_sum,
        action_value_target_prior=action_value_target_prior,
    )


def _dirichlet_search_output_from_backend(
    *,
    alpha_v: jax.Array,
    backend_output: _DirichletSearchBackendOutput,
    legal_action_mask: jax.Array,
    posterior_key: jax.Array,
    search_cfg: GumbelSearchConfig | DirichletThompsonSearchConfig,
    use_search_weights_as_policy_target: bool,
    q_loss_weight_mode: str,
) -> SearchOutput:
    q_evidence_sum = backend_output.q_evidence_sum

    policy_samples = int(
        getattr(search_cfg, "policy_samples", _POSTERIOR_POLICY_TARGET_SAMPLES)
    )
    if policy_samples == 0:
        posterior_policy_target = backend_output.action_weights
    else:
        chunk_size = search_cfg.policy_sample_chunk_size
        posterior_policy_target = posterior_best_policy_target(
            posterior_key,
            backend_output.action_alpha_post,
            legal_action_mask,
            policy_samples,
            chunk_size=policy_samples if chunk_size is None else int(chunk_size),
        )
    if use_search_weights_as_policy_target:
        policy_target = backend_output.action_weights
    else:
        policy_target = posterior_policy_target
    beta_Q_target, beta_V_target = posterior_targets(
        alpha_v,
        backend_output.action_value_target_prior,
        q_evidence_sum,
        posterior_policy_target,
    )
    return SearchOutput(
        posterior=PosteriorTargets(
            prediction=PosteriorPrediction(
                policy=policy_target,
                alpha_v=beta_V_target,
                alpha_q=beta_Q_target,
            ),
            metadata=TargetMetadata(
                mask=_search_loss_mask(policy_target),
                q_weight=_q_loss_weight_from_mode(
                    q_loss_weight_mode,
                    q_evidence_sum,
                    posterior_policy_target,
                ),
                search_action=backend_output.action,
            ),
        ),
    )


def _run_dirichlet_backend_search_output(
    *,
    env_state: pgx.State,
    prediction: EvaluatorOutput,
    expand_fn,
    rng_key: jax.Array,
    search_cfg: GumbelSearchConfig | DirichletThompsonSearchConfig,
    run_backend: Callable[..., _DirichletSearchBackendOutput],
    use_search_weights_as_policy_target: bool,
    q_loss_weight_mode: str,
) -> SearchOutput:
    root_context = _make_dirichlet_root_context(env_state, prediction)
    search_key, posterior_key, _action_key = jax.random.split(rng_key, 3)
    backend_output = run_backend(
        env_state=env_state,
        root=root_context.root,
        expand_fn=expand_fn,
        search_key=search_key,
        search_cfg=search_cfg,
        action_value_prior=root_context.action_value_prior,
    )
    return _dirichlet_search_output_from_backend(
        alpha_v=root_context.alpha_v,
        backend_output=backend_output,
        legal_action_mask=env_state.legal_action_mask,
        posterior_key=posterior_key,
        search_cfg=search_cfg,
        use_search_weights_as_policy_target=use_search_weights_as_policy_target,
        q_loss_weight_mode=q_loss_weight_mode,
    )


def _run_scalar_gumbel_search_output(
    *,
    env_state: pgx.State,
    prediction: EvaluatorOutput,
    expand_fn,
    rng_key: jax.Array,
    search_cfg: GumbelSearchConfig,
) -> SearchOutput:
    value = _required_output(prediction.value, "value")
    root = mctx.RootFnOutput(
        prior_logits=prediction.logits,
        value=value,
        embedding=env_state,
    )
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
    return SearchOutput(
        posterior=PosteriorTargets(
            prediction=PosteriorPrediction(
                policy=policy_target,
                value=value,
            ),
            metadata=TargetMetadata(
                mask=_search_loss_mask(policy_target),
                search_action=legalize_action(
                    search_action,
                    env_state.legal_action_mask,
                ),
            ),
        ),
    )


def _run_policy_search_output(
    *,
    env_state: pgx.State,
    prediction: EvaluatorOutput,
    rng_key: jax.Array,
    search_cfg: PolicySearchConfig,
) -> SearchOutput:
    masked_logits = _masked_logits(prediction.logits, env_state.legal_action_mask)
    policy = jax.nn.softmax(masked_logits / float(search_cfg.temperature), axis=-1)
    has_legal_action = jnp.any(env_state.legal_action_mask, axis=-1, keepdims=True)
    policy = jnp.where(has_legal_action, policy, jnp.zeros_like(policy))
    # TODO: this search action should just be the action committed by the player. it shoudln't even be here in the first place
    search_action = posterior_sample_action(rng_key, policy, env_state.legal_action_mask)

    return SearchOutput(
        posterior=PosteriorTargets(
            prediction=PosteriorPrediction(policy, prediction.value, prediction.alpha_v, prediction.alpha_q),
            metadata=TargetMetadata(mask=_search_loss_mask(policy), search_action=legalize_action(search_action, env_state.legal_action_mask)),
        ),
    )


def _make_policy_search(
    env,
    evaluator: Callable[[jax.Array], EvaluatorOutput],
    search_cfg: PolicySearchConfig,
    *args, **kwargs,
) -> Callable[..., SearchOutput]:
    def search(*, root_state: pgx.State, rng_key: jax.Array) -> SearchOutput:
        prediction = evaluator(root_state.observation)
        return _run_policy_search_output(
            env_state=root_state,
            prediction=prediction,
            rng_key=rng_key,
            search_cfg=search_cfg,
        )

    return search


def _make_gumbel_search(
    env,
    evaluator: Callable[[jax.Array], EvaluatorOutput],
    search_cfg: GumbelSearchConfig,
    *,
    q_loss_weight_mode: str,
) -> Callable[..., SearchOutput]:
    scalar_expand_fn = make_expand_fn(env, evaluator)
    dirichlet_expand_fn = make_dirichlet_expand_fn_from_constants(
        env,
        evaluator,
        search_cfg.constants,
    )

    def search(*, root_state: pgx.State, rng_key: jax.Array) -> SearchOutput:
        prediction = evaluator(root_state.observation)
        if prediction.alpha_q is None:
            return _run_scalar_gumbel_search_output(
                env_state=root_state,
                prediction=prediction,
                expand_fn=scalar_expand_fn,
                rng_key=rng_key,
                search_cfg=search_cfg,
            )
        return _run_dirichlet_backend_search_output(
            env_state=root_state,
            prediction=prediction,
            expand_fn=dirichlet_expand_fn,
            rng_key=rng_key,
            search_cfg=search_cfg,
            run_backend=_run_dirichlet_gumbel_backend,
            use_search_weights_as_policy_target=isinstance(search_cfg, GumbelSearchConfig),
            q_loss_weight_mode=q_loss_weight_mode,
        )

    return search


def _make_dirichlet_thompson_search(
    env,
    evaluator: Callable[[jax.Array], EvaluatorOutput],
    search_cfg: DirichletThompsonSearchConfig,
    *,
    q_loss_weight_mode: str,
) -> Callable[..., SearchOutput]:
    expand_fn = make_dirichlet_expand_fn_from_constants(
        env,
        evaluator,
        search_cfg.constants,
    )

    def search(*, root_state: pgx.State, rng_key: jax.Array) -> SearchOutput:
        prediction = evaluator(root_state.observation)
        if prediction.alpha_q is None:
            raise ValueError(
                f"{SearchKind.dirichlet_thompson!r} search requires a Dirichlet "
                "evaluator."
            )
        return _run_dirichlet_backend_search_output(
            env_state=root_state,
            prediction=prediction,
            expand_fn=expand_fn,
            rng_key=rng_key,
            search_cfg=search_cfg,
            run_backend=_run_dirichlet_thompson_backend,
            use_search_weights_as_policy_target=isinstance(
                search_cfg,
                GumbelSearchConfig,
            ),
            q_loss_weight_mode=q_loss_weight_mode,
        )

    return search


def make_search(
    env,
    evaluator: Callable[[jax.Array], EvaluatorOutput],
    search_cfg: SearchConfig,
    *,
    q_loss_weight_mode: str = "policy",
) -> Callable[..., SearchOutput]:
    
    active_search_cfg, _make_search_function = {
        SearchKind.policy: (search_cfg.policy, _make_policy_search),
        SearchKind.gumbel: (search_cfg.gumbel, _make_gumbel_search),
        SearchKind.dirichlet_thompson: (search_cfg.dirichlet_thompson, _make_dirichlet_thompson_search),
    }[search_cfg.kind]
    
    return _make_search_function(env, evaluator, active_search_cfg, q_loss_weight_mode=q_loss_weight_mode)  # ty:ignore[invalid-argument-type]
