from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import mctx
import pgx

from .dirichlet_q_search import (
    NO_PARENT,
    NodeEmbedding,
    dirichlet_q_policy,
    outcome_mean,
    outcome_utility,
    posterior_best_action,
    posterior_best_policy_target,
    posterior_sample_action,
    posterior_targets,
    q_evidence_sum_from_tree,
    root_action_value_priors_from_tree,
)
from .dirichlet_tree.native import native_fields_from_beta
from .dirichlet_tree.types import SearchDiagnostics, TreeTrainingData
from .network import policy_value_from_output
from .posterior_tree import (
    run_posterior_tree_search,
    run_posterior_tree_search_state_batch,
    split_batched_state,
)


class _SearchStepOutput(NamedTuple):
    action_weights: jax.Array
    played_action: jax.Array
    beta_Q_target: jax.Array
    beta_V_target: jax.Array
    q_loss_weight: jax.Array
    search_loss_mask: jax.Array
    tree_data: TreeTrainingData | None = None
    search_diagnostics: SearchDiagnostics | None = None
    q_target_kind: jax.Array | None = None
    q_target_weight: jax.Array | None = None
    q_target_outcome: jax.Array | None = None
    q_target_distance: jax.Array | None = None
    v_target_kind: jax.Array | None = None
    v_target_weight: jax.Array | None = None
    v_target_outcome: jax.Array | None = None
    v_target_distance: jax.Array | None = None


_NATIVE_TARGET_FIELD_NAMES = (
    "q_target_kind",
    "q_target_weight",
    "q_target_outcome",
    "q_target_distance",
    "v_target_kind",
    "v_target_weight",
    "v_target_outcome",
    "v_target_distance",
)


def _empty_posterior_targets(
    policy_target: jax.Array,
    num_outcomes: int,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    batch_size, num_actions = policy_target.shape
    beta_q = jnp.zeros(
        (batch_size, num_actions, num_outcomes),
        dtype=policy_target.dtype,
    )
    beta_v = jnp.zeros((batch_size, num_outcomes), dtype=policy_target.dtype)
    q_loss_weight = jnp.zeros((batch_size, num_actions), dtype=policy_target.dtype)
    return beta_q, beta_v, q_loss_weight


def _num_outcomes_for_config(config) -> int:
    num_outcomes = getattr(config, "num_outcomes", None)
    if num_outcomes is None:
        return 2 if config.env_id == "hex" else 3
    return num_outcomes


def _search_loss_mask(action_weights: jax.Array) -> jax.Array:
    return jnp.sum(action_weights, axis=-1) > 0


def _native_target_kwargs_from_output(output: Any) -> dict[str, jax.Array]:
    native_defaults = native_fields_from_beta(
        output.beta_Q_target,
        output.beta_V_target,
    )

    def field_or_default(name: str) -> jax.Array:
        value = getattr(output, name, None)
        return native_defaults[name] if value is None else value

    return {name: field_or_default(name) for name in _NATIVE_TARGET_FIELD_NAMES}


def _select_played_action(
    action_source: str,
    rng_key: jax.Array,
    action_weights: jax.Array,
    legal_action_mask: jax.Array,
    search_action: jax.Array,
) -> jax.Array:
    if action_source in ("posterior_best", "posterior_argmax"):
        return posterior_best_action(action_weights, legal_action_mask)
    if action_source == "posterior_sample":
        return posterior_sample_action(rng_key, action_weights, legal_action_mask)
    if action_source == "search_action":
        return search_action
    raise ValueError(f"unknown selfplay_action_source: {action_source!r}")


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
        return jnp.sum(q_evidence_sum, axis=-1)
    if mode == "policy":
        return posterior_policy_target
    raise ValueError(f"unknown q_loss_weight_mode: {mode!r}")


def _run_scalar_gumbel_search(
    *,
    env_state: pgx.State,
    model_output,
    recurrent_fn,
    rng_key: jax.Array,
    config,
) -> _SearchStepOutput:
    logits, value = policy_value_from_output(model_output)
    root = mctx.RootFnOutput(
        prior_logits=logits,
        value=value,
        embedding=env_state,
    )
    policy_output = mctx.gumbel_muzero_policy(
        params=(),
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=config.num_simulations,
        invalid_actions=~env_state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=1.0,
    )
    policy_target = policy_output.action_weights
    beta_Q_target, beta_V_target, q_loss_weight = _empty_posterior_targets(
        policy_target,
        _num_outcomes_for_config(config),
    )
    return _SearchStepOutput(
        action_weights=policy_target,
        played_action=policy_output.action,
        beta_Q_target=beta_Q_target,
        beta_V_target=beta_V_target,
        q_loss_weight=q_loss_weight,
        search_loss_mask=_search_loss_mask(policy_target),
    )


def _run_dirichlet_search(
    *,
    env_state: pgx.State,
    model_output,
    recurrent_fn,
    search_key: jax.Array,
    posterior_key: jax.Array,
    action_key: jax.Array,
    config,
    action_source: str | None = None,
) -> _SearchStepOutput:
    logits, alpha_v, alpha_q = model_output
    root = _make_dirichlet_root(env_state, logits, alpha_v, alpha_q)
    action_value_prior = alpha_q

    if config.search_policy == "dirichlet_thompson":
        policy_output = dirichlet_q_policy(
            params=(),
            rng_key=search_key,
            root=root,
            recurrent_fn=recurrent_fn,
            action_value_prior=action_value_prior,
            num_simulations=config.num_simulations,
            invalid_actions=~env_state.legal_action_mask,
            num_search_blocks=getattr(config, "num_search_blocks", 1),
        )
        q_evidence_sum = policy_output.q_evidence_sum
        action_alpha_post = policy_output.alpha_search
        action_value_target_prior = action_alpha_post - q_evidence_sum
    else:
        policy_output = mctx.gumbel_muzero_policy(
            params=(),
            rng_key=search_key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            invalid_actions=~env_state.legal_action_mask,
            qtransform=mctx.qtransform_completed_by_mix_value,
            gumbel_scale=1.0,
        )
        q_evidence_sum = q_evidence_sum_from_tree(policy_output.search_tree)
        action_value_target_prior = root_action_value_priors_from_tree(
            policy_output.search_tree,
            action_value_prior,
        )
        action_alpha_post = action_value_target_prior + q_evidence_sum

    posterior_policy_target = posterior_best_policy_target(
        posterior_key,
        action_alpha_post,
        env_state.legal_action_mask,
        config.policy_mc_samples,
    )
    if config.search_policy == "gumbel":
        policy_target = policy_output.action_weights
    else:
        policy_target = posterior_policy_target
    beta_Q_target, beta_V_target = posterior_targets(
        alpha_v,
        action_value_target_prior,
        q_evidence_sum,
        posterior_policy_target,
    )
    played_action = _select_played_action(
        config.selfplay_action_source if action_source is None else action_source,
        action_key,
        posterior_policy_target,
        env_state.legal_action_mask,
        policy_output.action,
    )
    return _SearchStepOutput(
        action_weights=policy_target,
        played_action=played_action,
        beta_Q_target=beta_Q_target,
        beta_V_target=beta_V_target,
        q_loss_weight=_q_loss_weight_from_mode(
            getattr(config, "q_loss_weight_mode", "policy"),
            q_evidence_sum,
            posterior_policy_target,
        ),
        search_loss_mask=_search_loss_mask(policy_target),
    )


def _run_model_search(
    *,
    env_state: pgx.State,
    model_output,
    scalar_recurrent_fn,
    dirichlet_recurrent_fn,
    search_key: jax.Array,
    posterior_key: jax.Array,
    action_key: jax.Array,
    config,
    action_source: str | None = None,
) -> _SearchStepOutput:
    if len(model_output) == 2:
        return _run_scalar_gumbel_search(
            env_state=env_state,
            model_output=model_output,
            recurrent_fn=scalar_recurrent_fn,
            rng_key=search_key,
            config=config,
        )
    return _run_dirichlet_search(
        env_state=env_state,
        model_output=model_output,
        recurrent_fn=dirichlet_recurrent_fn,
        search_key=search_key,
        posterior_key=posterior_key,
        action_key=action_key,
        config=config,
        action_source=action_source,
    )


def _run_model_eval_search(
    *,
    env_state: pgx.State,
    model_output,
    scalar_recurrent_fn,
    dirichlet_recurrent_fn,
    rng_key: jax.Array,
    config,
) -> _SearchStepOutput:
    if (
        len(model_output) == 3
        and getattr(config, "search_policy", "gumbel") == "dirichlet_thompson"
    ):
        search_key, posterior_key = jax.random.split(rng_key)
        return _run_dirichlet_search(
            env_state=env_state,
            model_output=model_output,
            recurrent_fn=dirichlet_recurrent_fn,
            search_key=search_key,
            posterior_key=posterior_key,
            action_key=posterior_key,
            config=config,
            action_source="posterior_argmax",
        )
    return _run_scalar_gumbel_search(
        env_state=env_state,
        model_output=model_output,
        recurrent_fn=scalar_recurrent_fn,
        rng_key=rng_key,
        config=config,
    )


def _run_posterior_tree_search_step(
    *,
    env,
    config,
    env_state: pgx.State,
    leaf_evaluator,
    search_key: jax.Array,
    action_key: jax.Array,
    use_wavefront_arena: bool,
    device_put_cpu: Callable[[Any], Any],
    action_source: str | None = None,
) -> _SearchStepOutput:
    if use_wavefront_arena:
        search_output = run_posterior_tree_search_state_batch(
            env=env,
            root_state_batch=env_state,
            leaf_evaluator=leaf_evaluator,
            rng_key=search_key,
            config=config,
        )
    else:
        search_output = run_posterior_tree_search(
            env=env,
            root_states=split_batched_state(env_state),
            leaf_evaluator=leaf_evaluator,
            rng_key=search_key,
            config=config,
        )

    search_action = (
        search_output.action
        if use_wavefront_arena
        else device_put_cpu(search_output.action)
    )
    played_action = (
        _select_played_action(
            config.selfplay_action_source if action_source is None else action_source,
            action_key,
            search_output.action_weights,
            env_state.legal_action_mask,
            search_action,
        )
        if use_wavefront_arena
        else search_action
    )
    root_search_mask = search_output.search_loss_mask
    if root_search_mask is None:
        root_search_mask = _search_loss_mask(search_output.action_weights)
    return _SearchStepOutput(
        action_weights=search_output.action_weights,
        played_action=played_action,
        beta_Q_target=search_output.beta_Q_target,
        beta_V_target=search_output.beta_V_target,
        q_loss_weight=search_output.q_loss_weight,
        search_loss_mask=root_search_mask,
        tree_data=search_output.tree_data,
        search_diagnostics=getattr(search_output, "diagnostics", None),
        **_native_target_kwargs_from_output(search_output),
    )
