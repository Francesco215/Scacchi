from typing import Any, Callable, NamedTuple

import jax
import jax.numpy as jnp
import mctx
import numpy as np
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
    PosteriorTree,
    PosteriorTreeBatchOutput,
    StepRequest,
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
    device_put_cpu: Callable[[Any], Any],
    transition_evaluator: Callable[[Any, jax.Array], Any] | None = None,
) -> _SearchStepOutput:
    if transition_evaluator is None:
        def transition_evaluator(states: Any, actions: jax.Array):
            child_states = jax.vmap(env.step)(states, actions)
            return child_states, leaf_evaluator(child_states.observation)

    search_output = _run_fused_posterior_tree_search(
        env=env,
        root_states=split_batched_state(env_state),
        leaf_evaluator=leaf_evaluator,
        transition_evaluator=transition_evaluator,
        search_key=search_key,
        config=config,
        device_put_cpu=device_put_cpu,
    )
    search_action = device_put_cpu(search_output.action)
    played_action = search_action
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


def _run_fused_posterior_tree_search(
    *,
    env: Any,
    root_states: list[Any],
    leaf_evaluator: Callable[[jax.Array], Any],
    transition_evaluator: Callable[[Any, jax.Array], Any],
    search_key: jax.Array,
    config,
    device_put_cpu: Callable[[Any], Any],
) -> PosteriorTreeBatchOutput:
    if not root_states:
        raise ValueError("root_states must not be empty")

    root_observations = jnp.stack([state.observation for state in root_states], axis=0)
    root_logits, root_alpha_v, root_alpha_q = leaf_evaluator(root_observations)
    root_logits, root_alpha_v, root_alpha_q = jax.device_get(
        (root_logits, root_alpha_v, root_alpha_q)
    )

    seed = int(
        jax.device_get(
            jax.random.randint(search_key, (), minval=0, maxval=np.iinfo(np.int32).max)
        )
    )
    rng = np.random.default_rng(seed)
    trees = tuple(
        PosteriorTree(
            env=env,
            root_state=state,
            root_logits=root_logits[ix],
            root_alpha_v=root_alpha_v[ix],
            root_alpha_q=root_alpha_q[ix],
            tree_index=ix,
            rng=rng,
            leaf_value_mode=getattr(config, "leaf_value_mode", "alpha"),
            kappa_leaf=float(getattr(config, "kappa_leaf", 1.0)),
            kappa_terminal=float(getattr(config, "kappa_terminal", 8.0)),
            epsilon_terminal=float(getattr(config, "epsilon_terminal", 1e-6)),
            state_posterior_kappa_n=float(
                getattr(config, "state_posterior_kappa_n", 9.0)
            ),
            policy_mc_samples=getattr(config, "policy_mc_samples"),
            backup_mc_samples=getattr(
                config,
                "backup_mc_samples",
                getattr(config, "policy_mc_samples"),
            ),
            commit=getattr(config, "selfplay_action_source"),
            categorical_draw_rule=getattr(config, "categorical_draw_rule", "policy_prior"),
        )
        for ix, state in enumerate(root_states)
    )

    _run_fused_search_loop(
        trees,
        transition_evaluator=transition_evaluator,
        num_simulations=int(getattr(config, "num_simulations")),
        eval_batch_size=_eval_batch_size(config, len(trees)),
        inflight_limit=int(getattr(config, "inflight_limit", 1)),
    )

    finished = [tree.finish_native() for tree in trees]
    (
        actions,
        policies,
        beta_q,
        beta_v,
        q_weight,
        alpha_root,
        q_kind,
        q_target_weight,
        q_outcome,
        q_distance,
        v_kind,
        v_target_weight,
        v_outcome,
        v_distance,
    ) = zip(*finished, strict=True)
    policy_array = np.stack(policies, axis=0)
    return PosteriorTreeBatchOutput(
        action=device_put_cpu(jnp.asarray(np.asarray(actions), dtype=jnp.int32)),
        action_weights=device_put_cpu(jnp.asarray(policy_array)),
        beta_Q_target=device_put_cpu(jnp.asarray(np.stack(beta_q, axis=0))),
        beta_V_target=device_put_cpu(jnp.asarray(np.stack(beta_v, axis=0))),
        q_loss_weight=device_put_cpu(jnp.asarray(np.stack(q_weight, axis=0))),
        alpha_root=device_put_cpu(jnp.asarray(np.stack(alpha_root, axis=0))),
        trees=trees,
        tree_data=None,
        search_loss_mask=device_put_cpu(jnp.asarray(np.sum(policy_array, axis=-1) > 0.0)),
        q_target_kind=device_put_cpu(jnp.asarray(np.stack(q_kind, axis=0), dtype=jnp.int8)),
        q_target_weight=device_put_cpu(
            jnp.asarray(np.stack(q_target_weight, axis=0), dtype=jnp.float32)
        ),
        q_target_outcome=device_put_cpu(jnp.asarray(np.stack(q_outcome, axis=0), dtype=jnp.int8)),
        q_target_distance=device_put_cpu(
            jnp.asarray(np.stack(q_distance, axis=0), dtype=jnp.int32)
        ),
        v_target_kind=device_put_cpu(jnp.asarray(np.stack(v_kind, axis=0), dtype=jnp.int8)),
        v_target_weight=device_put_cpu(
            jnp.asarray(np.stack(v_target_weight, axis=0), dtype=jnp.float32)
        ),
        v_target_outcome=device_put_cpu(jnp.asarray(np.stack(v_outcome, axis=0), dtype=jnp.int8)),
        v_target_distance=device_put_cpu(
            jnp.asarray(np.stack(v_distance, axis=0), dtype=jnp.int32)
        ),
    )


def _run_fused_search_loop(
    trees: tuple[PosteriorTree, ...],
    *,
    transition_evaluator: Callable[[Any, jax.Array], Any],
    num_simulations: int,
    eval_batch_size: int,
    inflight_limit: int,
) -> None:
    while any(tree.done < num_simulations for tree in trees):
        step_requests, made_progress = _build_fused_step_batch(
            trees,
            num_simulations=num_simulations,
            inflight_limit=inflight_limit,
            eval_batch_size=eval_batch_size,
        )
        if not step_requests:
            if made_progress:
                continue
            if all(tree.done >= num_simulations for tree in trees):
                break
            unfinished = [tree.tree_index for tree in trees if tree.done < num_simulations]
            raise RuntimeError(f"posterior tree search stalled for roots {unfinished}")

        _consume_fused_step_requests(
            trees,
            step_requests,
            transition_evaluator=transition_evaluator,
            eval_batch_size=eval_batch_size,
        )


def _build_fused_step_batch(
    trees: tuple[PosteriorTree, ...],
    *,
    num_simulations: int,
    inflight_limit: int,
    eval_batch_size: int,
) -> tuple[list[StepRequest], bool]:
    requests: list[StepRequest] = []
    made_progress = False
    for tree in trees:
        if tree.done + tree.inflight >= num_simulations or tree.inflight >= inflight_limit:
            continue
        before = (tree.done, tree.inflight, len(tree.nodes))
        request = tree.next_step_request()
        after = (tree.done, tree.inflight, len(tree.nodes))
        if request is not None:
            requests.append(request)
        if request is not None or after != before:
            made_progress = True
        if len(requests) >= eval_batch_size:
            break
    return requests, made_progress


def _consume_fused_step_requests(
    trees: tuple[PosteriorTree, ...],
    requests: list[StepRequest],
    *,
    transition_evaluator: Callable[[Any, jax.Array], Any],
    eval_batch_size: int,
) -> None:
    if not requests:
        return
    fallback = requests[0]
    padded = requests + [fallback] * (eval_batch_size - len(requests))
    states = [request.state for request in padded]
    actions = jnp.asarray([request.action for request in padded], dtype=jnp.int32)
    transition_output = transition_evaluator(_stack_states(states), actions)
    child_state_batch, logits, alpha_v, alpha_q = _unpack_transition_output(
        transition_output
    )
    child_states = split_batched_state(child_state_batch)
    logits, alpha_v, alpha_q = jax.device_get((logits, alpha_v, alpha_q))

    for ix, request in enumerate(requests):
        eval_request = trees[request.tree_index].consume_step_result(
            request,
            child_states[ix],
        )
        if eval_request is None:
            continue
        trees[eval_request.tree_index].consume_result(
            eval_request,
            logits=logits[ix],
            alpha_v=alpha_v[ix],
            alpha_q=alpha_q[ix],
        )


def _unpack_transition_output(output: Any) -> tuple[Any, jax.Array, jax.Array, jax.Array]:
    if len(output) == 2:
        child_states, model_output = output
        if len(model_output) != 3:
            raise ValueError(
                "posterior-tree search requires transition_evaluator to return "
                "(child_states, (logits, alpha_V, alpha_Q))."
            )
        logits, alpha_v, alpha_q = model_output
        return child_states, logits, alpha_v, alpha_q
    if len(output) == 4:
        child_states, logits, alpha_v, alpha_q = output
        return child_states, logits, alpha_v, alpha_q
    raise ValueError(
        "posterior-tree search requires transition_evaluator to return "
        "(child_states, model_output) or (child_states, logits, alpha_V, alpha_Q)."
    )


def _stack_states(states: list[Any]) -> Any:
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


def _eval_batch_size(config: Any, num_trees: int) -> int:
    configured = getattr(config, "search_eval_batch_size", None)
    if configured is None:
        return max(1, num_trees)
    return max(1, int(configured))
