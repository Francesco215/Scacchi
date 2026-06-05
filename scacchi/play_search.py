import os
import time
from functools import partial
from typing import Any, Callable, NamedTuple

import dqaz
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
from .dirichlet_tree.native import (
    NO_DISTANCE,
    NO_OUTCOME,
    TARGET_PAD,
    native_fields_from_beta,
)
from .dirichlet_tree.types import SearchDiagnostics, TreeTrainingData
from .dqaz_jax_backup import BackupArrays, apply_batched_backup_block
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


class _DQAZBackupInputs(NamedTuple):
    tree_ids: np.ndarray
    node_ids: np.ndarray
    edge_b: np.ndarray
    edge_completed: np.ndarray
    edge_r_count: np.ndarray
    q_alpha: np.ndarray
    value_alpha: np.ndarray
    legal_mask: np.ndarray
    node_players: np.ndarray
    path_slots: np.ndarray
    path_edges: np.ndarray
    path_mask: np.ndarray
    leaf_alpha: np.ndarray
    leaf_players: np.ndarray


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
    num_outcomes = config.env.num_outcomes
    if num_outcomes is None:
        return 2 if config.env.id == "hex" else 3
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
        selected = posterior_best_action(action_weights, legal_action_mask)
    elif action_source == "posterior_sample":
        selected = posterior_sample_action(rng_key, action_weights, legal_action_mask)
    elif action_source == "search_action":
        selected = search_action
    else:
        raise ValueError(f"unknown selfplay_action_source: {action_source!r}")
    return _legalize_played_action(selected, legal_action_mask)


def _legalize_played_action(
    action: jax.Array,
    legal_action_mask: jax.Array,
) -> jax.Array:
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
        num_simulations=config.search.num_simulations,
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
        played_action=_legalize_played_action(
            policy_output.action,
            env_state.legal_action_mask,
        ),
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

    if config.search.policy == "dirichlet_thompson":
        policy_output = dirichlet_q_policy(
            params=(),
            rng_key=search_key,
            root=root,
            recurrent_fn=recurrent_fn,
            action_value_prior=action_value_prior,
            num_simulations=config.search.num_simulations,
            invalid_actions=~env_state.legal_action_mask,
            num_search_blocks=config.search.num_blocks,
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
            num_simulations=config.search.num_simulations,
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
        config.search.monte_carlo.policy_samples,
    )
    if config.search.policy == "gumbel":
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
        config.selfplay.action_source if action_source is None else action_source,
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
            config.training.losses.q_loss_weight_mode,
            q_evidence_sum,
            posterior_policy_target,
        ),
        search_loss_mask=_search_loss_mask(policy_target),
    )


def _run_model_search(
    env_state: pgx.State,
    model_output,
    scalar_recurrent_fn,
    dirichlet_recurrent_fn,
    search_key: jax.Array,
    posterior_key: jax.Array,
    action_key: jax.Array,
    config,
    *,
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
        and config.search.policy == "dirichlet_thompson"
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

    search_backend = getattr(config, "search_backend", None)
    if search_backend == "dqaz":
        search_output = _run_dqaz_posterior_tree_search(
            env=env,
            root_state_batch=env_state,
            leaf_evaluator=leaf_evaluator,
            transition_evaluator=transition_evaluator,
            search_key=search_key,
            config=config,
            device_put_cpu=device_put_cpu,
        )
    else:
        root_states = split_batched_state(env_state)
        search_output = _run_fused_posterior_tree_search(
            env=env,
            root_states=root_states,
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
            leaf_value_mode=config.search.leaf_value_mode,
            kappa_leaf=float(config.search.constants.kappa_leaf),
            kappa_terminal=float(config.search.constants.kappa_terminal),
            epsilon_terminal=float(config.search.constants.epsilon_terminal),
            state_posterior_kappa_n=float(
                config.search.constants.state_posterior_kappa_n
            ),
            policy_mc_samples=config.search.monte_carlo.policy_samples,
            backup_mc_samples=config.search.monte_carlo.backup_samples,
            commit=config.selfplay.action_source,
            categorical_draw_rule=config.search.constants.categorical_draw_rule,
        )
        for ix, state in enumerate(root_states)
    )

    _run_fused_search_loop(
        trees,
        transition_evaluator=transition_evaluator,
        num_simulations=int(config.search.num_simulations),
        eval_batch_size=_eval_batch_size(config, len(trees)),
        inflight_limit=int(config.search.inflight_limit),
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


_DQAZ_JAX_BACKUP_BLOCK_DEPTH = 8
_DQAZ_STATE_REPLAY_BLOCK_DEPTH = 8
_DQAZ_STATE_REPLAY_CACHE: dict[int, Any] = {}


def _pad_to_shape(array: np.ndarray, target_shape: tuple[int, ...], value: Any) -> np.ndarray:
    if array.shape == target_shape:
        return array
    if len(array.shape) != len(target_shape):
        raise ValueError("cannot pad array to shape with different rank")
    if any(size > target for size, target in zip(array.shape, target_shape, strict=True)):
        raise ValueError("cannot pad array to a smaller shape")
    padded = np.empty(target_shape, dtype=array.dtype)
    padded[...] = value
    slices = tuple(slice(0, size) for size in array.shape)
    padded[slices] = array
    return padded


def _dqaz_backup_width_bucket(
    actual_width: int,
    *,
    eval_batch_size: int,
    root_batch_size: int,
    num_simulations: int,
) -> int:
    root_batch_size = max(1, int(root_batch_size))
    per_root_eval_width = min(
        max(1, int(num_simulations)),
        max(1, (int(eval_batch_size) + root_batch_size - 1) // root_batch_size),
    )
    if actual_width <= per_root_eval_width:
        return per_root_eval_width
    bucket = 1 << (int(actual_width) - 1).bit_length()
    return min(bucket, max(1, int(num_simulations)))


def _dqaz_backup_root_bucket(actual_roots: int, root_batch_size: int) -> int:
    actual_roots = max(1, int(actual_roots))
    root_batch_size = max(1, int(root_batch_size))
    if root_batch_size < 32:
        return root_batch_size
    if actual_roots >= root_batch_size:
        return root_batch_size
    min_bucket = min(root_batch_size, 8)
    return min(root_batch_size, max(min_bucket, 1 << (actual_roots - 1).bit_length()))


def _pad_dqaz_backup_inputs(
    backup_batch: Any,
    *,
    root_batch_size: int,
    eval_batch_size: int,
    num_simulations: int,
) -> _DQAZBackupInputs:
    tree_ids = np.asarray(backup_batch.tree_ids, dtype=np.uint64)
    node_ids = np.asarray(backup_batch.node_ids, dtype=np.int64)
    root_target = _dqaz_backup_root_bucket(
        int(node_ids.shape[0]),
        int(root_batch_size),
    )
    width_target = _dqaz_backup_width_bucket(
        int(node_ids.shape[2]),
        eval_batch_size=eval_batch_size,
        root_batch_size=root_target,
        num_simulations=num_simulations,
    )

    def pad_roots_and_slots(array: np.ndarray, value: Any) -> np.ndarray:
        return _pad_to_shape(
            array,
            (root_target, array.shape[1], width_target, *array.shape[3:]),
            value,
        )

    def pad_roots_and_leaf_slots(array: np.ndarray, value: Any) -> np.ndarray:
        return _pad_to_shape(
            array,
            (root_target, width_target, *array.shape[2:]),
            value,
        )

    return _DQAZBackupInputs(
        tree_ids=_pad_to_shape(tree_ids, (root_target,), 0),
        node_ids=pad_roots_and_slots(node_ids, -1),
        edge_b=pad_roots_and_slots(np.asarray(backup_batch.edge_b, dtype=np.float32), 1.0),
        edge_completed=pad_roots_and_slots(
            np.asarray(backup_batch.edge_completed, dtype=bool),
            False,
        ),
        edge_r_count=pad_roots_and_slots(
            np.asarray(backup_batch.edge_r_count, dtype=np.int32),
            0,
        ),
        q_alpha=pad_roots_and_slots(np.asarray(backup_batch.q_alpha, dtype=np.float32), 1.0),
        value_alpha=pad_roots_and_slots(
            np.asarray(backup_batch.value_alpha, dtype=np.float32),
            1.0,
        ),
        legal_mask=pad_roots_and_slots(np.asarray(backup_batch.legal_mask, dtype=bool), False),
        node_players=pad_roots_and_slots(
            np.asarray(backup_batch.node_players, dtype=np.int32),
            0,
        ),
        path_slots=pad_roots_and_slots(np.asarray(backup_batch.path_slots, dtype=np.int32), 0),
        path_edges=pad_roots_and_slots(np.asarray(backup_batch.path_edges, dtype=np.int32), 0),
        path_mask=pad_roots_and_slots(np.asarray(backup_batch.path_mask, dtype=bool), False),
        leaf_alpha=pad_roots_and_leaf_slots(
            np.asarray(backup_batch.leaf_alpha, dtype=np.float32),
            1.0,
        ),
        leaf_players=pad_roots_and_leaf_slots(
            np.asarray(backup_batch.leaf_players, dtype=np.int32),
            0,
        ),
    )


def _path_state_replay(env: Any):
    cache_key = id(env)
    replay = _DQAZ_STATE_REPLAY_CACHE.get(cache_key)
    if replay is not None:
        return replay

    @jax.jit
    def replay(root_states: Any, root_indices: jax.Array, paths: jax.Array, lengths: jax.Array):
        states = jax.tree_util.tree_map(lambda x: x[root_indices], root_states)

        def body(depth: int, current_states: Any) -> Any:
            active = depth < lengths

            def active_body(current_states: Any) -> Any:
                stepped = jax.vmap(env.step)(current_states, paths[:, depth])
                return _select_active_states(stepped, current_states, active)

            return jax.lax.cond(
                jnp.any(active),
                active_body,
                lambda current_states: current_states,
                current_states,
            )

        return jax.lax.fori_loop(0, paths.shape[1], body, states)

    _DQAZ_STATE_REPLAY_CACHE[cache_key] = replay
    return replay


@partial(jax.jit, static_argnames=("sample_count",))
def _apply_dqaz_jax_backup_block(
    rng_key: jax.Array,
    edge_b: jax.Array,
    edge_completed: jax.Array,
    edge_r_count: jax.Array,
    q_alpha: jax.Array,
    value_alpha: jax.Array,
    legal_mask: jax.Array,
    node_players: jax.Array,
    path_nodes: jax.Array,
    path_edges: jax.Array,
    path_mask: jax.Array,
    c_v: jax.Array,
    n_down: jax.Array,
    policy: jax.Array,
    beta: jax.Array,
    beta_players: jax.Array,
    block_start: jax.Array,
    kappa_n: float,
    *,
    sample_count: int,
):
    return apply_batched_backup_block(
        rng_key,
        edge_b,
        edge_completed,
        edge_r_count,
        q_alpha,
        value_alpha,
        legal_mask,
        node_players,
        path_nodes,
        path_edges,
        path_mask,
        c_v,
        n_down,
        policy,
        beta,
        beta_players,
        block_start,
        kappa_n,
        sample_count,
    )


def _apply_dqaz_jax_backup(
    rng_key: jax.Array,
    edge_b: jax.Array,
    edge_completed: jax.Array,
    edge_r_count: jax.Array,
    q_alpha: jax.Array,
    value_alpha: jax.Array,
    legal_mask: jax.Array,
    node_players: jax.Array,
    path_nodes: jax.Array,
    path_edges: jax.Array,
    path_mask: jax.Array,
    leaf_alpha: jax.Array,
    leaf_players: jax.Array,
    kappa_n: float,
    *,
    sample_count: int,
    max_depth: int | None = None,
) -> BackupArrays:
    path_depth = int(path_nodes.shape[1])
    if path_depth % _DQAZ_JAX_BACKUP_BLOCK_DEPTH != 0:
        raise ValueError(
            f"dqaz jax backup path depth {path_depth} is not divisible by "
            f"{_DQAZ_JAX_BACKUP_BLOCK_DEPTH}"
        )

    c_v = value_alpha
    n_down = jnp.sum(
        jnp.where(legal_mask, edge_r_count, 0),
        axis=-1,
        dtype=jnp.int32,
    )
    policy = jnp.zeros_like(legal_mask, dtype=value_alpha.dtype)
    beta = leaf_alpha
    beta_players = leaf_players

    active_depth = path_depth if max_depth is None else int(max_depth)
    active_depth = max(1, min(active_depth, path_depth))
    active_depth = (
        (active_depth + _DQAZ_JAX_BACKUP_BLOCK_DEPTH - 1)
        // _DQAZ_JAX_BACKUP_BLOCK_DEPTH
        * _DQAZ_JAX_BACKUP_BLOCK_DEPTH
    )

    for block_start in range(
        active_depth - _DQAZ_JAX_BACKUP_BLOCK_DEPTH,
        -1,
        -_DQAZ_JAX_BACKUP_BLOCK_DEPTH,
    ):
        path_slice = (
            (slice(None), slice(block_start, block_start + _DQAZ_JAX_BACKUP_BLOCK_DEPTH), slice(None))
            if path_nodes.ndim == 3
            else (slice(None), slice(block_start, block_start + _DQAZ_JAX_BACKUP_BLOCK_DEPTH))
        )
        block = _apply_dqaz_jax_backup_block(
            rng_key,
            edge_b,
            edge_completed,
            edge_r_count,
            q_alpha,
            value_alpha,
            legal_mask,
            node_players,
            path_nodes[path_slice],
            path_edges[path_slice],
            path_mask[path_slice],
            c_v,
            n_down,
            policy,
            beta,
            beta_players,
            jnp.asarray(block_start, dtype=jnp.int32),
            kappa_n,
            sample_count=sample_count,
        )
        edge_b = block.edge_b
        edge_completed = block.edge_completed
        edge_r_count = block.edge_r_count
        c_v = block.c_v
        n_down = block.n_down
        policy = block.policy
        beta = block.beta
        beta_players = block.beta_players

    return BackupArrays(
        edge_b=edge_b,
        edge_completed=edge_completed,
        edge_r_count=edge_r_count,
        c_v=c_v,
        n_down=n_down,
        policy=policy,
    )


def _run_dqaz_posterior_tree_search(
    *,
    env: Any,
    root_state_batch: Any,
    leaf_evaluator: Callable[[jax.Array], Any],
    transition_evaluator: Callable[[Any, jax.Array], Any],
    search_key: jax.Array,
    config,
    device_put_cpu: Callable[[Any], Any],
) -> PosteriorTreeBatchOutput:
    root_observations = root_state_batch.observation
    batch_size = int(root_observations.shape[0])
    if batch_size == 0:
        raise ValueError("root_states must not be empty")

    _root_logits, root_alpha_v, root_alpha_q = leaf_evaluator(root_observations)
    (
        root_alpha_v,
        root_alpha_q,
        root_observations_np,
        root_legal_mask,
        root_current_players,
    ) = jax.device_get(
        (
            root_alpha_v,
            root_alpha_q,
            root_observations,
            root_state_batch.legal_action_mask,
            root_state_batch.current_player,
        )
    )
    action_size = int(root_alpha_q.shape[-2])
    empty_policy_logits = np.zeros((0,), dtype=np.float32)
    seed = int(
        jax.device_get(
            jax.random.randint(search_key, (), minval=0, maxval=np.iinfo(np.int32).max)
        )
    )
    engine = dqaz.SearchEngine(
        dqaz.SearchConfig(
            action_size=action_size,
            observation_shape=tuple(root_observations.shape[1:]),
            simulations_per_root=int(config.search.num_simulations),
            posterior_best_samples=int(config.search.monte_carlo.policy_samples),
            kappa_n=float(config.search.constants.state_posterior_kappa_n),
            seed=seed,
            debug=bool(getattr(config, "debug", False)),
            max_pending_requests_per_root=int(config.search.inflight_limit),
        )
    )
    root_offsets, root_actions, root_q = _compact_valid_actions_and_q_from_mask_np(
        root_legal_mask,
        root_alpha_q,
    )
    state_store = _PathStateStore(env, root_state_batch)
    root_handles = state_store.add_roots(batch_size)
    tree_ids = engine.add_roots(
        root_handles,
        np.asarray(root_observations_np, dtype=np.float32),
        root_offsets,
        root_actions,
        np.asarray(root_current_players, dtype=np.int32),
        empty_policy_logits,
        np.asarray(root_alpha_v, dtype=np.float32),
        root_q,
    )
    profile_search = os.environ.get("DQAZ_PROFILE_SEARCH") is not None
    profile_times = {
        "request": 0.0,
        "state_batch": 0.0,
        "transition_dispatch": 0.0,
        "device_get": 0.0,
        "flatten": 0.0,
        "state_store_add": 0.0,
        "terminal_alpha": 0.0,
        "submit_prepare": 0.0,
        "pad_backup": 0.0,
        "jax_backup_dispatch": 0.0,
        "jax_backup_get": 0.0,
        "apply_jax": 0.0,
    }
    profile_waves = 0
    profile_active = 0
    profile_paths = 0
    profile_nodes = 0
    profile_shapes: dict[tuple[int, int, int, int], int] = {}

    eval_batch_size = _eval_batch_size(config, batch_size)
    pad_to = eval_batch_size if bool(getattr(config, "search_pad_to_eval_batch", False)) else None
    use_jax_backup = bool(getattr(config, "search_jax_backup", True))
    jax_backup_step = 0
    while not engine.is_done(tree_ids):
        start = time.perf_counter() if profile_search else 0.0
        batch = engine.request_transitions(
            max_batch_size=eval_batch_size,
            pad_to=pad_to,
            include_parent_states=False,
        )
        if profile_search:
            profile_times["request"] += time.perf_counter() - start
        if batch.size == 0:
            raise RuntimeError("dqaz posterior tree search stalled")
        active_size = int(batch.size)
        if profile_search:
            profile_waves += 1
            profile_active += active_size
        parent_handles = np.asarray(batch.parent_state_handles, dtype=np.int64)
        action_array = np.asarray(batch.actions, dtype=np.int32)
        start = time.perf_counter() if profile_search else 0.0
        parent_states = state_store.batch(parent_handles)
        if profile_search:
            profile_times["state_batch"] += time.perf_counter() - start
        actions = jnp.asarray(action_array, dtype=jnp.int32)
        start = time.perf_counter() if profile_search else 0.0
        transition_output = transition_evaluator(parent_states, actions)
        child_state_batch, _logits, alpha_v, alpha_q = _unpack_transition_output(
            transition_output
        )
        active_mask = jnp.asarray(np.asarray(batch.active_mask), dtype=jnp.bool_)
        (
            padded_actions,
            padded_q_alpha,
            valid_counts,
        ) = _padded_valid_actions_and_q_from_mask(
            child_state_batch.legal_action_mask,
            alpha_q,
            child_state_batch.terminated,
            active_mask,
        )
        if profile_search:
            profile_times["transition_dispatch"] += time.perf_counter() - start
        start = time.perf_counter() if profile_search else 0.0
        (
            alpha_v,
            padded_actions,
            padded_q_alpha,
            valid_counts,
            terminated,
            current_players,
            child_observations,
            child_rewards,
        ) = jax.device_get(
            (
                alpha_v,
                padded_actions,
                padded_q_alpha,
                valid_counts,
                child_state_batch.terminated,
                child_state_batch.current_player,
                child_state_batch.observation,
                child_state_batch.rewards,
            )
        )
        if profile_search:
            profile_times["device_get"] += time.perf_counter() - start
        terminated = np.asarray(terminated[:active_size], dtype=bool)
        current_players = np.asarray(current_players[:active_size], dtype=np.int32)
        start = time.perf_counter() if profile_search else 0.0
        offsets, legal_actions, q_alpha = _flatten_padded_valid_actions_and_q_np(
            padded_actions[:active_size],
            padded_q_alpha[:active_size],
            valid_counts[:active_size],
        )
        if profile_search:
            profile_times["flatten"] += time.perf_counter() - start
        start = time.perf_counter() if profile_search else 0.0
        child_handles = state_store.add_transitions(
            parent_handles[:active_size],
            action_array[:active_size],
        )
        if profile_search:
            profile_times["state_store_add"] += time.perf_counter() - start
        child_observations_np = np.asarray(child_observations[:active_size], dtype=np.float32)
        start = time.perf_counter() if profile_search else 0.0
        terminal_alpha = _terminal_alpha_from_arrays(
            np.asarray(child_rewards[:active_size], dtype=np.float32),
            current_players,
            epsilon=float(config.search.constants.epsilon_terminal),
            kappa=float(config.search.constants.kappa_terminal),
        )
        if profile_search:
            profile_times["terminal_alpha"] += time.perf_counter() - start
        value_alpha_np = np.asarray(alpha_v[:active_size], dtype=np.float32)
        if use_jax_backup:
            start = time.perf_counter() if profile_search else 0.0
            backup_batch = engine.submit_transitions_jax_prepare(
                batch.token,
                child_handles,
                child_observations_np,
                offsets,
                legal_actions,
                current_players,
                terminated,
                terminal_alpha,
                empty_policy_logits,
                value_alpha_np,
                q_alpha,
            )
            if profile_search:
                profile_times["submit_prepare"] += time.perf_counter() - start
            if bool(backup_batch.used_jax):
                if profile_search:
                    profile_paths += int(backup_batch.path_count)
                    profile_nodes += int(backup_batch.node_count)
                start = time.perf_counter() if profile_search else 0.0
                backup_inputs = _pad_dqaz_backup_inputs(
                    backup_batch,
                    root_batch_size=batch_size,
                    eval_batch_size=eval_batch_size,
                    num_simulations=int(config.search.num_simulations),
                )
                if profile_search:
                    profile_times["pad_backup"] += time.perf_counter() - start
                    shape_key = (
                        int(backup_inputs.edge_b.shape[0]),
                        int(backup_inputs.edge_b.shape[1]),
                        int(backup_inputs.edge_b.shape[2]),
                        int(backup_batch.max_depth),
                    )
                    profile_shapes[shape_key] = profile_shapes.get(shape_key, 0) + 1
                backup_key = jax.random.fold_in(search_key, jax_backup_step)
                jax_backup_step += 1
                start = time.perf_counter() if profile_search else 0.0
                backup = _apply_dqaz_jax_backup(
                    backup_key,
                    jnp.asarray(backup_inputs.edge_b, dtype=jnp.float32),
                    jnp.asarray(backup_inputs.edge_completed, dtype=jnp.bool_),
                    jnp.asarray(backup_inputs.edge_r_count, dtype=jnp.int32),
                    jnp.asarray(backup_inputs.q_alpha, dtype=jnp.float32),
                    jnp.asarray(backup_inputs.value_alpha, dtype=jnp.float32),
                    jnp.asarray(backup_inputs.legal_mask, dtype=jnp.bool_),
                    jnp.asarray(backup_inputs.node_players, dtype=jnp.int32),
                    jnp.asarray(backup_inputs.path_slots, dtype=jnp.int32),
                    jnp.asarray(backup_inputs.path_edges, dtype=jnp.int32),
                    jnp.asarray(backup_inputs.path_mask, dtype=jnp.bool_),
                    jnp.asarray(backup_inputs.leaf_alpha, dtype=jnp.float32),
                    jnp.asarray(backup_inputs.leaf_players, dtype=jnp.int32),
                    float(config.search.constants.state_posterior_kappa_n),
                    sample_count=int(config.search.monte_carlo.policy_samples),
                    max_depth=int(backup_batch.max_depth),
                )
                if profile_search:
                    profile_times["jax_backup_dispatch"] += time.perf_counter() - start
                start = time.perf_counter() if profile_search else 0.0
                (
                    edge_b_out,
                    edge_completed_out,
                    edge_r_count_out,
                    c_v_out,
                    n_down_out,
                    policy_out,
                ) = jax.device_get(
                    (
                        backup.edge_b,
                        backup.edge_completed,
                        backup.edge_r_count,
                        backup.c_v,
                        backup.n_down,
                        backup.policy,
                    )
                )
                if profile_search:
                    profile_times["jax_backup_get"] += time.perf_counter() - start
                start = time.perf_counter() if profile_search else 0.0
                engine.apply_jax_backup(
                    backup_inputs.tree_ids,
                    backup_inputs.node_ids,
                    np.asarray(edge_b_out, dtype=np.float32),
                    np.asarray(edge_completed_out, dtype=bool),
                    np.asarray(edge_r_count_out, dtype=np.int32),
                    np.asarray(c_v_out, dtype=np.float32),
                    np.asarray(n_down_out, dtype=np.int32),
                    np.asarray(policy_out, dtype=np.float32),
                    int(backup_batch.node_count),
                )
                if profile_search:
                    profile_times["apply_jax"] += time.perf_counter() - start
        else:
            engine.submit_transitions(
                batch.token,
                child_handles,
                child_observations_np,
                offsets,
                legal_actions,
                current_players,
                terminated,
                terminal_alpha,
                empty_policy_logits,
                value_alpha_np,
                q_alpha,
            )

    if profile_search:
        total = sum(profile_times.values())
        fields = " ".join(
            f"{name}={value:.6f}s" for name, value in profile_times.items()
        )
        shape_fields = ",".join(
            f"{root}x{depth}x{width}/m{max_depth}:{count}"
            for (root, depth, width, max_depth), count in sorted(profile_shapes.items())
        )
        print(
            "DQAZ_PROFILE_SEARCH "
            f"roots={batch_size} waves={profile_waves} active={profile_active} "
            f"paths={profile_paths} nodes={profile_nodes} total={total:.6f}s {fields}",
            f"shapes={shape_fields}",
            flush=True,
        )

    commit = config.selfplay.action_source
    if commit in ("posterior_best", "search_action"):
        commit = "posterior_argmax"
    results = engine.finish(tree_ids, commit=commit)
    return _dqaz_output_to_posterior_batch(
        results,
        batch_size=batch_size,
        action_size=action_size,
        device_put_cpu=device_put_cpu,
    )


def _compact_valid_actions_np(
    states: list[Any],
    policy_logits: Any,
    alpha_q: Any,
    *,
    terminated: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    policy_logits = np.asarray(policy_logits, dtype=np.float32)
    alpha_q = np.asarray(alpha_q, dtype=np.float32)
    if terminated is None:
        terminated = np.zeros((len(states),), dtype=bool)
    offsets = [0]
    legal_actions: list[int] = []
    compact_logits: list[float] = []
    compact_q: list[np.ndarray] = []
    for row, state in enumerate(states):
        if bool(terminated[row]):
            offsets.append(len(legal_actions))
            continue
        legal = np.flatnonzero(np.asarray(jax.device_get(state.legal_action_mask), dtype=bool))
        legal_actions.extend(int(action) for action in legal)
        compact_logits.extend(float(value) for value in policy_logits[row, legal])
        compact_q.extend(np.asarray(alpha_q[row, legal], dtype=np.float32))
        offsets.append(len(legal_actions))
    if compact_q:
        q_alpha = np.stack(compact_q, axis=0).astype(np.float32)
    else:
        q_alpha = np.zeros((0, alpha_q.shape[-1]), dtype=np.float32)
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(legal_actions, dtype=np.int32),
        np.asarray(compact_logits, dtype=np.float32),
        q_alpha,
    )


def _compact_valid_actions_from_mask_np(
    legal_action_mask: np.ndarray,
    policy_logits: Any,
    alpha_q: Any,
    *,
    terminated: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    policy_logits = np.asarray(policy_logits, dtype=np.float32)
    alpha_q = np.asarray(alpha_q, dtype=np.float32)
    legal_action_mask = np.asarray(legal_action_mask, dtype=bool)
    if terminated is None:
        terminated = np.zeros((legal_action_mask.shape[0],), dtype=bool)
    offsets = [0]
    legal_actions: list[int] = []
    compact_logits: list[float] = []
    compact_q: list[np.ndarray] = []
    for row in range(legal_action_mask.shape[0]):
        if bool(terminated[row]):
            offsets.append(len(legal_actions))
            continue
        legal = np.flatnonzero(legal_action_mask[row])
        legal_actions.extend(int(action) for action in legal)
        compact_logits.extend(float(value) for value in policy_logits[row, legal])
        compact_q.extend(np.asarray(alpha_q[row, legal], dtype=np.float32))
        offsets.append(len(legal_actions))
    if compact_q:
        q_alpha = np.stack(compact_q, axis=0).astype(np.float32)
    else:
        q_alpha = np.zeros((0, alpha_q.shape[-1]), dtype=np.float32)
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(legal_actions, dtype=np.int32),
        np.asarray(compact_logits, dtype=np.float32),
        q_alpha,
    )


def _compact_valid_actions_and_q_from_mask_np(
    legal_action_mask: np.ndarray,
    alpha_q: Any,
    *,
    terminated: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    alpha_q = np.asarray(alpha_q, dtype=np.float32)
    legal_action_mask = np.asarray(legal_action_mask, dtype=bool)
    if terminated is None:
        terminated = np.zeros((legal_action_mask.shape[0],), dtype=bool)
    offsets = [0]
    legal_actions: list[int] = []
    compact_q: list[np.ndarray] = []
    for row in range(legal_action_mask.shape[0]):
        if bool(terminated[row]):
            offsets.append(len(legal_actions))
            continue
        legal = np.flatnonzero(legal_action_mask[row])
        legal_actions.extend(int(action) for action in legal)
        compact_q.extend(np.asarray(alpha_q[row, legal], dtype=np.float32))
        offsets.append(len(legal_actions))
    if compact_q:
        q_alpha = np.stack(compact_q, axis=0).astype(np.float32)
    else:
        q_alpha = np.zeros((0, alpha_q.shape[-1]), dtype=np.float32)
    return (
        np.asarray(offsets, dtype=np.int64),
        np.asarray(legal_actions, dtype=np.int32),
        q_alpha,
    )


@jax.jit
def _padded_valid_actions_from_mask(
    legal_action_mask: jax.Array,
    policy_logits: jax.Array,
    alpha_q: jax.Array,
    terminated: jax.Array,
    active_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    action_size = legal_action_mask.shape[-1]
    action_ids = jnp.arange(action_size, dtype=jnp.int32)
    valid_mask = (
        legal_action_mask
        & active_mask[:, None]
        & (~terminated)[:, None]
    )
    rank = jnp.where(valid_mask, action_ids[None, :], action_size)
    order = jnp.argsort(rank, axis=-1, stable=True)
    action_table = jnp.broadcast_to(action_ids[None, :], legal_action_mask.shape)
    return (
        jnp.take_along_axis(action_table, order, axis=-1),
        jnp.take_along_axis(policy_logits, order, axis=-1),
        jnp.take_along_axis(alpha_q, order[..., None], axis=-2),
        jnp.sum(valid_mask, axis=-1, dtype=jnp.int32),
    )


@jax.jit
def _padded_valid_actions_and_q_from_mask(
    legal_action_mask: jax.Array,
    alpha_q: jax.Array,
    terminated: jax.Array,
    active_mask: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    action_size = legal_action_mask.shape[-1]
    action_ids = jnp.arange(action_size, dtype=jnp.int32)
    valid_mask = (
        legal_action_mask
        & active_mask[:, None]
        & (~terminated)[:, None]
    )
    rank = jnp.where(valid_mask, action_ids[None, :], action_size)
    order = jnp.argsort(rank, axis=-1, stable=True)
    action_table = jnp.broadcast_to(action_ids[None, :], legal_action_mask.shape)
    return (
        jnp.take_along_axis(action_table, order, axis=-1),
        jnp.take_along_axis(alpha_q, order[..., None], axis=-2),
        jnp.sum(valid_mask, axis=-1, dtype=jnp.int32),
    )


def _flatten_padded_valid_actions_np(
    padded_actions: Any,
    padded_policy_logits: Any,
    padded_q_alpha: Any,
    valid_counts: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    padded_actions = np.asarray(padded_actions, dtype=np.int32)
    padded_policy_logits = np.asarray(padded_policy_logits, dtype=np.float32)
    padded_q_alpha = np.asarray(padded_q_alpha, dtype=np.float32)
    valid_counts = np.asarray(valid_counts, dtype=np.int64)

    offsets = np.empty((valid_counts.shape[0] + 1,), dtype=np.int64)
    offsets[0] = 0
    np.cumsum(valid_counts, out=offsets[1:])

    action_slots = np.arange(padded_actions.shape[1])[None, :]
    valid = action_slots < valid_counts[:, None]
    return (
        offsets,
        padded_actions[valid].astype(np.int32, copy=False),
        padded_policy_logits[valid].astype(np.float32, copy=False),
        padded_q_alpha[valid].astype(np.float32, copy=False),
    )


def _flatten_padded_valid_actions_and_q_np(
    padded_actions: Any,
    padded_q_alpha: Any,
    valid_counts: Any,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    padded_actions = np.asarray(padded_actions, dtype=np.int32)
    padded_q_alpha = np.asarray(padded_q_alpha, dtype=np.float32)
    valid_counts = np.asarray(valid_counts, dtype=np.int64)

    offsets = np.empty((valid_counts.shape[0] + 1,), dtype=np.int64)
    offsets[0] = 0
    np.cumsum(valid_counts, out=offsets[1:])

    action_slots = np.arange(padded_actions.shape[1])[None, :]
    valid = action_slots < valid_counts[:, None]
    return (
        offsets,
        padded_actions[valid].astype(np.int32, copy=False),
        padded_q_alpha[valid].astype(np.float32, copy=False),
    )


class _StateStore:
    def __init__(self):
        self._treedef = None
        self._states: list[Any] = []
        self._size = 0

    def add_batch(self, state_batch: Any) -> list[int]:
        leaves, treedef = jax.tree_util.tree_flatten(state_batch)
        batch_size = int(leaves[0].shape[0])
        if self._treedef is None:
            self._treedef = treedef
        elif treedef != self._treedef:
            raise ValueError("state batch tree structure changed")
        start = self._size
        self._states.extend(split_batched_state(state_batch))
        self._size += batch_size
        return list(range(start, start + batch_size))

    def batch(self, handles: list[int]) -> Any:
        if not handles:
            raise ValueError("state handle batch must not be empty")
        return _stack_states([self._states[handle] for handle in handles])


class _PathStateStore:
    def __init__(self, env: Any, root_state_batch: Any):
        self._root_state_batch = root_state_batch
        self._root_indices: list[int] = []
        self._paths: list[tuple[int, ...]] = []
        self._replay = _path_state_replay(env)

    def add_roots(self, count: int) -> list[int]:
        start = len(self._paths)
        self._root_indices.extend(range(count))
        self._paths.extend(() for _ in range(count))
        return list(range(start, start + count))

    def add_transitions(self, parent_handles: Any, actions: np.ndarray) -> list[int]:
        parent_handles = np.asarray(parent_handles, dtype=np.int64)
        start = len(self._paths)
        for parent_handle, action in zip(parent_handles, actions, strict=True):
            self._root_indices.append(self._root_indices[int(parent_handle)])
            self._paths.append((*self._paths[int(parent_handle)], int(action)))
        return list(range(start, len(self._paths)))

    def batch(self, handles: Any) -> Any:
        handles = np.asarray(handles, dtype=np.int64)
        if len(handles) == 0:
            raise ValueError("state handle batch must not be empty")
        paths = [self._paths[int(handle)] for handle in handles]
        max_depth = max((len(path) for path in paths), default=0)
        replay_depth = (
            (max_depth + _DQAZ_STATE_REPLAY_BLOCK_DEPTH - 1)
            // _DQAZ_STATE_REPLAY_BLOCK_DEPTH
            * _DQAZ_STATE_REPLAY_BLOCK_DEPTH
        )
        dense_paths = np.zeros((len(paths), replay_depth), dtype=np.int32)
        lengths = np.empty((len(paths),), dtype=np.int32)
        root_indices = np.empty((len(paths),), dtype=np.int32)
        for row, (handle, path) in enumerate(zip(handles, paths, strict=True)):
            root_indices[row] = self._root_indices[int(handle)]
            lengths[row] = len(path)
            dense_paths[row, : len(path)] = path
        if max_depth == 0:
            index = jnp.asarray(root_indices, dtype=jnp.int32)
            return jax.tree_util.tree_map(lambda x: x[index], self._root_state_batch)
        return self._replay(
            self._root_state_batch,
            jnp.asarray(root_indices, dtype=jnp.int32),
            jnp.asarray(dense_paths, dtype=jnp.int32),
            jnp.asarray(lengths, dtype=jnp.int32),
        )


def _terminal_alpha_from_state_batch(
    states: Any,
    *,
    epsilon: float,
    kappa: float,
) -> np.ndarray:
    rewards = np.asarray(jax.device_get(states.rewards), dtype=np.float32)
    current_players = np.asarray(jax.device_get(states.current_player), dtype=np.int32)
    return _terminal_alpha_from_arrays(
        rewards,
        current_players,
        epsilon=epsilon,
        kappa=kappa,
    )


def _terminal_alpha_from_arrays(
    rewards: np.ndarray,
    current_players: np.ndarray,
    *,
    epsilon: float,
    kappa: float,
) -> np.ndarray:
    reward = rewards[np.arange(rewards.shape[0]), current_players]
    outcome = np.where(reward > 0.0, 2, np.where(reward < 0.0, 0, 1))
    alpha = np.full((rewards.shape[0], 3), float(epsilon), dtype=np.float32)
    alpha[np.arange(rewards.shape[0]), outcome] += np.float32(kappa)
    return alpha


def _dqaz_output_to_posterior_batch(
    results: Any,
    *,
    batch_size: int,
    action_size: int,
    device_put_cpu: Callable[[Any], Any],
) -> PosteriorTreeBatchOutput:
    action_offsets = np.asarray(results.action_offsets, dtype=np.int64)
    legal_actions = np.asarray(results.legal_actions, dtype=np.int32)
    action_weights = np.zeros((batch_size, action_size), dtype=np.float32)
    beta_q = np.zeros((batch_size, action_size, 3), dtype=np.float32)
    q_loss_weight = np.zeros((batch_size, action_size), dtype=np.float32)
    q_kind = np.full((batch_size, action_size), int(TARGET_PAD), dtype=np.int8)
    q_weight = np.zeros((batch_size, action_size), dtype=np.float32)
    q_outcome = np.full((batch_size, action_size), int(NO_OUTCOME), dtype=np.int8)
    q_distance = np.full((batch_size, action_size), int(NO_DISTANCE), dtype=np.int32)

    sparse_policy = np.asarray(results.pi_search, dtype=np.float32)
    sparse_q = np.asarray(results.root_alpha, dtype=np.float32)
    sparse_q_kind = np.asarray(results.q_target_kind, dtype=np.int8)
    sparse_q_weight = np.asarray(results.q_target_weight, dtype=np.float32)
    sparse_q_outcome = np.asarray(results.q_target_outcome, dtype=np.int8)
    sparse_q_distance = np.asarray(results.q_target_distance, dtype=np.int32)
    for row in range(batch_size):
        start = int(action_offsets[row])
        end = int(action_offsets[row + 1])
        actions = legal_actions[start:end]
        action_weights[row, actions] = sparse_policy[start:end]
        beta_q[row, actions] = sparse_q[start:end]
        q_loss_weight[row, actions] = sparse_policy[start:end]
        q_kind[row, actions] = sparse_q_kind[start:end]
        q_weight[row, actions] = sparse_q_weight[start:end]
        q_outcome[row, actions] = sparse_q_outcome[start:end]
        q_distance[row, actions] = sparse_q_distance[start:end]

    policy_array = np.asarray(action_weights, dtype=np.float32)
    return PosteriorTreeBatchOutput(
        action=device_put_cpu(jnp.asarray(np.asarray(results.actions), dtype=jnp.int32)),
        action_weights=device_put_cpu(jnp.asarray(policy_array)),
        beta_Q_target=device_put_cpu(jnp.asarray(beta_q)),
        beta_V_target=device_put_cpu(
            jnp.asarray(np.asarray(results.beta_v, dtype=np.float32))
        ),
        q_loss_weight=device_put_cpu(jnp.asarray(q_loss_weight)),
        alpha_root=device_put_cpu(jnp.asarray(beta_q)),
        trees=(),
        tree_data=None,
        search_loss_mask=device_put_cpu(jnp.asarray(np.sum(policy_array, axis=-1) > 0.0)),
        q_target_kind=device_put_cpu(jnp.asarray(q_kind, dtype=jnp.int8)),
        q_target_weight=device_put_cpu(jnp.asarray(q_weight, dtype=jnp.float32)),
        q_target_outcome=device_put_cpu(jnp.asarray(q_outcome, dtype=jnp.int8)),
        q_target_distance=device_put_cpu(jnp.asarray(q_distance, dtype=jnp.int32)),
        v_target_kind=device_put_cpu(
            jnp.asarray(np.asarray(results.v_target_kind), dtype=jnp.int8)
        ),
        v_target_weight=device_put_cpu(
            jnp.asarray(np.asarray(results.v_target_weight), dtype=jnp.float32)
        ),
        v_target_outcome=device_put_cpu(
            jnp.asarray(np.asarray(results.v_target_outcome), dtype=jnp.int8)
        ),
        v_target_distance=device_put_cpu(
            jnp.asarray(np.asarray(results.v_target_distance), dtype=jnp.int32)
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


def _select_active_states(
    stepped_state: Any,
    original_state: Any,
    active_mask: jax.Array,
) -> Any:
    def select_leaf(stepped_leaf: jax.Array, original_leaf: jax.Array) -> jax.Array:
        mask = jnp.reshape(
            active_mask,
            active_mask.shape + (1,) * (stepped_leaf.ndim - 1),
        )
        return jnp.where(mask, stepped_leaf, original_leaf)

    return jax.tree_util.tree_map(select_leaf, stepped_state, original_state)


def _eval_batch_size(config: Any, num_trees: int) -> int:
    configured = config.search.eval_batch_size
    if configured is None:
        return max(1, num_trees)
    return max(1, int(configured))
