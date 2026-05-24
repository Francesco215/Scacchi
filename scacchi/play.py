import weakref
from typing import Any, Callable, NamedTuple

import chex
from flax import nnx
import jax
import jax.numpy as jnp
import mctx
import pgx
from pgx.experimental import auto_reset

from .dirichlet_q_search import (
    NO_PARENT,
    NodeEmbedding,
    flip_outcome,
    outcome_mean,
    outcome_utility,
    dirichlet_q_policy,
    posterior_best_action,
    posterior_best_policy_target,
    posterior_sample_action,
    posterior_targets,
    q_evidence_sum_from_tree,
    root_action_value_priors_from_tree,
    terminal_outcome_from_reward,
)
from .dirichlet_tree.types import TreeTrainingData
from .network import policy_value_from_output
from .posterior_tree import (
    is_posterior_tree_policy,
    run_posterior_tree_search,
    run_posterior_tree_search_state_batch,
    split_batched_state,
)


BatchedEnvInit = Callable[[jax.Array], Any]
BatchedEnvStep = Callable[[Any, jax.Array, jax.Array], Any]

_CPU_ENV_INIT_CACHE: dict[int, tuple[weakref.ReferenceType[Any] | None, BatchedEnvInit]] = {}
_CPU_ENV_STEP_CACHE: dict[int, tuple[weakref.ReferenceType[Any] | None, BatchedEnvStep]] = {}


class SelfplayOutput(NamedTuple):
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    action_weights: chex.Array
    played_action: jax.Array
    legal_action_mask: jax.Array
    beta_Q_target: jax.Array
    beta_V_target: jax.Array
    q_loss_weight: jax.Array
    discount: jax.Array
    tree_data: TreeTrainingData | None = None
    search_loss_mask: jax.Array | None = None

    @property
    def q_evidence_mass(self) -> jax.Array:
        return self.q_loss_weight


def _cpu_device() -> jax.Device:
    try:
        return jax.devices("cpu")[0]
    except RuntimeError as exc:
        raise RuntimeError(
            "posterior_tree selfplay requires the JAX CPU platform for PGX env "
            "initialization and stepping. Use JAX_PLATFORMS=cuda,cpu when running "
            "with a GPU."
        ) from exc


def _device_put_cpu(value: Any) -> Any:
    return jax.device_put(value, _cpu_device())


def _env_ref(env: Any) -> weakref.ReferenceType[Any] | None:
    try:
        return weakref.ref(env)
    except TypeError:
        return None


def _cached_cpu_env_init(env: Any) -> BatchedEnvInit:
    cache_key = id(env)
    cached = _CPU_ENV_INIT_CACHE.get(cache_key)
    if cached is not None:
        env_ref, init_fn = cached
        if env_ref is None or env_ref() is env:
            return init_fn
    init_fn = jax.jit(jax.vmap(env.init))
    _CPU_ENV_INIT_CACHE[cache_key] = (_env_ref(env), init_fn)
    return init_fn


def _cached_cpu_env_step(env: Any) -> BatchedEnvStep:
    cache_key = id(env)
    cached = _CPU_ENV_STEP_CACHE.get(cache_key)
    if cached is not None:
        env_ref, step_fn = cached
        if env_ref is None or env_ref() is env:
            return step_fn
    step_fn = jax.jit(jax.vmap(auto_reset(env.step, env.init)))
    _CPU_ENV_STEP_CACHE[cache_key] = (_env_ref(env), step_fn)
    return step_fn


def _cached_default_env_init(env: Any) -> BatchedEnvInit:
    return _cached_cpu_env_init(env)


def _cached_default_env_step(env: Any) -> BatchedEnvStep:
    return _cached_cpu_env_step(env)


def make_recurrent_fn(env, predict_fn):
    def recurrent_fn(
        _,
        rng_key: chex.PRNGKey,
        action: chex.Array,
        env_state: pgx.State,
    ):
        del rng_key

        current_player = env_state.current_player
        env_state = jax.vmap(env.step)(env_state, action)
        logits, value = policy_value_from_output(predict_fn(env_state.observation))
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(
            env_state.legal_action_mask,
            logits,
            jnp.finfo(logits.dtype).min,
        )

        reward = env_state.rewards[
            jnp.arange(env_state.rewards.shape[0]),
            current_player,
        ]
        value = jnp.where(env_state.terminated, 0.0, value)
        discount = -jnp.ones_like(value)
        discount = jnp.where(env_state.terminated, 0.0, discount)

        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=logits,
                value=value,
            ),
            env_state,
        )

    return recurrent_fn


def make_dirichlet_recurrent_fn(env, predict_fn, config):
    def recurrent_fn(
        _,
        rng_key: chex.PRNGKey,
        action: chex.Array,
        embedding: NodeEmbedding,
    ):
        del rng_key

        current_player = embedding.state.current_player
        env_state = jax.vmap(env.step)(embedding.state, action)
        logits, alpha_v, alpha_q = predict_fn(env_state.observation)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(
            env_state.legal_action_mask,
            logits,
            jnp.finfo(logits.dtype).min,
        )

        reward = env_state.rewards[
            jnp.arange(env_state.rewards.shape[0]),
            current_player,
        ]
        nonterminal_outcome = outcome_mean(alpha_v)
        terminal_parent_outcome = terminal_outcome_from_reward(
            reward,
            alpha_v.shape[-1],
        )
        terminal_child_outcome = flip_outcome(terminal_parent_outcome)
        outcome_dist = jnp.where(
            env_state.terminated[..., None],
            terminal_child_outcome,
            nonterminal_outcome,
        )
        evidence_weight = jnp.where(
            env_state.terminated,
            jnp.asarray(config.kappa_terminal, dtype=outcome_dist.dtype),
            jnp.asarray(config.kappa_leaf, dtype=outcome_dist.dtype),
        )
        root_action = jnp.where(
            embedding.root_action == NO_PARENT,
            action,
            embedding.root_action,
        )
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
        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=logits,
                value=value,
            ),
            next_embedding,
        )

    return recurrent_fn


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


def _select_posterior_tree_played_action(
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
    if action_source in ("search_action", "scalar_q_argmax"):
        return search_action
    raise ValueError(f"unknown selfplay_action_source: {action_source!r}")


def make_posterior_tree_selfplay(env, config):
    @nnx.jit
    def evaluate_leaves(model: nnx.Module, obs: jax.Array):
        return model(obs, train=False)

    def selfplay(model: nnx.Module, rng_key: jax.Array) -> SelfplayOutput:
        def leaf_evaluator(obs: jax.Array):
            output = evaluate_leaves(model, obs)
            if len(output) != 3:
                raise ValueError(
                    "posterior-tree search requires a Dirichlet model "
                    "returning (logits, alpha_V, alpha_Q)."
                )
            return output

        use_wavefront_arena = config.search_policy == "posterior_tree_wavefront"
        rng_key, init_key = jax.random.split(rng_key)
        if use_wavefront_arena:
            init_keys = jax.random.split(init_key, config.selfplay_batch_size)
            env_init = _cached_default_env_init(env)
            env_step = _cached_default_env_step(env)
        else:
            init_keys = _device_put_cpu(jax.random.split(init_key, config.selfplay_batch_size))
            env_init = _cached_cpu_env_init(env)
            env_step = _cached_cpu_env_step(env)
        env_state = env_init(init_keys)

        obs_seq = []
        reward_seq = []
        terminated_seq = []
        action_weights_seq = []
        played_action_seq = []
        legal_action_mask_seq = []
        beta_q_seq = []
        beta_v_seq = []
        q_loss_weight_seq = []
        search_loss_mask_seq = []
        discount_seq = []
        tree_data_seq = []

        for _ in range(config.max_num_steps):
            rng_key, search_key, action_key, reset_key = jax.random.split(rng_key, 4)
            observation = env_state.observation
            legal_action_mask = env_state.legal_action_mask
            actor = env_state.current_player
            if use_wavefront_arena:
                search_output = run_posterior_tree_search_state_batch(
                    env=env,
                    root_state_batch=env_state,
                    leaf_evaluator=leaf_evaluator,
                    rng_key=search_key,
                    config=config,
                )
            else:
                root_states = split_batched_state(env_state)
                search_output = run_posterior_tree_search(
                    env=env,
                    root_states=root_states,
                    leaf_evaluator=leaf_evaluator,
                    rng_key=search_key,
                    config=config,
                )
            search_action = (
                search_output.action
                if use_wavefront_arena
                else _device_put_cpu(search_output.action)
            )
            played_action = (
                _select_posterior_tree_played_action(
                    config.selfplay_action_source,
                    action_key,
                    search_output.action_weights,
                    legal_action_mask,
                    search_action,
                )
                if use_wavefront_arena
                else search_action
            )

            reset_keys = jax.random.split(reset_key, config.selfplay_batch_size)
            if not use_wavefront_arena:
                reset_keys = _device_put_cpu(reset_keys)
            env_state = env_step(env_state, played_action, reset_keys)
            reward = env_state.rewards[jnp.arange(env_state.rewards.shape[0]), actor]
            discount = -jnp.ones((config.selfplay_batch_size,), dtype=reward.dtype)
            discount = jnp.where(env_state.terminated, 0.0, discount)

            obs_seq.append(observation)
            action_weights_seq.append(search_output.action_weights)
            played_action_seq.append(played_action)
            legal_action_mask_seq.append(legal_action_mask)
            beta_q_seq.append(search_output.beta_Q_target)
            beta_v_seq.append(search_output.beta_V_target)
            q_loss_weight_seq.append(search_output.q_loss_weight)
            root_search_mask = search_output.search_loss_mask
            if root_search_mask is None:
                root_search_mask = jnp.sum(search_output.action_weights, axis=-1) > 0
            search_loss_mask_seq.append(root_search_mask)
            if search_output.tree_data is not None:
                tree_data_seq.append(search_output.tree_data)
            reward_seq.append(reward)
            terminated_seq.append(env_state.terminated)
            discount_seq.append(discount)

        tree_data = None
        if tree_data_seq:
            tree_data = jax.tree_util.tree_map(
                lambda *xs: jnp.stack(xs, axis=0),
                *tree_data_seq,
            )

        return SelfplayOutput(
            obs=jnp.stack(obs_seq, axis=0),
            reward=jnp.stack(reward_seq, axis=0),
            terminated=jnp.stack(terminated_seq, axis=0),
            action_weights=jnp.stack(action_weights_seq, axis=0),
            played_action=jnp.stack(played_action_seq, axis=0),
            legal_action_mask=jnp.stack(legal_action_mask_seq, axis=0),
            beta_Q_target=jnp.stack(beta_q_seq, axis=0),
            beta_V_target=jnp.stack(beta_v_seq, axis=0),
            q_loss_weight=jnp.stack(q_loss_weight_seq, axis=0),
            discount=jnp.stack(discount_seq, axis=0),
            tree_data=tree_data,
            search_loss_mask=jnp.stack(search_loss_mask_seq, axis=0),
        )

    return selfplay


def make_selfplay(env, config):
    if is_posterior_tree_policy(config.search_policy):
        return make_posterior_tree_selfplay(env, config)

    @nnx.jit
    def selfplay(model: nnx.Module, rng_key: jax.Array) -> SelfplayOutput:
        predict_fn = lambda obs: model(obs, train=False)
        recurrent_fn = make_recurrent_fn(env, predict_fn)
        dirichlet_recurrent_fn = make_dirichlet_recurrent_fn(env, predict_fn, config)

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def step_fn(
            env_state: pgx.State,
            key: jax.Array,
        ) -> tuple[pgx.State, SelfplayOutput]:
            search_key, posterior_key, action_key, reset_key = jax.random.split(key, 4)
            observation = env_state.observation
            legal_action_mask = env_state.legal_action_mask
            model_output = predict_fn(observation)

            if len(model_output) == 2:
                logits, value = policy_value_from_output(model_output)
                root = mctx.RootFnOutput(
                    prior_logits=logits,
                    value=value,
                    embedding=env_state,
                )
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
                policy_target = policy_output.action_weights
                played_action = policy_output.action
                num_outcomes = config.num_outcomes
                if num_outcomes is None:
                    num_outcomes = 2 if config.env_id == "hex" else 3
                beta_Q_target, beta_V_target, q_loss_weight = (
                    _empty_posterior_targets(policy_target, num_outcomes)
                )
            else:
                logits, alpha_v, alpha_q = model_output
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
                root = mctx.RootFnOutput(
                    prior_logits=logits,
                    value=value,
                    embedding=root_embedding,
                )
                action_value_prior = alpha_q
                if config.search_policy == "dirichlet_thompson":
                    policy_output = dirichlet_q_policy(
                        params=(),
                        rng_key=search_key,
                        root=root,
                        recurrent_fn=dirichlet_recurrent_fn,
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
                        recurrent_fn=dirichlet_recurrent_fn,
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
                policy_target = posterior_best_policy_target(
                    posterior_key,
                    action_alpha_post,
                    legal_action_mask,
                    config.policy_mc_samples,
                )
                beta_Q_target, beta_V_target = (
                    posterior_targets(
                        alpha_v,
                        action_value_target_prior,
                        q_evidence_sum,
                        policy_target,
                    )
                )
                q_loss_weight = policy_target
                if config.selfplay_action_source in ("posterior_best", "posterior_argmax"):
                    posterior_action = posterior_best_action(
                        policy_target,
                        legal_action_mask,
                    )
                    played_action = posterior_action
                elif config.selfplay_action_source == "posterior_sample":
                    posterior_action = posterior_sample_action(
                        action_key,
                        policy_target,
                        legal_action_mask,
                    )
                    played_action = posterior_action
                else:
                    played_action = policy_output.action

            actor = env_state.current_player
            reset_keys = jax.random.split(reset_key, config.selfplay_batch_size)
            env_state = jax.vmap(auto_reset(env.step, env.init))(
                env_state,
                played_action,
                reset_keys,
            )
            discount = -jnp.ones_like(value)
            discount = jnp.where(env_state.terminated, 0.0, discount)
            return env_state, SelfplayOutput(
                obs=observation,
                action_weights=policy_target,
                played_action=played_action,
                legal_action_mask=legal_action_mask,
                beta_Q_target=beta_Q_target,
                beta_V_target=beta_V_target,
                q_loss_weight=q_loss_weight,
                reward=env_state.rewards[jnp.arange(env_state.rewards.shape[0]), actor],
                terminated=env_state.terminated,
                discount=discount,
                search_loss_mask=jnp.sum(policy_target, axis=-1) > 0,
            )

        rng_key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, config.selfplay_batch_size)
        env_state = jax.vmap(env.init)(init_keys)
        step_keys = jax.random.split(rng_key, config.max_num_steps)
        _, data = step_fn(env_state, step_keys)
        return data

    return selfplay
