from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from .play import make_dirichlet_recurrent_fn, make_recurrent_fn
from .play_search import _run_model_eval_search, _run_posterior_tree_search_step
from .posterior_tree import is_posterior_tree_policy


def _predict(model: Any, obs: jax.Array):
    if isinstance(model, nnx.Module):
        return model(obs, train=False)
    return model(obs)


def _with_eval_num_simulations(config, num_simulations: int | None):
    if num_simulations is None or num_simulations == config.num_simulations:
        return config
    if hasattr(config, "model_copy"):
        return config.model_copy(update={"num_simulations": num_simulations})
    from types import SimpleNamespace

    values = dict(vars(config))
    values["num_simulations"] = num_simulations
    return SimpleNamespace(**values)


def _replace_legal_action_mask(env_state, legal_action_mask: jax.Array):
    if hasattr(env_state, "replace"):
        return env_state.replace(legal_action_mask=legal_action_mask)
    if hasattr(env_state, "_replace"):
        return env_state._replace(legal_action_mask=legal_action_mask)
    raise TypeError("env_state must support replacing legal_action_mask")


def _searchable_eval_state(env_state):
    """Give completed eval rows a dummy legal action; their moves are discarded."""

    dummy_legal_action_mask = (
        jnp.zeros_like(env_state.legal_action_mask)
        .at[..., 0]
        .set(True)
    )
    legal_action_mask = jnp.where(
        env_state.terminated[..., None],
        dummy_legal_action_mask,
        env_state.legal_action_mask,
    )
    return _replace_legal_action_mask(env_state, legal_action_mask)


def _step_active_eval_rows(env, env_state, action: jax.Array):
    active = ~env_state.terminated
    action = jnp.asarray(action, dtype=jnp.int32)
    num_actions = env_state.legal_action_mask.shape[-1]
    in_bounds = (0 <= action) & (action < num_actions)
    safe_action = jnp.clip(action, 0, num_actions - 1)
    selected_is_legal = jnp.take_along_axis(
        env_state.legal_action_mask,
        safe_action[..., None],
        axis=-1,
    )[..., 0]
    invalid_action = jnp.any(active & ~(in_bounds & selected_is_legal))

    def step_one(state, row_action, row_active):
        def step_state(state):
            return env.step(state, row_action)

        should_step = row_active & ~invalid_action
        return jax.lax.cond(should_step, step_state, lambda state: state, state)

    return jax.vmap(step_one)(env_state, action, active), active, invalid_action


def _poison_eval_returns(returns: jax.Array, invalid_action: jax.Array) -> jax.Array:
    return jnp.where(invalid_action, jnp.full_like(returns, jnp.nan), returns)


def _active_eval_state(env_state):
    active_indices = np.flatnonzero(
        np.asarray(jax.device_get(~env_state.terminated))
    )
    return active_indices, jax.tree_util.tree_map(
        lambda value: value[active_indices],
        env_state,
    )


def _scatter_eval_actions(
    batch_size: int,
    active_indices: np.ndarray,
    active_actions: jax.Array,
) -> jax.Array:
    indices = jnp.asarray(active_indices, dtype=jnp.int32)
    actions = jnp.zeros((batch_size,), dtype=jnp.int32)
    return actions.at[indices].set(active_actions)


def _make_model_mcts_policy(
    env,
    config,
    model,
    rng_key,
    env_state,
    num_simulations=None,
):
    search_config = _with_eval_num_simulations(config, num_simulations)
    predict = lambda obs: _predict(model, obs)
    search_state = _searchable_eval_state(env_state)
    model_output = predict(search_state.observation)
    return _run_model_eval_search(
        env_state=search_state,
        model_output=model_output,
        scalar_recurrent_fn=make_recurrent_fn(env, predict),
        dirichlet_recurrent_fn=make_dirichlet_recurrent_fn(env, predict, search_config),
        rng_key=rng_key,
        config=search_config,
    )


def _model_eval_action(env, config, model, rng_key, env_state):
    action = _make_model_mcts_policy(
        env,
        config,
        model,
        rng_key,
        env_state,
    ).played_action
    return action


def _posterior_tree_eval_action(
    *,
    env,
    config,
    env_state,
    leaf_evaluator,
    transition_evaluator=None,
    rng_key: jax.Array,
) -> jax.Array:
    return _run_posterior_tree_search_step(
        env=env,
        config=config,
        env_state=env_state,
        leaf_evaluator=leaf_evaluator,
        transition_evaluator=transition_evaluator,
        search_key=rng_key,
        device_put_cpu=lambda value: value,
    ).played_action


def make_mcts_evaluate(env, config, baseline_model):
    search_config = config
    eval_batch_size = int(
        getattr(config, "eval_batch_size", config.selfplay_batch_size)
    )

    if is_posterior_tree_policy(getattr(config, "search_policy", "gumbel")):
        @nnx.jit
        def evaluate_leaves(model: Any, obs: jax.Array):
            return _predict(model, obs)

        @nnx.jit
        def evaluate_transitions(model: Any, states, actions: jax.Array):
            child_states = jax.vmap(env.step)(states, actions)
            return child_states, _predict(model, child_states.observation)

        @nnx.jit
        def scalar_mcts_actions(model: Any, rng_key: jax.Array, env_state):
            return _model_eval_action(env, search_config, model, rng_key, env_state)

        def model_actions(model: Any, rng_key: jax.Array, env_state):
            active_indices, active_state = _active_eval_state(env_state)
            if active_indices.size == 0:
                return jnp.zeros((eval_batch_size,), dtype=jnp.int32)

            sample_output = evaluate_leaves(
                model,
                active_state.observation[:1],
            )
            if len(sample_output) != 3:
                active_action = scalar_mcts_actions(model, rng_key, active_state)
                return _scatter_eval_actions(
                    eval_batch_size,
                    active_indices,
                    active_action,
                )

            def leaf_evaluator(obs: jax.Array):
                return evaluate_leaves(model, obs)

            def transition_evaluator(states, actions: jax.Array):
                return evaluate_transitions(model, states, actions)

            action = _posterior_tree_eval_action(
                env=env,
                config=search_config,
                env_state=active_state,
                leaf_evaluator=leaf_evaluator,
                transition_evaluator=transition_evaluator,
                rng_key=rng_key,
            )
            return _scatter_eval_actions(
                eval_batch_size,
                active_indices,
                action,
            )

        def evaluate(rng_key: jax.Array, model: Any):
            my_player = 0
            key, init_key = jax.random.split(rng_key)
            init_keys = jax.random.split(init_key, eval_batch_size)
            env_state = jax.vmap(env.init)(init_keys)
            returns = jnp.zeros(eval_batch_size)

            while (
                not bool(np.asarray(jax.device_get(env_state.terminated)).all())
                and not bool(np.asarray(jax.device_get(jnp.isnan(returns).any())))
            ):
                key, my_key, opp_key = jax.random.split(key, 3)
                my_action = model_actions(model, my_key, env_state)
                opp_action = model_actions(baseline_model, opp_key, env_state)
                action = jnp.where(
                    env_state.current_player == my_player,
                    my_action,
                    opp_action,
                )
                env_state, active, invalid_action = _step_active_eval_rows(
                    env,
                    env_state,
                    action,
                )
                reward = env_state.rewards[
                    jnp.arange(eval_batch_size),
                    my_player,
                ]
                returns = returns + jnp.where(active, reward, 0.0)
                returns = _poison_eval_returns(returns, invalid_action)
            return returns

        return evaluate

    @nnx.jit
    def evaluate(rng_key: jax.Array, model: nnx.Module):
        """MCTS evaluation: model search vs pretrained opponent."""
        my_player = 0

        key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, eval_batch_size)
        env_state = jax.vmap(env.init)(init_keys)

        def body_fn(val):
            key, env_state, returns = val
            key, my_key, opp_key = jax.random.split(key, 3)

            my_action = _model_eval_action(env, config, model, my_key, env_state)
            opp_action = _model_eval_action(
                env,
                config,
                baseline_model,
                opp_key,
                env_state,
            )

            is_my_turn = env_state.current_player == my_player
            action = jnp.where(is_my_turn, my_action, opp_action)

            env_state, active, invalid_action = _step_active_eval_rows(
                env,
                env_state,
                action,
            )
            reward = env_state.rewards[
                jnp.arange(eval_batch_size),
                my_player,
            ]
            returns = returns + jnp.where(active, reward, 0.0)
            returns = _poison_eval_returns(returns, invalid_action)
            return key, env_state, returns

        _, _, returns = nnx.while_loop(
            lambda x: ~(x[1].terminated.all()) & ~(jnp.isnan(x[2]).any()),
            body_fn,
            (key, env_state, jnp.zeros(eval_batch_size)),
        )
        return returns

    return evaluate
