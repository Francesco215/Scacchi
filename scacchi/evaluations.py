from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np

from .play import make_dirichlet_recurrent_fn, make_recurrent_fn
from .play_search import (
    _run_model_eval_search,
    _run_posterior_tree_search_step,
)
from .posterior_tree import is_posterior_tree_policy


def _with_eval_num_simulations(config, num_simulations: int | None):
    if num_simulations is None or num_simulations == config.num_simulations:
        return config
    if hasattr(config, "model_copy"):
        return config.model_copy(update={"num_simulations": num_simulations})
    from types import SimpleNamespace

    values = dict(vars(config))
    values["num_simulations"] = num_simulations
    return SimpleNamespace(**values)


def _make_model_mcts_policy(env, config, model, rng_key, env_state, num_simulations=None):
    search_config = _with_eval_num_simulations(config, num_simulations)
    predict = lambda obs: model(obs, train=False)
    model_output = predict(env_state.observation)
    return _run_model_eval_search(
        env_state=env_state,
        model_output=model_output,
        scalar_recurrent_fn=make_recurrent_fn(env, predict),
        dirichlet_recurrent_fn=make_dirichlet_recurrent_fn(env, predict, search_config),
        rng_key=rng_key,
        config=search_config,
    )


def _model_eval_action(env, config, model, rng_key, env_state):
    return _make_model_mcts_policy(
        env,
        config,
        model,
        rng_key,
        env_state,
    ).played_action


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
            return model(obs, train=False)

        @nnx.jit
        def evaluate_transitions(model: Any, states, actions: jax.Array):
            child_states = jax.vmap(env.step)(states, actions)
            return child_states, model(child_states.observation, train=False)

        @nnx.jit
        def scalar_mcts_actions(model: Any, rng_key: jax.Array, env_state):
            return _model_eval_action(env, search_config, model, rng_key, env_state)

        def model_actions(model: Any, rng_key: jax.Array, env_state):
            sample_output = evaluate_leaves(
                model,
                env_state.observation[:1],
            )
            if len(sample_output) != 3:
                return scalar_mcts_actions(model, rng_key, env_state)

            def leaf_evaluator(obs: jax.Array):
                return evaluate_leaves(model, obs)

            def transition_evaluator(states, actions: jax.Array):
                return evaluate_transitions(model, states, actions)

            return _posterior_tree_eval_action(
                env=env,
                config=search_config,
                env_state=env_state,
                leaf_evaluator=leaf_evaluator,
                transition_evaluator=transition_evaluator,
                rng_key=rng_key,
            )

        def evaluate(rng_key: jax.Array, model: Any):
            my_player = 0
            key, init_key = jax.random.split(rng_key)
            init_keys = jax.random.split(init_key, eval_batch_size)
            env_state = jax.vmap(env.init)(init_keys)
            returns = jnp.zeros(eval_batch_size)

            while not bool(np.asarray(jax.device_get(env_state.terminated)).all()):
                key, my_key, opp_key = jax.random.split(key, 3)
                my_action = model_actions(model, my_key, env_state)
                opp_action = model_actions(baseline_model, opp_key, env_state)
                action = jnp.where(
                    env_state.current_player == my_player,
                    my_action,
                    opp_action,
                )
                env_state = jax.vmap(env.step)(env_state, action)
                returns = returns + env_state.rewards[
                    jnp.arange(eval_batch_size),
                    my_player,
                ]
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

            env_state = jax.vmap(env.step)(env_state, action)
            returns = returns + env_state.rewards[
                jnp.arange(eval_batch_size),
                my_player,
            ]
            return key, env_state, returns

        _, _, returns = nnx.while_loop(
            lambda x: ~(x[1].terminated.all()),
            body_fn,
            (key, env_state, jnp.zeros(eval_batch_size)),
        )
        return returns

    return evaluate
