from typing import Any

from flax import nnx
import jax
import jax.numpy as jnp
import mctx
import numpy as np

from .network import policy_value_from_output
from .play import make_recurrent_fn
from .dirichlet_tree.search import run_wavefront_posterior_tree_search_state_batch


def _make_mcts_policy(predict, recurrent_fn, rng_key, env_state, num_simulations):
    logits, value = policy_value_from_output(predict(env_state.observation))
    root = mctx.RootFnOutput(
        prior_logits=logits,
        value=value,
        embedding=env_state,
    )
    return mctx.gumbel_muzero_policy(
        params=(),
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        invalid_actions=~env_state.legal_action_mask,
        qtransform=mctx.qtransform_completed_by_mix_value,
        gumbel_scale=1.0,
    )


def make_mcts_evaluate(env, eval_config, search_config, baseline_model):
    eval_batch_size = eval_config.batch_size

    @nnx.jit
    def evaluate_leaves(model: Any, obs: jax.Array):
        return model(obs, train=False)

    @nnx.jit
    def scalar_mcts_actions(model: Any, rng_key: jax.Array, env_state):
        predict = lambda obs: model(obs, train=False)
        return _make_mcts_policy(
            predict,
            make_recurrent_fn(env, predict),
            rng_key,
            env_state,
            search_config.num_simulations,
        ).action

    def model_actions(model: Any, rng_key: jax.Array, env_state):
        sample_output = evaluate_leaves(model, env_state.observation[:1])
        if len(sample_output) != 3:
            return scalar_mcts_actions(model, rng_key, env_state)

        def leaf_evaluator(obs: jax.Array):
            return evaluate_leaves(model, obs)

        return run_wavefront_posterior_tree_search_state_batch(
            env=env,
            root_state_batch=env_state,
            leaf_evaluator=leaf_evaluator,
            rng_key=rng_key,
            config=search_config,
        ).action

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
