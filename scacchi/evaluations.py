from flax import nnx
import jax
import jax.numpy as jnp
import mctx
import numpy as np

from .dirichlet_q_search import (
    DirichletQSearchOutput,
    NO_PARENT,
    NodeEmbedding,
    dirichlet_q_policy,
    outcome_mean,
    outcome_utility,
    posterior_best_action,
    posterior_best_policy_target,
)
from .network import policy_value_from_output
from .play import make_dirichlet_recurrent_fn, make_recurrent_fn
from .posterior_tree import (
    is_posterior_tree_policy,
    run_posterior_tree_search,
    run_posterior_tree_search_state_batch,
    split_batched_state,
)


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
        gumbel_scale=1.,
    )


def _make_model_mcts_policy(env, config, model, rng_key, env_state, num_simulations):
    predict = lambda obs: model(obs, train=False)
    model_output = predict(env_state.observation)
    if (
        len(model_output) == 3
        and getattr(config, "search_policy", "gumbel") == "dirichlet_thompson"
    ):
        search_key, posterior_key = jax.random.split(rng_key)
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
        policy = dirichlet_q_policy(
            params=(),
            rng_key=search_key,
            root=root,
            recurrent_fn=make_dirichlet_recurrent_fn(env, predict, config),
            action_value_prior=action_value_prior,
            num_simulations=num_simulations,
            invalid_actions=~env_state.legal_action_mask,
            num_search_blocks=getattr(config, "num_search_blocks", 1),
        )
        policy_target = posterior_best_policy_target(
            posterior_key,
            policy.alpha_search,
            env_state.legal_action_mask,
            config.policy_mc_samples,
        )
        return DirichletQSearchOutput(
            action=posterior_best_action(policy_target, env_state.legal_action_mask),
            action_weights=policy_target,
            search_tree=policy.search_tree,
            q_evidence_sum=policy.q_evidence_sum,
            alpha_search=policy.alpha_search,
            explored_action_mask=policy.explored_action_mask,
        )
    return _make_mcts_policy(
        predict,
        make_recurrent_fn(env, predict),
        rng_key,
        env_state,
        num_simulations,
    )


def make_mcts_evaluate(env, config, baseline_model):
    eval_batch_size = int(getattr(config, "eval_batch_size", config.selfplay_batch_size))

    if is_posterior_tree_policy(getattr(config, "search_policy", "gumbel")):
        @nnx.jit
        def evaluate_leaves(model: nnx.Module, obs: jax.Array):
            return model(obs, train=False)

        @nnx.jit
        def scalar_mcts_actions(model: nnx.Module, rng_key: jax.Array, env_state):
            predict = lambda obs: model(obs, train=False)
            return _make_mcts_policy(
                predict,
                make_recurrent_fn(env, predict),
                rng_key,
                env_state,
                config.num_simulations,
            ).action

        def model_actions(model: nnx.Module, rng_key: jax.Array, env_state):
            sample_output = evaluate_leaves(
                model,
                env_state.observation[:1],
            )
            if len(sample_output) != 3:
                return scalar_mcts_actions(model, rng_key, env_state)

            def leaf_evaluator(obs: jax.Array):
                return evaluate_leaves(model, obs)

            if getattr(config, "search_policy", "gumbel") == "posterior_tree_wavefront":
                return run_posterior_tree_search_state_batch(
                    env=env,
                    root_state_batch=env_state,
                    leaf_evaluator=leaf_evaluator,
                    rng_key=rng_key,
                    config=config,
                ).action

            return run_posterior_tree_search(
                env=env,
                root_states=split_batched_state(env_state),
                leaf_evaluator=leaf_evaluator,
                rng_key=rng_key,
                config=config,
            ).action

        def evaluate(rng_key: jax.Array, model: nnx.Module):
            my_player = 0
            key, init_key = jax.random.split(rng_key)
            init_keys = jax.random.split(init_key, eval_batch_size)
            env_state = jax.vmap(env.init)(init_keys)
            returns = jnp.zeros(eval_batch_size)

            while not bool(np.asarray(jax.device_get(env_state.terminated)).all()):
                key, my_key, opp_key = jax.random.split(key, 3)
                my_action = model_actions(model, my_key, env_state)
                opp_action = model_actions(baseline_model, opp_key, env_state)
                action = jnp.where(env_state.current_player == my_player, my_action, opp_action)
                env_state = jax.vmap(env.step)(env_state, action)
                returns = returns + env_state.rewards[jnp.arange(eval_batch_size), my_player]
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

            my_policy = _make_model_mcts_policy(env, config, model, my_key, env_state, config.num_simulations)
            opp_policy = _make_model_mcts_policy(env, config, baseline_model, opp_key, env_state, config.num_simulations)
            
            is_my_turn = env_state.current_player == my_player
            action = jnp.where(is_my_turn, my_policy.action, opp_policy.action)

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
