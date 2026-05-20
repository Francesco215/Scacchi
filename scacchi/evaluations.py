from flax import nnx
import jax
import jax.numpy as jnp
import mctx

from .dirichlet_q_search import (
    NO_PARENT,
    NodeEmbedding,
    dirichlet_q_policy,
    outcome_mean,
    outcome_utility,
)
from .network import policy_value_from_output
from .play import make_dirichlet_recurrent_fn, make_recurrent_fn


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
        return dirichlet_q_policy(
            params=(),
            rng_key=rng_key,
            root=root,
            recurrent_fn=make_dirichlet_recurrent_fn(env, predict, config),
            action_value_prior=action_value_prior,
            num_simulations=num_simulations,
            invalid_actions=~env_state.legal_action_mask,
            num_search_blocks=getattr(config, "num_search_blocks", 1),
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
