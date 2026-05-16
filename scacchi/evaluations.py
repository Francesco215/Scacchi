from flax import nnx
import jax
import jax.numpy as jnp
import mctx

from .play import make_recurrent_fn


def _make_mcts_policy(predict, recurrent_fn, rng_key, env_state, num_simulations):
    logits, value = predict(env_state.observation)
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
        gumbel_scale=0.0,
    )


def _make_model_mcts_policy(env, model, rng_key, env_state, num_simulations):
    predict = lambda obs: model(obs, train=False)
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

            my_policy = _make_model_mcts_policy(env, model, my_key, env_state, config.num_simulations)
            opp_policy = _make_model_mcts_policy(env, baseline_model, opp_key, env_state, config.num_simulations)
            
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
