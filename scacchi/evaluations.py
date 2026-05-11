from flax import nnx
import jax
import jax.numpy as jnp
import mctx

from .dirichlet_q_search import (
    dirichlet_q_policy,
    make_dirichlet_recurrent_fn,
    make_root_embedding,
    utility,
    wdl_mean,
)
from .network import AZNet
from .play import slice_obs


def _baseline_as_dirichlet(baseline):
    """Adapt a baseline returning (logits, scalar_value) to the (logits, alpha_V, alpha_Q) shape.

    The synthetic alpha_V preserves U(mean(alpha_V)) ~= scalar_value. alpha_Q is uniform; it is
    not consumed during MCTS backup for the baseline side of evaluation.
    """

    def wrapped(obs):
        logits, value = baseline(obs)
        eps = jnp.asarray(1e-6, dtype=value.dtype)
        v = jnp.clip(value, -1.0 + eps, 1.0 - eps)
        p_D = jnp.full_like(v, eps)
        p_W = (1.0 - eps + v) / 2.0
        p_L = (1.0 - eps - v) / 2.0
        wdl_mean = jnp.stack([p_L, p_D, p_W], axis=-1)
        alpha_V = 10.0 * wdl_mean
        num_actions = logits.shape[-1]
        alpha_Q = jnp.broadcast_to(
            jnp.ones((1, 1, 3), dtype=logits.dtype),
            (logits.shape[0], num_actions, 3),
        )
        return logits, alpha_V, alpha_Q

    return wrapped


def make_evaluate(env, baseline, config):
    nhs = config.num_history_steps

    @nnx.jit
    def evaluate(rng_key: jax.Array, model: AZNet):
        """A simplified evaluation by sampling. Only for debugging.
        Please use MCTS and run tournaments for serious evaluation."""
        my_player = 0

        key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, config.selfplay_batch_size)
        env_state = jax.vmap(env.init)(init_keys)

        def body_fn(val):
            key, env_state, returns = val
            my_logits, _, _ = model(slice_obs(env_state.observation, nhs), train=False)
            opp_logits, _ = baseline(env_state.observation)
            is_my_turn = (env_state.current_player == my_player).reshape((-1, 1))
            logits = jnp.where(is_my_turn, my_logits, opp_logits)
            key, action_key = jax.random.split(key)
            action = jax.random.categorical(action_key, logits, axis=-1)
            env_state = jax.vmap(env.step)(env_state, action)
            returns = returns + env_state.rewards[
                jnp.arange(config.selfplay_batch_size),
                my_player,
            ]
            return key, env_state, returns

        _, _, returns = nnx.while_loop(
            lambda x: ~(x[1].terminated.all()),
            body_fn,
            (key, env_state, jnp.zeros(config.selfplay_batch_size)),
        )
        return returns

    return evaluate


def make_gumbel_recurrent_fn(env, baseline):
    """Recurrent function for Gumbel MuZero search using a scalar-value baseline."""

    def recurrent_fn(_, rng_key, action, embedding):
        del rng_key
        current_player = embedding.current_player
        env_state = jax.vmap(env.step)(embedding, action)
        logits, value = baseline(env_state.observation)
        logits = jnp.where(env_state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        reward = env_state.rewards[jnp.arange(env_state.rewards.shape[0]), current_player]
        value = jnp.where(env_state.terminated, 0.0, value)
        discount = jnp.where(env_state.terminated, 0.0, -jnp.ones_like(value))
        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        ), env_state

    return recurrent_fn


def make_mcts_evaluate(env, baseline, config):
    # WARNING: THIS HAS NOT BEEN TESTED
    nhs = config.num_history_steps

    @nnx.jit
    def evaluate(rng_key: jax.Array, model: AZNet):
        """MCTS evaluation: model (Dirichlet-Q search) vs baseline (Gumbel MuZero search)."""
        my_player = 0
        my_predict = lambda obs: model(slice_obs(obs, nhs), train=False)
        my_recurrent_fn = make_dirichlet_recurrent_fn(env, my_predict, config.c_terminal, config.c_leaf)
        opp_recurrent_fn = make_gumbel_recurrent_fn(env, baseline)

        key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, config.selfplay_batch_size)
        env_state = jax.vmap(env.init)(init_keys)

        def body_fn(val):
            key, env_state, returns = val
            observation = env_state.observation

            my_logits, my_alpha_V, my_alpha_Q = my_predict(observation)
            opp_logits, opp_value = baseline(observation)

            my_value = utility(wdl_mean(my_alpha_V))

            key, my_search_key, opp_search_key = jax.random.split(key, 3)
            invalid_actions = ~env_state.legal_action_mask
            interior_selector = getattr(config, "inference_interior_selector", "wdl")

            my_root = mctx.RootFnOutput(
                prior_logits=my_logits,
                value=my_value,
                embedding=make_root_embedding(env_state, wdl_mean(my_alpha_V), my_alpha_Q, config.c_leaf),
            )
            my_search = dirichlet_q_policy(
                params=(),
                rng_key=my_search_key,
                root=my_root,
                recurrent_fn=my_recurrent_fn,
                num_simulations=config.num_simulations,
                alpha_Q_prior=my_alpha_Q,
                invalid_actions=invalid_actions,
                policy_mc_samples=config.policy_mc_samples,
                interior_selector=interior_selector,
                num_search_blocks=getattr(config, "num_search_blocks", 1),
                action_selection_mode="argmax",
            )

            opp_root = mctx.RootFnOutput(
                prior_logits=opp_logits,
                value=opp_value,
                embedding=env_state,
            )
            opp_search = mctx.gumbel_muzero_policy(
                params=(),
                rng_key=opp_search_key,
                root=opp_root,
                recurrent_fn=opp_recurrent_fn,
                num_simulations=config.num_simulations,
                invalid_actions=invalid_actions,
            )

            is_my_turn = env_state.current_player == my_player
            action = jnp.where(
                is_my_turn,
                my_search.action,
                opp_search.action,
            )

            env_state = jax.vmap(env.step)(env_state, action)
            returns = returns + env_state.rewards[
                jnp.arange(config.selfplay_batch_size),
                my_player,
            ]
            return key, env_state, returns

        _, _, returns = nnx.while_loop(
            lambda x: ~(x[1].terminated.all()),
            body_fn,
            (key, env_state, jnp.zeros(config.selfplay_batch_size)),
        )
        return returns

    return evaluate
