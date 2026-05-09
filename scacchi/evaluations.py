from flax import nnx
import jax
import jax.numpy as jnp
import mctx

from .network import AZNet
from .play import (
    NodeEmbedding,
    _dirichlet_thompson_search,
    _mc_posterior_best,
    _q_evidence_sum,
    make_recurrent_fn,
)


def _root_embedding(env_state, alpha_V_mean, alpha_Q, c_leaf):
    batch_size = alpha_V_mean.shape[0]
    return NodeEmbedding(
        state=env_state,
        wdl_dist=alpha_V_mean,
        evidence_weight=jnp.full((batch_size,), c_leaf, dtype=alpha_V_mean.dtype),
        root_action=jnp.full((batch_size,), mctx.Tree.NO_PARENT, dtype=jnp.int32),
        depth_parity=jnp.zeros((batch_size,), dtype=jnp.int32),
        alpha_Q_prior=alpha_Q,
    )


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


def _wdl_mean(alpha):
    return alpha / jnp.sum(alpha, axis=-1, keepdims=True)


def _utility(wdl_mean):
    return wdl_mean[..., 2] - wdl_mean[..., 0]


def make_evaluate(env, baseline, config):
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
            my_logits, _, _ = model(env_state.observation, train=False)
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


def make_mcts_evaluate(env, baseline, config):
    # WARNING: THIS HAS NOT BEEN TESTED
    @nnx.jit
    def evaluate(rng_key: jax.Array, model: AZNet):
        """MCTS evaluation: model vs baseline. Both sides search with
        Dirichlet-Q Thompson search using their own network."""
        my_player = 0
        my_predict = lambda obs: model(obs, train=False)
        baseline_predict = _baseline_as_dirichlet(baseline)
        my_recurrent_fn = make_recurrent_fn(env, my_predict, config.c_terminal, config.c_leaf)
        opp_recurrent_fn = make_recurrent_fn(env, baseline_predict, config.c_terminal, config.c_leaf)

        key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, config.selfplay_batch_size)
        env_state = jax.vmap(env.init)(init_keys)

        def body_fn(val):
            key, env_state, returns = val
            observation = env_state.observation

            my_logits, my_alpha_V, my_alpha_Q = my_predict(observation)
            opp_logits, opp_alpha_V, opp_alpha_Q = baseline_predict(observation)

            my_value = _utility(_wdl_mean(my_alpha_V))
            opp_value = _utility(_wdl_mean(opp_alpha_V))

            key, my_search_key, my_mc_key, opp_search_key, opp_mc_key = jax.random.split(key, 5)
            invalid_actions = ~env_state.legal_action_mask
            interior_selector = getattr(config, "inference_interior_selector", "wdl")

            my_root = mctx.RootFnOutput(
                prior_logits=my_logits,
                value=my_value,
                embedding=_root_embedding(env_state, _wdl_mean(my_alpha_V), my_alpha_Q, config.c_leaf),
            )
            my_tree = _dirichlet_thompson_search(
                rng_key=my_search_key,
                root=my_root,
                recurrent_fn=my_recurrent_fn,
                alpha_Q_prior=my_alpha_Q,
                num_simulations=config.num_simulations,
                invalid_actions=invalid_actions,
                interior_selector=interior_selector,
            )
            my_q_evidence_sum = _q_evidence_sum(my_tree, my_alpha_Q.shape[1], my_alpha_Q.dtype)
            my_policy_target = _mc_posterior_best(
                my_mc_key,
                my_alpha_Q + my_q_evidence_sum,
                invalid_actions,
                env_state.legal_action_mask,
                config.policy_mc_samples,
            )

            opp_root = mctx.RootFnOutput(
                prior_logits=opp_logits,
                value=opp_value,
                embedding=_root_embedding(env_state, _wdl_mean(opp_alpha_V), opp_alpha_Q, config.c_leaf),
            )
            opp_tree = _dirichlet_thompson_search(
                rng_key=opp_search_key,
                root=opp_root,
                recurrent_fn=opp_recurrent_fn,
                alpha_Q_prior=opp_alpha_Q,
                num_simulations=config.num_simulations,
                invalid_actions=invalid_actions,
                interior_selector=interior_selector,
            )
            opp_q_evidence_sum = _q_evidence_sum(opp_tree, opp_alpha_Q.shape[1], opp_alpha_Q.dtype)
            opp_policy_target = _mc_posterior_best(
                opp_mc_key,
                opp_alpha_Q + opp_q_evidence_sum,
                invalid_actions,
                env_state.legal_action_mask,
                config.policy_mc_samples,
            )

            is_my_turn = env_state.current_player == my_player
            action = jnp.where(
                is_my_turn,
                jnp.argmax(my_policy_target, axis=-1),
                jnp.argmax(opp_policy_target, axis=-1),
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
