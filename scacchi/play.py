from typing import NamedTuple

import chex
from flax import nnx
import jax
import jax.numpy as jnp
import mctx
import pgx
from pgx.experimental import auto_reset

from .network import AZNet


WDL_DIM = 3
W_IDX = 2
L_IDX = 0


class NodeEmbedding(NamedTuple):
    state: pgx.State
    y: jax.Array            # [B, 3] WDL distribution at this node, local (player-to-move) perspective
    c: jax.Array            # [B] evidence weight: c_terminal at terminal nodes, else c_leaf
    root_action: jax.Array  # [B] int32, NO_PARENT at root, action_taken at depth 1, inherited deeper
    depth_parity: jax.Array # [B] int32, 0 at root, flipped each ply


class SelfplayOutput(NamedTuple):
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    policy_target: chex.Array
    played_action: jax.Array
    discount: jax.Array
    q_evidence_sum: chex.Array  # [B, A, 3] Σ_n c_n · y_n^aligned per root action


def _wdl_mean(alpha: jax.Array) -> jax.Array:
    return alpha / jnp.sum(alpha, axis=-1, keepdims=True)


def _utility(wdl_mean: jax.Array) -> jax.Array:
    return wdl_mean[..., W_IDX] - wdl_mean[..., L_IDX]


def _flip_wdl(wdl: jax.Array) -> jax.Array:
    return wdl[..., ::-1]


def make_recurrent_fn(env, predict_fn, c_terminal: float, c_leaf: float):
    def recurrent_fn(
        _,
        rng_key: chex.PRNGKey,
        action: chex.Array,
        embedding: NodeEmbedding,
    ):
        del rng_key

        env_state = embedding.state
        current_player = env_state.current_player
        env_state = jax.vmap(env.step)(env_state, action)
        logits, alpha_V, _ = predict_fn(env_state.observation) # model(observation)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(env_state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

        alpha_V_mean = _wdl_mean(alpha_V)
        reward = env_state.rewards[jnp.arange(env_state.rewards.shape[0]), current_player]
        terminal_y_parent = jax.nn.one_hot(
            jnp.round(reward).astype(jnp.int32) + 1, WDL_DIM, dtype=alpha_V_mean.dtype,
        )
        terminal_y_child = _flip_wdl(terminal_y_parent)
        y = jnp.where(env_state.terminated[..., None], terminal_y_child, alpha_V_mean)
        c = jnp.where(env_state.terminated, c_terminal, c_leaf).astype(alpha_V_mean.dtype)
        new_root_action = jnp.where(
            embedding.root_action == mctx.Tree.NO_PARENT, action, embedding.root_action,
        )
        new_depth_parity = 1 - embedding.depth_parity

        value = _utility(alpha_V_mean)
        value = jnp.where(env_state.terminated, 0.0, value)
        discount = jnp.where(env_state.terminated, 0.0, -jnp.ones_like(value))

        new_embedding = NodeEmbedding(
            state=env_state,
            y=y,
            c=c,
            root_action=new_root_action,
            depth_parity=new_depth_parity,
        )

        return (
            mctx.RecurrentFnOutput(
                reward=reward,
                discount=discount,
                prior_logits=logits,
                value=value,
            ),
            new_embedding,
        )

    return recurrent_fn


def _mc_posterior_best(
    rng_key: jax.Array,
    alpha_Q_post: jax.Array,
    invalid_actions: jax.Array,
    legal_action_mask: jax.Array,
    num_samples: int,
) -> jax.Array:
    """MC estimator of the posterior-best policy (math.md §6, §7).

    alpha_Q_post: [B, A, 3]
    invalid_actions: [B, A] bool — True for invalid
    legal_action_mask: [B, A] bool — True for legal
    Returns: [B, A] probabilities, Laplace-smoothed over legal actions.
    """
    batch_size, num_actions, _ = alpha_Q_post.shape
    phi = jax.random.dirichlet(
        rng_key, alpha=alpha_Q_post, shape=(num_samples, batch_size, num_actions)
    )
    utilities = phi[..., W_IDX] - phi[..., L_IDX]
    utilities = jnp.where(invalid_actions[None, :, :], -jnp.inf, utilities)
    argmax_action = jnp.argmax(utilities, axis=-1)
    counts = jax.nn.one_hot(argmax_action, num_actions).sum(axis=0)
    legal_f = legal_action_mask.astype(counts.dtype)
    num_legal = jnp.maximum(legal_f.sum(axis=-1, keepdims=True), 1.0)
    smoothed = (counts + legal_f) / (num_samples + num_legal)
    return smoothed


def _q_evidence_sum(tree: mctx.Tree, num_actions: int, dtype) -> jax.Array:
    """Σ_n c_n · y_n^aligned per root action, shape [B, A, 3].

    Math: math.md §4 (evidence update), §6 (search-tree posterior).
    Routes every expanded node into its root-action bucket via NodeEmbedding.root_action
    and aligns to root-player perspective via NodeEmbedding.depth_parity.
    """
    emb = tree.embeddings
    y_aligned = jnp.where(emb.depth_parity[..., None] == 1, _flip_wdl(emb.y), emb.y)
    valid = (emb.root_action != mctx.Tree.NO_PARENT) & (tree.node_visits > 0)
    weight = jnp.where(valid, emb.c, jnp.zeros((), dtype=emb.c.dtype)).astype(dtype)

    batch_size, _ = tree.node_visits.shape
    batch_range = jnp.arange(batch_size)[:, None]
    safe_root_action = jnp.where(valid, emb.root_action, 0)
    out = jnp.zeros((batch_size, num_actions, WDL_DIM), dtype=dtype)
    return out.at[batch_range, safe_root_action].add(
        weight[..., None] * y_aligned.astype(dtype)
    )


def make_selfplay(env, config):
    def selfplay(model: AZNet, rng_key: jax.Array) -> SelfplayOutput:
        recurrent_fn = make_recurrent_fn(
            env,
            lambda obs: model(obs, train=False),
            config.c_terminal,
            config.c_leaf,
        )

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def step_fn(
            env_state: pgx.State,
            key: jax.Array,
        ) -> tuple[pgx.State, SelfplayOutput]:
            search_key, mc_key, reset_key = jax.random.split(key, 3)
            observation = env_state.observation
            logits, alpha_V, alpha_Q = model(observation, train=False)
            alpha_V_mean = _wdl_mean(alpha_V)
            value = _utility(alpha_V_mean)
            batch_size = alpha_V_mean.shape[0]
            root = mctx.RootFnOutput(
                prior_logits=logits,
                value=value,
                embedding=NodeEmbedding(
                    state=env_state,
                    y=alpha_V_mean,
                    c=jnp.full((batch_size,), config.c_leaf, dtype=alpha_V_mean.dtype),
                    root_action=jnp.full((batch_size,), mctx.Tree.NO_PARENT, dtype=jnp.int32),
                    depth_parity=jnp.zeros((batch_size,), dtype=jnp.int32),
                ),
            )

            invalid_actions = ~env_state.legal_action_mask
            policy_output = mctx.gumbel_muzero_policy(
                params=(),
                rng_key=search_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=config.num_simulations,
                invalid_actions=invalid_actions,
                qtransform=mctx.qtransform_completed_by_mix_value,
                gumbel_scale=1.0,
            )

            q_evidence_sum = _q_evidence_sum(
                policy_output.search_tree, alpha_Q.shape[1], alpha_Q.dtype,
            )
            alpha_Q_post = alpha_Q + q_evidence_sum
            policy_target = _mc_posterior_best(
                mc_key,
                alpha_Q_post,
                invalid_actions,
                env_state.legal_action_mask,
                config.policy_mc_samples,
            )

            actor = env_state.current_player
            reset_keys = jax.random.split(reset_key, config.selfplay_batch_size)
            env_state = jax.vmap(auto_reset(env.step, env.init))(
                env_state,
                policy_output.action,
                reset_keys,
            )
            discount = -jnp.ones_like(value)
            discount = jnp.where(env_state.terminated, 0.0, discount)
            return env_state, SelfplayOutput(
                obs=observation,
                policy_target=policy_target,
                played_action=policy_output.action,
                reward=env_state.rewards[jnp.arange(env_state.rewards.shape[0]), actor],
                terminated=env_state.terminated,
                discount=discount,
                q_evidence_sum=q_evidence_sum,
            )

        rng_key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, config.selfplay_batch_size)
        env_state = jax.vmap(env.init)(init_keys)
        step_keys = jax.random.split(rng_key, config.max_num_steps)
        _, data = step_fn(env_state, step_keys)
        return data

    return selfplay
