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
    alpha_V_mean: jax.Array  # [B, 3] WDL mean from this node's α^V


class SelfplayOutput(NamedTuple):
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    policy_target: chex.Array
    played_action: jax.Array
    discount: jax.Array


def _wdl_mean(alpha: jax.Array) -> jax.Array:
    return alpha / jnp.sum(alpha, axis=-1, keepdims=True)


def _utility(wdl_mean: jax.Array) -> jax.Array:
    return wdl_mean[..., W_IDX] - wdl_mean[..., L_IDX]


def _flip_wdl(wdl: jax.Array) -> jax.Array:
    return wdl[..., ::-1]


def make_recurrent_fn(env, predict_fn):
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
        logits, alpha_V, _ = predict_fn(env_state.observation)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(
            env_state.legal_action_mask,
            logits,
            jnp.finfo(logits.dtype).min,
        )

        alpha_V_mean = _wdl_mean(alpha_V)
        value = _utility(alpha_V_mean)

        reward = env_state.rewards[
            jnp.arange(env_state.rewards.shape[0]),
            current_player,
        ]
        value = jnp.where(env_state.terminated, 0.0, value)
        discount = -jnp.ones_like(value)
        discount = jnp.where(env_state.terminated, 0.0, discount)

        new_embedding = NodeEmbedding(state=env_state, alpha_V_mean=alpha_V_mean)

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
    """MC estimator of the posterior-best policy (math.md §6, §12).

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


def _gather_child_wdl(
    tree: mctx.Tree,
    root_alpha_Q_mean: jax.Array,
) -> jax.Array:
    """For each root action, return y_a [B, A, 3] from the player-at-root's perspective.

    Visited child actions: gather the child node's alpha_V_mean and flip W↔L.
    Unvisited actions: fall back to the root's α^Q mean for that action.
    """
    children_idx = tree.children_index[:, mctx.Tree.ROOT_INDEX, :]  # [B, A]
    visited = children_idx >= 0
    safe_idx = jnp.where(visited, children_idx, 0)
    batch_size = children_idx.shape[0]
    batch_range = jnp.arange(batch_size)[:, None]
    child_wdl = tree.embeddings.alpha_V_mean[batch_range, safe_idx]  # [B, A, 3]
    flipped = _flip_wdl(child_wdl)
    return jnp.where(visited[..., None], flipped, root_alpha_Q_mean)


def make_selfplay(env, config):
    def selfplay(model: AZNet, rng_key: jax.Array) -> SelfplayOutput:
        recurrent_fn = make_recurrent_fn(
            env, lambda obs: model(obs, train=False)
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
            root = mctx.RootFnOutput(
                prior_logits=logits,
                value=value,
                embedding=NodeEmbedding(state=env_state, alpha_V_mean=alpha_V_mean),
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

            tree = policy_output.search_tree
            visit_counts = tree.children_visits[:, mctx.Tree.ROOT_INDEX, :].astype(alpha_Q.dtype)
            root_alpha_Q_mean = _wdl_mean(alpha_Q)  # [B, A, 3]
            y_a = _gather_child_wdl(tree, root_alpha_Q_mean)  # [B, A, 3]
            evidence = config.search_evidence_rho * jnp.sqrt(visit_counts + 1.0)  # [B, A]
            alpha_Q_post = alpha_Q + evidence[..., None] * y_a  # [B, A, 3]
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
            )

        rng_key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, config.selfplay_batch_size)
        env_state = jax.vmap(env.init)(init_keys)
        step_keys = jax.random.split(rng_key, config.max_num_steps)
        _, data = step_fn(env_state, step_keys)
        return data

    return selfplay
