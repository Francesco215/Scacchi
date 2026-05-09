from typing import NamedTuple

import chex
from flax import nnx
import jax
import jax.numpy as jnp
import mctx
from mctx._src import action_selection
from mctx._src import search as mctx_search
import pgx
from pgx.experimental import auto_reset

from .network import AZNet


WDL_DIM = 3
W_IDX = 2
L_IDX = 0


class NodeEmbedding(NamedTuple):
    state: pgx.State
    wdl_dist: jax.Array  # [B, 3] WDL distribution at this node, local player-to-move perspective
    evidence_weight: jax.Array  # [B] c_terminal at terminal nodes, else c_leaf
    root_action: jax.Array  # [B] int32, NO_PARENT at root, action_taken at depth 1, inherited deeper
    depth_parity: jax.Array # [B] int32, 0 at root, flipped each ply
    alpha_Q_prior: jax.Array  # [B, A, 3] Q Dirichlet prior for actions at this node


class SelfplayOutput(NamedTuple):
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    policy_target: chex.Array
    played_action: jax.Array
    discount: jax.Array
    q_evidence_sum: chex.Array  # [B, A, 3] Σ_n evidence_weight_n · wdl_dist_n^aligned per root action


class DirichletRootExtra(NamedTuple):
    alpha_Q_prior: jax.Array  # [B, A, 3] outside search, [A, 3] inside vmapped selectors


def _wdl_mean(alpha: jax.Array) -> jax.Array:
    return alpha / jnp.sum(alpha, axis=-1, keepdims=True)


def _utility(wdl_mean: jax.Array) -> jax.Array:
    return wdl_mean[..., W_IDX] - wdl_mean[..., L_IDX]


def _flip_wdl(wdl: jax.Array) -> jax.Array:
    return wdl[..., ::-1]


def make_recurrent_fn(env, predict_fn, c_terminal: float, c_leaf: float):
    """Build the MuZero recurrent function used to expand search-tree nodes.

    The returned function applies one environment action from a stored
    NodeEmbedding, evaluates the resulting observation with predict_fn, and
    packages the transition, priors, value, and updated embedding in the shape
    expected by MCTX search.
    """
    def recurrent_fn(_, rng_key: chex.PRNGKey, action: chex.Array, embedding: NodeEmbedding):
        """Advance a batch of tree nodes by one action during MCTS expansion and evaluates the result"""
        del rng_key

        env_state = embedding.state
        current_player = env_state.current_player
        env_state = jax.vmap(env.step)(env_state, action)
        logits, alpha_V, alpha_Q = predict_fn(env_state.observation) # model(observation)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(env_state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

        alpha_V_mean = _wdl_mean(alpha_V)
        reward = env_state.rewards[jnp.arange(env_state.rewards.shape[0]), current_player]
        terminal_wdl_parent = jax.nn.one_hot(
            jnp.round(reward).astype(jnp.int32) + 1, WDL_DIM, dtype=alpha_V_mean.dtype,
        )
        terminal_wdl_child = _flip_wdl(terminal_wdl_parent)
        wdl_dist = jnp.where(env_state.terminated[..., None], terminal_wdl_child, alpha_V_mean)
        evidence_weight = jnp.where(env_state.terminated, c_terminal, c_leaf).astype(alpha_V_mean.dtype)
        new_root_action = jnp.where(
            embedding.root_action == mctx.Tree.NO_PARENT, action, embedding.root_action,
        )
        new_depth_parity = 1 - embedding.depth_parity

        value = _utility(alpha_V_mean)
        value = jnp.where(env_state.terminated, 0.0, value)
        discount = jnp.where(env_state.terminated, 0.0, -jnp.ones_like(value))

        new_embedding = NodeEmbedding(
            state=env_state,
            wdl_dist=wdl_dist,
            evidence_weight=evidence_weight,
            root_action=new_root_action,
            depth_parity=new_depth_parity,
            alpha_Q_prior=alpha_Q,
        )

        mctx_output = mctx.RecurrentFnOutput(reward=reward, discount=discount, prior_logits=logits, value=value)
        return mctx_output, new_embedding

    return recurrent_fn


def _mc_posterior_best(
    rng_key: jax.Array,
    alpha_Q_post: jax.Array,
    invalid_actions: jax.Array,
    legal_action_mask: jax.Array,
    num_samples: int,
) -> jax.Array:
    """MC estimator of the posterior-best policy (math.md §6, §7).

    Samples WDL outcomes from each action posterior, counts how often each legal
    action has the highest sampled utility, and returns those counts as a
    smoothed policy target.

    alpha_Q_post: [B, A, 3]
    invalid_actions: [B, A] bool — True for invalid
    legal_action_mask: [B, A] bool — True for legal
    Returns: [B, A] probabilities, Laplace-smoothed over legal actions.
    """
    batch_size, num_actions, _ = alpha_Q_post.shape
    phi = jax.random.dirichlet(rng_key, alpha=alpha_Q_post, shape=(num_samples, batch_size, num_actions))
    utilities = phi[..., W_IDX] - phi[..., L_IDX]
    utilities = jnp.where(invalid_actions[None, :, :], -jnp.inf, utilities)
    argmax_action = jnp.argmax(utilities, axis=-1)
    counts = jax.nn.one_hot(argmax_action, num_actions).sum(axis=0)
    legal_f = legal_action_mask.astype(counts.dtype)
    num_legal = jnp.maximum(legal_f.sum(axis=-1, keepdims=True), 1.0)
    smoothed = (counts + legal_f) / (num_samples + num_legal)
    return smoothed


def _q_evidence_sum(tree: mctx.Tree, num_actions: int, dtype) -> jax.Array:
    """Σ_n evidence_weight_n · wdl_dist_n^aligned per root action, shape [B, A, 3].

    Aggregates value evidence gathered by search so it can be added to the
    network's alpha_Q prior before computing the training policy target.

    Math: math.md §4 (evidence update), §6 (search-tree posterior).
    Routes every expanded node into its root-action bucket via NodeEmbedding.root_action
    and aligns to root-player perspective via NodeEmbedding.depth_parity.
    """
    emb = tree.embeddings
    wdl_aligned = jnp.where(
        emb.depth_parity[..., None] == 1,
        _flip_wdl(emb.wdl_dist),
        emb.wdl_dist,
    )
    valid = (emb.root_action != mctx.Tree.NO_PARENT) & (tree.node_visits > 0)
    weight = jnp.where(
        valid,
        emb.evidence_weight,
        jnp.zeros((), dtype=emb.evidence_weight.dtype),
    ).astype(dtype)

    batch_size, _ = tree.node_visits.shape
    batch_range = jnp.arange(batch_size)[:, None]
    safe_root_action = jnp.where(valid, emb.root_action, 0)
    out = jnp.zeros((batch_size, num_actions, WDL_DIM), dtype=dtype)
    return out.at[batch_range, safe_root_action].add(
        weight[..., None] * wdl_aligned.astype(dtype)
    )


def _q_evidence_sum_unbatched(tree: mctx.Tree, num_actions: int, dtype) -> jax.Array:
    """Single-tree version of _q_evidence_sum, shape [A, 3]."""
    emb = tree.embeddings
    wdl_aligned = jnp.where(
        emb.depth_parity[..., None] == 1,
        _flip_wdl(emb.wdl_dist),
        emb.wdl_dist,
    )
    valid = (emb.root_action != mctx.Tree.NO_PARENT) & (tree.node_visits > 0)
    weight = jnp.where(
        valid,
        emb.evidence_weight,
        jnp.zeros((), dtype=emb.evidence_weight.dtype),
    ).astype(dtype)

    safe_root_action = jnp.where(valid, emb.root_action, 0)
    out = jnp.zeros((num_actions, WDL_DIM), dtype=dtype)
    return out.at[safe_root_action].add(weight[..., None] * wdl_aligned.astype(dtype))


def _dirichlet_root_action_selection(
    rng_key: jax.Array,
    tree: mctx.Tree,
    node_index: chex.Array,
) -> jax.Array:
    """Thompson-sample root actions from live Dirichlet-Q posteriors."""
    del node_index
    alpha_Q_prior = tree.extra_data.alpha_Q_prior
    q_evidence_sum = _q_evidence_sum_unbatched(
        tree, alpha_Q_prior.shape[0], alpha_Q_prior.dtype,
    )
    alpha_Q_post = alpha_Q_prior + q_evidence_sum
    phi = jax.random.dirichlet(rng_key, alpha_Q_post)
    score = _utility(phi)
    return action_selection.masked_argmax(score, tree.root_invalid_actions)


def _child_action_from_ancestor(
    tree: mctx.Tree,
    ancestor_index: chex.Array,
    node_index: chex.Array,
) -> tuple[jax.Array, jax.Array]:
    """Find which child action under ancestor_index reaches node_index."""
    int_dtype = tree.parents.dtype
    no_parent = jnp.asarray(mctx.Tree.NO_PARENT, dtype=int_dtype)
    node_index = node_index.astype(int_dtype)
    ancestor_index = ancestor_index.astype(int_dtype)

    def cond_fn(carry):
        _, _, _, done = carry
        return jnp.logical_not(done)

    def body_fn(carry):
        current, child_action, found, _ = carry
        parent = tree.parents[current]
        action = tree.action_from_parent[current]
        is_direct_child = parent == ancestor_index
        reached_top = parent == no_parent
        return (
            parent,
            jnp.where(is_direct_child, action, child_action),
            found | is_direct_child,
            is_direct_child | reached_top,
        )

    _, child_action, found, _ = jax.lax.while_loop(
        cond_fn,
        body_fn,
        (
            node_index,
            jnp.zeros((), dtype=int_dtype),
            jnp.asarray(False),
            node_index == ancestor_index,
        ),
    )
    return child_action, found


def _child_evidence_sum_unbatched(
    tree: mctx.Tree,
    node_index: chex.Array,
    num_actions: int,
    dtype,
) -> jax.Array:
    """Σ evidence under each child action of an interior node, shape [A, 3]."""
    emb = tree.embeddings
    node_indices = jnp.arange(tree.node_visits.shape[0], dtype=tree.parents.dtype)
    child_action, is_descendant = jax.vmap(
        lambda candidate: _child_action_from_ancestor(tree, node_index, candidate)
    )(node_indices)

    node_parity = emb.depth_parity[node_index]
    wdl_aligned = jnp.where(
        emb.depth_parity[..., None] != node_parity,
        _flip_wdl(emb.wdl_dist),
        emb.wdl_dist,
    )
    valid = is_descendant & (tree.node_visits > 0)
    weight = jnp.where(
        valid,
        emb.evidence_weight,
        jnp.zeros((), dtype=emb.evidence_weight.dtype),
    ).astype(dtype)

    safe_child_action = jnp.where(valid, child_action, 0)
    out = jnp.zeros((num_actions, WDL_DIM), dtype=dtype)
    return out.at[safe_child_action].add(weight[..., None] * wdl_aligned.astype(dtype))


def _wdl_interior_action_selection(
    rng_key: jax.Array,
    tree: mctx.Tree,
    node_index: chex.Array,
    depth: chex.Array,
) -> jax.Array:
    """Thompson-sample interior actions from reconstructed WDL posteriors."""
    del depth
    prior_logits = tree.children_prior_logits[node_index]
    alpha_Q_prior = tree.embeddings.alpha_Q_prior[node_index]
    child_evidence_sum = _child_evidence_sum_unbatched(
        tree, node_index, alpha_Q_prior.shape[0], alpha_Q_prior.dtype,
    )
    alpha_Q_post = alpha_Q_prior + child_evidence_sum
    phi = jax.random.dirichlet(rng_key, alpha_Q_post)
    score = _utility(phi)
    invalid_actions = prior_logits <= (jnp.finfo(prior_logits.dtype).min / 2)
    return action_selection.masked_argmax(score, invalid_actions)


def _mask_invalid_logits(logits: jax.Array, invalid_actions: jax.Array | None) -> jax.Array:
    if invalid_actions is None:
        return logits
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    return jnp.where(invalid_actions, jnp.finfo(logits.dtype).min, logits)


def _dirichlet_thompson_search(
    rng_key: jax.Array,
    root: mctx.RootFnOutput,
    recurrent_fn: mctx.RecurrentFn,
    alpha_Q_prior: jax.Array,
    num_simulations: int,
    invalid_actions: jax.Array,
    max_depth: int | None = None,
) -> mctx.Tree:
    """Run MCTX search with Dirichlet-Q Thompson root selection.

    This is intentionally the only call site for mctx._src.search.search, so an
    MCTX upgrade has one obvious integration point.
    """
    root = root.replace(
        prior_logits=_mask_invalid_logits(root.prior_logits, invalid_actions),
    )
    return mctx_search.search(
        params=(),
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        root_action_selection_fn=_dirichlet_root_action_selection,
        interior_action_selection_fn=_wdl_interior_action_selection,
        num_simulations=num_simulations,
        max_depth=max_depth,
        invalid_actions=invalid_actions,
        extra_data=DirichletRootExtra(alpha_Q_prior=alpha_Q_prior),
    )


def make_selfplay(env, config):
    """Build a self-play rollout function for the configured environment.

    The returned function initializes a vectorized batch of games, repeatedly
    selects moves with Dirichlet-Q Thompson root search, converts search evidence into
    training targets, and returns the per-step data used by training.
    """
    def selfplay(model: AZNet, rng_key: jax.Array) -> SelfplayOutput:
        """Generate one batched self-play trajectory with the current model."""
        recurrent_fn = make_recurrent_fn(env, lambda obs: model(obs, train=False), config.c_terminal, config.c_leaf)

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def step_fn(
            env_state: pgx.State,
            key: jax.Array,
        ) -> tuple[pgx.State, SelfplayOutput]:
            """Run one environment/search step and emit its training sample."""
            search_key, mc_key, action_key, reset_key = jax.random.split(key, 4)
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
                    wdl_dist=alpha_V_mean,
                    evidence_weight=jnp.full((batch_size,), config.c_leaf, dtype=alpha_V_mean.dtype),
                    root_action=jnp.full((batch_size,), mctx.Tree.NO_PARENT, dtype=jnp.int32),
                    depth_parity=jnp.zeros((batch_size,), dtype=jnp.int32),
                    alpha_Q_prior=alpha_Q,
                ),
            )

            invalid_actions = ~env_state.legal_action_mask
            search_tree = _dirichlet_thompson_search(
                rng_key=search_key,
                root=root,
                recurrent_fn=recurrent_fn,
                alpha_Q_prior=alpha_Q,
                num_simulations=config.num_simulations,
                invalid_actions=invalid_actions,
            )

            q_evidence_sum = _q_evidence_sum(
                search_tree, alpha_Q.shape[1], alpha_Q.dtype,
            )
            alpha_Q_post = alpha_Q + q_evidence_sum
            policy_target = _mc_posterior_best(
                mc_key,
                alpha_Q_post,
                invalid_actions,
                env_state.legal_action_mask,
                config.policy_mc_samples,
            )
            action_logits = jnp.log(jnp.clip(policy_target, 1e-8, 1.0))
            action_logits = jnp.where(
                env_state.legal_action_mask,
                action_logits,
                jnp.finfo(action_logits.dtype).min,
            )
            action = jax.random.categorical(action_key, action_logits, axis=-1)

            actor = env_state.current_player
            reset_keys = jax.random.split(reset_key, config.selfplay_batch_size)
            env_state = jax.vmap(auto_reset(env.step, env.init))(
                env_state,
                action,
                reset_keys,
            )
            discount = -jnp.ones_like(value)
            discount = jnp.where(env_state.terminated, 0.0, discount)
            return env_state, SelfplayOutput(
                obs=observation,
                policy_target=policy_target,
                played_action=action,
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
