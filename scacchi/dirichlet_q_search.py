from typing import Any, Literal, NamedTuple, cast

import chex
import jax
import jax.numpy as jnp
import mctx
from mctx._src import action_selection
from mctx._src import search as mctx_search
import pgx


WDL_DIM = 3
W_IDX = 2
L_IDX = 0


class NodeEmbedding(NamedTuple):
    state: pgx.State
    wdl_dist: jax.Array  # [B, 3] WDL distribution at this node, local player-to-move perspective
    evidence_weight: jax.Array  # [B] c_terminal at terminal nodes, else c_leaf
    root_action: jax.Array  # [B] int32, NO_PARENT at root, action_taken at depth 1, inherited deeper
    depth_parity: jax.Array  # [B] int32, 0 at root, flipped each ply
    alpha_Q_prior: jax.Array  # [B, A, 3] Q Dirichlet prior for actions at this node


class DirichletRootExtra(NamedTuple):
    alpha_Q_prior: jax.Array  # [B, A, 3] outside search, [A, 3] inside vmapped selectors


class DirichletQPolicyOutput(NamedTuple):
    action: jax.Array
    action_weights: jax.Array
    search_tree: mctx.Tree
    q_evidence_sum: jax.Array
    alpha_Q_post: jax.Array


def wdl_mean(alpha: jax.Array) -> jax.Array:
    return alpha / jnp.sum(alpha, axis=-1, keepdims=True)


def utility(wdl_mean_: jax.Array) -> jax.Array:
    return wdl_mean_[..., W_IDX] - wdl_mean_[..., L_IDX]


def flip_wdl(wdl: jax.Array) -> jax.Array:
    return wdl[..., ::-1]


def make_root_embedding(
    env_state: pgx.State,
    alpha_V_mean: jax.Array,
    alpha_Q: jax.Array,
    c_leaf: float,
) -> NodeEmbedding:
    batch_size = alpha_V_mean.shape[0]
    return NodeEmbedding(
        state=env_state,
        wdl_dist=alpha_V_mean,
        evidence_weight=jnp.full((batch_size,), c_leaf, dtype=alpha_V_mean.dtype),
        root_action=jnp.full((batch_size,), mctx.Tree.NO_PARENT, dtype=jnp.int32),
        depth_parity=jnp.zeros((batch_size,), dtype=jnp.int32),
        alpha_Q_prior=alpha_Q,
    )


def make_dirichlet_recurrent_fn(env, predict_fn, c_terminal: float, c_leaf: float):
    """Build the MuZero recurrent function used to expand search-tree nodes."""

    def recurrent_fn(_, rng_key: chex.PRNGKey, action: chex.Array, embedding: NodeEmbedding):
        """Advance tree nodes by one action during MCTS expansion and evaluate the result."""
        del rng_key

        env_state = embedding.state
        current_player = env_state.current_player
        env_state = jax.vmap(env.step)(env_state, action)
        logits, alpha_V, alpha_Q = predict_fn(env_state.observation)
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(env_state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

        alpha_V_mean = wdl_mean(alpha_V)
        reward = env_state.rewards[jnp.arange(env_state.rewards.shape[0]), current_player]
        terminal_wdl_parent = jax.nn.one_hot(
            jnp.round(reward).astype(jnp.int32) + 1,
            WDL_DIM,
            dtype=alpha_V_mean.dtype,
        )
        terminal_wdl_child = flip_wdl(terminal_wdl_parent)
        wdl_dist = jnp.where(env_state.terminated[..., None], terminal_wdl_child, alpha_V_mean)
        evidence_weight = jnp.where(env_state.terminated, c_terminal, c_leaf).astype(alpha_V_mean.dtype)
        new_root_action = jnp.where(
            embedding.root_action == mctx.Tree.NO_PARENT,
            action,
            embedding.root_action,
        )
        new_depth_parity = 1 - embedding.depth_parity

        value = utility(alpha_V_mean)
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

        mctx_output = mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        )
        return mctx_output, new_embedding

    return recurrent_fn


def posterior_best_policy_target(
    rng_key: jax.Array,
    alpha_Q_post: jax.Array,
    invalid_actions: jax.Array,
    num_samples: int,
) -> jax.Array:
    """MC estimator of the posterior-best policy from Dirichlet-Q posteriors."""
    batch_size, num_actions, _ = alpha_Q_post.shape
    phi = jax.random.dirichlet(rng_key, alpha=alpha_Q_post, shape=(num_samples, batch_size, num_actions))
    utilities = phi[..., W_IDX] - phi[..., L_IDX]
    utilities = jnp.where(invalid_actions[None, :, :], -jnp.inf, utilities)
    argmax_action = jnp.argmax(utilities, axis=-1)
    counts = jax.nn.one_hot(argmax_action, num_actions).sum(axis=0)
    return counts / num_samples


def posterior_targets(
    alpha_V_prior: jax.Array,
    alpha_Q_prior: jax.Array,
    q_evidence_sum: jax.Array,
    policy_target: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    beta_Q_target = alpha_Q_prior + q_evidence_sum
    q_target_mask = q_evidence_sum.sum(axis=-1) > 0
    v_evidence_sum = (
        policy_target.astype(q_evidence_sum.dtype)[..., None] * q_evidence_sum
    ).sum(axis=-2)
    beta_V_target = alpha_V_prior + v_evidence_sum
    value_target_mask = v_evidence_sum.sum(axis=-1) > 0
    return beta_V_target, value_target_mask, beta_Q_target, q_target_mask


def q_evidence_sum(tree: mctx.Tree, num_actions: int, dtype) -> jax.Array:
    """Sum search evidence by root action, shape [B, A, 3]."""
    emb = tree.embeddings
    wdl_aligned = jnp.where(
        emb.depth_parity[..., None] == 1,
        flip_wdl(emb.wdl_dist),
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
    """Single-tree version of q_evidence_sum, shape [A, 3]."""
    emb = tree.embeddings
    wdl_aligned = jnp.where(
        emb.depth_parity[..., None] == 1,
        flip_wdl(emb.wdl_dist),
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
    tree: Any,
    node_index: chex.Array,
) -> jax.Array:
    """Thompson-sample root actions from live Dirichlet-Q posteriors."""
    del node_index
    alpha_Q_prior = tree.extra_data.alpha_Q_prior
    q_evidence = _q_evidence_sum_unbatched(
        tree,
        alpha_Q_prior.shape[0],
        alpha_Q_prior.dtype,
    )
    alpha_Q_post = alpha_Q_prior + q_evidence
    phi = jax.random.dirichlet(rng_key, alpha_Q_post)
    score = utility(phi)
    return cast(jax.Array, action_selection.masked_argmax(score, tree.root_invalid_actions))


def _policy_prior_interior_action_selection(
    rng_key: jax.Array,
    tree: Any,
    node_index: chex.Array,
    depth: chex.Array,
) -> jax.Array:
    """Policy-prior visit balancing for non-root traversal."""
    del rng_key, depth
    prior_logits = tree.children_prior_logits[node_index]
    visit_counts = tree.children_visits[node_index]
    probs = jax.nn.softmax(prior_logits)
    score = probs - visit_counts / (1 + jnp.sum(visit_counts, keepdims=True))
    invalid_actions = prior_logits <= (jnp.finfo(prior_logits.dtype).min / 2)
    return cast(jax.Array, action_selection.masked_argmax(score, invalid_actions))


def _child_action_from_ancestor(
    tree: Any,
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
    tree: Any,
    node_index: chex.Array,
    num_actions: int,
    dtype,
) -> jax.Array:
    """Sum evidence under each child action of an interior node, shape [A, 3]."""
    emb = tree.embeddings
    node_indices = jnp.arange(tree.node_visits.shape[0], dtype=tree.parents.dtype)
    child_action, is_descendant = jax.vmap(
        lambda candidate: _child_action_from_ancestor(tree, node_index, candidate)
    )(node_indices)

    node_parity = emb.depth_parity[node_index]
    wdl_aligned = jnp.where(
        emb.depth_parity[..., None] != node_parity,
        flip_wdl(emb.wdl_dist),
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
    tree: Any,
    node_index: chex.Array,
    depth: chex.Array,
) -> jax.Array:
    """Thompson-sample interior actions from reconstructed WDL posteriors."""
    del depth
    prior_logits = tree.children_prior_logits[node_index]
    alpha_Q_prior = tree.embeddings.alpha_Q_prior[node_index]
    child_evidence = _child_evidence_sum_unbatched(
        tree,
        node_index,
        alpha_Q_prior.shape[0],
        alpha_Q_prior.dtype,
    )
    alpha_Q_post = alpha_Q_prior + child_evidence
    phi = jax.random.dirichlet(rng_key, alpha_Q_post)
    score = utility(phi)
    invalid_actions = prior_logits <= (jnp.finfo(prior_logits.dtype).min / 2)
    return cast(jax.Array, action_selection.masked_argmax(score, invalid_actions))


def _get_interior_action_selection_fn(interior_selector: str):
    if interior_selector == "policy_prior":
        return _policy_prior_interior_action_selection
    if interior_selector == "wdl":
        return _wdl_interior_action_selection
    raise ValueError(f"Unknown interior selector: {interior_selector}")


def _mask_invalid_logits(logits: jax.Array, invalid_actions: jax.Array | None) -> jax.Array:
    if invalid_actions is None:
        return logits
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    return jnp.where(invalid_actions, jnp.finfo(logits.dtype).min, logits)


def dirichlet_q_search(
    params,
    rng_key: jax.Array,
    root: mctx.RootFnOutput,
    recurrent_fn: mctx.RecurrentFn,
    alpha_Q_prior: jax.Array,
    num_simulations: int,
    invalid_actions: jax.Array | None,
    interior_selector: str,
    max_depth: int | None = None,
    loop_fn=jax.lax.fori_loop,
) -> mctx.Tree:
    """Run MCTX search with Dirichlet-Q Thompson root selection.

    This is intentionally the only call site for mctx._src.search.search, so an
    MCTX upgrade has one obvious integration point.
    """
    root = mctx.RootFnOutput(
        prior_logits=_mask_invalid_logits(cast(jax.Array, root.prior_logits), invalid_actions),
        value=root.value,
        embedding=root.embedding,
    )
    return mctx_search.search(
        params=params,
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        root_action_selection_fn=_dirichlet_root_action_selection,
        interior_action_selection_fn=_get_interior_action_selection_fn(interior_selector),
        num_simulations=num_simulations,
        max_depth=max_depth,
        invalid_actions=invalid_actions,
        extra_data=DirichletRootExtra(alpha_Q_prior=alpha_Q_prior),
        loop_fn=loop_fn,
    )


def repeated_dirichlet_q_search(
    params,
    rng_key: jax.Array,
    root: mctx.RootFnOutput,
    recurrent_fn: mctx.RecurrentFn,
    alpha_Q_prior: jax.Array,
    num_simulations: int,
    invalid_actions: jax.Array | None,
    interior_selector: str,
    num_search_blocks: int,
    max_depth: int | None = None,
    loop_fn=jax.lax.fori_loop,
) -> tuple[jax.Array, jax.Array, mctx.Tree]:
    """Run repeated MCTX blocks, carrying the root posterior between blocks."""
    if num_search_blocks < 1:
        raise ValueError("num_search_blocks must be >= 1")

    alpha_Q_post = alpha_Q_prior
    q_evidence_total = jnp.zeros_like(alpha_Q_prior)
    search_tree: mctx.Tree | None = None

    for block_index in range(num_search_blocks):
        block_key = jax.random.fold_in(rng_key, block_index)
        block_root = mctx.RootFnOutput(
            prior_logits=root.prior_logits,
            value=root.value,
            embedding=root.embedding._replace(alpha_Q_prior=alpha_Q_post),
        )
        search_tree = dirichlet_q_search(
            params=params,
            rng_key=block_key,
            root=block_root,
            recurrent_fn=recurrent_fn,
            alpha_Q_prior=alpha_Q_post,
            num_simulations=num_simulations,
            invalid_actions=invalid_actions,
            interior_selector=interior_selector,
            max_depth=max_depth,
            loop_fn=loop_fn,
        )
        block_q_evidence = q_evidence_sum(search_tree, alpha_Q_prior.shape[1], alpha_Q_prior.dtype)
        q_evidence_total = q_evidence_total + block_q_evidence
        alpha_Q_post = alpha_Q_post + block_q_evidence

    return q_evidence_total, alpha_Q_post, cast(mctx.Tree, search_tree)


def dirichlet_q_policy(
    params,
    rng_key: jax.Array,
    root: mctx.RootFnOutput,
    recurrent_fn: mctx.RecurrentFn,
    num_simulations: int,
    *,
    alpha_Q_prior: jax.Array,
    invalid_actions: jax.Array | None = None,
    max_depth: int | None = None,
    loop_fn=jax.lax.fori_loop,
    policy_mc_samples: int,
    interior_selector: str = "policy_prior",
    num_search_blocks: int = 1,
    action_selection_mode: Literal["sample", "argmax"] = "sample",
) -> DirichletQPolicyOutput:
    """Run Dirichlet-Q search and return an MCTX-like policy output."""
    if invalid_actions is None:
        invalid_actions = jnp.zeros_like(root.prior_logits, dtype=bool)

    search_key, mc_key, action_key = jax.random.split(rng_key, 3)
    q_evidence, alpha_Q_post, search_tree = repeated_dirichlet_q_search(
        params=params,
        rng_key=search_key,
        root=root,
        recurrent_fn=recurrent_fn,
        alpha_Q_prior=alpha_Q_prior,
        num_simulations=num_simulations,
        invalid_actions=invalid_actions,
        interior_selector=interior_selector,
        num_search_blocks=num_search_blocks,
        max_depth=max_depth,
        loop_fn=loop_fn,
    )
    policy_target = posterior_best_policy_target(
        mc_key,
        alpha_Q_post,
        invalid_actions,
        policy_mc_samples,
    )

    if action_selection_mode == "sample":
        action_logits = jnp.where(
            policy_target > 0,
            jnp.log(jnp.clip(policy_target, 1e-8, 1.0)),
            jnp.finfo(policy_target.dtype).min,
        )
        action_logits = jnp.where(
            ~invalid_actions,
            action_logits,
            jnp.finfo(action_logits.dtype).min,
        )
        action = jax.random.categorical(action_key, action_logits, axis=-1)
    elif action_selection_mode == "argmax":
        action = cast(jax.Array, action_selection.masked_argmax(policy_target, invalid_actions))
    else:
        raise ValueError(f"Unknown action_selection_mode: {action_selection_mode}")

    return DirichletQPolicyOutput(
        action=action,
        action_weights=policy_target,
        search_tree=search_tree,
        q_evidence_sum=q_evidence,
        alpha_Q_post=alpha_Q_post,
    )
