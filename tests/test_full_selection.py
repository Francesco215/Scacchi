from typing import Any, cast

import jax
import jax.numpy as jnp
import mctx

from scacchi.play import (
    DirichletRootExtra,
    NodeEmbedding,
    _child_evidence_sum_unbatched,
    _dirichlet_root_action_selection,
    _get_interior_action_selection_fn,
    _policy_prior_interior_action_selection,
    _q_evidence_sum_unbatched,
    _wdl_interior_action_selection,
)


def _make_tree(
    *,
    wdl_dist,
    evidence_weight,
    root_action,
    depth_parity,
    node_visits,
    children_prior_logits=None,
    children_visits=None,
    parents=None,
    action_from_parent=None,
    root_invalid_actions=None,
    alpha_Q_prior=None,
    embedding_alpha_Q_prior=None,
):
    num_nodes = node_visits.shape[0]
    dtype = wdl_dist.dtype
    if alpha_Q_prior is not None:
        num_actions = alpha_Q_prior.shape[0]
    elif children_prior_logits is not None:
        num_actions = children_prior_logits.shape[-1]
    else:
        raise ValueError("alpha_Q_prior or children_prior_logits is required")
    if children_prior_logits is None:
        children_prior_logits = jnp.zeros((num_nodes, num_actions), dtype=dtype)
    if children_visits is None:
        children_visits = jnp.zeros((num_nodes, num_actions), dtype=jnp.int32)
    if root_invalid_actions is None:
        root_invalid_actions = jnp.zeros((num_actions,), dtype=bool)
    if alpha_Q_prior is None:
        alpha_Q_prior = jnp.ones((num_actions, 3), dtype=dtype)
    if embedding_alpha_Q_prior is None:
        embedding_alpha_Q_prior = jnp.broadcast_to(
            alpha_Q_prior[None, :, :],
            (num_nodes, num_actions, 3),
        )
    if parents is None:
        parents = jnp.full((num_nodes,), mctx.Tree.NO_PARENT, dtype=jnp.int32)
    if action_from_parent is None:
        action_from_parent = jnp.full((num_nodes,), mctx.Tree.NO_PARENT, dtype=jnp.int32)

    return mctx.Tree(
        node_visits=node_visits.astype(jnp.int32),
        raw_values=jnp.zeros((num_nodes,), dtype=dtype),
        node_values=jnp.zeros((num_nodes,), dtype=dtype),
        parents=parents,
        action_from_parent=action_from_parent,
        children_index=jnp.full(
            (num_nodes, num_actions),
            mctx.Tree.UNVISITED,
            dtype=jnp.int32,
        ),
        children_prior_logits=children_prior_logits,
        children_visits=children_visits,
        children_rewards=jnp.zeros((num_nodes, num_actions), dtype=dtype),
        children_discounts=jnp.zeros((num_nodes, num_actions), dtype=dtype),
        children_values=jnp.zeros((num_nodes, num_actions), dtype=dtype),
        embeddings=NodeEmbedding(
            state=cast(Any, jnp.zeros((num_nodes,), dtype=jnp.int32)),
            wdl_dist=wdl_dist,
            evidence_weight=evidence_weight,
            root_action=root_action,
            depth_parity=depth_parity,
            alpha_Q_prior=embedding_alpha_Q_prior,
        ),
        root_invalid_actions=root_invalid_actions,
        extra_data=DirichletRootExtra(alpha_Q_prior=alpha_Q_prior),
    )


def test_q_evidence_sum_unbatched_routes_and_flips_wdl():
    tree = _make_tree(
        wdl_dist=jnp.array(
            [
                [0.2, 0.3, 0.5],
                [0.1, 0.2, 0.7],
                [0.3, 0.4, 0.3],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=jnp.float32,
        ),
        evidence_weight=jnp.array([0.0, 2.0, 3.0, 4.0, 5.0], dtype=jnp.float32),
        root_action=jnp.array([mctx.Tree.NO_PARENT, 0, 1, 0, 2], dtype=jnp.int32),
        depth_parity=jnp.array([0, 1, 0, 0, 1], dtype=jnp.int32),
        node_visits=jnp.array([1, 1, 1, 0, 1], dtype=jnp.int32),
        alpha_Q_prior=jnp.ones((3, 3), dtype=jnp.float32),
    )

    evidence = _q_evidence_sum_unbatched(tree, 3, jnp.float32)

    expected = jnp.array(
        [
            2.0 * jnp.array([0.7, 0.2, 0.1]),
            3.0 * jnp.array([0.3, 0.4, 0.3]),
            5.0 * jnp.array([0.0, 1.0, 0.0]),
        ],
        dtype=jnp.float32,
    )
    assert jnp.allclose(evidence, expected)


def test_dirichlet_root_action_selection_matches_sampled_utility_argmax():
    alpha_Q_prior = jnp.array(
        [
            [2.0, 1.0, 5.0],
            [4.0, 1.0, 2.0],
            [1.0, 1.0, 20.0],
        ],
        dtype=jnp.float32,
    )
    tree = _make_tree(
        wdl_dist=jnp.array(
            [
                [0.2, 0.3, 0.5],
                [0.1, 0.2, 0.7],
                [0.6, 0.2, 0.2],
            ],
            dtype=jnp.float32,
        ),
        evidence_weight=jnp.array([0.0, 2.0, 3.0], dtype=jnp.float32),
        root_action=jnp.array([mctx.Tree.NO_PARENT, 0, 1], dtype=jnp.int32),
        depth_parity=jnp.array([0, 1, 0], dtype=jnp.int32),
        node_visits=jnp.array([1, 1, 1], dtype=jnp.int32),
        root_invalid_actions=jnp.array([False, False, True]),
        alpha_Q_prior=alpha_Q_prior,
    )
    key = jax.random.PRNGKey(7)

    alpha_Q_post = alpha_Q_prior + _q_evidence_sum_unbatched(
        tree, alpha_Q_prior.shape[0], alpha_Q_prior.dtype,
    )
    phi = jax.random.dirichlet(key, alpha_Q_post)
    score = phi[..., 2] - phi[..., 0]
    score = jnp.where(tree.root_invalid_actions, -jnp.inf, score)
    expected = jnp.argmax(score).astype(jnp.int32)

    actual = _dirichlet_root_action_selection(key, tree, jnp.array(0, dtype=jnp.int32))

    assert int(actual) == int(expected)
    assert int(actual) != 2


def test_child_evidence_sum_unbatched_routes_descendants_under_interior_node():
    tree = _make_tree(
        wdl_dist=jnp.array(
            [
                [0.0, 1.0, 0.0],
                [0.2, 0.2, 0.6],
                [0.1, 0.2, 0.7],
                [0.8, 0.1, 0.1],
                [0.0, 0.3, 0.7],
                [1.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        ),
        evidence_weight=jnp.array([0.0, 0.0, 2.0, 3.0, 5.0, 7.0], dtype=jnp.float32),
        root_action=jnp.array(
            [mctx.Tree.NO_PARENT, 0, 0, 0, 0, 1],
            dtype=jnp.int32,
        ),
        depth_parity=jnp.array([0, 1, 0, 0, 1, 1], dtype=jnp.int32),
        node_visits=jnp.ones((6,), dtype=jnp.int32),
        parents=jnp.array(
            [mctx.Tree.NO_PARENT, 0, 1, 1, 2, 0],
            dtype=jnp.int32,
        ),
        action_from_parent=jnp.array(
            [mctx.Tree.NO_PARENT, 0, 2, 1, 0, 1],
            dtype=jnp.int32,
        ),
        alpha_Q_prior=jnp.ones((3, 3), dtype=jnp.float32),
    )

    evidence = _child_evidence_sum_unbatched(
        tree,
        jnp.array(1, dtype=jnp.int32),
        3,
        jnp.float32,
    )

    expected = jnp.array(
        [
            [0.0, 0.0, 0.0],
            3.0 * jnp.array([0.1, 0.1, 0.8]),
            2.0 * jnp.array([0.7, 0.2, 0.1]) + 5.0 * jnp.array([0.0, 0.3, 0.7]),
        ],
        dtype=jnp.float32,
    )
    assert jnp.allclose(evidence, expected)


def test_wdl_interior_action_selection_matches_sampled_utility_argmax():
    alpha_Q_prior = jnp.array(
        [
            [2.0, 1.0, 2.0],
            [1.0, 1.0, 1.0],
            [1.0, 2.0, 5.0],
        ],
        dtype=jnp.float32,
    )
    embedding_alpha_Q_prior = jnp.broadcast_to(
        jnp.ones((3, 3), dtype=jnp.float32)[None, :, :],
        (5, 3, 3),
    ).at[1].set(alpha_Q_prior)
    tree = _make_tree(
        wdl_dist=jnp.array(
            [
                [0.0, 1.0, 0.0],
                [0.2, 0.2, 0.6],
                [0.1, 0.2, 0.7],
                [0.7, 0.1, 0.2],
                [0.0, 0.2, 0.8],
            ],
            dtype=jnp.float32,
        ),
        evidence_weight=jnp.array([0.0, 0.0, 2.0, 3.0, 4.0], dtype=jnp.float32),
        root_action=jnp.array([mctx.Tree.NO_PARENT, 0, 0, 0, 0], dtype=jnp.int32),
        depth_parity=jnp.array([0, 1, 0, 0, 1], dtype=jnp.int32),
        node_visits=jnp.ones((5,), dtype=jnp.int32),
        parents=jnp.array([mctx.Tree.NO_PARENT, 0, 1, 1, 2], dtype=jnp.int32),
        action_from_parent=jnp.array([mctx.Tree.NO_PARENT, 0, 2, 1, 0], dtype=jnp.int32),
        children_prior_logits=jnp.zeros((5, 3), dtype=jnp.float32),
        alpha_Q_prior=jnp.ones((3, 3), dtype=jnp.float32),
        embedding_alpha_Q_prior=embedding_alpha_Q_prior,
    )
    key = jax.random.PRNGKey(11)

    alpha_Q_post = alpha_Q_prior + _child_evidence_sum_unbatched(
        tree,
        jnp.array(1, dtype=jnp.int32),
        alpha_Q_prior.shape[0],
        alpha_Q_prior.dtype,
    )
    phi = jax.random.dirichlet(key, alpha_Q_post)
    expected = jnp.argmax(phi[..., 2] - phi[..., 0]).astype(jnp.int32)

    actual = _wdl_interior_action_selection(
        key,
        tree,
        jnp.array(1, dtype=jnp.int32),
        jnp.array(1, dtype=jnp.int32),
    )

    assert int(actual) == int(expected)


def test_wdl_interior_action_selection_ignores_masked_logits():
    min_logit = jnp.finfo(jnp.float32).min
    alpha_Q_prior = jnp.array(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1000.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=jnp.float32,
    )
    embedding_alpha_Q_prior = jnp.broadcast_to(
        alpha_Q_prior[None, :, :],
        (1, 4, 3),
    )
    tree = _make_tree(
        wdl_dist=jnp.array([[0.2, 0.3, 0.5]], dtype=jnp.float32),
        evidence_weight=jnp.array([0.0], dtype=jnp.float32),
        root_action=jnp.array([mctx.Tree.NO_PARENT], dtype=jnp.int32),
        depth_parity=jnp.array([0], dtype=jnp.int32),
        node_visits=jnp.array([1], dtype=jnp.int32),
        children_prior_logits=jnp.array([[0.0, min_logit, 2.0, 1.0]], dtype=jnp.float32),
        children_visits=jnp.array([[100, 0, 0, 0]], dtype=jnp.int32),
        alpha_Q_prior=alpha_Q_prior,
        embedding_alpha_Q_prior=embedding_alpha_Q_prior,
    )

    action = _wdl_interior_action_selection(
        jax.random.PRNGKey(0),
        tree,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(1, dtype=jnp.int32),
    )

    assert int(action) != 1


def test_policy_prior_interior_action_selection_ignores_masked_logits():
    min_logit = jnp.finfo(jnp.float32).min
    tree = _make_tree(
        wdl_dist=jnp.array([[0.2, 0.3, 0.5]], dtype=jnp.float32),
        evidence_weight=jnp.array([0.0], dtype=jnp.float32),
        root_action=jnp.array([mctx.Tree.NO_PARENT], dtype=jnp.int32),
        depth_parity=jnp.array([0], dtype=jnp.int32),
        node_visits=jnp.array([1], dtype=jnp.int32),
        children_prior_logits=jnp.array([[0.0, min_logit, 2.0, 1.0]], dtype=jnp.float32),
        children_visits=jnp.array([[100, 0, 0, 0]], dtype=jnp.int32),
        alpha_Q_prior=jnp.ones((4, 3), dtype=jnp.float32),
    )

    action = _policy_prior_interior_action_selection(
        jax.random.PRNGKey(0),
        tree,
        jnp.array(0, dtype=jnp.int32),
        jnp.array(1, dtype=jnp.int32),
    )

    assert int(action) == 2


def test_get_interior_action_selection_fn_dispatches_modes():
    assert _get_interior_action_selection_fn("policy_prior") is _policy_prior_interior_action_selection
    assert _get_interior_action_selection_fn("wdl") is _wdl_interior_action_selection
