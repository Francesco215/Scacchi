import jax
import jax.numpy as jnp
import mctx

from scacchi.play import (
    DirichletRootExtra,
    NodeEmbedding,
    _dirichlet_root_action_selection,
    _policy_prior_interior_action_selection,
    _q_evidence_sum_unbatched,
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
    root_invalid_actions=None,
    alpha_Q_prior=None,
):
    num_nodes = node_visits.shape[0]
    num_actions = (
        alpha_Q_prior.shape[0]
        if alpha_Q_prior is not None
        else children_prior_logits.shape[-1]
    )
    dtype = wdl_dist.dtype
    if children_prior_logits is None:
        children_prior_logits = jnp.zeros((num_nodes, num_actions), dtype=dtype)
    if children_visits is None:
        children_visits = jnp.zeros((num_nodes, num_actions), dtype=jnp.int32)
    if root_invalid_actions is None:
        root_invalid_actions = jnp.zeros((num_actions,), dtype=bool)
    if alpha_Q_prior is None:
        alpha_Q_prior = jnp.ones((num_actions, 3), dtype=dtype)

    return mctx.Tree(
        node_visits=node_visits.astype(jnp.int32),
        raw_values=jnp.zeros((num_nodes,), dtype=dtype),
        node_values=jnp.zeros((num_nodes,), dtype=dtype),
        parents=jnp.full((num_nodes,), mctx.Tree.NO_PARENT, dtype=jnp.int32),
        action_from_parent=jnp.full((num_nodes,), mctx.Tree.NO_PARENT, dtype=jnp.int32),
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
            state=jnp.zeros((num_nodes,), dtype=jnp.int32),
            wdl_dist=wdl_dist,
            evidence_weight=evidence_weight,
            root_action=root_action,
            depth_parity=depth_parity,
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
