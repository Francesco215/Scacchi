from types import SimpleNamespace
import jax
import jax.numpy as jnp
import mctx

from scacchi.dirichlet_q_search import (
    DirichletRootExtraData,
    NO_PARENT,
    NodeEmbedding,
    _dirichlet_q_search_block,
    _q_evidence_sum_from_unbatched_tree,
    dirichlet_q_policy,
    dirichlet_root_action_selection,
    flip_outcome,
    outcome_utility,
    posterior_best_action,
    posterior_best_policy_target,
    posterior_sample_action,
    posterior_targets,
    q_evidence_sum_from_tree,
    root_action_value_priors_from_tree,
    terminal_outcome_from_reward,
)


class _FakeTree(SimpleNamespace):
    @property
    def num_actions(self):
        return self.children_index.shape[-1]


def _fake_tree(embedding: NodeEmbedding, node_visits: jax.Array, num_actions: int):
    return _FakeTree(
        embeddings=embedding,
        node_visits=node_visits,
        children_index=jnp.zeros((*node_visits.shape, num_actions), dtype=jnp.int32),
    )


def _fake_unbatched_tree(
    embedding: NodeEmbedding,
    node_visits: jax.Array,
    num_actions: int,
    *,
    action_value_prior: jax.Array | None = None,
    root_invalid_actions: jax.Array | None = None,
    children_index: jax.Array | None = None,
):
    if action_value_prior is None:
        action_value_prior = jnp.ones((num_actions, embedding.outcome_dist.shape[-1]))
    if root_invalid_actions is None:
        root_invalid_actions = jnp.zeros((num_actions,), dtype=bool)
    if children_index is None:
        children_index = jnp.full((node_visits.shape[0], num_actions), NO_PARENT, dtype=jnp.int32)
    return _FakeTree(
        embeddings=embedding,
        node_visits=node_visits,
        children_index=children_index,
        extra_data=DirichletRootExtraData(
            action_value_prior=action_value_prior,
            explored_action_mask=jnp.zeros((num_actions,), dtype=bool),
        ),
        root_invalid_actions=root_invalid_actions,
    )


def _toy_root(num_actions: int = 2):
    root_embedding = NodeEmbedding(
        state=jnp.zeros((1,), dtype=jnp.int32),
        outcome_dist=jnp.full((1, 2), 0.5),
        alpha_V_prior=jnp.full((1, 2), 1.0),
        evidence_weight=jnp.zeros((1,), dtype=jnp.float32),
        root_action=jnp.full((1,), NO_PARENT),
        depth_parity=jnp.zeros((1,), dtype=jnp.int32),
        alpha_Q_prior=jnp.ones((1, num_actions, 2)),
    )
    return mctx.RootFnOutput(
        prior_logits=jnp.zeros((1, num_actions)),
        value=jnp.zeros((1,)),
        embedding=root_embedding,
    )


def _toy_recurrent_fn(_, rng_key, action, embedding: NodeEmbedding):
    del rng_key
    batch_size = action.shape[0]
    win_prob = jnp.where(action == 0, 0.75, 0.25).astype(jnp.float32)
    outcome_dist = jnp.stack([1.0 - win_prob, win_prob], axis=-1)
    evidence_weight = jnp.ones((batch_size,), dtype=jnp.float32)
    root_action = jnp.where(embedding.root_action == NO_PARENT, action, embedding.root_action)
    depth_parity = 1 - embedding.depth_parity
    next_embedding = NodeEmbedding(
        state=embedding.state + 1,
        outcome_dist=outcome_dist,
        alpha_V_prior=1.0 + outcome_dist,
        evidence_weight=evidence_weight,
        root_action=root_action,
        depth_parity=depth_parity,
        alpha_Q_prior=embedding.alpha_Q_prior,
    )
    return (
        mctx.RecurrentFnOutput(
            reward=jnp.zeros((batch_size,)),
            discount=jnp.zeros((batch_size,)),
            prior_logits=jnp.zeros((batch_size, 2)),
            value=outcome_utility(outcome_dist),
        ),
        next_embedding,
    )


def test_root_thompson_selector_matches_dirichlet_utility_draw():
    action_value_prior = jnp.array([[2.0, 1.0], [1.0, 3.0], [3.0, 1.0]])
    embedding = NodeEmbedding(
        state=None,
        outcome_dist=jnp.array([[0.5, 0.5], [0.1, 0.9], [0.8, 0.2]]),
        alpha_V_prior=jnp.ones((3, 2)),
        evidence_weight=jnp.array([0.0, 2.0, 3.0]),
        root_action=jnp.array([NO_PARENT, 0, 1]),
        depth_parity=jnp.array([0, 0, 1]),
        alpha_Q_prior=jnp.zeros((3, 3, 2)),
    )
    tree = _fake_unbatched_tree(
        embedding,
        jnp.array([1, 1, 1]),
        num_actions=3,
        action_value_prior=action_value_prior,
    )
    rng_key = jax.random.PRNGKey(7)
    alpha_post = action_value_prior + _q_evidence_sum_from_unbatched_tree(tree)
    phi = jax.random.dirichlet(rng_key, alpha_post)
    expected = jnp.argmax(outcome_utility(phi), axis=-1)

    action = dirichlet_root_action_selection(rng_key, tree, 0)

    assert int(action) == int(expected)


def test_root_thompson_selector_masks_invalid_high_utility_action():
    action_value_prior = jnp.array([[1000.0, 1.0], [1.0, 1000.0], [500.0, 1.0]])
    root_invalid_actions = jnp.array([False, True, False])
    embedding = NodeEmbedding(
        state=None,
        outcome_dist=jnp.array([[0.5, 0.5]]),
        alpha_V_prior=jnp.ones((1, 2)),
        evidence_weight=jnp.array([0.0]),
        root_action=jnp.array([NO_PARENT]),
        depth_parity=jnp.array([0]),
        alpha_Q_prior=jnp.zeros((1, 3, 2)),
    )
    tree = _fake_unbatched_tree(
        embedding,
        jnp.array([1]),
        num_actions=3,
        action_value_prior=action_value_prior,
        root_invalid_actions=root_invalid_actions,
    )
    rng_key = jax.random.PRNGKey(0)
    scores = outcome_utility(jax.random.dirichlet(rng_key, action_value_prior))
    expected = jnp.argmax(jnp.where(root_invalid_actions, -jnp.inf, scores), axis=-1)

    action = dirichlet_root_action_selection(rng_key, tree, 0)

    assert int(jnp.argmax(scores, axis=-1)) == 1
    assert int(action) == int(expected)


def test_root_action_prior_uses_child_v_only_after_action_is_explored():
    embedding = NodeEmbedding(
        state=None,
        outcome_dist=jnp.full((1, 3, 2), 0.5),
        alpha_V_prior=jnp.array([[[9.0, 9.0], [3.0, 1.0], [7.0, 5.0]]]),
        evidence_weight=jnp.zeros((1, 3)),
        root_action=jnp.array([[NO_PARENT, 0, 1]]),
        depth_parity=jnp.array([[0, 1, 1]]),
        alpha_Q_prior=jnp.zeros((1, 3, 2, 2)),
    )
    tree = _FakeTree(
        embeddings=embedding,
        node_visits=jnp.array([[1, 1, 0]]),
        children_index=jnp.array([[[1, NO_PARENT], [NO_PARENT, NO_PARENT], [NO_PARENT, NO_PARENT]]]),
    )
    action_value_prior = jnp.array([[[1.0, 10.0], [2.0, 20.0]]])

    mixed_prior = root_action_value_priors_from_tree(tree, action_value_prior)

    assert jnp.allclose(mixed_prior, jnp.array([[[1.0, 3.0], [2.0, 20.0]]]))


def test_root_action_prior_keeps_carried_posterior_for_previously_explored_action():
    embedding = NodeEmbedding(
        state=None,
        outcome_dist=jnp.full((1, 2, 2), 0.5),
        alpha_V_prior=jnp.array([[[9.0, 9.0], [3.0, 1.0]]]),
        evidence_weight=jnp.zeros((1, 2)),
        root_action=jnp.array([[NO_PARENT, 0]]),
        depth_parity=jnp.array([[0, 1]]),
        alpha_Q_prior=jnp.zeros((1, 2, 2, 2)),
    )
    tree = _FakeTree(
        embeddings=embedding,
        node_visits=jnp.array([[1, 1]]),
        children_index=jnp.array([[[1, NO_PARENT], [NO_PARENT, NO_PARENT]]]),
    )
    carried_posterior = jnp.array([[[5.0, 6.0], [2.0, 20.0]]])
    explored_action_mask = jnp.array([[True, False]])

    mixed_prior = root_action_value_priors_from_tree(
        tree,
        carried_posterior,
        explored_action_mask,
    )

    assert jnp.allclose(mixed_prior, carried_posterior)


def test_batched_and_unbatched_root_evidence_sums_agree_on_toy_tree():
    embedding = NodeEmbedding(
        state=None,
        outcome_dist=jnp.array(
            [
                [[0.5, 0.5], [0.2, 0.8], [0.7, 0.3]],
                [[0.5, 0.5], [0.1, 0.9], [0.4, 0.6]],
            ]
        ),
        alpha_V_prior=jnp.ones((2, 3, 2)),
        evidence_weight=jnp.array([[0.0, 2.0, 3.0], [0.0, 5.0, 7.0]]),
        root_action=jnp.array([[NO_PARENT, 0, 2], [NO_PARENT, 1, 1]]),
        depth_parity=jnp.array([[0, 0, 1], [0, 1, 0]]),
        alpha_Q_prior=jnp.zeros((2, 3, 3, 2)),
    )
    node_visits = jnp.array([[1, 1, 1], [1, 1, 1]])
    tree = _fake_tree(embedding, node_visits, num_actions=3)

    batched = q_evidence_sum_from_tree(tree)
    unbatched = []
    for batch_index in range(2):
        unbatched_embedding = NodeEmbedding(
            state=None,
            outcome_dist=embedding.outcome_dist[batch_index],
            alpha_V_prior=embedding.alpha_V_prior[batch_index],
            evidence_weight=embedding.evidence_weight[batch_index],
            root_action=embedding.root_action[batch_index],
            depth_parity=embedding.depth_parity[batch_index],
            alpha_Q_prior=embedding.alpha_Q_prior[batch_index],
        )
        unbatched.append(
            _q_evidence_sum_from_unbatched_tree(
                _fake_unbatched_tree(
                    unbatched_embedding,
                    node_visits[batch_index],
                    num_actions=3,
                )
            )
        )

    assert jnp.allclose(batched, jnp.stack(unbatched))


def test_repeated_search_blocks_aggregate_evidence_and_carry_posterior():
    rng_key = jax.random.PRNGKey(11)
    block_keys = jax.random.split(rng_key, 2)
    root = _toy_root()
    action_value_prior = jnp.full((1, 2, 2), 2.0)
    invalid_actions = jnp.array([[False, False]])

    block_1 = _dirichlet_q_search_block(
        params=(),
        rng_key=block_keys[0],
        root=root,
        recurrent_fn=_toy_recurrent_fn,
        action_value_prior=action_value_prior,
        explored_action_mask=jnp.zeros(action_value_prior.shape[:-1], dtype=bool),
        num_simulations=1,
        invalid_actions=invalid_actions,
    )
    block_2 = _dirichlet_q_search_block(
        params=(),
        rng_key=block_keys[1],
        root=root,
        recurrent_fn=_toy_recurrent_fn,
        action_value_prior=block_1.alpha_search,
        explored_action_mask=block_1.explored_action_mask,
        num_simulations=1,
        invalid_actions=invalid_actions,
    )

    repeated = dirichlet_q_policy(
        params=(),
        rng_key=rng_key,
        root=root,
        recurrent_fn=_toy_recurrent_fn,
        action_value_prior=action_value_prior,
        num_simulations=1,
        invalid_actions=invalid_actions,
        num_search_blocks=2,
    )

    expected_evidence = block_1.q_evidence_sum + block_2.q_evidence_sum
    assert jnp.allclose(repeated.q_evidence_sum, expected_evidence)
    assert jnp.allclose(repeated.alpha_search, block_2.alpha_search)
    assert jnp.allclose(
        repeated.search_tree.extra_data.action_value_prior,
        block_1.alpha_search,
    )


def test_single_search_block_matches_one_block_policy():
    rng_key = jax.random.PRNGKey(3)
    root = _toy_root()
    action_value_prior = jnp.full((1, 2, 2), 2.0)
    invalid_actions = jnp.array([[False, False]])

    block = _dirichlet_q_search_block(
        params=(),
        rng_key=rng_key,
        root=root,
        recurrent_fn=_toy_recurrent_fn,
        action_value_prior=action_value_prior,
        explored_action_mask=jnp.zeros(action_value_prior.shape[:-1], dtype=bool),
        num_simulations=2,
        invalid_actions=invalid_actions,
    )
    policy = dirichlet_q_policy(
        params=(),
        rng_key=rng_key,
        root=root,
        recurrent_fn=_toy_recurrent_fn,
        action_value_prior=action_value_prior,
        num_simulations=2,
        invalid_actions=invalid_actions,
        num_search_blocks=1,
    )

    assert jnp.array_equal(policy.action, block.action)
    assert jnp.allclose(policy.action_weights, block.action_weights)
    assert jnp.allclose(policy.q_evidence_sum, block.q_evidence_sum)
    assert jnp.allclose(policy.alpha_search, block.alpha_search)


def test_zero_simulation_policy_uses_q_prior_without_search_tree():
    rng_key = jax.random.PRNGKey(17)
    root = _toy_root(num_actions=3)
    action_value_prior = jnp.array(
        [[[1.0, 4.0], [4.0, 1.0], [2.0, 2.0]]],
        dtype=jnp.float32,
    )
    invalid_actions = jnp.array([[False, False, True]])

    policy = dirichlet_q_policy(
        params=(),
        rng_key=rng_key,
        root=root,
        recurrent_fn=_toy_recurrent_fn,
        action_value_prior=action_value_prior,
        num_simulations=0,
        invalid_actions=invalid_actions,
    )

    assert policy.search_tree is None
    assert jnp.allclose(policy.q_evidence_sum, jnp.zeros_like(action_value_prior))
    assert jnp.allclose(policy.alpha_search, action_value_prior)
    assert not bool(policy.explored_action_mask.any())
    assert policy.action.shape == (1,)
    assert policy.action_weights.shape == (1, 3)
    assert float(policy.action_weights[0, 2]) == 0.0
    assert float(policy.action_weights.sum()) == 1.0


def test_q_evidence_routes_by_root_action_and_aligns_parity():
    embedding = NodeEmbedding(
        state=None,
        outcome_dist=jnp.array([[[0.5, 0.5], [0.2, 0.8], [0.7, 0.3], [0.1, 0.9]]]),
        alpha_V_prior=jnp.ones((1, 4, 2)),
        evidence_weight=jnp.array([[0.0, 2.0, 3.0, 5.0]]),
        root_action=jnp.array([[NO_PARENT, 0, 2, 0]]),
        depth_parity=jnp.array([[0, 0, 1, 1]]),
        alpha_Q_prior=jnp.zeros((1, 4, 3, 2)),
    )
    tree = _fake_tree(embedding, jnp.array([[1, 1, 1, 1]]), num_actions=3)

    evidence_sum = q_evidence_sum_from_tree(tree)

    expected = jnp.array(
        [
            [
                2.0 * jnp.array([0.2, 0.8]) + 5.0 * jnp.array([0.9, 0.1]),
                jnp.array([0.0, 0.0]),
                3.0 * jnp.array([0.3, 0.7]),
            ]
        ]
    )
    assert evidence_sum.shape == (1, 3, 2)
    assert jnp.allclose(evidence_sum, expected)


def test_terminal_child_outcome_scatters_back_to_root_perspective():
    terminal_parent = terminal_outcome_from_reward(jnp.array([1.0]), 2)
    terminal_child = flip_outcome(terminal_parent)
    embedding = NodeEmbedding(
        state=None,
        outcome_dist=jnp.array([[[0.5, 0.5], terminal_child[0]]]),
        alpha_V_prior=jnp.ones((1, 2, 2)),
        evidence_weight=jnp.array([[0.0, 8.0]]),
        root_action=jnp.array([[NO_PARENT, 1]]),
        depth_parity=jnp.array([[0, 1]]),
        alpha_Q_prior=jnp.zeros((1, 2, 2, 2)),
    )
    tree = _fake_tree(embedding, jnp.array([[1, 1]]), num_actions=2)

    evidence_sum = q_evidence_sum_from_tree(tree)

    assert jnp.allclose(terminal_parent, jnp.array([[0.0, 1.0]]))
    assert jnp.allclose(terminal_child, jnp.array([[1.0, 0.0]]))
    assert jnp.allclose(evidence_sum[0, 1], jnp.array([0.0, 8.0]))


def test_posterior_best_policy_target_masks_invalid_actions():
    alpha_Q_post = jnp.array([[[1.0, 2.0], [1.0, 1000.0], [2.0, 1.0]]])
    legal_action_mask = jnp.array([[True, False, True]])

    policy_target = posterior_best_policy_target(
        jax.random.PRNGKey(0),
        alpha_Q_post,
        legal_action_mask,
        num_samples=128,
    )

    assert policy_target.shape == (1, 3)
    assert jnp.allclose(policy_target[0, 1], 0.0)
    assert jnp.allclose(policy_target.sum(axis=-1), 1.0)


def test_posterior_best_action_is_argmax_policy_target_and_masks_invalid():
    policy_target = jnp.array([[0.2, 0.7, 0.1], [0.6, 0.4, 0.0]])
    legal_action_mask = jnp.array([[True, False, True], [True, True, False]])

    action = posterior_best_action(policy_target, legal_action_mask)

    assert jnp.array_equal(action, jnp.array([0, 0], dtype=jnp.int32))


def test_posterior_sample_action_samples_policy_target_and_masks_invalid():
    policy_target = jnp.array([[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    legal_action_mask = jnp.array([[True, False, True], [False, True, False]])

    action = posterior_sample_action(
        jax.random.PRNGKey(0),
        policy_target,
        legal_action_mask,
    )

    assert bool(legal_action_mask[0, action[0]])
    assert int(action[1]) == 1


def test_posterior_targets_add_q_evidence_to_action_value_prior_and_weight_value_evidence():
    alpha_V_prior = jnp.array([[1.0, 1.0]])
    action_value_prior = jnp.array([[[1.0, 2.0], [2.0, 1.0], [1.0, 2.0]]])
    q_evidence_sum = jnp.array([[[2.0, 0.0], [0.0, 0.0], [0.5, 1.5]]])
    policy_target = jnp.array([[0.25, 0.0, 0.75]])

    beta_Q_target, beta_V_target = posterior_targets(
        alpha_V_prior,
        action_value_prior,
        q_evidence_sum,
        policy_target,
    )

    assert jnp.allclose(beta_Q_target, action_value_prior + q_evidence_sum)
    expected_v_evidence = 0.25 * jnp.array([2.0, 0.0]) + 0.75 * jnp.array([0.5, 1.5])
    assert jnp.allclose(beta_V_target, alpha_V_prior + expected_v_evidence)
