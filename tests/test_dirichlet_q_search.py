from typing import NamedTuple

import jax
import jax.numpy as jnp

from scacchi import dirichlet_mctx
from scacchi.dirichlet_q_search import (
    NO_PARENT,
    NodeEmbedding,
    adapt_dirichlet_expand_fn_to_mctx,
    make_dirichlet_expand_fn_from_constants,
    posterior_best_action,
    posterior_best_policy_target,
    posterior_sample_action,
    posterior_targets,
    q_loss_weight_from_mode,
    terminal_outcome_from_reward,
)
from scacchi.types import SearchConstantsConfig


def _root(
    action_values: jax.Array,
    *,
    prior_logits: jax.Array | None = None,
    value: jax.Array | None = None,
    to_play: int = 0,
    terminal: bool = False,
) -> dirichlet_mctx.RootFnOutput:
    action_values = jnp.asarray(action_values, dtype=jnp.float32)
    if action_values.ndim == 2:
        action_values = action_values[None, ...]
    batch_size, num_actions, num_outcomes = action_values.shape
    if prior_logits is None:
        prior_logits = jnp.zeros((batch_size, num_actions), dtype=jnp.float32)
    if value is None:
        value = jnp.ones((batch_size, num_outcomes), dtype=jnp.float32)
    return dirichlet_mctx.RootFnOutput(
        prior_logits=jnp.asarray(prior_logits, dtype=jnp.float32),
        value=jnp.asarray(value, dtype=jnp.float32),
        action_values=action_values,
        embedding=jnp.zeros((batch_size,), dtype=jnp.int32),
        terminal=jnp.full((batch_size,), terminal, dtype=jnp.bool_),
        to_play=jnp.full((batch_size,), to_play, dtype=jnp.int32),
    )


def _constant_recurrent_fn(
    *,
    num_actions: int,
    value: tuple[float, ...],
    outcome: tuple[float, ...],
    evidence_weight: float,
    to_play: int,
    terminal: bool = False,
    prior_logits: tuple[float, ...] | None = None,
):
    if prior_logits is None:
        prior_logits = (0.0,) * num_actions

    def recurrent_fn(_, rng_key, action, depth):
        del rng_key
        batch_size = action.shape[0]
        step = dirichlet_mctx.RecurrentFnOutput(
            prior_logits=jnp.broadcast_to(
                jnp.asarray(prior_logits, dtype=jnp.float32),
                (batch_size, num_actions),
            ),
            value=jnp.broadcast_to(
                jnp.asarray(value, dtype=jnp.float32),
                (batch_size, len(value)),
            ),
            outcome=jnp.broadcast_to(
                jnp.asarray(outcome, dtype=jnp.float32),
                (batch_size, len(outcome)),
            ),
            evidence_weight=jnp.full(
                (batch_size,), evidence_weight, dtype=jnp.float32
            ),
            terminal=jnp.full((batch_size,), terminal, dtype=jnp.bool_),
            to_play=jnp.full((batch_size,), to_play, dtype=jnp.int32),
        )
        return step, depth + 1

    return recurrent_fn


def _policy(
    root: dirichlet_mctx.RootFnOutput,
    recurrent_fn,
    *,
    rng_key: jax.Array = jax.random.PRNGKey(0),
    num_simulations: int,
    invalid_actions: jax.Array | None = None,
    max_depth: int | None = None,
    num_search_blocks: int = 1,
    policy_samples: int = 0,
):
    if invalid_actions is None:
        invalid_actions = jnp.zeros_like(root.prior_logits, dtype=bool)
    return dirichlet_mctx.dirichlet_thompson_policy(
        params=(),
        rng_key=rng_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        invalid_actions=invalid_actions,
        max_depth=max_depth,
        num_search_blocks=num_search_blocks,
        policy_samples=policy_samples,
    )


class _ExpandState(NamedTuple):
    observation: jax.Array
    legal_action_mask: jax.Array
    current_player: jax.Array
    rewards: jax.Array
    terminated: jax.Array


class _ExpandEnv:
    def step(self, state: _ExpandState, action: jax.Array) -> _ExpandState:
        del action
        return _ExpandState(
            observation=state.observation + 1.0,
            legal_action_mask=state.legal_action_mask,
            current_player=1 - state.current_player,
            rewards=state.rewards,
            terminated=state.terminated,
        )


class _ExpandPrediction(NamedTuple):
    logits: jax.Array
    alpha_v: jax.Array


def _expand_evaluator(observation: jax.Array) -> _ExpandPrediction:
    batch_size = observation.shape[0]
    return _ExpandPrediction(
        logits=jnp.zeros((batch_size, 2), dtype=jnp.float32),
        alpha_v=jnp.broadcast_to(
            jnp.array([2.0, 6.0], dtype=jnp.float32),
            (batch_size, 2),
        ),
    )


def test_shared_expand_fn_drives_thompson_and_mctx_adapter():
    root_state = _ExpandState(
        observation=jnp.zeros((1, 1), dtype=jnp.float32),
        legal_action_mask=jnp.array([[True, True]]),
        current_player=jnp.array([0], dtype=jnp.int32),
        rewards=jnp.zeros((1, 2), dtype=jnp.float32),
        terminated=jnp.array([False]),
    )
    expand_fn = make_dirichlet_expand_fn_from_constants(
        _ExpandEnv(),
        _expand_evaluator,
        SearchConstantsConfig(kappa_leaf=2.0, kappa_terminal=8.0),
    )
    action = jnp.array([0], dtype=jnp.int32)
    step, child_state = expand_fn((), jax.random.PRNGKey(1), action, root_state)

    embedding = NodeEmbedding(
        state=root_state,
        outcome_dist=jnp.array([[0.5, 0.5]]),
        alpha_V_prior=jnp.ones((1, 2)),
        evidence_weight=jnp.zeros((1,)),
        root_action=jnp.array([NO_PARENT], dtype=jnp.int32),
        root_player=root_state.current_player,
    )
    mctx_step, mctx_child = adapt_dirichlet_expand_fn_to_mctx(expand_fn)(
        (),
        jax.random.PRNGKey(1),
        action,
        embedding,
    )
    assert all(
        bool(jnp.array_equal(actual, expected))
        for actual, expected in zip(
            jax.tree.leaves(mctx_child.state),
            jax.tree.leaves(child_state),
            strict=True,
        )
    )
    assert jnp.array_equal(mctx_child.alpha_V_prior, step.value)
    assert jnp.array_equal(mctx_child.outcome_dist, step.outcome)
    assert jnp.array_equal(mctx_step.prior_logits, step.prior_logits)
    assert jnp.allclose(mctx_step.value, dirichlet_mctx.outcome_utility(step.outcome))

    root = dirichlet_mctx.RootFnOutput(
        prior_logits=jnp.zeros((1, 2)),
        value=jnp.ones((1, 2)),
        action_values=jnp.array([[[9.0, 1.0], [1.0, 9.0]]]),
        embedding=root_state,
        terminal=root_state.terminated,
        to_play=root_state.current_player,
    )
    output = _policy(
        root,
        expand_fn,
        num_simulations=1,
        invalid_actions=jnp.array([[False, True]]),
    )
    assert jnp.allclose(
        output.search_tree.posterior.base[0, 0],
        jnp.array([6.0, 2.0]),
    )
    assert jnp.allclose(
        output.search_tree.posterior.evidence[0, 0],
        jnp.array([1.5, 0.5]),
    )


def test_root_thompson_selector_matches_draw_and_masks_invalid_action():
    action_values = jnp.array(
        [[[1000.0, 1.0], [1.0, 1000.0], [500.0, 1.0]]],
        dtype=jnp.float32,
    )
    invalid_actions = jnp.array([[False, True, False]])
    tree = dirichlet_mctx.instantiate_tree_from_root(
        _root(action_values),
        num_simulations=0,
        root_invalid_actions=invalid_actions,
    )
    unbatched_tree = jax.tree.map(lambda leaf: leaf[0], tree)
    rng_key = jax.random.PRNGKey(0)
    scores = dirichlet_mctx.outcome_utility(
        jax.random.dirichlet(rng_key, action_values[0])
    )
    expected = jnp.argmax(
        jnp.where(invalid_actions[0], -jnp.inf, scores)
    ).astype(jnp.int32)

    action = dirichlet_mctx.thompson_root_action_selection(
        rng_key,
        unbatched_tree,
        jnp.asarray(0, dtype=jnp.int32),
    )

    assert int(jnp.argmax(scores)) == 1
    assert int(action) == int(expected)
    assert not bool(invalid_actions[0, action])


def test_first_root_child_replaces_q_fallback_and_adds_aligned_evidence():
    root = _root([[9.0, 1.0], [1.0, 9.0]], to_play=0)
    recurrent_fn = _constant_recurrent_fn(
        num_actions=2,
        value=(2.0, 6.0),
        outcome=(0.25, 0.75),
        evidence_weight=4.0,
        to_play=1,
    )

    output = _policy(
        root,
        recurrent_fn,
        num_simulations=1,
        invalid_actions=jnp.array([[False, True]]),
        max_depth=4,
    )

    tree = output.search_tree
    expected_base = jnp.array([[[6.0, 2.0], [1.0, 9.0]]])
    expected_evidence = jnp.array([[[3.0, 1.0], [0.0, 0.0]]])
    assert jnp.allclose(tree.posterior.base, expected_base)
    assert jnp.allclose(tree.posterior.evidence, expected_evidence)
    assert jnp.allclose(tree.posterior.alpha, expected_base + expected_evidence)
    assert jnp.array_equal(tree.posterior.explored, jnp.array([[True, False]]))
    assert jnp.array_equal(tree.summary().visit_counts, jnp.array([[1.0, 0.0]]))
    assert int(tree.children_index[0, 0, 0]) == 1


def test_repeated_depth_cutoff_counts_every_simulation_without_new_nodes():
    root = _root([[1.0, 1.0]], to_play=0)
    recurrent_fn = _constant_recurrent_fn(
        num_actions=1,
        value=(3.0, 1.0),
        outcome=(0.25, 0.75),
        evidence_weight=2.0,
        to_play=0,
    )

    output = _policy(
        root,
        recurrent_fn,
        num_simulations=4,
        max_depth=1,
    )

    tree = output.search_tree
    assert jnp.array_equal(tree.summary().visit_counts, jnp.array([[4.0]]))
    assert jnp.allclose(tree.posterior.base, jnp.array([[[3.0, 1.0]]]))
    assert jnp.allclose(tree.posterior.evidence, jnp.array([[[2.0, 6.0]]]))
    assert int(tree.children_index[0, 0, 0]) == 1
    assert jnp.array_equal(
        tree.parents[0],
        jnp.array([-1, 0, -1, -1, -1], dtype=jnp.int32),
    )


def test_terminal_child_has_no_descendants_and_repeats_terminal_evidence():
    root = _root([[1.0, 1.0], [1.0, 1.0]], to_play=0)
    recurrent_fn = _constant_recurrent_fn(
        num_actions=2,
        value=(4.0, 2.0),
        outcome=(1.0, 0.0),
        evidence_weight=8.0,
        to_play=1,
        terminal=True,
        prior_logits=(-jnp.inf, -jnp.inf),
    )

    output = _policy(
        root,
        recurrent_fn,
        num_simulations=3,
        invalid_actions=jnp.array([[True, False]]),
        max_depth=6,
    )

    tree = output.search_tree
    assert int(tree.children_index[0, 0, 1]) == 1
    assert bool(tree.node_terminal[0, 1])
    assert bool(jnp.all(tree.children_index[0, 1] == tree.UNVISITED))
    assert jnp.array_equal(
        tree.parents[0],
        jnp.array([-1, 0, -1, -1], dtype=jnp.int32),
    )
    assert jnp.array_equal(tree.summary().visit_counts, jnp.array([[0.0, 3.0]]))
    assert jnp.allclose(
        tree.posterior.evidence,
        jnp.array([[[0.0, 0.0], [0.0, 24.0]]]),
    )
    assert jnp.allclose(
        tree.posterior.base,
        jnp.array([[[1.0, 1.0], [2.0, 4.0]]]),
    )


def test_depth_evidence_uses_node_players_and_routes_to_first_root_action():
    root = _root(
        [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
        to_play=0,
    )

    def recurrent_fn(_, rng_key, action, depth):
        del rng_key
        batch_size = action.shape[0]
        next_depth = depth + 1
        first = next_depth == 1
        value = jnp.where(
            first[:, None],
            jnp.array([[2.0, 5.0]]),
            jnp.array([[7.0, 3.0]]),
        )
        outcome = jnp.where(
            first[:, None],
            jnp.array([[0.2, 0.8]]),
            jnp.array([[0.1, 0.9]]),
        )
        step = dirichlet_mctx.RecurrentFnOutput(
            prior_logits=jnp.broadcast_to(
                jnp.array([-jnp.inf, 0.0, -jnp.inf]),
                (batch_size, 3),
            ),
            value=value,
            outcome=outcome,
            evidence_weight=jnp.where(first, 1.0, 2.0),
            terminal=jnp.zeros((batch_size,), dtype=bool),
            # Both descendants have player 1. At depth two the evidence still
            # needs one flip, so a simple depth-parity rule would be wrong.
            to_play=jnp.ones((batch_size,), dtype=jnp.int32),
        )
        return step, next_depth

    output = _policy(
        root,
        recurrent_fn,
        num_simulations=2,
        invalid_actions=jnp.array([[True, True, False]]),
        max_depth=2,
    )

    tree = output.search_tree
    expected_evidence = jnp.array(
        [[[0.0, 0.0], [0.0, 0.0], [2.6, 0.4]]]
    )
    assert jnp.allclose(tree.posterior.evidence, expected_evidence)
    assert jnp.allclose(tree.posterior.base[0, 2], jnp.array([5.0, 2.0]))
    assert jnp.array_equal(tree.summary().visit_counts, jnp.array([[0.0, 0.0, 2.0]]))
    root_child = tree.children_index[0, 0, 2]
    assert int(root_child) == 1
    assert int(tree.children_index[0, root_child, 1]) == 2
    assert int(tree.children_visits[0, root_child, 1]) == 1


def test_search_blocks_carry_posterior_but_rebuild_block_topology():
    root = _root([[1.0, 1.0]], to_play=0)
    recurrent_fn = _constant_recurrent_fn(
        num_actions=1,
        value=(4.0, 1.0),
        outcome=(0.6, 0.4),
        evidence_weight=2.0,
        to_play=0,
    )

    output = _policy(
        root,
        recurrent_fn,
        rng_key=jax.random.PRNGKey(11),
        num_simulations=1,
        max_depth=1,
        num_search_blocks=3,
    )

    tree = output.search_tree
    assert jnp.allclose(tree.posterior.base, jnp.array([[[4.0, 1.0]]]))
    assert jnp.allclose(tree.posterior.evidence, jnp.array([[[3.6, 2.4]]]))
    assert jnp.array_equal(tree.posterior.explored, jnp.array([[True]]))
    # Tree topology and visit counts describe the final block; its posterior
    # carries evidence from all three blocks.
    assert jnp.array_equal(tree.summary().visit_counts, jnp.array([[1.0]]))
    assert int(tree.children_index[0, 0, 0]) == 1


def test_policy_accepts_custom_posterior_update_rule():
    root = _root([[1.0, 1.0]])
    expand_fn = _constant_recurrent_fn(
        num_actions=1,
        value=(2.0, 2.0),
        outcome=(0.25, 0.75),
        evidence_weight=1.0,
        to_play=0,
    )

    def triple_evidence(posterior, **update):
        return dirichlet_mctx.update_posterior(
            posterior,
            **{
                **update,
                "evidence_weight": 3.0 * update["evidence_weight"],
            },
        )

    output = dirichlet_mctx.dirichlet_thompson_policy(
        params=(),
        rng_key=jax.random.PRNGKey(4),
        root=root,
        recurrent_fn=expand_fn,
        num_simulations=1,
        invalid_actions=jnp.array([[False]]),
        posterior_update=triple_evidence,
        policy_samples=0,
    )

    assert jnp.allclose(
        output.search_tree.posterior.evidence,
        jnp.array([[[0.75, 2.25]]]),
    )


def test_zero_simulation_policy_samples_q_prior_without_tree_updates():
    rng_key = jax.random.PRNGKey(17)
    action_values = jnp.array(
        [[[1.0, 4.0], [4.0, 1.0], [2.0, 2.0]]],
        dtype=jnp.float32,
    )
    invalid_actions = jnp.array([[False, False, True]])
    root = _root(action_values)
    recurrent_fn = _constant_recurrent_fn(
        num_actions=3,
        value=(1.0, 1.0),
        outcome=(0.5, 0.5),
        evidence_weight=1.0,
        to_play=1,
    )
    search_key, _, _ = jax.random.split(rng_key, 3)
    sampled = jax.random.dirichlet(search_key, action_values)
    expected_action = dirichlet_mctx.masked_argmax(
        dirichlet_mctx.outcome_utility(sampled),
        invalid_actions,
    )

    output = _policy(
        root,
        recurrent_fn,
        rng_key=rng_key,
        num_simulations=0,
        invalid_actions=invalid_actions,
    )

    tree = output.search_tree
    assert jnp.array_equal(output.action, expected_action)
    assert jnp.array_equal(
        output.action_weights,
        jax.nn.one_hot(expected_action, 3, dtype=action_values.dtype),
    )
    assert jnp.array_equal(tree.posterior.base, action_values)
    assert jnp.array_equal(tree.posterior.evidence, jnp.zeros_like(action_values))
    assert not bool(tree.posterior.explored.any())
    assert bool(jnp.all(tree.children_index == tree.UNVISITED))


def test_dirichlet_policy_jits_with_heterogeneous_root_masks():
    root = _root(
        jnp.array(
            [
                [[2.0, 1.0], [1.0, 2.0]],
                [[1.0, 3.0], [3.0, 1.0]],
            ]
        )
    )
    recurrent_fn = _constant_recurrent_fn(
        num_actions=2,
        value=(2.0, 4.0),
        outcome=(0.25, 0.75),
        evidence_weight=1.0,
        to_play=1,
    )
    invalid_actions = jnp.array([[False, True], [True, False]])

    @jax.jit
    def run(root_output, rng_key):
        return _policy(
            root_output,
            recurrent_fn,
            rng_key=rng_key,
            num_simulations=2,
            invalid_actions=invalid_actions,
            max_depth=2,
            policy_samples=4,
        )

    with jax.debug_key_reuse(True):
        output = run(root, jax.random.key(9))
    assert jnp.array_equal(output.action, jnp.array([0, 1], dtype=jnp.int32))
    assert jnp.allclose(output.action_weights.sum(axis=-1), 1.0)
    assert jnp.all(output.action_weights[invalid_actions] == 0.0)
    assert jnp.array_equal(
        output.search_tree.summary().visit_counts,
        jnp.array([[2.0, 0.0], [0.0, 2.0]]),
    )


def test_posterior_best_policy_target_masks_invalid_actions():
    alpha_q_post = jnp.array([[[1.0, 2.0], [1.0, 1000.0], [2.0, 1.0]]])
    legal_action_mask = jnp.array([[True, False, True]])

    policy_target = posterior_best_policy_target(
        jax.random.PRNGKey(0),
        alpha_q_post,
        legal_action_mask,
        num_samples=128,
    )

    assert policy_target.shape == (1, 3)
    assert jnp.allclose(policy_target[0, 1], 0.0)
    assert jnp.allclose(policy_target.sum(axis=-1), 1.0)


def test_posterior_best_policy_target_chunk_size_matches_full_chunk():
    alpha_q_post = jnp.array(
        [
            [[1.0, 2.0], [5.0, 1.0], [2.0, 2.0]],
            [[3.0, 1.0], [1.0, 4.0], [2.0, 1.0]],
        ]
    )
    legal_action_mask = jnp.array(
        [
            [True, True, False],
            [True, False, True],
        ]
    )
    key = jax.random.PRNGKey(3)

    full_chunk = posterior_best_policy_target(
        key,
        alpha_q_post,
        legal_action_mask,
        num_samples=7,
        chunk_size=7,
    )
    chunked = posterior_best_policy_target(
        key,
        alpha_q_post,
        legal_action_mask,
        num_samples=7,
        chunk_size=3,
    )

    assert jnp.allclose(chunked, full_chunk)


def test_posterior_action_helpers_respect_legal_mask():
    policy_target = jnp.array([[0.2, 0.7, 0.1], [1.0, 0.0, 0.0]])
    legal_action_mask = jnp.array([[True, False, True], [False, True, False]])

    best = posterior_best_action(policy_target, legal_action_mask)
    sampled = posterior_sample_action(
        jax.random.PRNGKey(0),
        policy_target,
        legal_action_mask,
    )

    assert jnp.array_equal(best, jnp.array([0, 1], dtype=jnp.int32))
    assert bool(jnp.all(legal_action_mask[jnp.arange(2), sampled]))


def test_posterior_targets_add_q_evidence_and_policy_weight_value_evidence():
    alpha_v_prior = jnp.array([[1.0, 1.0]])
    action_value_prior = jnp.array(
        [[[1.0, 2.0], [2.0, 1.0], [1.0, 2.0]]]
    )
    q_evidence_sum = jnp.array(
        [[[2.0, 0.0], [0.0, 0.0], [0.5, 1.5]]]
    )
    policy_target = jnp.array([[0.25, 0.0, 0.75]])

    beta_q_target, beta_v_target = posterior_targets(
        alpha_v_prior,
        action_value_prior,
        q_evidence_sum,
        policy_target,
    )

    expected_v_evidence = (
        0.25 * jnp.array([2.0, 0.0])
        + 0.75 * jnp.array([0.5, 1.5])
    )
    assert jnp.allclose(beta_q_target, action_value_prior + q_evidence_sum)
    assert jnp.allclose(beta_v_target, alpha_v_prior + expected_v_evidence)


def test_q_loss_weights_support_policy_and_evidence_mass_modes():
    evidence = jnp.array([[[2.0, 0.0], [0.5, 1.5], [0.0, 0.0]]])
    policy = jnp.array([[0.25, 0.75, 0.0]])

    assert jnp.array_equal(
        q_loss_weight_from_mode("policy", evidence, policy),
        policy,
    )
    assert jnp.allclose(
        q_loss_weight_from_mode("evidence_mass", evidence, policy),
        jnp.array([[2.0, 2.0, 0.0]]),
    )


def test_terminal_reward_maps_two_and_three_outcome_spaces():
    reward = jnp.array([-1.0, 0.0, 1.0])

    assert jnp.array_equal(
        terminal_outcome_from_reward(reward, 3),
        jnp.eye(3, dtype=reward.dtype),
    )
    assert jnp.array_equal(
        terminal_outcome_from_reward(jnp.array([-1.0, 1.0]), 2),
        jnp.eye(2, dtype=reward.dtype),
    )
