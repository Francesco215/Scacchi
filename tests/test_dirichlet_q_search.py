from typing import NamedTuple

import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
import pytest

from scacchi import dirichlet_mctx
from scacchi.dirichlet_mctx.outcomes import NO_OUTCOME
from scacchi.dirichlet_q_search import (
    make_dirichlet_expand_fn,
    posterior_best_action,
    posterior_sample_action,
    q_loss_weight_from_mode,
    terminal_outcome_from_reward,
)
from scacchi.types import load_config


def test_tiny_kappa_cache_mix_remains_a_positive_dirichlet():
    mixed = dirichlet_mctx.mix_value_prior(
        value_prior=jnp.array([[1.0, 1.0, 1.0]], dtype=jnp.float32),
        effective_action_alpha=jnp.array(
            [[[0.0, 0.0, 3.0]]],
            dtype=jnp.float32,
        ),
        search_policy=jnp.ones((1, 1), dtype=jnp.float32),
        n_down=jnp.array([32], dtype=jnp.int32),
        kappa=1e-6,
    )

    assert jnp.all(mixed > 0.0)
    assert jnp.all(jnp.isfinite(mixed))


def test_huge_finite_kappa_cache_mix_stays_at_the_value_prior():
    value_prior = jnp.array([[1.0, 2.0, 3.0]], dtype=jnp.float32)
    mixed = dirichlet_mctx.mix_value_prior(
        value_prior=value_prior,
        effective_action_alpha=jnp.array(
            [[[0.0, 0.0, 3.0]]],
            dtype=jnp.float32,
        ),
        search_policy=jnp.ones((1, 1), dtype=jnp.float32),
        n_down=jnp.array([32], dtype=jnp.int32),
        kappa=1e300,
    )

    assert jnp.all(jnp.isfinite(mixed))
    assert jnp.allclose(mixed, value_prior)


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
    action_values: tuple[tuple[float, ...], ...] | None = None,
    to_play: int,
    terminal_outcome: int = int(NO_OUTCOME),
    invalid_actions: tuple[bool, ...] | None = None,
):
    if invalid_actions is None:
        invalid_actions = (False,) * num_actions
    if action_values is None:
        action_values = (value,) * num_actions

    def recurrent_fn(_, rng_key, action, depth):
        del rng_key
        batch_size = action.shape[0]
        step = dirichlet_mctx.RecurrentFnOutput(
            value=jnp.broadcast_to(
                jnp.asarray(value, dtype=jnp.float32),
                (batch_size, len(value)),
            ),
            action_values=jnp.broadcast_to(
                jnp.asarray(action_values, dtype=jnp.float32),
                (batch_size, num_actions, len(value)),
            ),
            invalid_actions=jnp.broadcast_to(
                jnp.asarray(invalid_actions, dtype=jnp.bool_),
                (batch_size, num_actions),
            ),
            terminal_outcome=jnp.full(
                (batch_size,), terminal_outcome, dtype=jnp.int8
            ),
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
    alpha_q: jax.Array


def _expand_evaluator(observation: jax.Array) -> _ExpandPrediction:
    batch_size = observation.shape[0]
    return _ExpandPrediction(
        logits=jnp.zeros((batch_size, 2), dtype=jnp.float32),
        alpha_v=jnp.broadcast_to(
            jnp.array([2.0, 6.0], dtype=jnp.float32),
            (batch_size, 2),
        ),
        alpha_q=jnp.broadcast_to(
            jnp.array([[3.0, 1.0], [1.0, 3.0]], dtype=jnp.float32),
            (batch_size, 2, 2),
        ),
    )


def test_native_expand_fn_preserves_learned_alpha_and_tags_nonterminal():
    root_state = _ExpandState(
        observation=jnp.zeros((1, 1), dtype=jnp.float32),
        legal_action_mask=jnp.array([[True, True]]),
        current_player=jnp.array([0], dtype=jnp.int32),
        rewards=jnp.zeros((1, 2), dtype=jnp.float32),
        terminated=jnp.array([False]),
    )
    expand_fn = make_dirichlet_expand_fn(_ExpandEnv(), _expand_evaluator)
    action = jnp.array([0], dtype=jnp.int32)
    step, child_state = expand_fn((), jax.random.PRNGKey(1), action, root_state)

    prediction = _expand_evaluator(child_state.observation)
    assert jnp.array_equal(step.value, prediction.alpha_v)
    assert jnp.array_equal(step.action_values, _expand_evaluator(child_state.observation).alpha_q)
    assert jnp.array_equal(
        step.terminal_outcome,
        jnp.asarray([int(NO_OUTCOME)], dtype=jnp.int8),
    )

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
    tree = output.search_tree
    assert jnp.allclose(tree.summary().alpha[0, 0], jnp.array([6.0, 2.0]))
    assert jnp.array_equal(
        tree.root_posterior.action_count > 0,
        jnp.array([[True, False]]),
    )
    assert jnp.array_equal(tree.root_posterior.action_count, jnp.array([[1, 0]]))
    assert jnp.allclose(tree.root_posterior.value_alpha, jnp.array([[2.0, 1.2]]))
    assert jnp.array_equal(
        tree.posterior.action_alpha[0, 1],
        step.action_values[0],
    )


def test_native_expand_fn_publishes_exact_terminal_without_rescaling_alpha():
    root_state = _ExpandState(
        observation=jnp.zeros((1, 1), dtype=jnp.float32),
        legal_action_mask=jnp.array([[True, False]]),
        current_player=jnp.array([0], dtype=jnp.int32),
        rewards=jnp.array([[1.0, -1.0]], dtype=jnp.float32),
        terminated=jnp.array([True]),
    )
    expand_fn = make_dirichlet_expand_fn(_ExpandEnv(), _expand_evaluator)
    action = jnp.array([0], dtype=jnp.int32)

    step, _ = expand_fn((), jax.random.PRNGKey(2), action, root_state)

    # Player 0 won, but the child is from player 1's perspective. The exact
    # terminal tag flips to loss while the network alpha remains untouched.
    assert jnp.array_equal(step.terminal_outcome, jnp.array([0], dtype=jnp.int8))
    assert jnp.array_equal(step.value, jnp.array([[2.0, 6.0]]))
    assert jnp.sum(step.value) == 8.0

    root = dirichlet_mctx.RootFnOutput(
        prior_logits=jnp.zeros((1, 2), dtype=jnp.float32),
        value=jnp.ones((1, 2), dtype=jnp.float32),
        action_values=jnp.ones((1, 2, 2), dtype=jnp.float32),
        embedding=root_state,
        terminal=jnp.zeros((1,), dtype=bool),
        to_play=root_state.current_player,
    )
    output = _policy(
        root,
        expand_fn,
        num_simulations=2,
        invalid_actions=jnp.array([[False, True]]),
        policy_samples=8,
    )
    summary = output.search_tree.summary()
    assert summary.q_categorical_outcome[0, 0] == 1
    assert summary.v_categorical_outcome[0] == 1
    assert jnp.array_equal(output.action_weights, jnp.array([[1.0, 0.0]]))
    # Exact categorical propagation does not overwrite the uncertain Q slot.
    # The evaluated child's learned alpha is retained on the child itself.
    assert jnp.array_equal(summary.alpha[0, 0], jnp.array([1.0, 1.0]))
    child = output.search_tree.children_index[0, 0, 0]
    assert jnp.array_equal(
        output.search_tree.node_value_priors[0, child],
        step.value[0],
    )


def test_single_thompson_selector_uses_interior_posterior_and_mask():
    child_action_values = (
        (1000.0, 1.0),
        (1.0, 1000.0),
        (500.0, 1.0),
    )
    output = _policy(
        _root([[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]]),
        _constant_recurrent_fn(
            num_actions=3,
            value=(2.0, 2.0),
            action_values=child_action_values,
            to_play=0,
            invalid_actions=(False, True, False),
        ),
        num_simulations=1,
        invalid_actions=jnp.array([[False, True, True]]),
    )
    tree = output.search_tree
    child_index = tree.children_index[0, 0, 0]
    assert int(child_index) == 1
    unbatched_tree = jax.tree.map(lambda leaf: leaf[0], tree)
    rng_key = jax.random.PRNGKey(0)
    child_alpha = jnp.asarray(child_action_values, dtype=jnp.float32)
    scores = dirichlet_mctx.outcome_utility(
        dirichlet_mctx.sample_dirichlet(rng_key, child_alpha)
    )
    invalid_actions = jnp.array([False, True, False])
    expected = jnp.argmax(
        jnp.where(invalid_actions, -jnp.inf, scores)
    ).astype(jnp.int32)

    action = dirichlet_mctx.thompson_action_selection(
        rng_key,
        unbatched_tree,
        child_index,
    )

    assert int(jnp.argmax(scores)) == 1
    assert int(action) == int(expected)
    assert not bool(invalid_actions[action])


def test_first_leaf_replaces_q_fallback_with_aligned_child_message():
    root = _root([[9.0, 1.0], [1.0, 9.0]], to_play=0)
    recurrent_fn = _constant_recurrent_fn(
        num_actions=2,
        value=(2.0, 6.0),
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
    assert jnp.allclose(
        tree.summary().alpha,
        jnp.array([[[6.0, 2.0], [1.0, 9.0]]]),
    )
    assert jnp.array_equal(
        tree.root_posterior.action_count > 0,
        jnp.array([[True, False]]),
    )
    assert jnp.array_equal(tree.root_posterior.action_count, jnp.array([[1, 0]]))
    assert jnp.allclose(tree.root_posterior.value_alpha, jnp.array([[2.0, 1.2]]))
    assert jnp.array_equal(tree.summary().visit_counts, jnp.array([[1.0, 0.0]]))
    assert int(tree.children_index[0, 0, 0]) == 1


def test_tree_stores_only_posterior_messages_counts_and_values():
    root = _root([[1.0, 1.0], [1.0, 1.0]])
    tree = dirichlet_mctx.instantiate_tree_from_root(
        root,
        num_simulations=3,
        root_invalid_actions=jnp.array([[False, True]]),
    )

    assert tree.posterior.action_alpha.shape == (1, 4, 2, 2)
    assert tree.posterior.action_count.shape == (1, 4, 2)
    assert tree.posterior.value_alpha.shape == (1, 4, 2)
    assert set(tree.posterior.__dataclass_fields__) == {
        "action_alpha",
        "action_count",
        "value_alpha",
    }


def test_posterior_best_policy_uses_current_alphas_not_previous_hits():
    alpha = jnp.array([[[100.0, 1.0], [1.0, 100.0]]])

    policy = dirichlet_mctx.posterior_best_policy(
        jax.random.PRNGKey(3),
        alpha,
        jnp.array([[False, False]]),
        num_samples=16,
    )

    assert jnp.array_equal(policy, jnp.array([[0.0, 1.0]]))


def test_fixed_work_dirichlet_sample_handles_tiny_terminal_components():
    alpha = jnp.array([0.01, 0.01, 80.01], dtype=jnp.float32)
    keys = jax.random.split(jax.random.PRNGKey(17), 1024)
    samples = jax.vmap(
        lambda key: dirichlet_mctx.sample_dirichlet(key, alpha)
    )(keys)

    assert bool(jnp.all(jnp.isfinite(samples)))
    assert jnp.allclose(jnp.sum(samples, axis=-1), 1.0)
    assert jnp.mean(samples[:, -1]) > 0.995


def test_fixed_work_selector_tracks_exact_dirichlet_action_probabilities():
    """Keep the production approximation small relative to K-sample noise."""

    alpha = jnp.array(
        [
            [0.3, 1.0, 0.7],
            [0.5, 1.0, 1.0],
            [0.8, 1.0, 1.2],
        ],
        dtype=jnp.float32,
    )
    invalid_actions = jnp.zeros((3,), dtype=bool)
    num_samples = 8192
    keys = jax.random.split(jax.random.PRNGKey(23), num_samples)
    approximate_best = jax.vmap(
        lambda key: dirichlet_mctx.thompson_sample(
            key,
            alpha,
            invalid_actions,
        )
    )(keys)
    exact_best = jax.vmap(
        lambda key: dirichlet_mctx.masked_argmax(
            dirichlet_mctx.outcome_utility(
                jax.random.dirichlet(key, alpha)
            ),
            invalid_actions,
        )
    )(keys)
    approximate_policy = jnp.bincount(
        approximate_best,
        length=alpha.shape[0],
    ) / num_samples
    exact_policy = jnp.bincount(
        exact_best,
        length=alpha.shape[0],
    ) / num_samples

    assert jnp.allclose(approximate_policy, exact_policy, atol=0.025)


def test_structural_edge_counts_do_not_define_search_policy():
    action_alpha = jnp.array([[[100.0, 1.0], [1.0, 100.0]]])
    posterior = dirichlet_mctx.NodePosterior(
        action_alpha=action_alpha,
        action_count=jnp.array([[64, 1]], dtype=jnp.int32),
        value_alpha=jnp.ones((1, 2)),
    )
    node = dirichlet_mctx.NodeView(
        index=jnp.array([0], dtype=jnp.int32),
        embedding=jnp.zeros((1,), dtype=jnp.int32),
        value_prior=jnp.ones((1, 2)),
        to_play=jnp.zeros((1,), dtype=jnp.int32),
        terminal=jnp.zeros((1,), dtype=bool),
        invalid_actions=jnp.zeros((1, 2), dtype=bool),
        posterior=posterior,
    )
    children = dirichlet_mctx.ChildrenView(
        index=jnp.full((1, 2), -1, dtype=jnp.int32),
        visited=jnp.zeros((1, 2), dtype=bool),
        embedding_table=jnp.zeros((1, 1), dtype=jnp.int32),
        value_prior=jnp.ones((1, 2, 2)),
        value_alpha=jnp.ones((1, 2, 2)),
        count=jnp.zeros((1, 2), dtype=jnp.int32),
        to_play=jnp.zeros((1, 2), dtype=jnp.int32),
        terminal=jnp.zeros((1, 2), dtype=bool),
    )
    context = dirichlet_mctx.PosteriorUpdateContext(
        node=node,
        children=children,
        leaf=dirichlet_mctx.LeafView(
            action=jnp.zeros((1,), dtype=jnp.int32),
            value_alpha=jnp.ones((1, 2)),
            to_play=jnp.zeros((1,), dtype=jnp.int32),
            active=jnp.zeros((1,), dtype=bool),
        ),
        active=jnp.array([True]),
    )

    update_key = jax.random.PRNGKey(0)
    updated = dirichlet_mctx.update_posterior(
        update_key,
        context,
    )

    assert jnp.array_equal(updated.action_count, posterior.action_count)
    # The first edge has 64 times more structural visits, but pi_search comes
    # only from the Dirichlet alphas and therefore selects the second edge.
    assert updated.value_alpha[0, 1] > updated.value_alpha[0, 0]


def test_repeated_depth_cutoff_counts_every_simulation_without_new_nodes():
    root = _root([[1.0, 1.0]], to_play=0)
    recurrent_fn = _constant_recurrent_fn(
        num_actions=1,
        value=(3.0, 1.0),
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
    assert jnp.allclose(tree.summary().alpha, jnp.array([[[3.0, 1.0]]]))
    assert jnp.allclose(tree.root_posterior.value_alpha, jnp.array([[2.0, 1.0]]))
    assert jnp.array_equal(tree.root_posterior.action_count > 0, jnp.array([[True]]))
    assert int(tree.children_index[0, 0, 0]) == 1
    assert jnp.array_equal(
        tree.parents[0],
        jnp.array([-1, 0, -1, -1, -1], dtype=jnp.int32),
    )


def test_terminal_child_has_no_descendants_and_short_circuits_search():
    root = _root([[1.0, 1.0], [1.0, 1.0]], to_play=0)
    recurrent_fn = _constant_recurrent_fn(
        num_actions=2,
        value=(4.0, 2.0),
        to_play=1,
        terminal_outcome=0,
        invalid_actions=(True, True),
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
    assert jnp.array_equal(tree.summary().visit_counts, jnp.array([[0.0, 1.0]]))
    assert jnp.allclose(
        tree.summary().alpha,
        jnp.array([[[1.0, 1.0], [1.0, 1.0]]]),
    )
    assert jnp.array_equal(
        tree.root_posterior.action_count > 0,
        jnp.array([[False, True]]),
    )
    # Cache repair projects the exact win onto the existing learned Q mass
    # (C=2) and mixes it with the root prior using gamma=1/(4+1).
    expected_value = jnp.array([0.8, 1.2])
    assert jnp.allclose(tree.root_posterior.value_alpha[0], expected_value)
    child = tree.children_index[0, 0, 1]
    assert jnp.array_equal(tree.node_value_priors[0, child], jnp.array([4.0, 2.0]))
    assert tree.node_categorical_outcome[0, 0] == 1
    assert tree.node_categorical_distance[0, 0] == 1


def test_default_max_depth_uses_complete_simulation_budget():
    terminal_depth = 5
    config = load_config(
        OmegaConf.create(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {
                        "num_simulations": 6,
                        "max_depth": None,
                        "policy_samples": 0,
                    },
                }
            }
        )
    )
    search_cfg = config.search.dirichlet_thompson

    def chain_recurrent_fn(_, rng_key, action, depth):
        del rng_key, action
        child_depth = depth + 1
        batch_size = child_depth.shape[0]
        terminal = child_depth >= terminal_depth
        return (
            dirichlet_mctx.RecurrentFnOutput(
                value=jnp.ones((batch_size, 2), dtype=jnp.float32),
                action_values=jnp.ones(
                    (batch_size, 1, 2),
                    dtype=jnp.float32,
                ),
                invalid_actions=terminal[:, None],
                terminal_outcome=jnp.where(
                    terminal,
                    jnp.asarray(1, dtype=jnp.int8),
                    jnp.asarray(int(NO_OUTCOME), dtype=jnp.int8),
                ),
                to_play=jnp.zeros((batch_size,), dtype=jnp.int32),
            ),
            child_depth,
        )

    output = _policy(
        _root([[1.0, 1.0]]),
        chain_recurrent_fn,
        num_simulations=search_cfg.num_simulations,
        max_depth=search_cfg.max_depth,
    )
    tree = output.search_tree

    assert bool(tree.node_terminal[0, terminal_depth]), (
        f"max_depth={search_cfg.max_depth} prevented "
        f"{search_cfg.num_simulations} simulations "
        f"from reaching depth {terminal_depth}"
    )
    assert jnp.array_equal(
        tree.parents[0, 1 : terminal_depth + 1],
        jnp.arange(terminal_depth, dtype=jnp.int32),
    )
    assert jnp.array_equal(
        tree.children_index[0, :terminal_depth, 0],
        jnp.arange(1, terminal_depth + 1, dtype=jnp.int32),
    )
    # The fifth active simulation reaches the terminal certificate. Exact
    # categorical descendants propagate through sidecars, not uncertain R/B,
    # so the root retains the four preceding uncertain messages.
    assert tree.summary().visit_counts[0, 0] == terminal_depth - 1
    assert tree.posterior.action_count[0, terminal_depth - 1, 0] == 1


def test_terminal_root_does_not_search_even_if_actions_are_marked_legal():
    root = _root([[1.0, 1.0]], terminal=True)
    output = _policy(
        root,
        _constant_recurrent_fn(
            num_actions=1,
            value=(2.0, 1.0),
            to_play=0,
        ),
        num_simulations=1,
        invalid_actions=jnp.array([[False]]),
    )

    assert jnp.array_equal(
        output.search_tree.root_posterior.action_count,
        jnp.array([[0]], dtype=jnp.int32),
    )


def test_bottom_up_repair_uses_updated_child_cache_and_node_players():
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
        step = dirichlet_mctx.RecurrentFnOutput(
            value=value,
            action_values=jnp.ones((batch_size, 3, 2), dtype=jnp.float32),
            invalid_actions=jnp.broadcast_to(
                jnp.array([True, False, True]),
                (batch_size, 3),
            ),
            terminal_outcome=jnp.full(
                (batch_size,), int(NO_OUTCOME), dtype=jnp.int8
            ),
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
    assert jnp.allclose(tree.summary().alpha[0, 2], jnp.array([4.6, 3.0]))
    assert jnp.array_equal(tree.summary().visit_counts, jnp.array([[0.0, 0.0, 2.0]]))
    root_child = tree.children_index[0, 0, 2]
    assert int(root_child) == 1
    assert int(tree.children_index[0, root_child, 1]) == 2
    assert jnp.allclose(
        tree.posterior.action_alpha[0, root_child, 1],
        jnp.array([7.0, 3.0]),
    )
    assert int(tree.posterior.action_count[0, root_child, 1]) == 1
    assert jnp.allclose(
        tree.posterior.value_alpha[0, root_child],
        jnp.array([3.0, 4.6]),
    )
    assert jnp.allclose(
        tree.root_posterior.value_alpha,
        jnp.array([[2.2, 5.0 / 3.0]]),
    )


def test_simulations_share_one_persistent_tree_and_posterior():
    root = _root([[1.0, 1.0]], to_play=0)
    recurrent_fn = _constant_recurrent_fn(
        num_actions=1,
        value=(4.0, 1.0),
        to_play=0,
    )

    output = _policy(
        root,
        recurrent_fn,
        rng_key=jax.random.PRNGKey(11),
        num_simulations=3,
        max_depth=1,
    )

    tree = output.search_tree
    assert jnp.allclose(tree.summary().alpha, jnp.array([[[4.0, 1.0]]]))
    assert jnp.array_equal(tree.root_posterior.action_count > 0, jnp.array([[True]]))
    assert jnp.array_equal(tree.summary().visit_counts, jnp.array([[3.0]]))
    assert jnp.allclose(
        tree.root_posterior.value_alpha,
        jnp.array([[16.0 / 7.0, 1.0]]),
    )
    assert int(tree.children_index[0, 0, 0]) == 1


def test_custom_posterior_update_sees_children_and_runs_bottom_up():
    root = _root([[1.0, 1.0]])
    expand_fn = _constant_recurrent_fn(
        num_actions=1,
        value=(2.0, 2.0),
        to_play=0,
    )

    def custom_update(rng_key, context):
        del rng_key
        old = context.node.posterior
        batch = jnp.arange(old.action_alpha.shape[0])
        action = jnp.zeros_like(batch)
        safe_child = jnp.where(
            context.children.visited,
            context.children.index,
            0,
        )
        child_embedding = context.children.embedding_table[batch[:, None], safe_child]
        selected_child_embedding = child_embedding[batch, action].astype(jnp.float32)
        direct_value = jnp.repeat(
            (10.0 + selected_child_embedding)[:, None],
            old.value_alpha.shape[-1],
            axis=-1,
        )
        repaired_child_value = context.children.value_alpha[batch, action] + 1.0
        child_has_message = context.children.count[batch, action] > 0
        value_alpha = jnp.where(
            child_has_message[:, None],
            repaired_child_value,
            jnp.where(context.leaf.active[:, None], direct_value, old.value_alpha),
        )
        selected_alpha = old.action_alpha[batch, action]
        replace_edge = context.leaf.active | child_has_message
        selected_alpha = jnp.where(replace_edge[:, None], value_alpha, selected_alpha)
        action_alpha = old.action_alpha.at[batch, action].set(selected_alpha)
        direct_count = old.action_count[batch, action] + context.leaf.active.astype(
            old.action_count.dtype
        )
        child_count = context.children.count[batch, action] + 1
        selected_count = jnp.where(
            child_has_message,
            child_count,
            direct_count,
        )
        action_count = old.action_count.at[batch, action].set(selected_count)
        return dirichlet_mctx.NodePosterior(
            action_alpha=action_alpha,
            action_count=action_count,
            value_alpha=jnp.where(
                context.active[:, None],
                value_alpha,
                old.value_alpha,
            ),
        )

    output = dirichlet_mctx.dirichlet_thompson_policy(
        params=(),
        rng_key=jax.random.PRNGKey(4),
        root=root,
        recurrent_fn=expand_fn,
        num_simulations=2,
        max_depth=2,
        invalid_actions=jnp.array([[False]]),
        posterior_update=custom_update,
        policy_samples=0,
    )

    tree = output.search_tree
    child = tree.children_index[0, 0, 0]
    assert jnp.allclose(tree.posterior.value_alpha[0, child], jnp.array([12.0, 12.0]))
    assert jnp.allclose(tree.root_posterior.value_alpha, jnp.array([[13.0, 13.0]]))
    assert jnp.array_equal(tree.root_posterior.action_count, jnp.array([[2]]))
    legal_actions = ~tree.invalid_actions
    assert jnp.array_equal(
        tree.node_n_down,
        jnp.sum(
            jnp.where(legal_actions, tree.posterior.action_count, 0),
            axis=-1,
        ),
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
        to_play=1,
    )
    _, policy_key = jax.random.split(rng_key)
    sampled = dirichlet_mctx.sample_dirichlet(policy_key, action_values)
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
    assert jnp.array_equal(tree.root_posterior.action_alpha, action_values)
    assert not bool(tree.root_posterior.action_count.any())
    assert jnp.array_equal(tree.root_posterior.value_alpha, root.value)
    assert not bool(tree.node_n_down.any())
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


def test_posterior_best_policy_masks_invalid_actions():
    alpha_q_post = jnp.array([[[1.0, 2.0], [1.0, 1000.0], [2.0, 1.0]]])
    legal_action_mask = jnp.array([[True, False, True]])

    policy_target = dirichlet_mctx.posterior_best_policy(
        jax.random.PRNGKey(0),
        alpha_q_post,
        ~legal_action_mask,
        num_samples=128,
    )

    assert policy_target.shape == (1, 3)
    assert jnp.allclose(policy_target[0, 1], 0.0)
    assert jnp.allclose(policy_target.sum(axis=-1), 1.0)


def test_posterior_best_policy_chunk_size_matches_full_chunk():
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

    full_chunk = dirichlet_mctx.posterior_best_policy(
        key,
        alpha_q_post,
        ~legal_action_mask,
        num_samples=7,
        chunk_size=7,
    )
    chunked = dirichlet_mctx.posterior_best_policy(
        key,
        alpha_q_post,
        ~legal_action_mask,
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


def test_posterior_sample_temperature_one_preserves_seeded_path():
    policy_target = jnp.array(
        [[0.05, 0.15, 0.8], [0.6, 0.3, 0.1]],
        dtype=jnp.float32,
    )
    legal_action_mask = jnp.array(
        [[True, True, True], [True, False, True]]
    )
    key = jax.random.PRNGKey(123)

    default = posterior_sample_action(key, policy_target, legal_action_mask)
    explicit_one = posterior_sample_action(
        key,
        policy_target,
        legal_action_mask,
        temperature=1.0,
    )
    legacy_logits = jnp.log(jnp.clip(policy_target, 1e-8, 1.0))
    legacy_logits = jnp.where(
        legal_action_mask,
        legacy_logits,
        jnp.finfo(legacy_logits.dtype).min,
    )
    legacy = jax.random.categorical(key, legacy_logits).astype(jnp.int32)

    assert jnp.array_equal(explicit_one, default)
    assert jnp.array_equal(explicit_one, legacy)


def test_posterior_sample_temperature_matches_power_transform():
    policy_target = jnp.array(
        [[0.1, 0.3, 0.6], [0.7, 0.2, 0.1]],
        dtype=jnp.float32,
    )
    legal_action_mask = jnp.array(
        [[True, True, True], [True, False, True]]
    )
    key = jax.random.PRNGKey(7)
    temperature = 0.5
    expected_logits = jnp.log(jnp.clip(policy_target, 1e-8, 1.0))
    expected_logits = expected_logits / temperature
    expected_logits = jnp.where(
        legal_action_mask,
        expected_logits,
        jnp.finfo(expected_logits.dtype).min,
    )

    sampled = posterior_sample_action(
        key,
        policy_target,
        legal_action_mask,
        temperature=temperature,
    )

    assert jnp.array_equal(
        sampled,
        jax.random.categorical(key, expected_logits).astype(jnp.int32),
    )


def test_nonunit_temperature_preserves_exact_zero_support():
    policy_target = jnp.asarray([[0.0, 0.2, 0.8]], dtype=jnp.float32)
    legal_action_mask = jnp.ones_like(policy_target, dtype=jnp.bool_)
    keys = jax.random.split(jax.random.PRNGKey(8), 512)

    samples = jax.vmap(
        lambda key: posterior_sample_action(
            key,
            policy_target,
            legal_action_mask,
            temperature=8.0,
        )[0]
    )(keys)

    assert bool(jnp.all(samples != 0))


@pytest.mark.parametrize("temperature", [0.0, -1.0, float("nan"), float("inf")])
def test_posterior_sample_rejects_invalid_temperature(temperature):
    with pytest.raises(ValueError, match="finite and > 0"):
        posterior_sample_action(
            jax.random.PRNGKey(0),
            jnp.array([[0.25, 0.75]], dtype=jnp.float32),
            jnp.array([[True, True]]),
            temperature=temperature,
        )


def test_q_loss_weights_support_policy_and_evidence_mass_modes():
    counts = jnp.array([[[2.0], [2.0], [0.0]]])
    policy = jnp.array([[0.25, 0.75, 0.0]])

    assert jnp.array_equal(
        q_loss_weight_from_mode("policy", counts, policy),
        policy,
    )
    assert jnp.allclose(
        q_loss_weight_from_mode("evidence_mass", counts, policy),
        jnp.array([[2.0, 2.0, 0.0]]),
    )


def test_terminal_reward_maps_two_and_three_outcome_spaces():
    reward = jnp.array([-1.0, 0.0, 1.0])

    assert jnp.array_equal(
        terminal_outcome_from_reward(reward, 3),
        jnp.array([0, 1, 2], dtype=jnp.int8),
    )
    assert jnp.array_equal(
        terminal_outcome_from_reward(jnp.array([-1.0, 1.0]), 2),
        jnp.array([0, 1], dtype=jnp.int8),
    )
