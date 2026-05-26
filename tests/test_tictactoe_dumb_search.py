import jax
import jax.numpy as jnp
import numpy as np
import pgx

from scacchi.dirichlet_tree.search import run_wavefront_posterior_tree_search_state_batch
from scacchi.train import SearchConfig, SearchConstantsConfig, SearchMonteCarloConfig


_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def _config():
    return SearchConfig(
        num_simulations=128,
        inflight_limit=8,
        max_depth=9,
        eval_batch_size=64,
        monte_carlo=SearchMonteCarloConfig(policy_samples=512),
        constants=SearchConstantsConfig(
            kappa_leaf=1.0,
            state_posterior_kappa_n=16.0,
        ),
    )


def _state_after(env, moves):
    state = env.init(jax.random.PRNGKey(0))
    for move in moves:
        state = env.step(state, jnp.asarray(move, dtype=jnp.int32))
    return state


def _stack_states(states):
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs, axis=0), *states)


def _uniform_leaf_evaluator(obs):
    batch = obs.shape[0]
    logits = jnp.zeros((batch, 9), dtype=jnp.float32)
    alpha_v = jnp.ones((batch, 3), dtype=jnp.float32)
    alpha_q = jnp.ones((batch, 9, 3), dtype=jnp.float32)
    return logits, alpha_v, alpha_q


def _winner(board):
    for line in _LINES:
        values = [board[ix] for ix in line]
        if values[0] != -1 and values[0] == values[1] == values[2]:
            return values[0]
    return None


def _score_for_root(board, player_to_move, root_player):
    winner = _winner(board)
    if winner is not None:
        return 1 if winner == root_player else -1
    if all(value != -1 for value in board):
        return 0

    legal = [ix for ix, value in enumerate(board) if value == -1]
    child_scores = []
    for action in legal:
        child = list(board)
        child[action] = player_to_move
        child_scores.append(_score_for_root(child, 1 - player_to_move, root_player))
    if player_to_move == root_player:
        return max(child_scores)
    return min(child_scores)


def _optimal_actions(state):
    board = [int(x) for x in np.asarray(state._x.board)]
    root_player = int(state.current_player)
    scores = {}
    for action, value in enumerate(board):
        if value != -1:
            continue
        child = list(board)
        child[action] = root_player
        scores[action] = _score_for_root(child, 1 - root_player, root_player)
    best = max(scores.values())
    return {action for action, score in scores.items() if score == best}


def test_uniform_model_posterior_tree_finds_tictactoe_tactical_moves():
    env = pgx.make("tic_tac_toe")
    states = [
        _state_after(env, [0, 3, 1, 4]),  # X wins immediately with 2.
        _state_after(env, [0, 3, 8, 4]),  # X must block O at 5.
        _state_after(env, [0, 4, 8, 2, 6, 3]),  # X preserves the draw with 7.
    ]

    output = run_wavefront_posterior_tree_search_state_batch(
        env=env,
        root_state_batch=_stack_states(states),
        leaf_evaluator=_uniform_leaf_evaluator,
        rng_key=jax.random.PRNGKey(0),
        config=_config(),
    )

    actions = [int(action) for action in np.asarray(output.action)]
    assert actions[0] == 2
    assert actions[1] == 5
    assert actions[2] == 7
    for action, state in zip(actions, states, strict=True):
        assert action in _optimal_actions(state)
