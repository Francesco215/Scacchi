from typing import NamedTuple

from flax import nnx
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf
import pgx

from scacchi.evaluations import baseline_search_config, make_mcts_evaluate
from scacchi.network import BoardlawNet
from scacchi.play import (
    EvalMetrics,
    play,
    play_eval,
    poison_eval_returns,
    searchable_eval_state,
    step_active_eval_rows,
)
from scacchi.play_search import PlayerOutput
from scacchi.types import (
    ActionCommitmentType,
    Config,
    EvalConfig,
    GumbelSearchConfig,
    ModelConfig,
    Network,
    PolicySearchConfig,
    SearchConfig,
    SearchKind,
    load_config,
)


class ToyState(NamedTuple):
    observation: jax.Array
    legal_action_mask: jax.Array
    current_player: jax.Array
    terminated: jax.Array
    rewards: jax.Array


class ToyEnv:
    def init(self, key: jax.Array) -> ToyState:
        del key
        return ToyState(
            observation=jnp.zeros((1,), dtype=jnp.float32),
            legal_action_mask=jnp.array([False, True, False]),
            current_player=jnp.array(0, dtype=jnp.int32),
            terminated=jnp.array(False),
            rewards=jnp.zeros((2,), dtype=jnp.float32),
        )

    def step(self, state: ToyState, action: jax.Array) -> ToyState:
        reward = jnp.where(action == 1, 1.0, -10.0)
        return state._replace(
            observation=state.observation + 1.0,
            legal_action_mask=jnp.zeros_like(state.legal_action_mask),
            terminated=jnp.array(True),
            rewards=jnp.array([reward, -reward], dtype=state.rewards.dtype),
        )


def winning_player(env_state: ToyState, key: jax.Array) -> PlayerOutput:
    del key
    return PlayerOutput(
        action=jnp.ones_like(env_state.current_player, dtype=jnp.int32),
        posterior=None,
        diagnostics=None,
    )


def losing_player(env_state: ToyState, key: jax.Array) -> PlayerOutput:
    del key
    return PlayerOutput(
        action=jnp.zeros_like(env_state.current_player, dtype=jnp.int32),
        posterior=None,
        diagnostics=None,
    )


def test_searchable_eval_state_adds_dummy_legal_action_only_for_terminal_rows():
    state = ToyState(
        observation=jnp.zeros((2, 1), dtype=jnp.float32),
        legal_action_mask=jnp.array(
            [
                [False, True, False],
                [False, False, False],
            ]
        ),
        current_player=jnp.zeros((2,), dtype=jnp.int32),
        terminated=jnp.array([False, True]),
        rewards=jnp.zeros((2, 2), dtype=jnp.float32),
    )

    search_state = searchable_eval_state(state)

    assert jnp.array_equal(
        search_state.legal_action_mask,
        jnp.array(
            [
                [False, True, False],
                [True, False, False],
            ]
        ),
    )


def test_step_active_eval_rows_reports_invalid_actions_and_skips_step():
    state = ToyState(
        observation=jnp.array([[0.0], [5.0]], dtype=jnp.float32),
        legal_action_mask=jnp.array(
            [
                [False, True, False],
                [False, False, False],
            ]
        ),
        current_player=jnp.zeros((2,), dtype=jnp.int32),
        terminated=jnp.array([False, True]),
        rewards=jnp.array([[0.0, 0.0], [7.0, -7.0]], dtype=jnp.float32),
    )

    next_state, active, invalid_action = step_active_eval_rows(
        ToyEnv(),
        state,
        jnp.array([2, 2], dtype=jnp.int32),
    )

    assert jnp.array_equal(active, jnp.array([True, False]))
    assert bool(invalid_action)
    assert jnp.array_equal(next_state.terminated, jnp.array([False, True]))
    assert jnp.array_equal(next_state.observation, jnp.array([[0.0], [5.0]]))
    assert jnp.array_equal(
        next_state.rewards,
        jnp.array([[0.0, 0.0], [7.0, -7.0]], dtype=jnp.float32),
    )


def test_step_active_eval_rows_ignores_invalid_terminal_row_actions():
    state = ToyState(
        observation=jnp.array([[0.0], [5.0]], dtype=jnp.float32),
        legal_action_mask=jnp.array(
            [
                [False, True, False],
                [False, False, False],
            ]
        ),
        current_player=jnp.zeros((2,), dtype=jnp.int32),
        terminated=jnp.array([False, True]),
        rewards=jnp.array([[0.0, 0.0], [7.0, -7.0]], dtype=jnp.float32),
    )

    next_state, active, invalid_action = step_active_eval_rows(
        ToyEnv(),
        state,
        jnp.array([1, 99], dtype=jnp.int32),
    )

    assert jnp.array_equal(active, jnp.array([True, False]))
    assert not bool(invalid_action)
    assert jnp.array_equal(next_state.observation, jnp.array([[1.0], [5.0]]))


def test_poison_eval_returns_makes_all_returns_nan_after_invalid_action():
    returns = poison_eval_returns(jnp.array([1.0, -1.0]), jnp.array(True))

    assert jnp.isnan(returns).all()


def test_play_eval_runs_two_player_loop_until_all_rows_done():
    metrics = play_eval(
        ToyEnv(),
        winning_player,
        losing_player,
        jax.random.PRNGKey(0),
        batch_size=3,
    )

    assert jnp.array_equal(metrics.returns, jnp.ones((3,), dtype=jnp.float32))
    assert jnp.allclose(metrics.avg_return, 1.0)
    assert jnp.allclose(metrics.win_rate, 1.0)


def test_play_dispatches_eval_mode():
    metrics = play(
        ToyEnv(),
        winning_player,
        losing_player,
        jax.random.PRNGKey(0),
        mode="eval",
        batch_size=3,
    )

    assert isinstance(metrics, EvalMetrics)
    assert jnp.array_equal(metrics.returns, jnp.ones((3,), dtype=jnp.float32))


def test_pgx_baseline_eval_uses_scalar_gumbel_search_with_same_budget():
    config = load_config(
        OmegaConf.create(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "selfplay": {
                    "search": {
                        "kind": "gumbel",
                        "gumbel": {"num_simulations": 2},
                    }
                },
                "eval": {
                    "baseline": "pgx",
                    "baseline_search": {
                        "kind": "dirichlet_thompson",
                        "dirichlet_thompson": {"num_simulations": 4},
                    },
                },
            }
        )
    )

    search_config = baseline_search_config(config)

    assert config.selfplay.search.kind == "gumbel"
    assert config.eval.baseline_search.kind == "dirichlet_thompson"
    assert search_config.kind == "gumbel"
    assert search_config.gumbel.num_simulations == 4


def test_pgx_baseline_eval_keeps_policy_search():
    config = load_config(
        OmegaConf.create(
            {
                "eval": {
                    "baseline": "pgx",
                    "baseline_search": {
                        "kind": "policy",
                        "policy": {"temperature": 0.75},
                    },
                },
            }
        )
    )

    search_config = baseline_search_config(config)

    assert search_config.kind == "policy"
    assert search_config.policy.temperature == 0.75


def test_legacy_top_level_eval_search_remains_compatible():
    config = load_config(
        OmegaConf.create(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {"num_simulations": 4},
                },
                "eval": {"baseline": "pgx"},
            }
        )
    )

    search_config = baseline_search_config(config)

    assert config.search.kind == "dirichlet_thompson"
    assert config.eval.baseline_search.kind == "dirichlet_thompson"
    assert search_config.kind == "gumbel"
    assert search_config.gumbel.num_simulations == 4


def test_checkpoint_baseline_eval_keeps_configured_search_kind():
    config = load_config(
        OmegaConf.create(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {"num_simulations": 4},
                },
                "eval": {"baseline": "checkpoint"},
            }
        )
    )

    search_config = baseline_search_config(config)

    assert search_config.kind == "dirichlet_thompson"
    assert search_config.dirichlet_thompson.num_simulations == 4


def test_make_mcts_evaluate_delegates_to_play_eval_smoke():
    env = pgx.make("tic_tac_toe")
    search = SearchConfig(gumbel=GumbelSearchConfig(num_simulations=1))
    config = Config(
        model=ModelConfig(
            network=Network.boardlaw,
            num_channels=8,
            num_layers=1,
        ),
        eval=EvalConfig(
            batch_size=2,
            player_search=search,
            baseline_search=search,
        ),
    )
    model = BoardlawNet(
        num_actions=env.num_actions,
        observation_shape=env.observation_shape,
        width=8,
        depth=1,
        rngs=nnx.Rngs(0),
    )
    baseline_model = BoardlawNet(
        num_actions=env.num_actions,
        observation_shape=env.observation_shape,
        width=8,
        depth=1,
        rngs=nnx.Rngs(1),
    )

    returns = make_mcts_evaluate(env, config, baseline_model)(
        jax.random.PRNGKey(2),
        model,
    )

    assert returns.shape == (2,)
    assert jnp.isfinite(returns).all()


def test_policy_eval_accepts_logits_only_baseline_smoke():
    env = pgx.make("tic_tac_toe")
    search = SearchConfig(
        kind=SearchKind.policy,
        policy=PolicySearchConfig(temperature=1.0),
    )
    config = Config(
        model=ModelConfig(
            network=Network.boardlaw,
            num_channels=8,
            num_layers=1,
        ),
        eval=EvalConfig(
            batch_size=2,
            player_search=search,
            baseline_search=search,
            player_action_commitment_type=ActionCommitmentType.posterior_argmax,
            baseline_action_commitment_type=ActionCommitmentType.posterior_argmax,
        ),
    )
    model = BoardlawNet(
        num_actions=env.num_actions,
        observation_shape=env.observation_shape,
        width=8,
        depth=1,
        rngs=nnx.Rngs(0),
    )

    def logits_only_baseline(obs: jax.Array) -> jax.Array:
        return jnp.zeros((obs.shape[0], env.num_actions), dtype=jnp.float32)

    returns = make_mcts_evaluate(env, config, logits_only_baseline)(
        jax.random.PRNGKey(2),
        model,
    )

    assert returns.shape == (2,)
    assert jnp.isfinite(returns).all()
