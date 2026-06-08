import argparse

from flax import nnx
import jax
import jax.numpy as jnp
import numpy as np
import pgx

from scacchi.network import BoardlawNet
from scripts.fig_8 import (
    _coerce_dqaz_wdl3_output,
    _nonnegative_int_csv,
    _tree_size_plot_position,
    _with_dqaz_eval_settings,
    make_stochastic_mcts_evaluate,
)
from scacchi.types import (
    Config,
    EnvConfig,
    EvalConfig,
    GumbelSearchConfig,
    ModelConfig,
    Network,
    SearchConfig,
)


def test_coerce_dqaz_wdl3_output_inserts_draw_channel_for_hex_lw_heads():
    logits = jnp.zeros((2, 3), dtype=jnp.float32)
    alpha_v = jnp.asarray([[2.0, 5.0], [3.0, 7.0]], dtype=jnp.float32)
    alpha_q = jnp.asarray(
        [
            [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]],
            [[7.0, 10.0], [8.0, 11.0], [9.0, 12.0]],
        ],
        dtype=jnp.float32,
    )

    out_logits, out_v, out_q = _coerce_dqaz_wdl3_output(
        (logits, alpha_v, alpha_q),
        draw_alpha=1e-4,
        min_alpha=0.0,
    )

    assert out_logits is logits
    assert out_v.shape == (2, 3)
    assert out_q.shape == (2, 3, 3)
    np.testing.assert_allclose(np.asarray(out_v[:, 0]), np.asarray(alpha_v[:, 0]))
    np.testing.assert_allclose(np.asarray(out_v[:, 1]), 1e-4)
    np.testing.assert_allclose(np.asarray(out_v[:, 2]), np.asarray(alpha_v[:, 1]))
    np.testing.assert_allclose(np.asarray(out_q[..., 0]), np.asarray(alpha_q[..., 0]))
    np.testing.assert_allclose(np.asarray(out_q[..., 1]), 1e-4)
    np.testing.assert_allclose(np.asarray(out_q[..., 2]), np.asarray(alpha_q[..., 1]))


def test_coerce_dqaz_wdl3_output_floors_tiny_alpha_values():
    logits = jnp.zeros((1, 2), dtype=jnp.float32)
    alpha_v = jnp.asarray([[1e-8, 0.2]], dtype=jnp.float32)
    alpha_q = jnp.asarray([[[1e-8, 0.3], [0.4, 1e-8]]], dtype=jnp.float32)

    _, out_v, out_q = _coerce_dqaz_wdl3_output(
        (logits, alpha_v, alpha_q),
        draw_alpha=1e-8,
        min_alpha=0.05,
    )

    assert float(jnp.min(out_v)) >= 0.05
    assert float(jnp.min(out_q)) >= 0.05


def test_fig_8_dqaz_settings_enable_jax_backup():
    config = _with_dqaz_eval_settings(
        Config(
            env=EnvConfig(id="hex", board_size=3),
            model=ModelConfig(network=Network.boardlaw_dirichlet),
        ),
        eval_batch_size=2,
        tree_size=4,
        search_eval_batch_size=2,
    )

    assert config.search.kind == "dqaz"
    assert config.search.dqaz.jax_backup is True
    assert config.search.dqaz.inflight_limit == 4


def test_tree_size_parser_accepts_zero_candidate_search():
    assert _nonnegative_int_csv("0,2,8") == (0, 2, 8)

    try:
        _nonnegative_int_csv("-1")
    except argparse.ArgumentTypeError:
        pass
    else:
        raise AssertionError("negative tree size should be rejected")


def test_zero_tree_size_has_custom_plot_position():
    assert _tree_size_plot_position(0) == -0.5
    assert _tree_size_plot_position(1) == 0.0
    assert _tree_size_plot_position(4) == 2.0


def test_stochastic_mcts_evaluate_uses_generic_eval_loop_smoke():
    env = pgx.make("tic_tac_toe")
    search = SearchConfig(gumbel=GumbelSearchConfig(num_simulations=1))
    config = Config(
        model=ModelConfig(
            network=Network.boardlaw,
            num_channels=8,
            num_layers=1,
        ),
        eval=EvalConfig(batch_size=2, player_search=search),
    )
    target_config = Config(
        model=ModelConfig(
            network=Network.boardlaw,
            num_channels=8,
            num_layers=1,
        ),
        eval=EvalConfig(batch_size=2, player_search=search),
    )
    model = BoardlawNet(
        num_actions=env.num_actions,
        observation_shape=env.observation_shape,
        width=8,
        depth=1,
        rngs=nnx.Rngs(0),
    )
    target_model = BoardlawNet(
        num_actions=env.num_actions,
        observation_shape=env.observation_shape,
        width=8,
        depth=1,
        rngs=nnx.Rngs(1),
    )

    returns = make_stochastic_mcts_evaluate(env, config, target_config, target_model)(
        jax.random.PRNGKey(2),
        model,
    )

    assert returns.shape == (2,)
    assert jnp.isfinite(returns).all()
