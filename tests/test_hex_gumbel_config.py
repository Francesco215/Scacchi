from pathlib import Path

from hydra import compose, initialize_config_dir
import pytest

from scacchi.types import load_config


@pytest.mark.parametrize("board_size", range(3, 10))
def test_numbered_hex_configs_accept_plain_gumbel_overlay(board_size: int):
    config_dir = Path(__file__).parents[1] / "scacchi" / "configs"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        dirichlet_config = load_config(compose(config_name=f"hex{board_size}"))
        config = load_config(
            compose(
                config_name=f"hex{board_size}",
                overrides=["+algorithm=gumbel"],
            )
        )

    assert config.env.board_size == board_size
    assert config.selfplay.max_num_steps == board_size**2
    assert config.model.network == "aznet"
    assert config.selfplay.search.kind == "gumbel"
    assert config.search.kind == "gumbel"
    assert config.selfplay.search.gumbel.num_simulations == 32
    assert config.selfplay.action_commitment.kind == "search_action"
    assert config.selfplay.action_commitment.posterior_update is None
    assert config.eval.player_search.kind == "gumbel"
    assert config.eval.baseline_search.kind == "gumbel"
    assert config.eval.player_action_commitment.kind == "search_action"
    assert config.training.losses.value_dir_kl_weight == 0.0
    assert config.training.losses.q_dir_kl_weight == 0.0
    assert not config.training.losses.terminal_edge_targets
    assert not config.training.losses.terminal_parent_targets

    # Keep the current Dirichlet-Thompson experiment's control variables.
    assert config.run == dirichlet_config.run
    assert config.env == dirichlet_config.env
    assert config.model.num_channels == dirichlet_config.model.num_channels
    assert config.model.num_layers == dirichlet_config.model.num_layers
    assert config.model.compute_dtype == dirichlet_config.model.compute_dtype
    assert config.selfplay.batch_size == dirichlet_config.selfplay.batch_size
    assert config.selfplay.max_num_steps == dirichlet_config.selfplay.max_num_steps
    assert config.training.batch_size == dirichlet_config.training.batch_size
    assert config.training.learning_rate == dirichlet_config.training.learning_rate
    assert config.training.grad_clip_norm == dirichlet_config.training.grad_clip_norm
