from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from scacchi.posterior_tree import is_posterior_tree_policy
from scacchi.train import Config, normalize_config_dict


def test_nested_hex_config_loads_into_runtime_config():
    cfg_path = Path(__file__).parents[1] / "scacchi" / "configs" / "hex.yaml"
    container = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)

    assert isinstance(container, dict)
    config = Config(**normalize_config_dict(container))

    assert config.env_id == "hex"
    assert config.board_size == 5
    assert config.network == "boardlaw_dirichlet"
    assert config.max_num_iters == 80
    assert config.num_channels == 512
    assert config.selfplay_batch_size == 4096
    assert config.max_num_steps == 25
    assert config.selfplay_action_source == "posterior_argmax"
    assert config.search_policy == "posterior_tree"
    assert config.num_simulations == 32
    assert config.search_eval_batch_size == 1024
    assert config.leaf_value_mode == "alpha"
    assert config.kappa_leaf == 1.0
    assert config.kappa_terminal == 8.0
    assert config.epsilon_terminal == 5e-2
    assert config.state_posterior_kappa_n == 16.0
    assert config.training_batch_size == 1024
    assert config.replay_buffer_size == 1
    assert config.learning_rate == 1e-3
    assert config.grad_clip_norm == 1.0
    assert config.policy_loss_weight == 0.05
    assert config.value_dir_kl_weight == 5.0
    assert config.q_dir_kl_weight == 5.0
    assert config.policy_target_mode == "search"
    assert config.dirichlet_concentration_clip == 8.0
    assert config.eval_interval == 1
    assert config.eval_batch_size == 512
    assert config.wandb_enabled is False
    assert config.ckpt_max_to_keep == 0


def test_hex8_terminal_targets_use_total_concentration_clip():
    cfg_path = Path(__file__).parents[1] / "scacchi" / "configs" / "hex8.yaml"
    container = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)

    assert isinstance(container, dict)
    config = Config(**normalize_config_dict(container))

    assert config.terminal_edge_targets is True
    assert config.terminal_parent_targets is True
    assert config.dirichlet_concentration_clip == 100.0


def test_flat_config_keys_take_precedence_over_nested_values():
    normalized = normalize_config_dict(
        {
            "env": {"board_size": 6},
            "model": {"num_channels": 64},
            "board_size": 7,
            "num_channels": 128,
        }
    )

    assert normalized["board_size"] == 7
    assert normalized["num_channels"] == 128


def test_deprecated_search_constants_are_rejected():
    with pytest.raises(ValueError, match="c_leaf"):
        normalize_config_dict({"search": {"constants": {"c_leaf": 1.0}}})

    with pytest.raises(ValidationError, match="c_state"):
        Config(network="boardlaw_dirichlet", c_state=0.1)


def test_scalar_network_rejects_dirichlet_loss_weights():
    with pytest.raises(ValidationError, match="network='boardlaw_dirichlet'"):
        Config(
            network="boardlaw",
            value_dir_kl_weight=1.0,
            q_dir_kl_weight=0.0,
        )


def test_scalar_network_allows_zero_dirichlet_loss_weights():
    config = Config(
        network="boardlaw",
        value_dir_kl_weight=0.0,
        q_dir_kl_weight=0.0,
    )

    assert config.network == "boardlaw"


def test_dirichlet_network_allows_dirichlet_loss_weights():
    config = Config(network="boardlaw_dirichlet", value_dir_kl_weight=1.0)

    assert config.network == "boardlaw_dirichlet"


def test_num_search_blocks_must_be_positive():
    with pytest.raises(ValidationError):
        Config(network="boardlaw_dirichlet", num_search_blocks=0)


def test_grad_clip_norm_must_be_positive_when_set():
    with pytest.raises(ValidationError):
        Config(network="boardlaw_dirichlet", grad_clip_norm=0.0)


def test_grad_clip_norm_can_be_disabled():
    config = Config(network="boardlaw_dirichlet", grad_clip_norm=None)

    assert config.grad_clip_norm is None


def test_dirichlet_concentration_clip_mode_is_rejected():
    with pytest.raises(ValueError, match="concentration logits are no longer clipped"):
        normalize_config_dict({"dirichlet_concentration_clip_mode": "logit"})

    with pytest.raises(ValueError, match="concentration logits are no longer clipped"):
        normalize_config_dict(
            {"model": {"dirichlet_concentration_clip_mode": "logit"}}
        )


def test_posterior_sample_action_source_is_valid():
    config = Config(
        network="boardlaw_dirichlet",
        selfplay_action_source="posterior_sample",
    )

    assert config.selfplay_action_source == "posterior_sample"


def test_scalar_q_argmax_action_source_is_rejected():
    with pytest.raises(ValidationError, match="selfplay_action_source"):
        Config(
            network="boardlaw_dirichlet",
            selfplay_action_source="scalar_q_argmax",
        )


def test_selfplay_action_source_must_be_known():
    with pytest.raises(ValidationError, match="selfplay_action_source"):
        Config(network="boardlaw_dirichlet", selfplay_action_source="unknown")


def test_search_policy_must_be_known():
    with pytest.raises(ValidationError, match="search_policy"):
        Config(network="boardlaw_dirichlet", search_policy="unknown")


def test_dirichlet_thompson_uses_jitted_dirichlet_q_path():
    assert not is_posterior_tree_policy("dirichlet_thompson")
    assert is_posterior_tree_policy("posterior_tree")
    assert not is_posterior_tree_policy("posterior_tree_wavefront")


def test_dirichlet_thompson_allows_legacy_two_outcome_hex_heads():
    config = Config(
        env_id="hex",
        network="boardlaw_dirichlet",
        search_policy="dirichlet_thompson",
        num_outcomes=2,
    )

    assert config.num_outcomes == 2


def test_policy_target_mode_must_be_known():
    with pytest.raises(ValidationError, match="policy_target_mode"):
        Config(network="boardlaw_dirichlet", policy_target_mode="unknown")


def test_posterior_tree_search_requires_dirichlet_network_and_wdl3():
    with pytest.raises(ValidationError, match="boardlaw_dirichlet"):
        Config(
            network="boardlaw",
            search_policy="posterior_tree",
            value_dir_kl_weight=0.0,
            q_dir_kl_weight=0.0,
        )
    with pytest.raises(ValidationError, match="WDL3"):
        Config(
            network="boardlaw_dirichlet",
            search_policy="posterior_tree",
            num_outcomes=2,
        )


def test_wavefront_posterior_tree_policy_is_removed():
    with pytest.raises(ValidationError, match="search_policy"):
        Config(network="boardlaw_dirichlet", search_policy="posterior_tree_wavefront")


def test_model_construction_context_allows_legacy_checkpoint_loss_weights():
    config = Config.model_validate(
        {
            "network": "boardlaw",
            "value_dir_kl_weight": 1.0,
            "q_dir_kl_weight": 1.0,
        },
        context={"model_construction_only": True},
    )

    assert config.network == "boardlaw"
