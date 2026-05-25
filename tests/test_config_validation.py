from pathlib import Path

import pytest
from omegaconf import OmegaConf
from pydantic import ValidationError

from scacchi.train import Config, normalize_config_dict


def test_nested_hex_config_loads_into_runtime_config():
    cfg_path = Path(__file__).parents[1] / "scacchi" / "configs" / "hex.yaml"
    container = OmegaConf.to_container(OmegaConf.load(cfg_path), resolve=True)

    assert isinstance(container, dict)
    config = Config(**normalize_config_dict(container))

    assert config.env_id == "hex"
    assert config.board_size == 4
    assert config.network == "boardlaw_dirichlet"
    assert config.max_num_iters == 30
    assert config.num_channels == 64
    assert config.selfplay_batch_size == 32
    assert config.search_policy == "posterior_tree_wavefront"
    assert config.num_simulations == 1
    assert config.search_eval_batch_size == 1024
    assert config.wavefront_num_lanes_per_root == 1
    assert config.wavefront_final_action_mode == "posterior_sample"
    assert config.leaf_value_mode == "alpha"
    assert config.kappa_leaf == 1.0
    assert config.kappa_terminal == 8.0
    assert config.epsilon_terminal == 5e-2
    assert config.state_posterior_kappa_n == 16.0
    assert config.train_tree_nodes is False
    assert config.train_tree_include_root is False
    assert config.exact_hex_solver_enabled is True
    assert config.exact_hex_solver_extra_batch_size == 128
    assert config.training_batch_size == 256
    assert config.replay_buffer_size == 4
    assert config.eval_interval == 1
    assert config.eval_batch_size == 128
    assert config.wandb_enabled is False
    assert config.ckpt_max_to_keep == 0


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
            value_outcome_weight=0.0,
            q_outcome_weight=0.0,
        )


def test_scalar_network_allows_zero_dirichlet_loss_weights():
    config = Config(
        network="boardlaw",
        value_dir_kl_weight=0.0,
        q_dir_kl_weight=0.0,
        value_outcome_weight=0.0,
        q_outcome_weight=0.0,
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


def test_posterior_tree_search_requires_dirichlet_network_and_wdl3():
    with pytest.raises(ValidationError, match="boardlaw_dirichlet"):
        Config(
            network="boardlaw",
            search_policy="posterior_tree",
            value_dir_kl_weight=0.0,
            q_dir_kl_weight=0.0,
            value_outcome_weight=0.0,
            q_outcome_weight=0.0,
        )
    with pytest.raises(ValidationError, match="WDL3"):
        Config(
            network="boardlaw_dirichlet",
            search_policy="posterior_tree",
            num_outcomes=2,
        )


def test_wavefront_posterior_tree_search_policy_is_valid():
    config = Config(network="boardlaw_dirichlet", search_policy="posterior_tree_wavefront")

    assert config.search_policy == "posterior_tree_wavefront"
    assert config.wavefront_backend == "arena"


def test_tree_node_training_requires_wavefront_policy():
    config = Config(
        network="boardlaw_dirichlet",
        search_policy="posterior_tree_wavefront",
        train_tree_nodes=True,
    )

    assert config.train_tree_nodes is True

    with pytest.raises(ValidationError, match="train_tree_nodes"):
        Config(
            network="boardlaw_dirichlet",
            search_policy="gumbel",
            train_tree_nodes=True,
        )


def test_wavefront_knobs_are_validated():
    config = Config(
        network="boardlaw_dirichlet",
        search_policy="posterior_tree_wavefront",
        wavefront_num_lanes_per_root=2,
        wavefront_max_depth=32,
        wavefront_final_action_mode="posterior_sample",
    )

    assert config.wavefront_num_lanes_per_root == 2
    assert config.wavefront_max_depth == 32
    assert config.wavefront_final_action_mode == "posterior_sample"
    assert config.wavefront_pad_eval_batches is True
    assert config.wavefront_pad_jax_select is False
    assert config.wavefront_np_select_below == 1024
    assert config.wavefront_grouped_expansion is True
    assert config.wavefront_lane_indexed_step is True
    assert config.wavefront_stable_lane_batch is True
    assert config.wavefront_pad_pending_observation_gather is True

    with pytest.raises(ValidationError, match="wavefront_final_action_mode"):
        Config(
            network="boardlaw_dirichlet",
            search_policy="posterior_tree_wavefront",
            wavefront_final_action_mode="scalar_q_argmax",
        )
    with pytest.raises(ValidationError, match="wavefront_final_action_mode"):
        Config(
            network="boardlaw_dirichlet",
            search_policy="posterior_tree_wavefront",
            wavefront_final_action_mode="argmax_q_mean",
        )

    with pytest.raises(ValidationError, match="wavefront_final_action_mode"):
        Config(
            network="boardlaw_dirichlet",
            search_policy="posterior_tree_wavefront",
            wavefront_final_action_mode="unknown",
        )


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
