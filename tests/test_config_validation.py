from pathlib import Path

import pytest
from omegaconf import OmegaConf
from omegaconf.errors import ConfigKeyError, ValidationError

from scacchi.envs import make_env
from scacchi.train import _load_eval_baseline
from scacchi.types import Config, load_config


def _config(values: dict) -> Config:
    return load_config(OmegaConf.create(values))


def test_config_yaml_loads_into_nested_runtime_config():
    cfg_path = Path(__file__).parents[1] / "scacchi" / "configs" / "config.yaml"
    config = load_config(OmegaConf.load(cfg_path))

    assert config.run.seed == 1
    assert config.run.max_num_iters == 2000
    assert config.env.id == "gardner_chess"
    assert config.env.board_size == 5
    assert config.env.num_outcomes == 3
    assert config.model.network == "aznet_dirichlet"
    assert config.model.num_channels == 128
    assert config.model.num_layers == 6
    assert config.selfplay.batch_size == 128
    assert config.selfplay.max_num_steps == 128
    assert config.selfplay.action_source == "posterior_sample"
    assert config.search.kind == "dirichlet_thompson"
    assert config.search.dirichlet_thompson.num_simulations == 4
    assert config.search.dirichlet_thompson.num_blocks == 8
    assert config.search.dirichlet_thompson.constants.kappa_leaf == 1.0
    assert config.search.dirichlet_thompson.constants.kappa_terminal == 8.0
    assert config.training.batch_size == 16_384
    assert config.training.max_updates_per_iter == 1
    assert config.training.learning_rate == 1e-3
    assert config.training.grad_clip_norm == 1.0
    assert config.training.losses.policy_weight == 0.5
    assert config.training.losses.value_dir_kl_weight == 5.0
    assert config.training.losses.q_dir_kl_weight == 5.0
    assert config.training.losses.terminal_edge_targets is True
    assert config.training.losses.terminal_parent_targets is True
    assert config.training.regularization.dirichlet_concentration_clip == 300.0
    assert config.eval.interval == 10
    assert config.eval.batch_size == 128
    assert config.eval.baseline == "pgx"
    assert config.eval.baseline_id == "gardner_chess_v0"
    assert config.logging.wandb.enabled is True
    assert config.checkpointing.max_to_keep == 50
    assert config.checkpointing.save_interval_steps == 100
    assert config.compatibility.rng_split_mode == "legacy_eval_train"


def test_make_env_supports_custom_go8():
    env = make_env("go", 8)

    assert env.id == "go_8x8"
    assert env.num_actions == 65
    assert env.observation_shape == (8, 8, 17)


def test_incompatible_pgx_eval_baseline_raises():
    env = make_env("gardner_chess")
    config = _config(
        {
            "env": {"id": "gardner_chess"},
            "model": {"network": "aznet_dirichlet"},
            "search": {"kind": "dirichlet_thompson"},
            "eval": {
                "interval": 1,
                "baseline": "pgx",
                "baseline_id": "go_9x9_v0",
            },
        }
    )

    with pytest.raises(ValueError, match="incompatible"):
        _load_eval_baseline(config, env)


def test_flat_config_keys_are_rejected():
    with pytest.raises(ConfigKeyError, match="network"):
        _config({"network": "boardlaw_dirichlet"})


def test_unknown_nested_config_keys_are_rejected():
    with pytest.raises(ConfigKeyError, match="unknown"):
        _config({"search": {"gumbel": {"constants": {"unknown": 1.0}}}})


def test_scalar_network_rejects_dirichlet_loss_weights():
    with pytest.raises(ValueError, match="Dirichlet loss weights"):
        _config(
            {
                "model": {"network": "boardlaw"},
                "training": {
                    "losses": {
                        "value_dir_kl_weight": 1.0,
                        "q_dir_kl_weight": 0.0,
                    }
                },
            }
        )


def test_scalar_network_allows_zero_dirichlet_loss_weights():
    config = _config(
        {
            "model": {"network": "boardlaw"},
            "training": {
                "losses": {
                    "value_dir_kl_weight": 0.0,
                    "q_dir_kl_weight": 0.0,
                }
            },
        }
    )

    assert config.model.network == "boardlaw"


def test_dirichlet_network_allows_dirichlet_loss_weights():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "training": {"losses": {"value_dir_kl_weight": 1.0}},
        }
    )

    assert config.model.network == "boardlaw_dirichlet"


def test_num_search_blocks_must_be_positive():
    with pytest.raises(ValueError, match="search.dirichlet_thompson.num_blocks"):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {"num_blocks": 0},
                },
            }
        )


def test_grad_clip_norm_must_be_positive_when_set():
    with pytest.raises(ValueError, match="training.grad_clip_norm"):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "training": {"grad_clip_norm": 0.0},
            }
        )


def test_grad_clip_norm_can_be_disabled():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "training": {"grad_clip_norm": None},
        }
    )

    assert config.training.grad_clip_norm is None


def test_posterior_sample_action_source_is_valid():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "selfplay": {"action_source": "posterior_sample"},
        }
    )

    assert config.selfplay.action_source == "posterior_sample"


def test_selfplay_action_source_must_be_known():
    with pytest.raises(ValidationError, match="posterior_sample"):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "selfplay": {"action_source": "unknown"},
            }
        )


def test_search_kind_must_be_known():
    with pytest.raises(ValidationError, match="dirichlet_thompson"):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {"kind": "unknown"},
            }
        )


def test_dirichlet_thompson_allows_legacy_two_outcome_hex_heads():
    config = _config(
        {
            "env": {"id": "hex", "num_outcomes": 2},
            "model": {"network": "boardlaw_dirichlet"},
            "search": {"kind": "dirichlet_thompson"},
        }
    )

    assert config.env.num_outcomes == 2


def test_policy_target_mode_must_be_known():
    with pytest.raises(ValidationError, match="winner_action"):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "training": {"losses": {"policy_target_mode": "unknown"}},
            }
        )


def test_posterior_tree_policy_is_not_part_of_training_config():
    with pytest.raises(ValidationError, match="posterior_tree"):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {"kind": "posterior_tree"},
            }
        )


def test_eval_baseline_none_requires_eval_disabled():
    with pytest.raises(ValueError, match="eval.baseline=none"):
        _config({"eval": {"baseline": "none", "interval": 1}})
