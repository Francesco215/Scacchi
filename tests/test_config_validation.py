from pathlib import Path

import jax.numpy as jnp
import pytest
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigKeyError, ValidationError

from scacchi.envs import make_env
from scacchi.train import _load_eval_baseline
from scacchi.types import Config, load_config


def _config(values: dict) -> Config:
    return load_config(OmegaConf.create(values))


def test_config_yaml_loads_into_nested_runtime_config():
    cfg_path = Path(__file__).parents[1] / "scacchi" / "configs" / "config.yaml"
    loaded = OmegaConf.load(cfg_path)
    assert isinstance(loaded, DictConfig)
    config = load_config(loaded)

    assert config.run.seed == 1
    assert config.run.max_num_iters == 2000
    assert config.env.id == "gardner_chess"
    assert config.env.board_size == 5
    assert config.env.num_outcomes == 3
    assert config.model.network == "aznet_dirichlet"
    assert config.model.num_channels == 128
    assert config.model.num_layers == 6
    assert config.model.compute_dtype == "float32"
    assert config.selfplay.batch_size == 2048
    assert config.selfplay.max_num_steps == 128
    assert config.selfplay.action_commitment_type == "posterior_sample"
    assert config.selfplay.search.kind == "dirichlet_thompson"
    assert config.selfplay.search.dirichlet_thompson.num_simulations == 16
    assert config.selfplay.search.dirichlet_thompson.max_depth == 16
    assert config.selfplay.search.dirichlet_thompson.policy_samples == 1
    assert config.selfplay.search.dirichlet_thompson.policy_sample_chunk_size == 1
    assert config.selfplay.search.dirichlet_thompson.constants.kappa_leaf == 1.0
    assert config.selfplay.search.dirichlet_thompson.constants.kappa_terminal == 8.0
    assert config.eval.player_search.kind == "dirichlet_thompson"
    assert config.eval.baseline_search.kind == "dirichlet_thompson"
    assert config.search.kind == "dirichlet_thompson"
    assert config.search.dirichlet_thompson.num_simulations == 16
    assert config.search.dirichlet_thompson.max_depth == 16
    assert config.search.dirichlet_thompson.policy_samples == 1
    assert config.search.dirichlet_thompson.policy_sample_chunk_size == 1
    assert config.search.dirichlet_thompson.constants.kappa_leaf == 1.0
    assert config.search.dirichlet_thompson.constants.kappa_terminal == 8.0
    assert config.training.batch_size == 4096
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
    assert config.eval.batch_size == 1024
    assert config.eval.baseline == "pgx"
    assert config.eval.baseline_id == "gardner_chess_v0"
    assert config.logging.wandb.enabled is True
    assert config.checkpointing.max_to_keep == 50
    assert config.checkpointing.save_interval_steps == 100


def test_go9x9_gumbel_config_matches_paper_level_recipe():
    cfg_path = Path(__file__).parents[1] / "scacchi" / "configs" / "go9x9_gumbel.yaml"
    config = load_config(OmegaConf.load(cfg_path))

    assert config.run.max_num_iters == 400
    assert config.env.id == "go_9x9"
    assert config.env.board_size == 9
    assert config.model.network == "aznet"
    assert config.model.num_channels == 128
    assert config.model.num_layers == 6
    assert config.model.resnet_v2 is True
    assert config.selfplay.batch_size == 1024
    assert config.selfplay.max_num_steps == 256
    assert config.selfplay.action_commitment_type == "search_action"
    assert config.selfplay.search.kind == "gumbel"
    assert config.selfplay.search.gumbel.num_simulations == 32
    assert config.selfplay.search.gumbel.completed_q_value_scale == 0.1
    assert config.selfplay.search.gumbel.completed_q_rescale_values is True
    assert config.training.batch_size == 4096
    assert config.training.max_updates_per_iter is None
    assert config.training.learning_rate == 1e-3
    assert config.eval.baseline == "pgx"
    assert config.eval.baseline_id == "go_9x9_v0"
    assert config.eval.player_action_commitment_type == "posterior_sample"
    assert config.eval.baseline_action_commitment_type == "posterior_sample"
    assert config.eval.player_search.kind == "policy"
    assert config.eval.player_search.policy.temperature == 1.0
    assert config.eval.baseline_search.kind == "policy"
    assert config.eval.baseline_search.policy.temperature == 1.0


def test_simple_policy_eval_fragment_overrides_eval_search_only():
    cfg_dir = Path(__file__).parents[1] / "scacchi" / "configs"
    base = OmegaConf.load(cfg_dir / "hex5.yaml")
    policy_eval = OmegaConf.load(cfg_dir / "eval_mode" / "simple_policy.yaml")

    config = load_config(OmegaConf.merge(base, policy_eval))

    assert config.selfplay.search.kind == "dirichlet_thompson"
    assert config.eval.player_search.kind == "policy"
    assert config.eval.player_search.policy.temperature == 1.0
    assert config.eval.baseline_search.kind == "policy"
    assert config.eval.baseline_search.policy.temperature == 1.0
    assert config.eval.player_action_commitment_type == "posterior_sample"
    assert config.eval.baseline_action_commitment_type == "posterior_sample"


def test_hex5_uses_corrected_dirichlet_search_recipe():
    cfg_path = Path(__file__).parents[1] / "scacchi" / "configs" / "hex5.yaml"
    config = load_config(OmegaConf.load(cfg_path))

    assert config.selfplay.search.dirichlet_thompson.num_simulations == 16
    assert config.selfplay.search.dirichlet_thompson.max_depth == 16
    assert config.selfplay.search.dirichlet_thompson.policy_samples == 32
    assert (
        config.selfplay.search.dirichlet_thompson.constants.kappa_terminal
        == 80.0
    )
    assert (
        config.selfplay.search.dirichlet_thompson.constants.categorical_epsilon
        == 0.01
    )
    assert config.eval.player_search.dirichlet_thompson.num_simulations == 32
    assert config.eval.player_search.dirichlet_thompson.max_depth == 32
    assert config.model.compute_dtype == "bfloat16"
    assert config.training.batch_size == 2048
    assert config.training.learning_rate == 2e-3
    assert config.training.losses.q_dir_kl_weight == 1.0
    assert config.training.losses.q_outcome_weight == 0.25
    assert (
        config.checkpointing.directory
        == "checkpoints/hex5_dirichlet_fresh_pi_seed9101_v1"
    )
    assert config.checkpointing.max_to_keep == 1
    assert config.checkpointing.save_interval_steps == 10


def test_policy_search_temperature_must_be_positive():
    with pytest.raises(ValueError, match="search.policy.temperature"):
        _config(
            {
                "search": {
                    "kind": "policy",
                    "policy": {"temperature": 0.0},
                },
            }
        )


@pytest.mark.parametrize(
    ("config_name", "eval_simulations"),
    [
        ("hex", 32),
        ("hex5", 4),
        ("hex6", 4),
        ("hex7", 32),
        ("hex8", 4),
    ],
)
def test_hex_checkpoint_baseline_configs_use_scalar_gumbel_eval_search(
    config_name: str,
    eval_simulations: int,
):
    cfg_path = Path(__file__).parents[1] / "scacchi" / "configs" / f"{config_name}.yaml"
    config = load_config(OmegaConf.load(cfg_path))

    assert config.selfplay.search.kind == "dirichlet_thompson"
    assert config.eval.baseline == "checkpoint"
    assert config.eval.baseline_search.kind == "gumbel"
    assert config.eval.baseline_search.gumbel.num_simulations == eval_simulations


def test_make_env_supports_custom_go8():
    env = make_env("go", 8)

    assert env.id == "go_8x8"
    assert env.num_actions == 65
    assert env.observation_shape == (8, 8, 17)


def test_gardner_pgx_eval_baseline_matches_env_action_space():
    env = make_env("gardner_chess")
    config = _config(
        {
            "env": {"id": "gardner_chess"},
            "model": {"network": "aznet_dirichlet"},
            "search": {"kind": "dirichlet_thompson"},
            "eval": {
                "interval": 1,
                "baseline": "pgx",
                "baseline_id": "gardner_chess_v0",
            },
        }
    )

    baseline = _load_eval_baseline(config, env)
    observation = jnp.zeros((1, *env.observation_shape), dtype=jnp.float32)
    output = baseline(observation)
    logits = output[0] if isinstance(output, tuple) else output
    assert logits.shape == (1, env.num_actions)


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


def test_legacy_top_level_search_populates_play_mode_search_configs():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "search": {
                "kind": "dirichlet_thompson",
                "dirichlet_thompson": {"num_simulations": 7},
            },
        }
    )

    assert config.search.kind == "dirichlet_thompson"
    assert config.selfplay.search.kind == "dirichlet_thompson"
    assert config.eval.player_search.kind == "dirichlet_thompson"
    assert config.eval.baseline_search.kind == "dirichlet_thompson"
    assert config.selfplay.search.dirichlet_thompson.num_simulations == 7
    assert config.eval.player_search.dirichlet_thompson.num_simulations == 7
    assert config.eval.baseline_search.dirichlet_thompson.num_simulations == 7
    assert config.selfplay.search.dirichlet_thompson.max_depth == 7
    assert config.eval.player_search.dirichlet_thompson.max_depth == 7
    assert config.eval.baseline_search.dirichlet_thompson.max_depth == 7


def test_dirichlet_thompson_search_accepts_explicit_max_depth():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "search": {
                "kind": "dirichlet_thompson",
                "dirichlet_thompson": {
                    "num_simulations": 7,
                    "max_depth": 3,
                },
            },
        }
    )

    assert config.search.dirichlet_thompson.max_depth == 3
    assert config.selfplay.search.dirichlet_thompson.max_depth == 3
    assert config.eval.player_search.dirichlet_thompson.max_depth == 3


def test_nested_selfplay_search_populates_top_level_compatibility_alias():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "selfplay": {
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {"num_simulations": 9},
                }
            },
        }
    )

    assert config.selfplay.search.kind == "dirichlet_thompson"
    assert config.search.kind == "dirichlet_thompson"
    assert config.eval.player_search.kind == "dirichlet_thompson"
    assert config.search.dirichlet_thompson.num_simulations == 9


@pytest.mark.parametrize(
    "key",
    [
        "sample_one_step_per_game",
        "sample_one_step_global",
        "return_suffix_mode",
        "replay_buffer_size",
    ],
)
def test_removed_sampled_rollout_training_keys_are_rejected(key):
    with pytest.raises(ConfigKeyError, match=key):
        _config({"training": {key: True}})


def test_removed_compatibility_config_is_rejected():
    with pytest.raises(ConfigKeyError, match="compatibility"):
        _config({"compatibility": {"rng_split_mode": "legacy_eval_train"}})


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


def test_dirichlet_mean_loss_mode_is_configurable():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "training": {"losses": {"dirichlet_loss_mode": "mean"}},
        }
    )

    assert config.training.losses.dirichlet_loss_mode == "mean"


def test_num_search_blocks_is_not_a_public_config_field():
    with pytest.raises(ConfigKeyError, match="num_blocks"):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {"num_blocks": 1},
                },
            }
        )


def test_dirichlet_thompson_allows_zero_simulations_for_prior_only_probe():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "search": {
                "kind": "dirichlet_thompson",
                "dirichlet_thompson": {"num_simulations": 0},
            },
        }
    )

    assert config.search.dirichlet_thompson.num_simulations == 0


def test_dirichlet_thompson_simulations_must_be_non_negative():
    with pytest.raises(ValueError, match="search.dirichlet_thompson.num_simulations"):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {"num_simulations": -1},
                },
            }
        )


def test_dirichlet_thompson_positive_search_rejects_zero_max_depth():
    with pytest.raises(ValueError, match="search.dirichlet_thompson.max_depth"):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {
                        "num_simulations": 1,
                        "max_depth": 0,
                    },
                },
            }
        )


def test_dirichlet_thompson_zero_search_allows_zero_max_depth():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "search": {
                "kind": "dirichlet_thompson",
                "dirichlet_thompson": {
                    "num_simulations": 0,
                    "max_depth": 0,
                },
            },
        }
    )

    assert config.search.dirichlet_thompson.max_depth == 0


def test_dirichlet_thompson_allows_zero_policy_samples_for_search_policy_targets():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "search": {
                "kind": "dirichlet_thompson",
                "dirichlet_thompson": {"policy_samples": 0},
            },
        }
    )

    assert config.search.dirichlet_thompson.policy_samples == 0


def test_dirichlet_thompson_policy_samples_must_be_non_negative():
    with pytest.raises(ValueError, match="search.dirichlet_thompson.policy_samples"):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {"policy_samples": -1},
                },
            }
        )


def test_dirichlet_thompson_accepts_separate_posterior_policy_budget():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "search": {
                "kind": "dirichlet_thompson",
                "dirichlet_thompson": {
                    "policy_samples": 32,
                    "posterior_policy_samples": 1,
                },
            },
        }
    )

    search = config.search.dirichlet_thompson
    assert search.policy_samples == 32
    assert search.posterior_policy_samples == 1


def test_dirichlet_thompson_posterior_policy_budget_must_be_positive():
    with pytest.raises(
        ValueError,
        match="search.dirichlet_thompson.posterior_policy_samples",
    ):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {"posterior_policy_samples": 0},
                },
            }
        )


def test_dirichlet_thompson_allows_null_policy_sample_chunk_size_for_full_chunk():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "search": {
                "kind": "dirichlet_thompson",
                "dirichlet_thompson": {"policy_sample_chunk_size": None},
            },
        }
    )

    assert config.search.dirichlet_thompson.policy_sample_chunk_size is None


def test_dirichlet_thompson_policy_sample_chunk_size_must_be_positive_when_set():
    with pytest.raises(
        ValueError,
        match="search.dirichlet_thompson.policy_sample_chunk_size",
    ):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {"policy_sample_chunk_size": 0},
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


def test_posterior_sample_action_commitment_type_is_valid():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "selfplay": {"action_commitment_type": "posterior_sample"},
        }
    )

    assert config.selfplay.action_commitment_type == "posterior_sample"


def test_legacy_action_source_is_rejected():
    with pytest.raises(ConfigKeyError, match="action_source"):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "selfplay": {"action_source": "posterior_sample"},
            }
        )


def test_selfplay_action_commitment_type_must_be_known():
    with pytest.raises(ValidationError, match="posterior_argmax"):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "selfplay": {"action_commitment_type": "posterior_best"},
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


def test_eval_baseline_none_requires_eval_disabled():
    with pytest.raises(ValueError, match="eval.baseline=none"):
        _config({"eval": {"baseline": "none", "interval": 1}})
