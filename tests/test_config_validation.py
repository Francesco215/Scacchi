import math
from pathlib import Path

import jax.numpy as jnp
import pytest
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import ConfigKeyError, ValidationError

from scacchi.envs import make_env
from scacchi.evaluations import load_eval_baseline
from scacchi.types import (
    Config,
    MonteCarloPosteriorUpdateConfig,
    NumericalPosteriorUpdateConfig,
    load_config,
)


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
    assert config.model.dirichlet_head_parameterization == "log_concentration"
    assert config.selfplay.batch_size == 2048
    assert config.selfplay.max_num_steps == 128
    assert config.selfplay.action_commitment.kind == "posterior_sample"
    assert (
        config.selfplay.action_commitment.posterior_update
        == "monte_carlo"
    )
    assert config.selfplay.search.kind == "dirichlet_thompson"
    assert config.selfplay.search.dirichlet_thompson.num_simulations == 16
    assert config.selfplay.search.dirichlet_thompson.max_depth == 16
    assert (
        config.selfplay.search.dirichlet_thompson.posterior_update.monte_carlo.policy_samples
        == 1
    )
    assert (
        config.selfplay.search.dirichlet_thompson.posterior_update.monte_carlo.kappa
        == 4.0
    )
    assert config.eval.player_search.kind == "dirichlet_thompson"
    assert config.eval.baseline_search.kind == "dirichlet_thompson"
    assert config.search.kind == "dirichlet_thompson"
    assert config.search.dirichlet_thompson.num_simulations == 16
    assert config.search.dirichlet_thompson.max_depth == 16
    assert (
        config.search.dirichlet_thompson.posterior_update.monte_carlo.policy_samples
        == 1
    )
    assert (
        config.search.dirichlet_thompson.posterior_update.monte_carlo.policy_sample_chunk_size
        == 1
    )
    assert config.search.dirichlet_thompson.posterior_update.monte_carlo.kappa == 4.0
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
    assert config.eval.player_action_commitment.kind == "posterior_sample"
    assert config.eval.baseline_action_commitment.kind == "posterior_sample"


def test_hex5_uses_corrected_dirichlet_search_recipe():
    cfg_path = Path(__file__).parents[1] / "scacchi" / "configs" / "hex5.yaml"
    config = load_config(OmegaConf.load(cfg_path))

    assert config.selfplay.search.dirichlet_thompson.num_simulations == 16
    assert config.selfplay.search.dirichlet_thompson.max_depth == 16
    assert (
        config.selfplay.search.dirichlet_thompson.posterior_update.monte_carlo.policy_samples
        == 32
    )
    assert (
        config.selfplay.search.dirichlet_thompson.posterior_update.monte_carlo.kappa
        == 4.0
    )
    assert config.eval.player_search.dirichlet_thompson.num_simulations == 32
    assert config.eval.player_search.dirichlet_thompson.max_depth == 32
    assert config.model.compute_dtype == "bfloat16"
    assert config.training.batch_size == 2048
    assert config.training.learning_rate == 2e-3
    assert config.training.losses.q_dir_kl_weight == 1.0
    assert config.training.losses.q_outcome_weight == 0.25
    assert config.model.dirichlet_head_parameterization == "log_concentration"
    assert config.training.regularization.dirichlet_concentration_clip == 8.0
    assert config.checkpointing.max_to_keep == 1
    assert config.checkpointing.save_interval_steps == 10


def test_hex9_restores_solved_run_policy_target_recipe():
    cfg_path = Path(__file__).parents[1] / "scacchi" / "configs" / "hex9.yaml"
    config = load_config(OmegaConf.load(cfg_path))

    selfplay_search = config.selfplay.search.dirichlet_thompson
    assert selfplay_search.root_policy_support == "search_evidence"
    assert selfplay_search.policy_target_temperature == 1.0 / 3.0
    assert (
        config.eval.player_search.dirichlet_thompson.root_policy_support
        == "search_evidence"
    )


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


def test_removed_categorical_draw_rule_is_rejected():
    with pytest.raises(ConfigKeyError, match="categorical_draw_rule"):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {
                        "categorical_draw_rule": "policy_prior"
                    },
                },
            }
        )


@pytest.mark.parametrize(
    "kappa",
    [0.0, -1.0, float("nan"), float("inf"), -float("inf")],
)
@pytest.mark.parametrize("kind", ["monte_carlo", "numerical"])
def test_dirichlet_thompson_kappa_must_be_finite_and_positive(
    kappa: float,
    kind: str,
):
    with pytest.raises(
        ValueError,
        match=rf"search.dirichlet_thompson.posterior_update.{kind}.kappa",
    ):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {
                        "posterior_update": {
                            "kind": kind,
                            kind: {"kappa": kappa},
                        },
                    },
                },
            }
        )


@pytest.mark.parametrize("legacy_key", ["kappa_leaf", "kappa_terminal"])
def test_legacy_dirichlet_kappa_keys_are_rejected(legacy_key: str):
    with pytest.raises(ConfigKeyError, match=legacy_key):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {legacy_key: 1.0},
                },
            }
        )


def test_legacy_dirichlet_constants_block_is_rejected():
    with pytest.raises(ConfigKeyError, match="constants"):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {
                        "constants": {"kappa_terminal": 80.0}
                    },
                },
            }
        )


def test_deprecated_categorical_epsilon_selects_legacy_head_compatibility():
    with pytest.warns(FutureWarning):
        config = _config(
            {"training": {"losses": {"categorical_epsilon": 0.01}}}
        )

    assert config.model.dirichlet_head_parameterization == "legacy"


def test_full_categorical_dispersion_requires_finite_reference_concentration():
    with pytest.raises(ValueError, match="finite.*concentration_clip"):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {"kind": "dirichlet_thompson"},
                "training": {
                    "losses": {"value_dir_kl_weight": 1.0},
                    "regularization": {"dirichlet_concentration_clip": None},
                },
            }
        )


def test_mean_categorical_training_does_not_require_reference_concentration():
    with pytest.warns(UserWarning, match="mean"):
        config = _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {"kind": "dirichlet_thompson"},
                "training": {
                    "losses": {
                        "value_dir_kl_weight": 1.0,
                        "dirichlet_loss_mode": "mean",
                    },
                    "regularization": {"dirichlet_concentration_clip": None},
                },
            }
        )

    assert config.training.regularization.dirichlet_concentration_clip is None


@pytest.mark.parametrize("clip", [math.inf, math.nan])
def test_dirichlet_concentration_cap_must_be_finite(clip: float):
    with pytest.raises(ValueError, match="concentration_clip.*finite"):
        _config(
            {
                "training": {
                    "regularization": {
                        "dirichlet_concentration_clip": clip,
                    }
                }
            }
        )


@pytest.mark.parametrize(
    "initial_concentration",
    [0.0, -1.0, math.inf, math.nan],
)
def test_direct_initial_concentration_must_be_finite_and_positive(
    initial_concentration: float,
):
    with pytest.raises(ValueError, match="initial_concentration"):
        _config(
            {
                "model": {
                    "network": "boardlaw_dirichlet",
                    "dirichlet_initial_concentration": (
                        initial_concentration
                    ),
                }
            }
        )


def test_direct_log_head_rejects_a_concentration_floor():
    with pytest.raises(ValueError, match="incompatible"):
        _config(
            {
                "model": {
                    "network": "boardlaw_dirichlet",
                    "dirichlet_head_parameterization": "log_concentration",
                    "dirichlet_concentration_floor": 2.0,
                }
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


def test_hex6_recipe_uses_q21_cubic_commitment_and_wandb():
    cfg_path = (
        Path(__file__).parents[1]
        / "scacchi"
        / "configs"
        / "hex6.yaml"
    )
    config = load_config(OmegaConf.load(cfg_path))

    assert config.run.seed == 9105
    assert config.run.max_num_iters == 201
    assert config.env.num_outcomes == 2
    assert config.selfplay.action_commitment.kind == "posterior_sample"
    assert (
        config.selfplay.action_commitment.posterior_update == "numerical"
    )
    assert (
        config.selfplay.action_commitment.posterior_sample_temperature
        == pytest.approx(1.0 / 3.0)
    )
    selfplay_search = config.selfplay.search.dirichlet_thompson
    assert selfplay_search.posterior_update.kind == "numerical"
    assert selfplay_search.posterior_update.numerical.half_width == 10
    assert (
        selfplay_search.posterior_update.numerical.fallback_policy_samples
        == 8
    )
    eval_search = config.eval.player_search.dirichlet_thompson
    assert eval_search.posterior_update.kind == "numerical"
    assert config.logging.wandb.enabled
    assert config.checkpointing.max_to_keep == 0


def test_hex7_uses_full_learned_concentration_head():
    cfg_path = Path(__file__).parents[1] / "scacchi" / "configs" / "hex7.yaml"
    config = load_config(OmegaConf.load(cfg_path))

    assert config.run.seed == 7104
    assert config.model.dirichlet_head_parameterization == "log_concentration"
    assert config.model.dirichlet_concentration_floor is None
    assert config.model.dirichlet_initial_concentration == 2.1
    assert config.training.losses.dirichlet_loss_mode == "full"
    assert config.selfplay.search.dirichlet_thompson.num_simulations == 64
    assert (
        config.selfplay.search.dirichlet_thompson.posterior_update.monte_carlo.policy_samples
        == 32
    )
    assert config.eval.player_search.dirichlet_thompson.num_simulations == 128
    assert (
        config.eval.player_search.dirichlet_thompson.posterior_update.monte_carlo.policy_samples
        == 32
    )


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

    baseline = load_eval_baseline(config, env)
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
        load_eval_baseline(config, env)


def test_flat_config_keys_are_rejected():
    with pytest.raises(ConfigKeyError, match="network"):
        _config({"network": "boardlaw_dirichlet"})


def test_unknown_nested_config_keys_are_rejected():
    with pytest.raises(ConfigKeyError, match="unknown"):
        _config({"search": {"gumbel": {"unknown": 1.0}}})


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


def test_full_dirichlet_loss_mode_is_configurable():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "training": {"losses": {"dirichlet_loss_mode": "full"}},
        }
    )

    assert config.training.losses.dirichlet_loss_mode == "full"


def test_mean_dirichlet_loss_mode_warns_that_it_is_comparison_only():
    with pytest.warns(
        UserWarning,
        match="Use it only for comparisons or ablations",
    ):
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


def test_monte_carlo_posterior_update_owns_the_policy_budget():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "search": {
                "kind": "dirichlet_thompson",
                "dirichlet_thompson": {
                    "posterior_update": {
                        "kind": "monte_carlo",
                        "monte_carlo": {
                            "policy_samples": 1,
                            "policy_sample_chunk_size": 1,
                        },
                    },
                },
            },
        }
    )

    search = config.search.dirichlet_thompson
    assert search.posterior_update.monte_carlo.policy_samples == 1
    assert (
        search.posterior_update.monte_carlo.policy_sample_chunk_size == 1
    )


def test_numerical_update_selects_numerical_root_readouts():
    config = _config(
        {
            "env": {"id": "hex", "board_size": 6, "num_outcomes": 2},
            "model": {"network": "boardlaw_dirichlet"},
            "selfplay": {
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {
                        "posterior_update": {"kind": "numerical"},
                    },
                },
            },
        }
    )

    search = config.selfplay.search.dirichlet_thompson
    assert search.posterior_update.kind == "numerical"
    assert isinstance(
        search.posterior_update.active(),
        NumericalPosteriorUpdateConfig,
    )
    assert search.posterior_update.numerical.half_width == 10


def test_numerical_update_infers_binary_outcomes_for_hex():
    config = _config(
        {
            "env": {"id": "hex", "board_size": 6},
            "model": {"network": "boardlaw_dirichlet"},
            "selfplay": {
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {
                        "posterior_update": {"kind": "numerical"},
                    },
                },
            },
        }
    )

    assert config.env.num_outcomes is None
    assert config.env.resolved_num_outcomes() == 2


def test_numerical_update_requires_binary_outcomes():
    with pytest.raises(
        ValueError,
        match="requires env.num_outcomes=2",
    ):
        _config(
            {
                "env": {
                    "id": "gardner_chess",
                    "board_size": 5,
                    "num_outcomes": 3,
                },
                "model": {"network": "boardlaw_dirichlet"},
                "selfplay": {
                    "search": {
                        "kind": "dirichlet_thompson",
                        "dirichlet_thompson": {
                            "posterior_update": {"kind": "numerical"},
                        },
                    },
                },
            }
        )


def test_action_commitment_selects_its_own_posterior_update():
    config = _config(
        {
            "env": {"id": "hex", "board_size": 6, "num_outcomes": 2},
            "model": {"network": "boardlaw_dirichlet"},
            "selfplay": {
                "action_commitment": {
                    "kind": "posterior_sample",
                    "posterior_update": "monte_carlo",
                },
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {
                        "posterior_update": {"kind": "numerical"},
                    },
                },
            },
        }
    )

    assert (
        config.selfplay.search.dirichlet_thompson.posterior_update.kind
        == "numerical"
    )
    assert (
        config.selfplay.action_commitment.posterior_update
        == "monte_carlo"
    )


def test_action_commitment_posterior_update_requires_dirichlet_search():
    with pytest.raises(
        ValueError,
        match="posterior_update requires dirichlet_thompson search",
    ):
        _config(
            {
                "selfplay": {
                    "action_commitment": {
                        "posterior_update": "monte_carlo",
                    },
                    "search": {"kind": "policy"},
                },
            }
        )


def test_monte_carlo_update_allows_non_binary_outcomes():
    config = _config(
        {
            "env": {
                "id": "gardner_chess",
                "board_size": 5,
                "num_outcomes": 3,
            },
            "model": {"network": "boardlaw_dirichlet"},
            "search": {"kind": "dirichlet_thompson"},
        }
    )

    search = config.selfplay.search.dirichlet_thompson
    assert search.posterior_update.kind == "monte_carlo"


def test_inactive_numerical_update_does_not_require_binary_outcomes():
    config = _config(
        {
            "env": {"num_outcomes": 3},
            "search": {
                "kind": "gumbel",
                "dirichlet_thompson": {
                    "posterior_update": {"kind": "numerical"},
                },
            },
        }
    )

    assert config.selfplay.search.kind == "gumbel"
    assert (
        config.selfplay.search.dirichlet_thompson.posterior_update.kind
        == "numerical"
    )


def test_numerical_posterior_update_parameters_are_configurable():
    config = _config(
        {
            "search": {
                "dirichlet_thompson": {
                    "posterior_update": {
                        "kind": "numerical",
                        "numerical": {
                            "half_width": 11,
                            "tail_scale": 7.0,
                            "min_half_range": 5.0,
                            "max_half_range": 9.0,
                            "fallback_policy_samples": 3,
                            "fallback_policy_sample_chunk_size": 2,
                        },
                    },
                },
            },
        }
    )

    numerical = config.search.dirichlet_thompson.posterior_update.numerical
    assert numerical.half_width == 11
    assert numerical.tail_scale == 7.0
    assert numerical.min_half_range == 5.0
    assert numerical.max_half_range == 9.0
    assert numerical.fallback_policy_samples == 3
    assert numerical.fallback_policy_sample_chunk_size == 2


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("half_width", 0),
        ("tail_scale", 0.0),
        ("tail_scale", math.nan),
        ("tail_scale", math.inf),
        ("min_half_range", 0.0),
        ("min_half_range", math.nan),
        ("min_half_range", math.inf),
        ("max_half_range", 5.0),
        ("max_half_range", math.nan),
        ("max_half_range", math.inf),
    ],
)
def test_numerical_posterior_update_parameters_are_validated(
    parameter: str,
    value: float,
):
    with pytest.raises(ValueError, match=parameter):
        _config(
            {
                "search": {
                    "dirichlet_thompson": {
                        "posterior_update": {
                            "kind": "numerical",
                            "numerical": {parameter: value},
                        },
                    },
                },
            }
        )


def test_monte_carlo_posterior_policy_budget_must_be_positive():
    with pytest.raises(
        ValueError,
        match=(
            "search.dirichlet_thompson.posterior_update.monte_carlo."
            "policy_samples"
        ),
    ):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {
                        "posterior_update": {
                            "monte_carlo": {"policy_samples": 0},
                        },
                    },
                },
            }
        )


def test_numerical_fallback_policy_budget_must_be_positive():
    with pytest.raises(
        ValueError,
        match="fallback_policy_samples",
    ):
        _config(
            {
                "search": {
                    "dirichlet_thompson": {
                        "posterior_update": {
                            "kind": "numerical",
                            "numerical": {
                                "fallback_policy_samples": 0,
                            },
                        },
                    },
                },
            }
        )


def test_monte_carlo_posterior_update_allows_null_chunk_size():
    config = _config(
        {
            "search": {
                "dirichlet_thompson": {
                    "posterior_update": {
                        "monte_carlo": {
                            "policy_sample_chunk_size": None,
                        },
                    },
                },
            },
        }
    )

    assert (
        config.search.dirichlet_thompson.posterior_update.monte_carlo.policy_sample_chunk_size
        is None
    )


def test_monte_carlo_posterior_update_chunk_size_must_be_positive_when_set():
    with pytest.raises(
        ValueError,
        match=(
            "search.dirichlet_thompson.posterior_update.monte_carlo."
            "policy_sample_chunk_size"
        ),
    ):
        _config(
            {
                "search": {
                    "dirichlet_thompson": {
                        "posterior_update": {
                            "monte_carlo": {
                                "policy_sample_chunk_size": 0,
                            },
                        },
                    },
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
            "selfplay": {
                "action_commitment": {"kind": "posterior_sample"},
            },
        }
    )

    assert config.selfplay.action_commitment.kind == "posterior_sample"


def test_posterior_sample_temperature_is_configurable():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "selfplay": {
                "action_commitment": {
                    "kind": "posterior_sample",
                    "posterior_sample_temperature": 1.0 / 3.0,
                },
            },
        }
    )

    assert (
        config.selfplay.action_commitment.posterior_sample_temperature
        == 1.0 / 3.0
    )


def test_posterior_sample_temperature_defaults_to_one():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "selfplay": {
                "action_commitment": {"kind": "posterior_sample"},
            },
        }
    )

    assert (
        config.selfplay.action_commitment.posterior_sample_temperature == 1.0
    )


def test_dirichlet_policy_target_controls_are_configurable():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "selfplay": {
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {
                        "root_policy_support": "search_evidence",
                        "policy_target_temperature": 1.0 / 3.0,
                    },
                },
            },
        }
    )

    search = config.selfplay.search.dirichlet_thompson
    assert search.root_policy_support == "search_evidence"
    assert search.policy_target_temperature == 1.0 / 3.0


@pytest.mark.parametrize(
    "temperature",
    [0.0, -1.0, math.nan, math.inf, -math.inf],
)
def test_policy_target_temperature_must_be_finite_and_positive(
    temperature: float,
):
    with pytest.raises(
        ValueError,
        match="search.dirichlet_thompson.policy_target_temperature",
    ):
        _config(
            {
                "model": {"network": "boardlaw_dirichlet"},
                "selfplay": {
                    "search": {
                        "kind": "dirichlet_thompson",
                        "dirichlet_thompson": {
                            "policy_target_temperature": temperature,
                        },
                    },
                },
            }
        )


def test_legacy_action_commitment_fields_migrate_to_player_config():
    config = _config(
        {
            "model": {"network": "boardlaw_dirichlet"},
            "selfplay": {
                "action_commitment_type": "posterior_sample",
                "search": {
                    "posterior_sample_temperature": 0.5,
                },
            },
        }
    )

    assert config.selfplay.action_commitment.kind == "posterior_sample"
    assert (
        config.selfplay.action_commitment.posterior_sample_temperature == 0.5
    )


def test_legacy_root_action_estimator_migrates_to_commitment_selector():
    config = _config(
        {
            "env": {"id": "hex", "board_size": 6, "num_outcomes": 2},
            "model": {"network": "boardlaw_dirichlet"},
            "selfplay": {
                "search": {
                    "kind": "dirichlet_thompson",
                    "dirichlet_thompson": {
                        "root_action_estimator": "prefix_cdf",
                    },
                },
            },
        }
    )

    assert (
        config.selfplay.action_commitment.posterior_update == "numerical"
    )


@pytest.mark.parametrize(
    "temperature",
    [0.0, -1.0, math.nan, math.inf, -math.inf],
)
def test_posterior_sample_temperature_must_be_finite_and_positive(
    temperature: float,
):
    with pytest.raises(
        ValueError,
        match="action_commitment.posterior_sample_temperature",
    ):
        _config(
            {
                "selfplay": {
                    "action_commitment": {
                        "posterior_sample_temperature": temperature,
                    },
                },
            }
        )


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
                "selfplay": {
                    "action_commitment": {"kind": "posterior_best"},
                },
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
