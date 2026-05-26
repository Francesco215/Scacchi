from pathlib import Path
from typing import cast

import pytest
from omegaconf import DictConfig, OmegaConf

from scacchi.train import ConfigError, SearchPolicy, load_config


def _load_config(name: str):
    path = Path(__file__).parents[1] / "scacchi" / "configs" / name
    return load_config(cast(DictConfig, OmegaConf.load(path)))


def test_hex_config_loads_native_posterior_tree_runtime_config():
    config = _load_config("hex.yaml")

    assert config.env.id == "hex"
    assert config.env.board_size == 5
    assert config.model.network == "boardlaw_dirichlet"
    assert config.search.policy == "posterior_tree_wavefront"
    assert config.selfplay.action_source == "search_action"
    assert config.env.num_outcomes == 3
    assert config.search.num_simulations == 32
    assert config.search.eval_batch_size == 1024
    assert config.search.inflight_limit == 4
    assert config.search.max_depth == 25
    assert config.search.final_action_mode == "posterior_argmax"
    assert config.search.leaf_value_mode == "alpha"
    assert config.search.constants.kappa_leaf == 1.0
    assert config.search.constants.state_posterior_kappa_n == 16.0
    assert config.search.categorical.epsilon == 1e-4
    assert config.search.categorical.draw_rule == "policy_prior"
    assert config.training.batch_size == 1024
    assert config.training.replay_buffer_size == 1
    assert config.training.learning_rate == 1e-3
    assert config.training.grad_clip_norm == 1.0
    assert config.training.losses.policy_weight == 0.05
    assert config.training.losses.value_dir_kl_weight == 5.0
    assert config.training.losses.q_dir_kl_weight == 5.0
    assert config.training.regularization.dirichlet_concentration_clip == 8.0
    assert config.eval.interval == 1
    assert config.eval.batch_size == 512
    assert config.logging.wandb.enabled is True
    assert config.checkpointing.max_to_keep == 0


def test_native_posterior_prototype_config_loads_like_default_hex():
    assert _load_config("hex_native_posterior_prototype.yaml").to_dict() == _load_config(
        "hex.yaml"
    ).to_dict()


def test_invalid_and_legacy_config_values_are_rejected():
    cases = [
        {"search": {"constants": {"c_leaf": 1.0}}},
        {"model": {"network": "boardlaw"}},
        {"model": {"network": "aznet"}},
        {"env": {"num_outcomes": 2}},
        {"search": {"policy": "gumbel"}},
        {"search": {"policy": "posterior_tree"}},
        {"selfplay": {"action_source": "posterior_sample"}},
        {"search": {"final_action_mode": "scalar_q_argmax"}},
        {"search": {"max_depth": 0}},
        {"search": {"inflight_limit": 0}},
        {"training": {"grad_clip_norm": 0.0}},
    ]
    for case in cases:
        with pytest.raises(ConfigError):
            load_config(case)


def test_search_policy_surface_is_only_native_posterior_tree():
    assert tuple(SearchPolicy) == (SearchPolicy.posterior_tree_wavefront,)
