from __future__ import annotations

from typing import Any, cast

import pytest
from omegaconf import DictConfig, OmegaConf

from scacchi.config import EvalConfig, ScacchiConfig, config_from_dict_config


def test_default_yaml_config_becomes_validated_dataclass():
    raw_cfg = OmegaConf.load("scacchi/configs/config.yaml")
    assert isinstance(raw_cfg, DictConfig)

    cfg = config_from_dict_config(raw_cfg)

    assert isinstance(cfg, ScacchiConfig)
    assert cfg.train.num_iters == 10_000
    assert isinstance(cfg.train.batch_size, int)
    assert cfg.train.search.num_simulations == 16
    assert cfg.train.search.gumbel_scale == 1.0
    assert cfg.eval.batch_size % 2 == 0
    assert cfg.eval.search.num_simulations == 4
    assert cfg.eval.search.gumbel_scale == 0.0
    assert cfg.runtime.num_devices == 1


def test_post_init_coerces_bool_and_numeric_values():
    eval_cfg = EvalConfig(
        enabled=cast(Any, "false"),
        batch_size=cast(Any, "3"),
        interval=cast(Any, "5"),
    )

    assert eval_cfg.enabled is False
    assert eval_cfg.batch_size == 3
    assert eval_cfg.interval == 5


def test_train_and_eval_search_configs_are_independent():
    raw_cfg = OmegaConf.create(
        {
            "train": {"search": {"num_simulations": 9, "gumbel_scale": 1.5}},
            "eval": {"search": {"num_simulations": 3, "gumbel_scale": 0.0}},
        }
    )

    cfg = config_from_dict_config(raw_cfg)

    assert cfg.train.search.num_simulations == 9
    assert cfg.train.search.gumbel_scale == 1.5
    assert cfg.eval.search.num_simulations == 3
    assert cfg.eval.search.gumbel_scale == 0.0


def test_enabled_eval_requires_even_batch_size():
    with pytest.raises(ValueError, match="eval.batch_size must be even"):
        EvalConfig(enabled=True, batch_size=3)


def test_unknown_top_level_config_section_is_rejected():
    raw_cfg = OmegaConf.create({"unknown": {"enabled": True}})

    with pytest.raises(ValueError, match="Unknown config section"):
        config_from_dict_config(raw_cfg)
