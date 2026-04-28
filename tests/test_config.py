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
    assert cfg.model.resnet_v2 is True
    assert cfg.model.batch_norm is True
    assert cfg.train.search.num_simulations == 16
    assert cfg.train.search.gumbel_scale == 1.0
    assert cfg.eval.batch_size % 2 == 0
    assert cfg.runtime.num_devices == 1
    assert cfg.logging.wandb_enabled is False


def test_post_init_coerces_bool_and_numeric_values():
    eval_cfg = EvalConfig(
        enabled=cast(Any, "false"),
        interval=cast(Any, "5"),
        batch_size=cast(Any, "3"),
        max_num_steps=cast(Any, "64"),
    )

    assert eval_cfg.enabled is False
    assert eval_cfg.batch_size == 3
    assert eval_cfg.interval == 5


def test_train_search_config_accepts_overrides():
    raw_cfg = OmegaConf.merge(
        OmegaConf.load("scacchi/configs/config.yaml"),
        OmegaConf.create(
            {
                "train": {"search": {"num_simulations": 9, "gumbel_scale": 1.5}},
            }
        ),
    )

    cfg = config_from_dict_config(cast(DictConfig, raw_cfg))

    assert cfg.train.search.num_simulations == 9
    assert cfg.train.search.gumbel_scale == 1.5


def test_enabled_eval_requires_even_batch_size():
    with pytest.raises(ValueError, match="eval.batch_size must be even"):
        EvalConfig(
            enabled=True,
            interval=10,
            batch_size=3,
            max_num_steps=64,
        )


def test_unknown_top_level_config_section_is_rejected():
    raw_cfg = OmegaConf.create({"unknown": {"enabled": True}})

    with pytest.raises(ValueError, match="Unknown config section"):
        config_from_dict_config(raw_cfg)
