"""Structured training configuration."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, fields
from typing import Any, Mapping, TypeVar, cast

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException

T = TypeVar("T")


def _as_str(value: Any, name: str) -> str:
    text = str(value).strip()
    if not text:
        msg = f"{name} must be non-empty."
        raise ValueError(msg)
    return text


def _as_bool(value: Any, name: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "f", "no", "n", "off"}:
            return False
    msg = f"{name} must be a boolean."
    raise ValueError(msg)


def _as_int(value: Any, name: str, *, min_value: int | None = None) -> int:
    if isinstance(value, bool):
        msg = f"{name} must be an integer."
        raise ValueError(msg)
    try:
        coerced = int(value)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be an integer."
        raise ValueError(msg) from exc
    if min_value is not None and coerced < min_value:
        msg = f"{name} must be at least {min_value}."
        raise ValueError(msg)
    return coerced


def _as_optional_int(value: Any, name: str, *, min_value: int | None = None) -> int | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
        return None
    return _as_int(value, name, min_value=min_value)


def _as_float(value: Any, name: str, *, min_value: float | None = None) -> float:
    if isinstance(value, bool):
        msg = f"{name} must be a finite number."
        raise ValueError(msg)
    try:
        coerced = float(value)
    except (TypeError, ValueError) as exc:
        msg = f"{name} must be a finite number."
        raise ValueError(msg) from exc
    if not math.isfinite(coerced):
        msg = f"{name} must be finite."
        raise ValueError(msg)
    if min_value is not None and coerced < min_value:
        msg = f"{name} must be at least {min_value}."
        raise ValueError(msg)
    return coerced


def _as_section(value: Any, cls: type[T], name: str) -> T:
    if isinstance(value, cls):
        return value
    if isinstance(value, DictConfig):
        raw_data = OmegaConf.to_container(value, resolve=True)
    elif isinstance(value, Mapping):
        raw_data = dict(value)
    else:
        msg = f"{name} must be a mapping or {cls.__name__}."
        raise ValueError(msg)
    if not isinstance(raw_data, Mapping):
        msg = f"{name} must be a mapping."
        raise ValueError(msg)
    data = cast(Mapping[str, Any], raw_data)
    return cls(**data)


@dataclass
class EnvConfig:
    id: str = "chess"

    def __post_init__(self) -> None:
        self.id = _as_str(self.id, "env.id")


@dataclass
class ModelConfig:
    name: str = "resnet"
    channels: int = 64
    blocks: int = 4
    policy_channels: int = 2
    value_channels: int = 1
    value_hidden: int = 64

    def __post_init__(self) -> None:
        self.name = _as_str(self.name, "model.name").lower()
        self.channels = _as_int(self.channels, "model.channels", min_value=1)
        self.blocks = _as_int(self.blocks, "model.blocks", min_value=1)
        self.policy_channels = _as_int(
            self.policy_channels, "model.policy_channels", min_value=1
        )
        self.value_channels = _as_int(self.value_channels, "model.value_channels", min_value=1)
        self.value_hidden = _as_int(self.value_hidden, "model.value_hidden", min_value=1)


@dataclass
class OptimizerConfig:
    name: str = "adamw"
    learning_rate: float = 1.0e-3
    weight_decay: float = 1.0e-4
    adam_weight_decay: float = 0.0

    def __post_init__(self) -> None:
        self.name = _as_str(self.name, "optimizer.name").lower()
        self.learning_rate = _as_float(
            self.learning_rate, "optimizer.learning_rate", min_value=0.0
        )
        self.weight_decay = _as_float(self.weight_decay, "optimizer.weight_decay", min_value=0.0)
        self.adam_weight_decay = _as_float(
            self.adam_weight_decay, "optimizer.adam_weight_decay", min_value=0.0
        )


def _validate_search_config(config: Any, prefix: str) -> None:
    config.name = _as_str(config.name, f"{prefix}.name").lower()
    config.num_simulations = _as_int(
        config.num_simulations, f"{prefix}.num_simulations", min_value=1
    )
    config.max_num_considered_actions = _as_int(
        config.max_num_considered_actions,
        f"{prefix}.max_num_considered_actions",
        min_value=1,
    )
    config.max_depth = _as_optional_int(config.max_depth, f"{prefix}.max_depth", min_value=1)
    config.gumbel_scale = _as_float(config.gumbel_scale, f"{prefix}.gumbel_scale", min_value=0.0)


@dataclass
class TrainSearchConfig:
    name: str = "gumbel"
    num_simulations: int = 16
    max_num_considered_actions: int = 16
    max_depth: int | None = None
    gumbel_scale: float = 1.0

    def __post_init__(self) -> None:
        _validate_search_config(self, "train.search")


@dataclass
class EvalSearchConfig:
    name: str = "gumbel"
    num_simulations: int = 4
    max_num_considered_actions: int = 8
    max_depth: int | None = 16
    gumbel_scale: float = 0.0

    def __post_init__(self) -> None:
        _validate_search_config(self, "eval.search")


@dataclass
class EvalConfig:
    enabled: bool = True
    interval: int = 10
    batch_size: int = 4
    max_num_steps: int = 64
    search: EvalSearchConfig = field(default_factory=EvalSearchConfig)
    anchor_interval: int = 50
    max_anchors: int = 4
    initial_anchor_elo: float = 0.0

    def __post_init__(self) -> None:
        self.enabled = _as_bool(self.enabled, "eval.enabled")
        self.interval = _as_int(self.interval, "eval.interval", min_value=1)
        self.batch_size = _as_int(self.batch_size, "eval.batch_size", min_value=1)
        if self.enabled and self.batch_size % 2 != 0:
            msg = "eval.batch_size must be even for color-balanced matches."
            raise ValueError(msg)
        self.max_num_steps = _as_int(self.max_num_steps, "eval.max_num_steps", min_value=1)
        self.search = _as_section(self.search, EvalSearchConfig, "eval.search")
        self.anchor_interval = _as_int(
            self.anchor_interval, "eval.anchor_interval", min_value=1
        )
        self.max_anchors = _as_int(self.max_anchors, "eval.max_anchors", min_value=1)
        self.initial_anchor_elo = _as_float(self.initial_anchor_elo, "eval.initial_anchor_elo")


@dataclass
class CheckpointConfig:
    dir: str = "checkpoints"
    max_to_keep: int = 5
    save_interval_steps: int = 1000
    resume: bool = True

    def __post_init__(self) -> None:
        self.dir = _as_str(self.dir, "checkpoint.dir")
        self.max_to_keep = _as_int(self.max_to_keep, "checkpoint.max_to_keep", min_value=0)
        self.save_interval_steps = _as_int(
            self.save_interval_steps, "checkpoint.save_interval_steps", min_value=1
        )
        self.resume = _as_bool(self.resume, "checkpoint.resume")


@dataclass
class RuntimeConfig:
    name: str = "single_mesh"
    axis_name: str = "data"
    num_devices: int = 1

    def __post_init__(self) -> None:
        self.name = _as_str(self.name, "runtime.name").lower()
        self.axis_name = _as_str(self.axis_name, "runtime.axis_name")
        self.num_devices = _as_int(self.num_devices, "runtime.num_devices", min_value=1)


@dataclass
class TrainConfig:
    seed: int = 0
    num_iters: int = 10_000
    selfplay_batch_size: int = 32
    max_num_steps: int = 16
    search: TrainSearchConfig = field(default_factory=TrainSearchConfig)
    batch_size: int = 128
    log_interval: int = 10

    def __post_init__(self) -> None:
        self.seed = _as_int(self.seed, "train.seed", min_value=0)
        self.num_iters = _as_int(self.num_iters, "train.num_iters", min_value=1)
        self.selfplay_batch_size = _as_int(
            self.selfplay_batch_size, "train.selfplay_batch_size", min_value=1
        )
        self.max_num_steps = _as_int(self.max_num_steps, "train.max_num_steps", min_value=1)
        self.search = _as_section(self.search, TrainSearchConfig, "train.search")
        self.batch_size = _as_int(self.batch_size, "train.batch_size", min_value=1)
        self.log_interval = _as_int(self.log_interval, "train.log_interval", min_value=1)


@dataclass
class ScacchiConfig:
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self) -> None:
        self.env = _as_section(self.env, EnvConfig, "env")
        self.model = _as_section(self.model, ModelConfig, "model")
        self.optimizer = _as_section(self.optimizer, OptimizerConfig, "optimizer")
        self.eval = _as_section(self.eval, EvalConfig, "eval")
        self.checkpoint = _as_section(self.checkpoint, CheckpointConfig, "checkpoint")
        self.runtime = _as_section(self.runtime, RuntimeConfig, "runtime")
        self.train = _as_section(self.train, TrainConfig, "train")


def config_from_dict_config(cfg: DictConfig | Mapping[str, Any] | ScacchiConfig) -> ScacchiConfig:
    """Convert a Hydra/OmegaConf config into validated dataclass config."""

    if isinstance(cfg, ScacchiConfig):
        return cfg

    if isinstance(cfg, DictConfig):
        data = OmegaConf.to_container(cfg, resolve=True)
    else:
        data = dict(cfg)

    if not isinstance(data, Mapping):
        msg = "Config must be a mapping."
        raise ValueError(msg)

    clean_data = dict(data)
    clean_data.pop("hydra", None)
    allowed_sections = {section.name for section in fields(ScacchiConfig)}
    unknown_sections = sorted(set(clean_data) - allowed_sections)
    if unknown_sections:
        msg = f"Unknown config section(s): {', '.join(unknown_sections)}."
        raise ValueError(msg)

    try:
        schema = OmegaConf.structured(ScacchiConfig)
        merged = OmegaConf.merge(schema, clean_data)
        return cast(ScacchiConfig, OmegaConf.to_object(merged))
    except OmegaConfBaseException as exc:
        msg = f"Invalid training config: {exc}"
        raise ValueError(msg) from exc
