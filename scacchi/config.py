# Copyright 2023 The Pgx Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException


class NetworkName(StrEnum):
    boardlaw_dirichlet = "boardlaw_dirichlet"


class SelfplayActionSource(StrEnum):
    search_action = "search_action"


class SearchPolicy(StrEnum):
    posterior_tree_wavefront = "posterior_tree_wavefront"


class LeafValueMode(StrEnum):
    alpha = "alpha"
    mean = "mean"


class FinalActionMode(StrEnum):
    posterior_argmax = "posterior_argmax"
    posterior_sample = "posterior_sample"


class CategoricalDrawRule(StrEnum):
    policy_prior = "policy_prior"
    fastest_draw = "fastest_draw"
    slowest_draw = "slowest_draw"
    fixed_order = "fixed_order"


class ConfigError(ValueError):
    """Raised when runtime configuration values are invalid."""


@dataclass(slots=True)
class RunConfig:
    seed: int = 0
    max_num_iters: int = 400


@dataclass(slots=True)
class EnvConfig:
    id: str = "go_9x9"
    board_size: int | None = None
    num_outcomes: int | None = None


@dataclass(slots=True)
class ModelConfig:
    network: NetworkName = NetworkName.boardlaw_dirichlet
    num_channels: int = 128
    num_layers: int = 6


@dataclass(slots=True)
class SelfplayConfig:
    batch_size: int = 1024
    max_num_steps: int = 256
    action_source: SelfplayActionSource = SelfplayActionSource.search_action


@dataclass(slots=True)
class SearchMonteCarloConfig:
    policy_samples: int = 32


@dataclass(slots=True)
class SearchConstantsConfig:
    kappa_leaf: float = 1.0
    state_posterior_kappa_n: float = 9.0


@dataclass(slots=True)
class SearchCategoricalConfig:
    epsilon: float = 1e-4
    draw_rule: CategoricalDrawRule = CategoricalDrawRule.policy_prior

    def __post_init__(self) -> None:
        if self.epsilon <= 0.0:
            raise ConfigError("search.categorical.epsilon must be positive.")
        if self.epsilon >= 0.5:
            raise ConfigError("search.categorical.epsilon must be less than 0.5.")


@dataclass(slots=True)
class SearchConfig:
    policy: SearchPolicy = SearchPolicy.posterior_tree_wavefront
    num_simulations: int = 32
    eval_batch_size: int | None = None
    inflight_limit: int = 1
    max_depth: int = 128
    final_action_mode: FinalActionMode = FinalActionMode.posterior_argmax
    leaf_value_mode: LeafValueMode = LeafValueMode.alpha
    monte_carlo: SearchMonteCarloConfig = field(default_factory=SearchMonteCarloConfig)
    constants: SearchConstantsConfig = field(default_factory=SearchConstantsConfig)
    categorical: SearchCategoricalConfig = field(default_factory=SearchCategoricalConfig)


@dataclass(slots=True)
class LossConfig:
    policy_weight: float = 1.0
    value_dir_kl_weight: float = 1.0
    q_dir_kl_weight: float = 1.0


@dataclass(slots=True)
class RegularizationConfig:
    dirichlet_concentration_clip: float | None = 8.0


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 4096
    replay_buffer_size: int = 1
    learning_rate: float = 0.001
    grad_clip_norm: float | None = None
    losses: LossConfig = field(default_factory=LossConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)


@dataclass(slots=True)
class EvalConfig:
    interval: int = 5
    batch_size: int = 16


@dataclass(slots=True)
class WandbConfig:
    enabled: bool = True
    project: str = "scacchi-az"


@dataclass(slots=True)
class LoggingConfig:
    interval: int = 1
    wandb: WandbConfig = field(default_factory=WandbConfig)


@dataclass(slots=True)
class CheckpointingConfig:
    max_to_keep: int = 3
    save_interval_steps: int = 50


@dataclass(slots=True)
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    selfplay: SelfplayConfig = field(default_factory=SelfplayConfig)
    search: SearchConfig = field(default_factory=SearchConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpointing: CheckpointingConfig = field(default_factory=CheckpointingConfig)

    def __post_init__(self) -> None:
        if self.env.num_outcomes not in (None, 3):
            raise ConfigError("native posterior-tree search uses WDL3 targets.")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_config(values: DictConfig | Mapping[str, Any]) -> Config:
    try:
        return cast(
            Config,
            OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Config), values)),
        )
    except ConfigError:
        raise
    except OmegaConfBaseException as exc:
        raise ConfigError(str(exc)) from exc
