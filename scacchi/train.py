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

import os
import time
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

# Training expects a GPU by default, but posterior_tree search keeps PGX env
# stepping on CPU, so both platforms need to be visible.
os.environ.setdefault("JAX_PLATFORMS", "cuda,cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from flax import nnx
import hydra
from hydra.utils import get_original_cwd
import jax
import numpy as np
import optax
from omegaconf import DictConfig, OmegaConf
from omegaconf.errors import OmegaConfBaseException
from tqdm import tqdm

from .checkpoint import build_checkpoint_manager, from_pretrained, maybe_save, restore
from .envs import make_env
from .evaluations import make_mcts_evaluate
from .logger import build_logger, returns_metrics
from .network import build_model
from .pipeline import make_training_iteration


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


def report_jax_backend() -> None:
    backend = jax.default_backend()
    devices = jax.devices()
    print(f"JAX backend: {backend}")
    print(f"JAX devices: {devices}")
    if os.environ.get("SCACCHI_ALLOW_CPU") != "1" and backend != "gpu":
        raise RuntimeError(
            "JAX is not using a GPU backend. Set SCACCHI_ALLOW_CPU=1 only for "
            "intentional CPU runs."
        )


@dataclass(slots=True)
class RunConfig:
    seed: int = 0
    max_num_iters: int = 400

    def __post_init__(self) -> None:
        _at_least(self.max_num_iters, "run.max_num_iters", 1)


@dataclass(slots=True)
class EnvConfig:
    id: str = "go_9x9"
    board_size: int | None = None
    num_outcomes: int | None = None

    def __post_init__(self) -> None:
        if self.board_size is not None:
            _at_least(self.board_size, "env.board_size", 1)


@dataclass(slots=True)
class ModelConfig:
    network: NetworkName = NetworkName.boardlaw_dirichlet
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True

    def __post_init__(self) -> None:
        self.network = _choice(NetworkName, self.network, "model.network")
        _at_least(self.num_channels, "model.num_channels", 1)
        _at_least(self.num_layers, "model.num_layers", 1)


@dataclass(slots=True)
class SelfplayConfig:
    batch_size: int = 1024
    max_num_steps: int = 256
    action_source: SelfplayActionSource = SelfplayActionSource.search_action

    def __post_init__(self) -> None:
        self.action_source = _choice(
            SelfplayActionSource,
            self.action_source,
            "selfplay.action_source",
        )
        _at_least(self.batch_size, "selfplay.batch_size", 1)
        _at_least(self.max_num_steps, "selfplay.max_num_steps", 1)


@dataclass(slots=True)
class SearchMonteCarloConfig:
    policy_samples: int = 32

    def __post_init__(self) -> None:
        _at_least(self.policy_samples, "search.monte_carlo.policy_samples", 1)


@dataclass(slots=True)
class SearchConstantsConfig:
    kappa_leaf: float = 1.0
    state_posterior_kappa_n: float = 9.0

    def __post_init__(self) -> None:
        _greater_than(self.kappa_leaf, "search.constants.kappa_leaf", 0.0)
        _greater_than(
            self.state_posterior_kappa_n,
            "search.constants.state_posterior_kappa_n",
            0.0,
        )


@dataclass(slots=True)
class SearchCategoricalConfig:
    epsilon: float = 1e-4
    draw_rule: CategoricalDrawRule = CategoricalDrawRule.policy_prior

    def __post_init__(self) -> None:
        self.draw_rule = _choice(
            CategoricalDrawRule,
            self.draw_rule,
            "search.categorical.draw_rule",
        )
        _greater_than(self.epsilon, "search.categorical.epsilon", 0.0)
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

    def __post_init__(self) -> None:
        self.policy = _choice(SearchPolicy, self.policy, "search.policy")
        self.final_action_mode = _choice(
            FinalActionMode,
            self.final_action_mode,
            "search.final_action_mode",
        )
        self.leaf_value_mode = _choice(
            LeafValueMode,
            self.leaf_value_mode,
            "search.leaf_value_mode",
        )
        _at_least(self.num_simulations, "search.num_simulations", 1)
        _at_least(self.inflight_limit, "search.inflight_limit", 1)
        _at_least(self.max_depth, "search.max_depth", 1)
        if self.eval_batch_size is not None:
            _at_least(self.eval_batch_size, "search.eval_batch_size", 1)


@dataclass(slots=True)
class LossConfig:
    policy_weight: float = 1.0
    value_dir_kl_weight: float = 1.0
    q_dir_kl_weight: float = 1.0

    def __post_init__(self) -> None:
        _at_least(self.policy_weight, "training.losses.policy_weight", 0.0)
        _at_least(self.value_dir_kl_weight, "training.losses.value_dir_kl_weight", 0.0)
        _at_least(self.q_dir_kl_weight, "training.losses.q_dir_kl_weight", 0.0)


@dataclass(slots=True)
class RegularizationConfig:
    dirichlet_concentration_clip: float | None = 8.0

    def __post_init__(self) -> None:
        if self.dirichlet_concentration_clip is not None:
            _greater_than(
                self.dirichlet_concentration_clip,
                "training.regularization.dirichlet_concentration_clip",
                0.0,
            )


@dataclass(slots=True)
class TrainingConfig:
    batch_size: int = 4096
    replay_buffer_size: int = 1
    learning_rate: float = 0.001
    grad_clip_norm: float | None = None
    losses: LossConfig = field(default_factory=LossConfig)
    regularization: RegularizationConfig = field(default_factory=RegularizationConfig)

    def __post_init__(self) -> None:
        _at_least(self.batch_size, "training.batch_size", 1)
        _at_least(self.replay_buffer_size, "training.replay_buffer_size", 1)
        _greater_than(self.learning_rate, "training.learning_rate", 0.0)
        if self.grad_clip_norm is not None:
            _greater_than(self.grad_clip_norm, "training.grad_clip_norm", 0.0)


@dataclass(slots=True)
class EvalConfig:
    interval: int = 5
    batch_size: int = 16

    def __post_init__(self) -> None:
        _at_least(self.interval, "eval.interval", 0)
        _at_least(self.batch_size, "eval.batch_size", 1)


@dataclass(slots=True)
class WandbConfig:
    enabled: bool = True
    project: str = "scacchi-az"


@dataclass(slots=True)
class LoggingConfig:
    interval: int = 1
    wandb: WandbConfig = field(default_factory=WandbConfig)

    def __post_init__(self) -> None:
        _at_least(self.interval, "logging.interval", 1)


@dataclass(slots=True)
class CheckpointingConfig:
    max_to_keep: int = 3
    save_interval_steps: int = 50

    def __post_init__(self) -> None:
        _at_least(self.max_to_keep, "checkpointing.max_to_keep", 0)
        _at_least(self.save_interval_steps, "checkpointing.save_interval_steps", 1)


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
        return cast(Config, OmegaConf.to_object(OmegaConf.merge(OmegaConf.structured(Config), values)))
    except ConfigError:
        raise
    except OmegaConfBaseException as exc:
        raise ConfigError(str(exc)) from exc


def _choice(enum: Any, value: Any, key: str) -> Any:
    try:
        return value if isinstance(value, enum) else enum(value)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum)
        raise ConfigError(f"{key} must be one of {allowed}; got {value!r}.") from exc


def _at_least(value: float | int, key: str, minimum: float | int) -> None:
    if value < minimum:
        raise ConfigError(f"{key} must be >= {minimum}; got {value!r}.")


def _greater_than(value: float | int, key: str, minimum: float | int) -> None:
    if value <= minimum:
        raise ConfigError(f"{key} must be > {minimum}; got {value!r}.")


@hydra.main(version_base=None, config_path="configs", config_name="hex")
def main(cfg: DictConfig) -> None:
    config = load_config(cfg)
    report_jax_backend()

    env = make_env(config.env.id, config.env.board_size)
    checkpoint_path = f"checkpoints/{config.env.board_size}_solved"
    baseline_model = from_pretrained(checkpoint_path, env, rngs=nnx.Rngs(0))

    model = build_model(
        config.model,
        num_outcomes=config.env.num_outcomes,
        dirichlet_concentration_clip=(
            config.training.regularization.dirichlet_concentration_clip
        ),
        num_actions=env.num_actions,
        observation_shape=env.observation_shape,
        rngs=nnx.Rngs(config.run.seed),
    )
    optimizer_transforms: list[optax.GradientTransformation] = []
    if config.training.grad_clip_norm is not None:
        optimizer_transforms.append(
            optax.clip_by_global_norm(config.training.grad_clip_norm)
        )
    optimizer_transforms.append(optax.adam(learning_rate=config.training.learning_rate))
    optimizer = nnx.Optimizer(
        model,
        optax.chain(*optimizer_transforms),
        wrt=nnx.Param,
    )

    training_iteration = make_training_iteration(
        env,
        config.selfplay,
        config.search,
        config.training,
    )
    
    
    evaluate = make_mcts_evaluate(env, config.eval, config.search, baseline_model)

    hours: float = 0.0
    frames: int = 0

    rng_key = jax.random.PRNGKey(config.run.seed)
    with build_logger(config.logging, config.run, config) as logger:
        eval_avg_return_history: list[float] = []
        previous_eval_avg_return: float | None = None
        board_size = (
            "none" if config.env.board_size is None else str(config.env.board_size)
        )
        ckpt_dir = (
            Path(get_original_cwd())
            / "checkpoints"
            / (
                f"{config.env.id}_bs{board_size}_{config.model.network}"
                f"_c{config.model.num_channels}_l{config.model.num_layers}"
                f"_seed{config.run.seed}"
            )
        ).resolve()
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        with build_checkpoint_manager(config.checkpointing, config.run, ckpt_dir) as ckpt_mgr:
            start_iter, rng_key, hours, frames = restore(ckpt_mgr, model, optimizer, rng_key)
            pbar = tqdm(range(start_iter, config.run.max_num_iters), desc="training", dynamic_ncols=True, total=config.run.max_num_iters, initial=start_iter)
            pbar.refresh()
            for iteration in pbar:
                dict_to_log = {}
                rng_key, eval_key, train_key = jax.random.split(rng_key, 3)
                if config.eval.interval > 0 and (
                    iteration == config.run.max_num_iters - 1
                    or iteration % config.eval.interval == 0
                ):
                    returns = evaluate(eval_key, model)
                    dict_to_log.update(returns_metrics("eval/vs_baseline", returns))
                    eval_avg_return = float(jax.device_get(returns.mean()))
                    eval_avg_return_history.append(eval_avg_return)
                    eval_window = eval_avg_return_history[-10:]
                    eval_mean_10 = float(np.mean(eval_window))
                    eval_std_10 = float(np.std(eval_window))
                    eval_delta = (
                        0.0
                        if previous_eval_avg_return is None
                        else abs(eval_avg_return - previous_eval_avg_return)
                    )
                    previous_eval_avg_return = eval_avg_return
                    dict_to_log.update(
                        {
                            "eval/vs_baseline/avg_R_rolling_mean_10": eval_mean_10,
                            "eval/vs_baseline/avg_R_rolling_std_10": eval_std_10,
                            "eval/vs_baseline/avg_R_step_delta_abs": eval_delta,
                        }
                    )

                st = time.time()
                train_metrics = training_iteration(model, optimizer, train_key)
                frames += config.selfplay.batch_size * config.selfplay.max_num_steps

                et = time.time()
                hours += (et - st) / 3600
                dict_to_log.update(
                    {
                        "train/policy_loss": train_metrics.policy_loss.mean().item(),
                        "train/value_loss": train_metrics.value_loss.mean().item(),
                        "train/policy_nll_loss": train_metrics.policy_nll_loss.mean().item(),
                        "train/policy_kl_hat": train_metrics.policy_kl_hat.mean().item(),
                        "train/policy_target_entropy": train_metrics.policy_target_entropy.mean().item(),
                        "train/value_dir_kl_loss": train_metrics.value_dir_kl_loss.mean().item(),
                        "train/q_dir_kl_loss": train_metrics.q_dir_kl_loss.mean().item(),
                        "train/alpha_V_concentration": train_metrics.alpha_V_concentration.mean().item(),
                        "train/alpha_Q_concentration": train_metrics.alpha_Q_concentration.mean().item(),
                        "train/q_loss_weight_mean": train_metrics.q_loss_weight_mean.mean().item(),
                        "search/path_depth_mean": train_metrics.search_path_depth_mean.mean().item(),
                        "search/path_depth_p50": train_metrics.search_path_depth_p50.mean().item(),
                        "search/path_depth_p90": train_metrics.search_path_depth_p90.mean().item(),
                        "search/path_depth_max": train_metrics.search_path_depth_max.mean().item(),
                        "search/expanded_nodes": train_metrics.search_expanded_nodes.mean().item(),
                        "search/terminal_fraction": train_metrics.search_terminal_fraction.mean().item(),
                        "search/root_policy_entropy": train_metrics.search_root_policy_entropy.mean().item(),
                        "search/root_gamma": train_metrics.search_root_gamma.mean().item(),
                        "search/root_downstream_eval_count": train_metrics.search_root_downstream_eval_count.mean().item(),
                        "search/root_q_concentration": train_metrics.search_root_q_concentration.mean().item(),
                        "train/hours": hours,
                        "train/frames": frames,
                    }
                )
                logger.log(
                    iteration,
                    dict_to_log,
                    pbar=pbar,
                    prefix="",
                    pbar_filter=r"loss|avg_R",
                )
                maybe_save(
                    ckpt_mgr,
                    iteration,
                    model,
                    optimizer,
                    rng_key,
                    config,
                    hours,
                    frames,
                )


if __name__ == "__main__":
    main()
