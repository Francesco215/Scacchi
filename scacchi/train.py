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
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    __package__ = "scacchi"

if any(Path("/dev").glob("accel*")):
    os.environ.setdefault("JAX_PLATFORMS", "tpu,cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import hydra
from hydra.utils import get_original_cwd
import jax

from .distributed import initialize_distributed, make_batch_parallel

initialize_distributed()

import jax.numpy as jnp
import numpy as np
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator
from tqdm import tqdm

from flax import nnx
import optax
import pgx

from .checkpoint import build_checkpoint_manager, from_pretrained, maybe_save, restore
from .envs import make_env
from .evaluations import make_mcts_evaluate
from .logger import build_logger, returns_metrics
from .network import build_model
from .pipeline import make_training_iteration


def report_jax_backend() -> None:
    backend = jax.default_backend()
    devices = jax.devices()
    process_index = jax.process_index()
    process_count = jax.process_count()
    if process_index == 0:
        print(f"JAX backend: {backend}")
        print(f"JAX process count: {process_count}")
        print(f"JAX global devices: {devices}")
    print(f"JAX process {process_index} local devices: {jax.local_devices()}")
    if os.environ.get("SCACCHI_ALLOW_CPU") != "1" and backend not in {"gpu", "tpu"}:
        raise RuntimeError(
            "JAX is not using a GPU or TPU backend. Set SCACCHI_ALLOW_CPU=1 "
            "only for intentional CPU runs."
        )

_NESTED_CONFIG_FIELDS: tuple[tuple[tuple[str, ...], str], ...] = (
    (("run", "seed"), "seed"),
    (("run", "max_num_iters"), "max_num_iters"),
    (("env", "id"), "env_id"),
    (("env", "board_size"), "board_size"),
    (("env", "num_outcomes"), "num_outcomes"),
    (("model", "network"), "network"),
    (("model", "num_channels"), "num_channels"),
    (("model", "num_layers"), "num_layers"),
    (("model", "resnet_v2"), "resnet_v2"),
    (("model", "legacy_dirichlet_head_init"), "legacy_dirichlet_head_init"),
    (("model", "rezero_kernel_init"), "rezero_kernel_init"),
    (("selfplay", "batch_size"), "selfplay_batch_size"),
    (("selfplay", "max_num_steps"), "max_num_steps"),
    (("selfplay", "chunk_size"), "selfplay_chunk_size"),
    (("selfplay", "action_source"), "selfplay_action_source"),
    (("search", "policy"), "search_policy"),
    (("search", "num_simulations"), "num_simulations"),
    (("search", "num_blocks"), "num_search_blocks"),
    (("search", "eval_batch_size"), "search_eval_batch_size"),
    (("search", "inflight_limit"), "inflight_limit"),
    (("search", "monte_carlo", "policy_samples"), "policy_mc_samples"),
    (("search", "monte_carlo", "backup_samples"), "backup_mc_samples"),
    (("search", "constants", "kappa_leaf"), "kappa_leaf"),
    (("search", "constants", "kappa_terminal"), "kappa_terminal"),
    (("search", "constants", "epsilon_terminal"), "epsilon_terminal"),
    (("search", "constants", "state_posterior_kappa_n"), "state_posterior_kappa_n"),
    (("search", "leaf_value_mode"), "leaf_value_mode"),
    (("training", "batch_size"), "training_batch_size"),
    (("training", "max_updates_per_iter"), "max_updates_per_iter"),
    (("training", "replay_buffer_size"), "replay_buffer_size"),
    (("training", "minibatch_sampling"), "minibatch_sampling"),
    (("training", "learning_rate"), "learning_rate"),
    (("training", "lr_decay_after_iters"), "lr_decay_after_iters"),
    (("training", "lr_decay_factor"), "lr_decay_factor"),
    (("training", "grad_clip_norm"), "grad_clip_norm"),
    (("training", "losses", "policy_weight"), "policy_loss_weight"),
    (("training", "losses", "value_dir_kl_weight"), "value_dir_kl_weight"),
    (("training", "losses", "q_dir_kl_weight"), "q_dir_kl_weight"),
    (("training", "losses", "value_outcome_weight"), "value_outcome_weight"),
    (("training", "losses", "q_outcome_weight"), "q_outcome_weight"),
    (("training", "losses", "q_loss_weight_mode"), "q_loss_weight_mode"),
    (("training", "losses", "q_dir_kl_reduction"), "q_dir_kl_reduction"),
    (("training", "losses", "loss_mask_mode"), "loss_mask_mode"),
    (("training", "losses", "terminal_edge_targets"), "terminal_edge_targets"),
    (("training", "losses", "terminal_parent_targets"), "terminal_parent_targets"),
    (("training", "losses", "policy_target_mode"), "policy_target_mode"),
    (
        ("training", "regularization", "dirichlet_concentration_clip"),
        "dirichlet_concentration_clip",
    ),
    (("eval", "interval"), "eval_interval"),
    (("eval", "batch_size"), "eval_batch_size"),
    (("eval", "baseline"), "eval_baseline"),
    (("eval", "baseline_id"), "eval_baseline_id"),
    (("eval", "checkpoint_path"), "eval_checkpoint_path"),
    (("logging", "interval"), "log_interval"),
    (("logging", "wandb", "enabled"), "wandb_enabled"),
    (("logging", "wandb", "project"), "wandb_project"),
    (("checkpointing", "max_to_keep"), "ckpt_max_to_keep"),
    (("checkpointing", "save_interval_steps"), "ckpt_save_interval_steps"),
    (("compatibility", "rng_split_mode"), "rng_split_mode"),
)
_NESTED_CONFIG_GROUPS = frozenset(path[0] for path, _ in _NESTED_CONFIG_FIELDS)
_NESTED_CONFIG_PATHS = frozenset(path for path, _ in _NESTED_CONFIG_FIELDS)
_DEPRECATED_CONFIG_KEYS = frozenset(
    {
        "c_leaf",
        "c_terminal",
        "c_state",
        "c_value_search",
        "dirichlet_concentration_clip_mode",
    }
)
_DEPRECATED_NESTED_CONFIG_PATHS = frozenset(
    [("search", "constants", key) for key in _DEPRECATED_CONFIG_KEYS]
    + [("model", "dirichlet_concentration_clip_mode")]
)


def _raise_deprecated_config(key: str) -> None:
    replacements = {
        "c_leaf": "kappa_leaf",
        "c_terminal": "kappa_terminal",
        "c_state": "state_posterior_kappa_n",
        "c_value_search": None,
        "dirichlet_concentration_clip_mode": "dirichlet_concentration_clip",
    }
    leaf = key.rsplit(".", 1)[-1]
    replacement = replacements.get(leaf)
    if leaf == "dirichlet_concentration_clip_mode":
        message = (
            f"{key!r} is deprecated; concentration logits are no longer clipped. "
            "Use 'dirichlet_concentration_clip' to cap total concentration."
        )
    elif replacement is None:
        message = f"{key!r} is deprecated and no longer used by posterior-tree search."
    else:
        message = f"{key!r} is deprecated; use {replacement!r} instead."
    raise ValueError(message)


def _nested_value(
    config: Mapping[str, Any],
    path: tuple[str, ...],
) -> tuple[bool, Any]:
    value: Any = config
    for part in path:
        if not isinstance(value, Mapping) or part not in value:
            return False, None
        value = value[part]
    return True, value


def _iter_nested_leaves(
    value: Any,
    prefix: tuple[str, ...],
) -> tuple[tuple[tuple[str, ...], Any], ...]:
    if not isinstance(value, Mapping):
        return ((prefix, value),)
    if not value:
        return ((prefix, value),)
    leaves: list[tuple[tuple[str, ...], Any]] = []
    for key, child in value.items():
        leaves.extend(_iter_nested_leaves(child, (*prefix, str(key))))
    return tuple(leaves)


def normalize_config_dict(config: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten the readable nested YAML shape into the runtime Config fields."""

    normalized: dict[str, Any] = {}
    for key, value in config.items():
        key = str(key)
        if key in _DEPRECATED_CONFIG_KEYS:
            _raise_deprecated_config(key)
        if key not in _NESTED_CONFIG_GROUPS:
            normalized[key] = value
            continue
        if not isinstance(value, Mapping):
            normalized[key] = value
            continue
        for path, leaf_value in _iter_nested_leaves(value, (key,)):
            if path in _DEPRECATED_NESTED_CONFIG_PATHS:
                _raise_deprecated_config(".".join(path))
            if path not in _NESTED_CONFIG_PATHS:
                normalized[".".join(path)] = leaf_value

    for path, target in _NESTED_CONFIG_FIELDS:
        found, value = _nested_value(config, path)
        if found and target not in normalized:
            normalized[target] = value

    return normalized


class Config(BaseModel):
    env_id: str = "go_9x9"
    board_size: int | None = None
    seed: int = 0
    max_num_iters: int = 400
    # network params
    network: str = "aznet"  # "aznet" | "aznet_dirichlet" | "boardlaw" | "boardlaw_dirichlet"
    num_outcomes: int | None = None
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    legacy_dirichlet_head_init: bool = False
    rezero_kernel_init: str = "variance_scaling"
    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    num_search_blocks: int = Field(default=1, ge=1)
    max_num_steps: int = 256
    selfplay_chunk_size: int | None = Field(default=None, ge=1)
    policy_mc_samples: int = 32
    backup_mc_samples: int = Field(default=16, ge=1)
    leaf_value_mode: str = "alpha"
    kappa_leaf: float = Field(default=1.0, gt=0.0)
    kappa_terminal: float = Field(default=8.0, gt=0.0)
    epsilon_terminal: float = Field(default=1e-6, gt=0.0)
    categorical_epsilon: float = Field(default=1e-4, gt=0.0, lt=0.5)
    categorical_draw_rule: str = "policy_prior"
    state_posterior_kappa_n: float = Field(default=9.0, gt=0.0)
    inflight_limit: int = Field(default=1, ge=1)
    search_eval_batch_size: int | None = Field(default=None, ge=1)
    selfplay_action_source: str = "posterior_best"
    search_policy: str = "gumbel"
    # training params
    training_batch_size: int = 4096
    max_updates_per_iter: int | None = Field(default=None, ge=1)
    replay_buffer_size: int = Field(default=1, ge=1)
    minibatch_sampling: str = "active_with_replacement"
    learning_rate: float = 0.001
    lr_decay_after_iters: int | None = Field(default=None, ge=1)
    lr_decay_factor: float = Field(default=1.0, gt=0.0)
    grad_clip_norm: float | None = Field(default=None, gt=0)
    policy_loss_weight: float = 1.0
    value_dir_kl_weight: float = 1.0
    q_dir_kl_weight: float = 1.0
    value_outcome_weight: float = 0.0
    q_outcome_weight: float = 0.0
    q_loss_weight_mode: str = "policy"
    q_dir_kl_reduction: str = "weighted"
    loss_mask_mode: str = "search"
    terminal_edge_targets: bool = False
    terminal_parent_targets: bool = False
    policy_target_mode: str = "search"
    dirichlet_concentration_clip: float | None = 8.0
    log_interval: int = 1
    # eval params
    eval_interval: int = 5
    eval_batch_size: int = 16
    eval_baseline: str = "checkpoint"
    eval_baseline_id: str | None = None
    eval_checkpoint_path: str | None = None
    # logging params
    wandb_enabled: bool = True
    wandb_project: str = "scacchi-az"
    # checkpoint params
    ckpt_max_to_keep: int = 3
    ckpt_save_interval_steps: int = 50
    rng_split_mode: str = "three_way"

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def normalize_search_config(cls, values: Any):
        if not isinstance(values, Mapping):
            return values
        values = dict(values)
        for key in sorted(_DEPRECATED_CONFIG_KEYS):
            if key in values:
                _raise_deprecated_config(key)
        return values

    @model_validator(mode="after")
    def require_dirichlet_network_for_dirichlet_losses(self, info: ValidationInfo):
        if isinstance(info.context, dict) and info.context.get("model_construction_only"):
            return self
        dirichlet_loss_weights = (
            "value_dir_kl_weight",
            "q_dir_kl_weight",
            "value_outcome_weight",
            "q_outcome_weight",
        )
        active_weights = [
            f"{name}={getattr(self, name)}"
            for name in dirichlet_loss_weights
            if getattr(self, name) != 0.0
        ]
        dirichlet_networks = {"aznet_dirichlet", "boardlaw_dirichlet"}
        if self.network not in dirichlet_networks and active_weights:
            weights = ", ".join(active_weights)
            raise ValueError(
                "Dirichlet loss weights require a Dirichlet network; "
                f"got network={self.network!r} with {weights}. Set these "
                "weights to 0.0 or use network='aznet_dirichlet' or "
                "network='boardlaw_dirichlet'."
            )
        valid_action_sources = {
            "posterior_best",
            "posterior_argmax",
            "posterior_sample",
            "search_action",
        }
        if self.selfplay_action_source not in valid_action_sources:
            allowed = ", ".join(sorted(valid_action_sources))
            raise ValueError(
                "selfplay_action_source must be one of "
                f"{allowed}; got {self.selfplay_action_source!r}."
            )
        valid_search_policies = {
            "gumbel",
            "dirichlet_thompson",
            "posterior_tree",
        }
        if self.search_policy not in valid_search_policies:
            allowed = ", ".join(sorted(valid_search_policies))
            raise ValueError(
                f"search_policy must be one of {allowed}; got {self.search_policy!r}."
            )
        if self.leaf_value_mode not in {"alpha", "mean"}:
            raise ValueError("leaf_value_mode must be 'alpha' or 'mean'.")
        if self.rezero_kernel_init not in {"variance_scaling", "orthogonal"}:
            raise ValueError(
                "rezero_kernel_init must be 'variance_scaling' or 'orthogonal'."
            )
        if self.minibatch_sampling not in {
            "as_is",
            "active_with_replacement",
            "permutation",
        }:
            raise ValueError(
                "minibatch_sampling must be 'active_with_replacement', 'as_is', "
                "or 'permutation'."
            )
        if self.q_loss_weight_mode not in {"policy", "evidence_mass"}:
            raise ValueError("q_loss_weight_mode must be 'policy' or 'evidence_mass'.")
        if self.q_dir_kl_reduction not in {"weighted", "masked_mean"}:
            raise ValueError("q_dir_kl_reduction must be 'weighted' or 'masked_mean'.")
        if self.loss_mask_mode not in {"search", "value"}:
            raise ValueError("loss_mask_mode must be 'search' or 'value'.")
        if self.rng_split_mode not in {"three_way", "legacy_eval_train"}:
            raise ValueError(
                "rng_split_mode must be 'three_way' or 'legacy_eval_train'."
            )
        if self.categorical_draw_rule not in {
            "policy_prior",
            "fastest_draw",
            "slowest_draw",
            "fixed_order",
        }:
            raise ValueError(
                "categorical_draw_rule must be one of "
                "'policy_prior', 'fastest_draw', 'slowest_draw', or 'fixed_order'."
            )
        if self.policy_target_mode not in {"search", "winner_action"}:
            raise ValueError("policy_target_mode must be 'search' or 'winner_action'.")
        if self.eval_baseline not in {"checkpoint", "pgx", "none"}:
            raise ValueError("eval_baseline must be 'checkpoint', 'pgx', or 'none'.")
        if self.search_policy in {
            "dirichlet_thompson",
            "posterior_tree",
        }:
            if self.network not in dirichlet_networks:
                raise ValueError(
                    "posterior-tree Dirichlet search requires "
                    "network='aznet_dirichlet' or network='boardlaw_dirichlet'."
                )
        if self.search_policy == "posterior_tree":
            if self.num_outcomes not in (None, 3):
                raise ValueError(
                    "posterior-tree Dirichlet search uses WDL3 targets; set "
                    "num_outcomes to null or 3."
                )
        return self


def _load_eval_baseline(config: Config, env: pgx.Env):
    if config.eval_interval <= 0:
        return None
    if config.eval_baseline == "none":
        raise ValueError("eval.baseline=none requires eval.interval=0.")
    if config.eval_baseline == "checkpoint":
        checkpoint_path = (
            config.eval_checkpoint_path
            if config.eval_checkpoint_path is not None
            else f"checkpoints/{config.board_size}_solved"
        )
        return from_pretrained(checkpoint_path, env, rngs=nnx.Rngs(0))
    if config.eval_baseline == "pgx":
        baseline_id = config.eval_baseline_id or f"{env.id}_v0"
        try:
            baseline_model = pgx.make_baseline_model(cast(Any, baseline_id))
        except AssertionError as exc:
            raise ValueError(
                f"PGX does not provide baseline model {baseline_id!r}. "
                "Use eval.baseline=none with eval.interval=0, or provide a "
                "checkpoint baseline."
            ) from exc
        _validate_pgx_eval_baseline(baseline_model, baseline_id, env)
        return baseline_model
    raise ValueError(f"unknown eval_baseline: {config.eval_baseline!r}")


def _validate_pgx_eval_baseline(
    baseline_model: Any,
    baseline_id: str,
    env: pgx.Env,
) -> None:
    observation = jnp.zeros((1, *env.observation_shape), dtype=jnp.float32)
    try:
        output = baseline_model(observation)
    except Exception as exc:
        raise ValueError(
            f"PGX baseline model {baseline_id!r} is incompatible with "
            f"env {env.id!r} observation_shape={env.observation_shape}."
        ) from exc

    logits = output[0] if isinstance(output, tuple) else output
    if tuple(logits.shape) != (1, env.num_actions):
        raise ValueError(
            f"PGX baseline model {baseline_id!r} returned logits shape "
            f"{tuple(logits.shape)} for env {env.id!r}; expected "
            f"{(1, env.num_actions)}."
        )


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    container = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    config = Config(**normalize_config_dict(container))
    initialize_distributed()
    report_jax_backend()
    parallel = make_batch_parallel(config)
    if jax.process_index() != 0:
        config.wandb_enabled = False

    with parallel.mesh_context():
        env = make_env(config.env_id, config.board_size)
        baseline_model = _load_eval_baseline(config, env)

        model = build_model(
            config,
            num_actions=env.num_actions,
            observation_shape=env.observation_shape,
            rngs=nnx.Rngs(config.seed),
        )
        optimizer_transforms: list[optax.GradientTransformation] = []
        if config.grad_clip_norm is not None:
            optimizer_transforms.append(optax.clip_by_global_norm(config.grad_clip_norm))
        learning_rate: float | optax.Schedule = config.learning_rate
        if config.lr_decay_after_iters is not None and config.lr_decay_factor != 1.0:
            rows_per_iter = max(
                1,
                config.selfplay_batch_size * config.max_num_steps,
            )
            updates_per_iter = max(1, rows_per_iter // config.training_batch_size)
            if config.max_updates_per_iter is not None:
                updates_per_iter = min(updates_per_iter, config.max_updates_per_iter)
            learning_rate = optax.piecewise_constant_schedule(
                init_value=config.learning_rate,
                boundaries_and_scales={
                    config.lr_decay_after_iters * updates_per_iter: config.lr_decay_factor,
                },
            )
        optimizer_transforms.append(optax.adam(learning_rate=learning_rate))
        optimizer = nnx.Optimizer(
            model,
            optax.chain(*optimizer_transforms),
            wrt=nnx.Param,
        )

        training_iteration = make_training_iteration(env, config, parallel=parallel)

        evaluate = (
            None
            if baseline_model is None
            else make_mcts_evaluate(env, config, baseline_model)
        )

        hours: float = 0.0
        frames: int = 0

        rng_key = jax.random.PRNGKey(config.seed)
        with build_logger(config) as logger:
            eval_avg_return_history: list[float] = []
            previous_eval_avg_return: float | None = None
            board_size = "none" if config.board_size is None else str(config.board_size)
            ckpt_dir = (
                Path(get_original_cwd())
                / "checkpoints"
                / (
                    f"{config.env_id}_bs{board_size}_{config.network}"
                    f"_c{config.num_channels}_l{config.num_layers}_seed{config.seed}"
                )
            ).resolve()
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            with build_checkpoint_manager(config, ckpt_dir) as ckpt_mgr:
                start_iter, rng_key, hours, frames = restore(ckpt_mgr, model, optimizer, rng_key)
                pbar = tqdm(
                    range(start_iter, config.max_num_iters),
                    desc="training",
                    dynamic_ncols=True,
                    total=config.max_num_iters,
                    initial=start_iter,
                    disable=jax.process_index() != 0,
                )
                pbar.refresh()
                for iteration in pbar:
                    dict_to_log = {}
                    legacy_rng_split = config.rng_split_mode == "legacy_eval_train"
                    if not legacy_rng_split:
                        rng_key, eval_key, train_key = jax.random.split(rng_key, 3)
                    if evaluate is not None and config.eval_interval > 0 and (
                        iteration == config.max_num_iters - 1
                        or iteration % config.eval_interval == 0
                    ):
                        if legacy_rng_split:
                            rng_key, eval_key = jax.random.split(rng_key)
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
                    if legacy_rng_split:
                        rng_key, train_key = jax.random.split(rng_key)
                    train_metrics = training_iteration(model, optimizer, train_key)
                    frames += config.selfplay_batch_size * config.max_num_steps

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
                            "train/value_outcome_loss": train_metrics.value_outcome_loss.mean().item(),
                            "train/q_outcome_loss": train_metrics.q_outcome_loss.mean().item(),
                            "train/alpha_V_concentration": train_metrics.alpha_V_concentration.mean().item(),
                            "train/alpha_Q_concentration": train_metrics.alpha_Q_concentration.mean().item(),
                            "train/q_evidence_mass_mean": train_metrics.q_evidence_mass_mean.mean().item(),
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
