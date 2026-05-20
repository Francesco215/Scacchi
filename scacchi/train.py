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
from pathlib import Path
from typing import Any, cast

# Training is GPU-only by default. Override JAX_PLATFORMS deliberately for
# diagnostics, and set SCACCHI_ALLOW_CPU=1 only when a CPU run is intentional.
os.environ.setdefault("JAX_PLATFORMS", "cuda")

from flax import nnx
import hydra
from hydra.utils import get_original_cwd
import jax
import optax
import pgx
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator
from tqdm import tqdm

from .checkpoint import build_checkpoint_manager, maybe_save, restore, from_pretrained
from .envs import make_env
from .evaluations import make_mcts_evaluate
from .logger import build_logger, returns_metrics
from .network import build_model
from .pipeline import make_training_iteration


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


class Config(BaseModel):
    env_id: pgx.EnvId = "go_9x9"
    board_size: int | None = None
    seed: int = 0
    max_num_iters: int = 400
    # network params
    network: str = "aznet"  # "aznet" | "boardlaw" | "boardlaw_dirichlet"
    num_outcomes: int | None = None
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    num_search_blocks: int = Field(default=1, ge=1)
    max_num_steps: int = 256
    policy_mc_samples: int = 32
    c_leaf: float = 1.0
    c_terminal: float = 8.0
    selfplay_action_source: str = "posterior_best"
    search_policy: str = "gumbel"
    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    policy_loss_weight: float = 1.0
    value_dir_kl_weight: float = 1.0
    q_dir_kl_weight: float = 1.0
    value_outcome_weight: float = 0.0
    q_outcome_weight: float = 0.0
    dirichlet_concentration_clip: float | None = 8.0
    log_interval: int = 1
    # eval params
    eval_interval: int = 5
    eval_batch_size: int = 16
    # logging params
    wandb_enabled: bool = True
    wandb_project: str = "scacchi-az"
    # checkpoint params
    ckpt_max_to_keep: int = 3
    ckpt_save_interval_steps: int = 50

    model_config = ConfigDict(extra="forbid")

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
        if self.network != "boardlaw_dirichlet" and active_weights:
            weights = ", ".join(active_weights)
            raise ValueError(
                "Dirichlet loss weights require network='boardlaw_dirichlet'; "
                f"got network={self.network!r} with {weights}. Set these "
                "weights to 0.0 or use network='boardlaw_dirichlet'."
            )
        return self


@hydra.main(version_base=None, config_path="configs", config_name="hex")
def main(cfg: DictConfig) -> None:
    container = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    config = Config(**container)
    report_jax_backend()

    env = make_env(config.env_id, config.board_size)
    checkpoint_path = f"checkpoints/{config.board_size}_solved"
    baseline_model = from_pretrained(checkpoint_path, env, rngs=nnx.Rngs(0))
    
    model = build_model(
        config,
        num_actions=env.num_actions,
        observation_shape=env.observation_shape,
        rngs=nnx.Rngs(config.seed),
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adam(learning_rate=config.learning_rate),
        wrt=nnx.Param,
    )

    training_iteration = make_training_iteration(env, config)
    
    
    evaluate = make_mcts_evaluate(env, config, baseline_model)

    hours: float = 0.0
    frames: int = 0

    rng_key = jax.random.PRNGKey(config.seed)
    with build_logger(config) as logger:
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
            pbar = tqdm(range(start_iter, config.max_num_iters), desc="training", dynamic_ncols=True, total=config.max_num_iters, initial=start_iter)
            pbar.refresh()
            for iteration in pbar:
                dict_to_log = {}
                if iteration == config.max_num_iters - 1 or (config.eval_interval > 0 and iteration % config.eval_interval == 0):
                    rng_key, subkey = jax.random.split(rng_key)
                    returns = evaluate(subkey, model)
                    dict_to_log.update(returns_metrics("eval/vs_baseline", returns))

                st = time.time()
                rng_key, subkey = jax.random.split(rng_key)
                train_metrics = training_iteration(model, optimizer, subkey)
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
