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
from pydantic import BaseModel, ConfigDict
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
    network: str = "aznet"  # "aznet" | "boardlaw"
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    max_num_steps: int = 256
    # training params
    training_batch_size: int = 4096
    learning_rate: float = 0.001
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


@hydra.main(version_base=None, config_path="configs", config_name="hex")
def main(cfg: DictConfig) -> None:
    container = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    config = Config(**container)
    report_jax_backend()

    env = make_env(config.env_id, config.board_size)
    
    # >> from pretrained here (aka load the model), get board size load eval model. <<
    #checkpoint_path = 'something' + str(config.board_size ) + 'something'
    #baseline_model = from_pretrained(checkpoint_path, env, rngs=nnx.Rngs(0))
    # random baseline test.
    baseline_model = build_model(
        config,
        num_actions=env.num_actions,
        observation_shape=env.observation_shape,
        rngs=nnx.Rngs(config.seed),
    )
    
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
    
    # pass eval model here.
    
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
                policy_losses, value_losses = training_iteration(model, optimizer, subkey)
                frames += config.selfplay_batch_size * config.max_num_steps

                et = time.time()
                hours += (et - st) / 3600
                dict_to_log.update(
                    {
                        "train/policy_loss": policy_losses.mean().item(),
                        "train/value_loss": value_losses.mean().item(),
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
