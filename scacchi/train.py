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
from typing import Any, cast

# Training is GPU-only by default. Override JAX_PLATFORMS deliberately for
# diagnostics, and set SCACCHI_ALLOW_CPU=1 only when a CPU run is intentional.
os.environ.setdefault("JAX_PLATFORMS", "cuda")

from flax import nnx
import hydra
import jax
import optax
import pgx
from omegaconf import DictConfig, OmegaConf
from pydantic import BaseModel, ConfigDict
from tqdm import tqdm

from .envs import make_env
from .evaluations import make_mcts_evaluate
from .logger import build_logger, returns_metrics
from .network import AZNet
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
    mohex_max_memory: int | None = None
    mohex_max_time: float | None = None
    mohex_max_games: int | None = None
    mohex_max_nodes: int | None = None
    mohex_num_processes: int = 1
    mohex_num_threads: int | None = None
    mohex_dfpn_threads: int | None = None
    mohex_parallel_solver: bool = False
    # logging params
    wandb_enabled: bool = True
    wandb_project: str = "scacchi-az"

    model_config = ConfigDict(extra="forbid")


@hydra.main(version_base=None, config_path="configs", config_name="hex")
def main(cfg: DictConfig) -> None:
    container = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    config = Config(**container)
    report_jax_backend()

    env = make_env(config.env_id, config.board_size)
    model = AZNet(
        num_actions=env.num_actions,
        observation_shape=env.observation_shape,
        num_channels=config.num_channels,
        num_blocks=config.num_layers,
        resnet_v2=config.resnet_v2,
        rngs=nnx.Rngs(config.seed),
    )
    optimizer = nnx.Optimizer(
        model,
        optax.adam(learning_rate=config.learning_rate),
        wrt=nnx.Param,
    )

    training_iteration = make_training_iteration(env, config)
    evaluate = make_mcts_evaluate(env, config)

    hours: float = 0.0
    frames: int = 0

    rng_key = jax.random.PRNGKey(config.seed)
    with build_logger(config) as logger:
        pbar = tqdm(
            range(config.max_num_iters),
            desc="training",
            dynamic_ncols=True,
            total=config.max_num_iters,
        )
        pbar.refresh()
        for iteration in pbar:
            dict_to_log = {}
            if config.eval_interval > 0 and iteration % config.eval_interval == 0:
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


if __name__ == "__main__":
    main()
