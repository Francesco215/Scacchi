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

import time
from typing import Any, cast

from flax import nnx
import hydra
import jax
import optax
import pgx
from omegaconf import DictConfig, OmegaConf
from pgx._src.baseline import BaselineModelId
from pydantic import BaseModel
from tqdm import tqdm

from .evaluations import make_evaluate
from .logger import build_logger
from .network import build_model
from .pipeline import make_training_iteration


class Config(BaseModel):
    env_id: pgx.EnvId = "go_9x9"
    seed: int = 0
    max_num_iters: int = 400
    # network params
    model_name: str = "resnet"
    num_channels: int = 128
    num_layers: int = 6
    resnet_v2: bool = True
    embed_dim: int = 128
    num_heads: int = 8
    mlp_dim: int = 512
    value_hidden_dim: int = 64
    position_encoding: str = "relative_2d"
    use_absolute_positions: bool = True
    num_history_steps: int = 8
    # selfplay params
    selfplay_batch_size: int = 1024
    num_simulations: int = 32
    max_num_steps: int = 256
    # training params
    optimizer_name: str = "adam"
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    log_interval: int = 1
    # eval params
    eval_interval: int = 5
    # logging params
    wandb_enabled: bool = True
    wandb_project: str = "scacchi-az"

    class Config:
        extra = "forbid"


@hydra.main(version_base=None, config_path="configs", config_name="gardner_chess")
def main(cfg: DictConfig) -> None:
    container = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    config: Config = Config(**container)
    if config.model_name == "transformer" and config.embed_dim % config.num_heads != 0:
        msg = "embed_dim must be divisible by num_heads."
        raise ValueError(msg)
    if config.model_name == "transformer" and config.position_encoding not in {
        "relative_2d",
        "basic_learned",
    }:
        msg = "position_encoding must be one of {'relative_2d', 'basic_learned'}."
        raise ValueError(msg)
    if config.optimizer_name not in {"adam", "adamw", "muon"}:
        msg = "optimizer_name must be one of {'adam', 'adamw', 'muon'}."
        raise ValueError(msg)

    env = pgx.make(config.env_id)
    baseline = pgx.make_baseline_model(cast(BaselineModelId, config.env_id + "_v0"))
    h, w, _ = env.observation_shape
    observation_shape = (h, w, config.num_history_steps * 14 + 3)
    model = build_model(
        config,
        num_actions=env.num_actions,
        observation_shape=observation_shape,
        rngs=nnx.Rngs(config.seed),
    )
    if config.optimizer_name == "adam":
        tx = optax.adam(learning_rate=config.learning_rate)
    elif config.optimizer_name == "adamw":
        tx = optax.adamw(learning_rate=config.learning_rate)
    else:
        tx = optax.contrib.muon(learning_rate=config.learning_rate)
    optimizer = nnx.Optimizer(model, tx, wrt=nnx.Param)

    training_iteration = make_training_iteration(env, config)
    evaluate = make_evaluate(env, baseline, config)

    hours: float = 0.0
    frames: int = 0

    rng_key = jax.random.PRNGKey(config.seed)
    with build_logger(config) as logger:
        pbar = tqdm(range(config.max_num_iters), desc="training", dynamic_ncols=True, total=config.max_num_iters)
        for iteration in pbar:
            if iteration % config.eval_interval == 0:
                rng_key, subkey = jax.random.split(rng_key)
                returns = evaluate(subkey, model)
                logger.log_returns(iteration, returns, prefix="eval/vs_baseline")

            st = time.time()
            rng_key, subkey = jax.random.split(rng_key)
            policy_losses, value_losses = training_iteration(model, optimizer, subkey)
            frames += config.selfplay_batch_size * config.max_num_steps

            et = time.time()
            hours += (et - st) / 3600
            dict_to_log = {"policy_loss": policy_losses.mean().item(), "value_loss": value_losses.mean().item(), "hours": hours, "frames": frames}
            logger.log(iteration, dict_to_log, pbar=pbar, prefix="train/", pbar_filter=r"loss")


if __name__ == "__main__":
    main()
