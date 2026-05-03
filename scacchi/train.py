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
from .network import AZNet
from .pipeline import make_training_iteration


class Config(BaseModel):
    env_id: pgx.EnvId = "go_9x9"
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
    # Dirichlet-Q params
    policy_mc_samples: int = 32
    policy_loss_weight: float = 1.0
    value_outcome_weight: float = 1.0
    q_outcome_weight: float = 1.0
    value_search_weight: float = 1.0
    q_search_weight: float = 1.0
    dir_kl_weight: float = 0.0
    c_terminal: float = 8.0
    c_leaf: float = 2.0
    # logging params
    wandb_enabled: bool = True
    wandb_project: str = "scacchi-az"

    class Config:
        extra = "forbid"


@hydra.main(version_base=None, config_path="configs", config_name="gardner_chess")
def main(cfg: DictConfig) -> None:
    container = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    config: Config = Config(**container)

    env = pgx.make(config.env_id)
    baseline = pgx.make_baseline_model(cast(BaselineModelId, config.env_id + "_v0"))
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
            losses = training_iteration(model, optimizer, subkey)
            frames += config.selfplay_batch_size * config.max_num_steps

            et = time.time()
            hours += (et - st) / 3600
            dict_to_log = {
                "policy_loss": losses.policy_loss.mean().item(),
                "value_outcome_loss": losses.value_outcome_loss.mean().item(),
                "q_outcome_loss": losses.q_outcome_loss.mean().item(),
                "value_search_mean_loss": losses.value_search_mean_loss.mean().item(),
                "q_search_mean_loss": losses.q_search_mean_loss.mean().item(),
                "value_dir_kl_loss": losses.value_dir_kl_loss.mean().item(),
                "q_dir_kl_loss": losses.q_dir_kl_loss.mean().item(),
                "hours": hours,
                "frames": frames,
            }
            logger.log(iteration, dict_to_log, pbar=pbar, prefix="train/", pbar_filter=r"loss")


if __name__ == "__main__":
    main()
