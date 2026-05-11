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
from pathlib import Path
from typing import Any, Literal, cast

from flax import nnx
import hydra
import jax
import optax
import pgx
from omegaconf import DictConfig, OmegaConf
from pgx._src.baseline import BaselineModelId
from pydantic import BaseModel
from tqdm import tqdm

from .checkpoint import build_checkpoint_manager, maybe_save, restore
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
    num_search_blocks: int = 1
    max_num_steps: int = 256
    # training params
    optimizer_name: str = "adam"
    training_batch_size: int = 4096
    learning_rate: float = 0.001
    log_interval: int = 1
    # eval params
    eval_interval: int = 5
    # Dirichlet-Q params
    policy_mc_samples: int = 32
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 1.0
    q_loss_weight: float = 1.0
    c_terminal: float = 8.0
    c_leaf: float = 2.0
    train_interior_selector: Literal["policy_prior", "wdl"] = "policy_prior"
    inference_interior_selector: Literal["policy_prior", "wdl"] = "wdl"
    # logging params
    wandb_enabled: bool = True
    wandb_project: str = "scacchi-az"
    # checkpoint params
    ckpt_dir: str | None = None          # None → hydra working dir / checkpoints
    ckpt_max_to_keep: int = 3            # 0 disables checkpointing
    ckpt_save_interval_steps: int = 10

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

    ckpt_dir = Path(config.ckpt_dir) if config.ckpt_dir else Path.cwd() / "checkpoints"
    ckpt_manager = build_checkpoint_manager(config, ckpt_dir)
    rng_key = jax.random.PRNGKey(config.seed)
    start_iter, rng_key, hours, frames = restore(ckpt_manager, model, optimizer, rng_key)

    with build_logger(config) as logger:
        pbar = tqdm(range(start_iter, config.max_num_iters), desc="training", dynamic_ncols=True, total=config.max_num_iters)
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
            maybe_save(ckpt_manager, iteration, model, optimizer, rng_key, config, hours, frames)
            dict_to_log = {
                "policy_nll_loss": losses.policy_nll_loss.mean().item(),
                "policy_kl_hat": losses.policy_kl_hat.mean().item(),
                "value_dir_kl_loss": losses.value_dir_kl_loss.mean().item(),
                "q_dir_kl_loss": losses.q_dir_kl_loss.mean().item(),
                "hours": hours,
                "frames": frames,
            }
            logger.log(iteration, dict_to_log, pbar=pbar, prefix="train/", pbar_filter=r"loss|kl_hat")


if __name__ == "__main__":
    main()
