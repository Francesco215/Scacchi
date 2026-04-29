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
import jax.numpy as jnp
import optax
import pgx
from omegaconf import DictConfig, OmegaConf
from pgx._src.baseline import BaselineModelId
from pydantic import BaseModel

from .loss import make_compute_loss_input, make_evaluate, train
from .network import AZNet
from .play import make_selfplay


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
    # eval params
    eval_interval: int = 5

    class Config:
        extra = "forbid"


@hydra.main(version_base=None, config_path="configs", config_name="gardner_chess")
def main(cfg: DictConfig) -> None:
    container = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    config: Config = Config(**container)
    print(config)

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

    selfplay = make_selfplay(env, config)
    compute_loss_input = make_compute_loss_input(config)
    evaluate = make_evaluate(env, baseline, config)

    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    log: dict[str, float] = {"iteration": iteration, "hours": hours, "frames": frames}

    rng_key = jax.random.PRNGKey(config.seed)
    while True:
        if iteration % config.eval_interval == 0:
            rng_key, subkey = jax.random.split(rng_key)
            returns = evaluate(subkey, model)
            log.update(
                {
                    "eval/vs_baseline/avg_R": returns.mean().item(),
                    "eval/vs_baseline/win_rate": (
                        (returns == 1).sum() / returns.size
                    ).item(),
                    "eval/vs_baseline/draw_rate": (
                        (returns == 0).sum() / returns.size
                    ).item(),
                    "eval/vs_baseline/lose_rate": (
                        (returns == -1).sum() / returns.size
                    ).item(),
                }
            )

        print(log)

        if iteration >= config.max_num_iters:
            break

        iteration += 1
        log: dict[str, float] = {"iteration": iteration}
        st = time.time()

        rng_key, subkey = jax.random.split(rng_key)
        data = selfplay(model, subkey)
        samples = compute_loss_input(data)

        samples = jax.device_get(samples)
        frames += samples.obs.shape[0] * samples.obs.shape[1]
        samples = jax.tree_util.tree_map(
            lambda x: x.reshape((-1, *x.shape[2:])),
            samples,
        )
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples.obs.shape[0]))
        samples = jax.tree_util.tree_map(lambda x: x[ixs], samples)

        num_updates = samples.obs.shape[0] // config.training_batch_size
        if num_updates == 0:
            raise ValueError(
                "training_batch_size must be less than or equal to "
                "selfplay_batch_size * max_num_steps"
            )
        num_train_samples = num_updates * config.training_batch_size
        samples = jax.tree_util.tree_map(lambda x: x[:num_train_samples], samples)
        minibatches = jax.tree_util.tree_map(
            lambda x: x.reshape(
                (num_updates, config.training_batch_size) + x.shape[1:]
            ),
            samples,
        )

        policy_losses, value_losses = [], []
        for i in range(num_updates):
            minibatch = jax.tree_util.tree_map(lambda x: x[i], minibatches)
            policy_loss, value_loss = train(model, optimizer, minibatch)
            policy_losses.append(policy_loss.item())
            value_losses.append(value_loss.item())

        et = time.time()
        hours += (et - st) / 3600
        log.update(
            {
                "train/policy_loss": sum(policy_losses) / len(policy_losses),
                "train/value_loss": sum(value_losses) / len(value_losses),
                "hours": hours,
                "frames": frames,
            }
        )


if __name__ == "__main__":
    main()
