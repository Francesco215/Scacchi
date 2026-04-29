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
from typing import NamedTuple

from flax import nnx
import hydra
import jax
import jax.numpy as jnp
import mctx
import optax
import pgx
from omegaconf import DictConfig, OmegaConf
from pgx.experimental import auto_reset
from pydantic import BaseModel

from .network import AZNet


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


class SelfplayOutput(NamedTuple):
    obs: jax.Array
    reward: jax.Array
    terminated: jax.Array
    action_weights: jax.Array
    discount: jax.Array


class Sample(NamedTuple):
    obs: jax.Array
    policy_tgt: jax.Array
    value_tgt: jax.Array
    mask: jax.Array


@hydra.main(version_base=None, config_path="configs", config_name="gardner_chess")
def main(cfg: DictConfig) -> None:
    config: Config = Config(**OmegaConf.to_container(cfg, resolve=True))
    print(config)

    env = pgx.make(config.env_id)
    baseline = pgx.make_baseline_model(config.env_id + "_v0")
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

    @nnx.jit
    def selfplay(model: AZNet, rng_key: jax.Array) -> SelfplayOutput:
        graphdef, model_state = nnx.split(model)

        def apply_model(state, observation):
            eval_model = nnx.merge(graphdef, state)
            return eval_model(observation, train=False)

        def recurrent_fn(
            state,
            rng_key: jax.Array,
            action: jax.Array,
            env_state: pgx.State,
        ):
            del rng_key

            current_player = env_state.current_player
            env_state = jax.vmap(env.step)(env_state, action)
            logits, value = apply_model(state, env_state.observation)
            logits = logits - jnp.max(logits, axis=-1, keepdims=True)
            logits = jnp.where(
                env_state.legal_action_mask,
                logits,
                jnp.finfo(logits.dtype).min,
            )

            reward = env_state.rewards[
                jnp.arange(env_state.rewards.shape[0]),
                current_player,
            ]
            value = jnp.where(env_state.terminated, 0.0, value)
            discount = -jnp.ones_like(value)
            discount = jnp.where(env_state.terminated, 0.0, discount)

            return (
                mctx.RecurrentFnOutput(
                    reward=reward,
                    discount=discount,
                    prior_logits=logits,
                    value=value,
                ),
                env_state,
            )

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def step_fn(
            env_state: pgx.State,
            key: jax.Array,
        ) -> tuple[pgx.State, SelfplayOutput]:
            search_key, reset_key = jax.random.split(key)
            observation = env_state.observation
            logits, value = apply_model(model_state, observation)
            root = mctx.RootFnOutput(
                prior_logits=logits,
                value=value,
                embedding=env_state,
            )

            policy_output = mctx.gumbel_muzero_policy(
                params=model_state,
                rng_key=search_key,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=config.num_simulations,
                invalid_actions=~env_state.legal_action_mask,
                qtransform=mctx.qtransform_completed_by_mix_value,
                gumbel_scale=1.0,
            )
            actor = env_state.current_player
            reset_keys = jax.random.split(reset_key, config.selfplay_batch_size)
            env_state = jax.vmap(auto_reset(env.step, env.init))(
                env_state,
                policy_output.action,
                reset_keys,
            )
            discount = -jnp.ones_like(value)
            discount = jnp.where(env_state.terminated, 0.0, discount)
            return env_state, SelfplayOutput(
                obs=observation,
                action_weights=policy_output.action_weights,
                reward=env_state.rewards[jnp.arange(env_state.rewards.shape[0]), actor],
                terminated=env_state.terminated,
                discount=discount,
            )

        rng_key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, config.selfplay_batch_size)
        env_state = jax.vmap(env.init)(init_keys)
        step_keys = jax.random.split(rng_key, config.max_num_steps)
        _, data = step_fn(env_state, step_keys)
        return data

    @nnx.jit
    def compute_loss_input(data: SelfplayOutput) -> Sample:
        value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1

        @nnx.scan(in_axes=(nnx.Carry, 0), out_axes=(nnx.Carry, 0))
        def body_fn(carry: jax.Array, i: jax.Array) -> tuple[jax.Array, jax.Array]:
            ix = config.max_num_steps - i - 1
            value = data.reward[ix] + data.discount[ix] * carry
            return value, value

        _, value_tgt = body_fn(
            jnp.zeros(config.selfplay_batch_size, dtype=data.reward.dtype),
            jnp.arange(config.max_num_steps),
        )
        value_tgt = value_tgt[::-1, :]

        return Sample(
            obs=data.obs,
            policy_tgt=data.action_weights,
            value_tgt=value_tgt,
            mask=value_mask,
        )

    @nnx.jit
    def train(model: AZNet, optimizer: nnx.Optimizer, data: Sample):
        def loss_fn(model: AZNet):
            logits, value = model(data.obs, train=True)
            policy_loss = optax.softmax_cross_entropy(logits, data.policy_tgt)
            policy_loss = jnp.mean(policy_loss)
            value_loss = optax.l2_loss(value, data.value_tgt)
            value_loss = jnp.mean(value_loss * data.mask)
            return policy_loss + value_loss, (policy_loss, value_loss)

        (_, (policy_loss, value_loss)), grads = nnx.value_and_grad(
            loss_fn,
            has_aux=True,
        )(model)
        optimizer.update(model, grads)
        return policy_loss, value_loss

    @nnx.jit
    def evaluate(rng_key: jax.Array, model: AZNet):
        """A simplified evaluation by sampling. Only for debugging.
        Please use MCTS and run tournaments for serious evaluation."""
        my_player = 0

        key, init_key = jax.random.split(rng_key)
        init_keys = jax.random.split(init_key, config.selfplay_batch_size)
        env_state = jax.vmap(env.init)(init_keys)

        def body_fn(val):
            key, env_state, returns = val
            my_logits, _ = model(env_state.observation, train=False)
            opp_logits, _ = baseline(env_state.observation)
            is_my_turn = (env_state.current_player == my_player).reshape((-1, 1))
            logits = jnp.where(is_my_turn, my_logits, opp_logits)
            key, action_key = jax.random.split(key)
            action = jax.random.categorical(action_key, logits, axis=-1)
            env_state = jax.vmap(env.step)(env_state, action)
            returns = returns + env_state.rewards[
                jnp.arange(config.selfplay_batch_size),
                my_player,
            ]
            return key, env_state, returns

        _, _, returns = nnx.while_loop(
            lambda x: ~(x[1].terminated.all()),
            body_fn,
            (key, env_state, jnp.zeros(config.selfplay_batch_size)),
        )
        return returns

    iteration: int = 0
    hours: float = 0.0
    frames: int = 0
    log = {"iteration": iteration, "hours": hours, "frames": frames}

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
        log = {"iteration": iteration}
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
