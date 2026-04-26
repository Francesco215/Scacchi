"""NNX policy/value networks."""

from __future__ import annotations

from typing import Protocol

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from omegaconf import DictConfig


class ResidualBlock(nnx.Module):
    def __init__(self, channels: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)

    def __call__(self, x: Float[Array, "batch height width channels"]) -> Float[
        Array, "batch height width channels"
    ]:
        residual = x
        x = jax.nn.relu(self.conv1(x))
        x = self.conv2(x)
        return jax.nn.relu(x + residual)


class AlphaZeroResNet(nnx.Module):
    """Small AlphaZero-style ResNet with swappable policy/value heads."""

    def __init__(
        self,
        cfg: DictConfig,
        *,
        observation_shape: tuple[int, int, int],
        num_actions: int,
        seed: int,
    ):
        if str(cfg.name) != "resnet":
            msg = f"Unknown model '{cfg.name}'."
            raise ValueError(msg)

        height, width, in_channels = observation_shape
        rngs = nnx.Rngs(seed)
        channels = int(cfg.channels)
        blocks = int(cfg.blocks)
        policy_channels = int(cfg.policy_channels)
        value_channels = int(cfg.value_channels)
        value_hidden = int(cfg.value_hidden)

        self.observation_shape = observation_shape
        self.num_actions = int(num_actions)
        self.policy_flat_dim = int(height * width * policy_channels)
        self.value_flat_dim = int(height * width * value_channels)
        self.trunk = nnx.Conv(
            in_channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs
        )
        self.blocks = nnx.List([ResidualBlock(channels, rngs=rngs) for _ in range(blocks)])
        self.policy_conv = nnx.Conv(
            channels, policy_channels, kernel_size=(1, 1), padding="SAME", rngs=rngs
        )
        self.policy_head = nnx.Linear(self.policy_flat_dim, num_actions, rngs=rngs)
        self.value_conv = nnx.Conv(
            channels, value_channels, kernel_size=(1, 1), padding="SAME", rngs=rngs
        )
        self.value_hidden = nnx.Linear(self.value_flat_dim, value_hidden, rngs=rngs)
        self.value_head = nnx.Linear(value_hidden, 1, rngs=rngs)

    def __call__(
        self, observation: Float[Array, "... height width channels"], *, train: bool
    ) -> tuple[Float[Array, "... action"], Float[Array, "..."]]:
        del train
        leading_shape = observation.shape[:-3]
        x = jnp.asarray(observation, dtype=jnp.float32)
        x = jnp.reshape(x, (-1, *self.observation_shape))

        x = jax.nn.relu(self.trunk(x))
        for block in self.blocks:
            x = block(x)

        policy = jax.nn.relu(self.policy_conv(x))
        policy = jnp.reshape(policy, (policy.shape[0], self.policy_flat_dim))
        policy_logits = self.policy_head(policy)

        value = jax.nn.relu(self.value_conv(x))
        value = jnp.reshape(value, (value.shape[0], self.value_flat_dim))
        value = jax.nn.relu(self.value_hidden(value))
        value = jnp.tanh(self.value_head(value))

        policy_logits = jnp.reshape(policy_logits, (*leading_shape, self.num_actions))
        value = jnp.reshape(value, leading_shape)
        return policy_logits, value
