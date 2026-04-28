"""NNX policy/value networks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from flax import nnx
from jaxtyping import Array, Float
from omegaconf import DictConfig

from scacchi.config import ModelConfig


class ResidualBlock(nnx.Module):
    def __init__(
        self,
        channels: int,
        *,
        resnet_v2: bool,
        batch_norm: bool,
        momentum: float,
        rngs: nnx.Rngs,
    ):
        self.resnet_v2 = resnet_v2
        self.conv1 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.conv2 = nnx.Conv(channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs)
        self.bn1 = _batch_norm(channels, batch_norm=batch_norm, momentum=momentum, rngs=rngs)
        self.bn2 = _batch_norm(channels, batch_norm=batch_norm, momentum=momentum, rngs=rngs)

    def __call__(
        self, x: Float[Array, "batch height width channels"], *, train: bool
    ) -> Float[Array, "batch height width channels"]:
        residual = x
        if self.resnet_v2:
            x = self.conv1(jax.nn.relu(_norm(self.bn1, x, train=train)))
            x = self.conv2(jax.nn.relu(_norm(self.bn2, x, train=train)))
            return x + residual
        x = jax.nn.relu(_norm(self.bn1, self.conv1(x), train=train))
        x = _norm(self.bn2, self.conv2(x), train=train)
        return jax.nn.relu(x + residual)


def _batch_norm(
    channels: int, *, batch_norm: bool, momentum: float, rngs: nnx.Rngs
) -> nnx.BatchNorm | None:
    return nnx.BatchNorm(channels, momentum=momentum, rngs=rngs) if batch_norm else None


def _norm(
    layer: nnx.BatchNorm | None,
    x: Float[Array, "batch height width channels"],
    *,
    train: bool,
) -> Float[Array, "batch height width channels"]:
    return x if layer is None else layer(x, use_running_average=not train)


class AlphaZeroResNet(nnx.Module):
    """AlphaZero ResNet ported from the PGX example to NNX."""

    def __init__(
        self,
        cfg: DictConfig | ModelConfig,
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
        batch_norm = bool(getattr(cfg, "batch_norm", True))
        momentum = float(getattr(cfg, "batch_norm_momentum", 0.9))

        self.observation_shape = observation_shape
        self.num_actions = int(num_actions)
        self.resnet_v2 = bool(getattr(cfg, "resnet_v2", True))
        self.policy_flat_dim = int(height * width * policy_channels)
        self.value_flat_dim = int(height * width * value_channels)
        self.trunk = nnx.Conv(
            in_channels, channels, kernel_size=(3, 3), padding="SAME", rngs=rngs
        )
        self.trunk_bn = _batch_norm(
            channels, batch_norm=batch_norm, momentum=momentum, rngs=rngs
        )
        self.blocks = nnx.List(
            [
                ResidualBlock(
                    channels,
                    resnet_v2=self.resnet_v2,
                    batch_norm=batch_norm,
                    momentum=momentum,
                    rngs=rngs,
                )
                for _ in range(blocks)
            ]
        )
        self.policy_conv = nnx.Conv(
            channels, policy_channels, kernel_size=(1, 1), padding="SAME", rngs=rngs
        )
        self.policy_bn = _batch_norm(
            policy_channels, batch_norm=batch_norm, momentum=momentum, rngs=rngs
        )
        self.policy_head = nnx.Linear(self.policy_flat_dim, num_actions, rngs=rngs)
        self.value_conv = nnx.Conv(
            channels, value_channels, kernel_size=(1, 1), padding="SAME", rngs=rngs
        )
        self.value_bn = _batch_norm(
            value_channels, batch_norm=batch_norm, momentum=momentum, rngs=rngs
        )
        self.value_hidden = nnx.Linear(self.value_flat_dim, value_hidden, rngs=rngs)
        self.value_head = nnx.Linear(value_hidden, 1, rngs=rngs)

    def __call__(
        self, observation: Float[Array, "... height width channels"], *, train: bool
    ) -> tuple[Float[Array, "... action"], Float[Array, "..."]]:
        leading_shape = observation.shape[:-3]
        x = jnp.asarray(observation, dtype=jnp.float32)
        x = jnp.reshape(x, (-1, *self.observation_shape))

        x = self.trunk(x)
        if not self.resnet_v2:
            x = jax.nn.relu(_norm(self.trunk_bn, x, train=train))
        for block in self.blocks:
            x = block(x, train=train)
        if self.resnet_v2:
            x = jax.nn.relu(_norm(self.trunk_bn, x, train=train))

        policy = jax.nn.relu(_norm(self.policy_bn, self.policy_conv(x), train=train))
        policy = jnp.reshape(policy, (policy.shape[0], self.policy_flat_dim))
        policy_logits = self.policy_head(policy)

        value = jax.nn.relu(_norm(self.value_bn, self.value_conv(x), train=train))
        value = jnp.reshape(value, (value.shape[0], self.value_flat_dim))
        value = jax.nn.relu(self.value_hidden(value))
        value = jnp.tanh(self.value_head(value))

        policy_logits = jnp.reshape(policy_logits, (*leading_shape, self.num_actions))
        value = jnp.reshape(value, leading_shape)
        return policy_logits, value
