from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx

if TYPE_CHECKING:
    from .train import ModelConfig


def _unit_dirichlet_concentration_logit(num_outcomes: int) -> float:
    """Logit whose squared softplus gives total concentration num_outcomes."""
    return math.log(math.expm1(math.sqrt(num_outcomes)))


def dirichlet_from_logits(
    mean_logits: jax.Array,
    concentration_logit: jax.Array,
    *,
    concentration_clip: float | None = None,
) -> jax.Array:
    concentration = jax.nn.softplus(concentration_logit)**2
    if concentration_clip is not None:
        concentration = jnp.minimum(
            concentration,
            jnp.asarray(concentration_clip, dtype=concentration.dtype),
        )
    return concentration[..., None] * jax.nn.softmax(mean_logits, axis=-1)


def outcome_mean(alpha: jax.Array) -> jax.Array:
    return alpha / jnp.sum(alpha, axis=-1, keepdims=True)


def outcome_utility(outcome_dist: jax.Array) -> jax.Array:
    return outcome_dist[..., -1] - outcome_dist[..., 0]


def policy_value_from_output(output):
    if len(output) == 2:
        return output
    logits, alpha_v, _alpha_q = output
    return logits, outcome_utility(outcome_mean(alpha_v))


class BlockV1(nnx.Module):
    def __init__(self, num_channels: int, *, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(num_channels, num_channels, kernel_size=3, padding="SAME", rngs=rngs)
        self.bn1 = nnx.BatchNorm(num_channels, momentum=0.9, rngs=rngs)
        self.conv2 = nnx.Conv(num_channels, num_channels, kernel_size=3, padding="SAME", rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_channels, momentum=0.9, rngs=rngs)

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        return jax.nn.relu(x + residual)


class BlockV2(nnx.Module):
    def __init__(self, num_channels: int, *, rngs: nnx.Rngs):
        self.bn1 = nnx.BatchNorm(num_channels, momentum=0.9, rngs=rngs)
        self.conv1 = nnx.Conv(num_channels, num_channels, kernel_size=3, padding="SAME", rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_channels, momentum=0.9, rngs=rngs)
        self.conv2 = nnx.Conv(num_channels, num_channels, kernel_size=3, padding="SAME", rngs=rngs)

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        residual = x
        x = self.bn1(x, use_running_average=not train)
        x = jax.nn.relu(x)
        x = self.conv1(x)
        x = self.bn2(x, use_running_average=not train)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        return x + residual


class AZNet(nnx.Module):
    """AlphaZero NN architecture retained for scalar baseline evaluation."""

    def __init__(
        self,
        num_actions: int,
        observation_shape: Sequence[int],
        *,
        num_channels: int = 64,
        num_blocks: int = 5,
        resnet_v2: bool = True,
        rngs: nnx.Rngs,
    ):
        height, width, input_channels = observation_shape
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.resnet_v2 = resnet_v2

        self.conv = nnx.Conv(input_channels, num_channels, kernel_size=3, padding="SAME", rngs=rngs)
        if not resnet_v2:
            self.bn = nnx.BatchNorm(num_channels, momentum=0.9, rngs=rngs)

        block_cls = BlockV2 if resnet_v2 else BlockV1
        self.blocks = nnx.List([block_cls(num_channels, rngs=rngs) for _ in range(num_blocks)])

        if resnet_v2:
            self.bn = nnx.BatchNorm(num_channels, momentum=0.9, rngs=rngs)

        self.policy_conv = nnx.Conv(num_channels, 2, kernel_size=1, padding="SAME", rngs=rngs)
        self.policy_bn = nnx.BatchNorm(2, momentum=0.9, rngs=rngs)
        self.policy_linear = nnx.Linear(
            height * width * 2,
            num_actions,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )

        self.value_conv = nnx.Conv(num_channels, 1, kernel_size=1, padding="SAME", rngs=rngs)
        self.value_bn = nnx.BatchNorm(1, momentum=0.9, rngs=rngs)
        self.value_linear = nnx.Linear(height * width, num_channels, rngs=rngs)
        self.value_out = nnx.Linear(num_channels, 1, rngs=rngs)

    def __call__(self, x: jax.Array, *, train: bool) -> tuple[jax.Array, jax.Array]:
        x = x.astype(jnp.float32)
        x = self.conv(x)

        if not self.resnet_v2:
            x = self.bn(x, use_running_average=not train)
            x = jax.nn.relu(x)

        for block in self.blocks:
            x = block(x, train=train)

        if self.resnet_v2:
            x = self.bn(x, use_running_average=not train)
            x = jax.nn.relu(x)

        logits = self.policy_conv(x)
        logits = self.policy_bn(logits, use_running_average=not train)
        logits = jax.nn.relu(logits)
        logits = logits.reshape((logits.shape[0], -1))
        logits = self.policy_linear(logits)

        value = self.value_conv(x)
        value = self.value_bn(value, use_running_average=not train)
        value = jax.nn.relu(value)
        value = value.reshape((value.shape[0], -1))
        value = self.value_linear(value)
        value = jax.nn.relu(value)
        value = self.value_out(value)
        value = jnp.tanh(value)
        value = value.reshape((-1,))

        return logits, value


class ReZeroResidual(nnx.Module):
    """ReZero residual block: x + α · Linear(ReLU(x)) with α initialized to 0."""

    def __init__(self, width: int, *, rngs: nnx.Rngs):
        self.linear = nnx.Linear(
            width,
            width,
            kernel_init=jax.nn.initializers.variance_scaling(
                scale=2.0,
                mode="fan_in",
                distribution="truncated_normal",
            ),
            rngs=rngs,
        )
        self.alpha = nnx.Param(jnp.zeros(()))

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.alpha * self.linear(jax.nn.relu(x))


class BoardlawNet(nnx.Module):
    """Scalar Boardlaw net retained for scalar baseline evaluation."""

    def __init__(
        self,
        num_actions: int,
        observation_shape: Sequence[int],
        *,
        width: int = 512,
        depth: int = 8,
        rngs: nnx.Rngs,
    ):
        self.num_actions = num_actions
        self.width = width
        self.depth = depth

        input_dim = math.prod(observation_shape)
        self.intake = nnx.Linear(input_dim, width, rngs=rngs)
        self.blocks = nnx.List([ReZeroResidual(width, rngs=rngs) for _ in range(depth)])
        self.policy_head = nnx.Linear(
            width,
            num_actions,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )
        self.value_head = nnx.Linear(width, 1, rngs=rngs)

    def __call__(self, x: jax.Array, *, train: bool) -> tuple[jax.Array, jax.Array]:
        del train
        x = x.astype(jnp.float32)
        x = x.reshape((x.shape[0], -1))
        x = self.intake(x)
        for block in self.blocks:
            x = block(x)
        logits = self.policy_head(x)
        value = jnp.tanh(self.value_head(x)).reshape((-1,))
        return logits, value


class BoardlawDirichletNet(nnx.Module):
    """Boardlaw-style MLP with policy, value-Dirichlet, and Q-Dirichlet heads."""

    def __init__(
        self,
        num_actions: int,
        observation_shape: Sequence[int],
        *,
        num_outcomes: int = 2,
        width: int = 512,
        depth: int = 8,
        dirichlet_concentration_clip: float | None = 8.0,
        rngs: nnx.Rngs,
    ):
        self.num_actions = num_actions
        self.num_outcomes = num_outcomes
        self.width = width
        self.depth = depth
        self.dirichlet_concentration_clip = dirichlet_concentration_clip

        input_dim = math.prod(observation_shape)
        self.intake = nnx.Linear(input_dim, width, rngs=rngs)
        self.blocks = nnx.List([ReZeroResidual(width, rngs=rngs) for _ in range(depth)])
        self.policy_head = nnx.Linear(
            width,
            num_actions,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )
        self.value_dir_head = nnx.Linear(
            width,
            num_outcomes,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )
        concentration_bias_init = jax.nn.initializers.constant(
            _unit_dirichlet_concentration_logit(num_outcomes)
        )
        self.value_conc_head = nnx.Linear(
            width,
            1,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=concentration_bias_init,
            rngs=rngs,
        )
        self.q_dir_head = nnx.Linear(
            width,
            num_actions * num_outcomes,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )
        self.q_conc_head = nnx.Linear(
            width,
            num_actions,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=concentration_bias_init,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, *, train: bool) -> tuple[jax.Array, jax.Array, jax.Array]:
        del train
        x = x.astype(jnp.float32)
        x = x.reshape((x.shape[0], -1))
        x = self.intake(x)
        for block in self.blocks:
            x = block(x)

        logits = self.policy_head(x)

        value_mean_logits = self.value_dir_head(x)
        value_concentration_logit = self.value_conc_head(x).reshape((x.shape[0],))
        alpha_v = dirichlet_from_logits(
            value_mean_logits,
            value_concentration_logit,
            concentration_clip=self.dirichlet_concentration_clip,
        )

        q_mean_logits = self.q_dir_head(x).reshape(
            (x.shape[0], self.num_actions, self.num_outcomes)
        )
        q_concentration_logit = self.q_conc_head(x)
        alpha_q = dirichlet_from_logits(
            q_mean_logits,
            q_concentration_logit,
            concentration_clip=self.dirichlet_concentration_clip,
        )
        return logits, alpha_v, alpha_q


def build_model(
    config: ModelConfig,
    *,
    num_outcomes: int | None,
    dirichlet_concentration_clip: float | None,
    num_actions: int,
    observation_shape: Sequence[int],
    rngs: nnx.Rngs,
) -> nnx.Module:
    if num_outcomes is None:
        num_outcomes = 3
    return BoardlawDirichletNet(
        num_actions=num_actions,
        observation_shape=observation_shape,
        num_outcomes=num_outcomes,
        width=config.num_channels,
        depth=config.num_layers,
        dirichlet_concentration_clip=dirichlet_concentration_clip,
        rngs=rngs,
    )

