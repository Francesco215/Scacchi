from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx

if TYPE_CHECKING:
    from .types import Config


def _dtype_from_name(name: str):
    if name in {"float32", "fp32", "f32"}:
        return jnp.float32
    if name in {"bfloat16", "bf16"}:
        return jnp.bfloat16
    if name in {"float16", "fp16", "f16"}:
        return jnp.float16
    raise ValueError(f"unknown model.compute_dtype: {name!r}")


def _unit_dirichlet_concentration_logit(num_outcomes: int) -> float:
    """Logit whose squared softplus gives total concentration num_outcomes."""
    return math.log(math.expm1(math.sqrt(num_outcomes)))


def dirichlet_from_logits(
    mean_logits: jax.Array,
    concentration_logit: jax.Array,
    *,
    concentration_clip: float | None = None,
) -> jax.Array:
    mean_logits = mean_logits.astype(jnp.float32)
    concentration_logit = concentration_logit.astype(jnp.float32)
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
    def __init__(self, num_channels: int, *, dtype=jnp.float32, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(num_channels, num_channels, kernel_size=3, padding="SAME", dtype=dtype, rngs=rngs)
        self.bn1 = nnx.BatchNorm(num_channels, momentum=0.9, dtype=dtype, rngs=rngs)
        self.conv2 = nnx.Conv(num_channels, num_channels, kernel_size=3, padding="SAME", dtype=dtype, rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_channels, momentum=0.9, dtype=dtype, rngs=rngs)

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        return jax.nn.relu(x + residual)


class BlockV2(nnx.Module):
    def __init__(self, num_channels: int, *, dtype=jnp.float32, rngs: nnx.Rngs):
        self.bn1 = nnx.BatchNorm(num_channels, momentum=0.9, dtype=dtype, rngs=rngs)
        self.conv1 = nnx.Conv(num_channels, num_channels, kernel_size=3, padding="SAME", dtype=dtype, rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_channels, momentum=0.9, dtype=dtype, rngs=rngs)
        self.conv2 = nnx.Conv(num_channels, num_channels, kernel_size=3, padding="SAME", dtype=dtype, rngs=rngs)

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
    """AlphaZero NN architecture."""

    def __init__(
        self,
        num_actions: int,
        observation_shape: Sequence[int],
        *,
        num_channels: int = 64,
        num_blocks: int = 5,
        resnet_v2: bool = True,
        dtype=jnp.float32,
        rngs: nnx.Rngs,
    ):
        height, width, input_channels = observation_shape
        self.num_actions = num_actions
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.resnet_v2 = resnet_v2
        self.dtype = dtype

        self.conv = nnx.Conv(input_channels, num_channels, kernel_size=3, padding="SAME", dtype=dtype, rngs=rngs)
        if not resnet_v2:
            self.bn = nnx.BatchNorm(num_channels, momentum=0.9, dtype=dtype, rngs=rngs)

        block_cls = BlockV2 if resnet_v2 else BlockV1
        self.blocks = nnx.List([block_cls(num_channels, dtype=dtype, rngs=rngs) for _ in range(num_blocks)])

        if resnet_v2:
            self.bn = nnx.BatchNorm(num_channels, momentum=0.9, dtype=dtype, rngs=rngs)

        self.policy_conv = nnx.Conv(num_channels, 2, kernel_size=1, padding="SAME", dtype=dtype, rngs=rngs)
        self.policy_bn = nnx.BatchNorm(2, momentum=0.9, dtype=dtype, rngs=rngs)
        self.policy_linear = nnx.Linear(
            height * width * 2,
            num_actions,
            dtype=dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )

        self.value_conv = nnx.Conv(num_channels, 1, kernel_size=1, padding="SAME", dtype=dtype, rngs=rngs)
        self.value_bn = nnx.BatchNorm(1, momentum=0.9, dtype=dtype, rngs=rngs)
        self.value_linear = nnx.Linear(height * width, num_channels, dtype=dtype, rngs=rngs)
        self.value_out = nnx.Linear(num_channels, 1, dtype=dtype, rngs=rngs)

    def __call__(self, x: jax.Array, *, train: bool) -> tuple[jax.Array, jax.Array]:
        x = x.astype(self.dtype)
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

        return logits.astype(jnp.float32), value.astype(jnp.float32)


class AZDirichletNet(nnx.Module):
    """AlphaZero convolutional trunk with policy, value-Dirichlet, and Q heads."""

    def __init__(
        self,
        num_actions: int,
        observation_shape: Sequence[int],
        *,
        num_outcomes: int = 3,
        num_channels: int = 64,
        num_blocks: int = 5,
        resnet_v2: bool = True,
        dtype=jnp.float32,
        dirichlet_concentration_clip: float | None = 8.0,
        rngs: nnx.Rngs,
    ):
        height, width, input_channels = observation_shape
        self.num_actions = num_actions
        self.num_outcomes = num_outcomes
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.resnet_v2 = resnet_v2
        self.dtype = dtype
        self.dirichlet_concentration_clip = dirichlet_concentration_clip

        self.conv = nnx.Conv(input_channels, num_channels, kernel_size=3, padding="SAME", dtype=dtype, rngs=rngs)
        if not resnet_v2:
            self.bn = nnx.BatchNorm(num_channels, momentum=0.9, dtype=dtype, rngs=rngs)

        block_cls = BlockV2 if resnet_v2 else BlockV1
        self.blocks = nnx.List([block_cls(num_channels, dtype=dtype, rngs=rngs) for _ in range(num_blocks)])

        if resnet_v2:
            self.bn = nnx.BatchNorm(num_channels, momentum=0.9, dtype=dtype, rngs=rngs)

        self.policy_conv = nnx.Conv(num_channels, 2, kernel_size=1, padding="SAME", dtype=dtype, rngs=rngs)
        self.policy_bn = nnx.BatchNorm(2, momentum=0.9, dtype=dtype, rngs=rngs)
        self.policy_linear = nnx.Linear(
            height * width * 2,
            num_actions,
            dtype=dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )

        self.value_conv = nnx.Conv(num_channels, 1, kernel_size=1, padding="SAME", dtype=dtype, rngs=rngs)
        self.value_bn = nnx.BatchNorm(1, momentum=0.9, dtype=dtype, rngs=rngs)
        self.value_linear = nnx.Linear(height * width, num_channels, dtype=dtype, rngs=rngs)
        self.value_dir_out = nnx.Linear(
            num_channels,
            num_outcomes,
            dtype=dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )
        concentration_bias_init = jax.nn.initializers.constant(
            _unit_dirichlet_concentration_logit(num_outcomes)
        )
        self.value_conc_out = nnx.Linear(
            num_channels,
            1,
            dtype=dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=concentration_bias_init,
            rngs=rngs,
        )

        self.q_dir_conv = nnx.Conv(num_channels, num_outcomes, kernel_size=1, padding="SAME", dtype=dtype, rngs=rngs)
        self.q_dir_bn = nnx.BatchNorm(num_outcomes, momentum=0.9, dtype=dtype, rngs=rngs)
        self.q_dir_linear = nnx.Linear(
            height * width * num_outcomes,
            num_actions * num_outcomes,
            dtype=dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )
        self.q_conc_conv = nnx.Conv(num_channels, 1, kernel_size=1, padding="SAME", dtype=dtype, rngs=rngs)
        self.q_conc_bn = nnx.BatchNorm(1, momentum=0.9, dtype=dtype, rngs=rngs)
        self.q_conc_linear = nnx.Linear(
            height * width,
            num_actions,
            dtype=dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=concentration_bias_init,
            rngs=rngs,
        )

    def _trunk(self, x: jax.Array, *, train: bool) -> jax.Array:
        x = x.astype(self.dtype)
        x = self.conv(x)

        if not self.resnet_v2:
            x = self.bn(x, use_running_average=not train)
            x = jax.nn.relu(x)

        for block in self.blocks:
            x = block(x, train=train)

        if self.resnet_v2:
            x = self.bn(x, use_running_average=not train)
            x = jax.nn.relu(x)

        return x

    def __call__(self, x: jax.Array, *, train: bool) -> tuple[jax.Array, jax.Array, jax.Array]:
        x = self._trunk(x, train=train)

        logits = self.policy_conv(x)
        logits = self.policy_bn(logits, use_running_average=not train)
        logits = jax.nn.relu(logits)
        logits = logits.reshape((logits.shape[0], -1))
        logits = self.policy_linear(logits)

        value_features = self.value_conv(x)
        value_features = self.value_bn(value_features, use_running_average=not train)
        value_features = jax.nn.relu(value_features)
        value_features = value_features.reshape((value_features.shape[0], -1))
        value_features = self.value_linear(value_features)
        value_features = jax.nn.relu(value_features)
        alpha_v = dirichlet_from_logits(
            self.value_dir_out(value_features),
            self.value_conc_out(value_features).reshape((value_features.shape[0],)),
            concentration_clip=self.dirichlet_concentration_clip,
        )

        q_mean_logits = self.q_dir_conv(x)
        q_mean_logits = self.q_dir_bn(q_mean_logits, use_running_average=not train)
        q_mean_logits = jax.nn.relu(q_mean_logits)
        q_mean_logits = q_mean_logits.reshape((q_mean_logits.shape[0], -1))
        q_mean_logits = self.q_dir_linear(q_mean_logits).reshape(
            (x.shape[0], self.num_actions, self.num_outcomes)
        )

        q_concentration_logit = self.q_conc_conv(x)
        q_concentration_logit = self.q_conc_bn(q_concentration_logit, use_running_average=not train)
        q_concentration_logit = jax.nn.relu(q_concentration_logit)
        q_concentration_logit = q_concentration_logit.reshape((q_concentration_logit.shape[0], -1))
        q_concentration_logit = self.q_conc_linear(q_concentration_logit)
        alpha_q = dirichlet_from_logits(
            q_mean_logits,
            q_concentration_logit,
            concentration_clip=self.dirichlet_concentration_clip,
        )
        return logits.astype(jnp.float32), alpha_v, alpha_q


class ReZeroResidual(nnx.Module):
    """ReZero residual block: x + α · Linear(ReLU(x)) with α initialized to 0."""

    def __init__(
        self,
        width: int,
        *,
        dtype=jnp.float32,
        kernel_init_mode: str = "variance_scaling",
        rngs: nnx.Rngs,
    ):
        if kernel_init_mode == "orthogonal":
            kernel_init = jax.nn.initializers.orthogonal(scale=math.sqrt(2.0))
        elif kernel_init_mode == "variance_scaling":
            kernel_init = jax.nn.initializers.variance_scaling(
                scale=2.0,
                mode="fan_in",
                distribution="truncated_normal",
            )
        else:
            raise ValueError(f"unknown ReZero kernel init mode: {kernel_init_mode!r}")
        self.linear = nnx.Linear(
            width,
            width,
            dtype=dtype,
            kernel_init=kernel_init,
            rngs=rngs,
        )
        self.alpha = nnx.Param(jnp.zeros(()))

    def __call__(self, x: jax.Array) -> jax.Array:
        return x + self.alpha * self.linear(jax.nn.relu(x))


class BoardlawNet(nnx.Module):
    """Fully-connected ReZero residual net from Jones (2021),
    "Scaling Scaling Laws with Board Games" (arXiv:2104.03113)."""

    def __init__(
        self,
        num_actions: int,
        observation_shape: Sequence[int],
        *,
        width: int = 512,
        depth: int = 8,
        dtype=jnp.float32,
        rezero_kernel_init: str = "variance_scaling",
        rngs: nnx.Rngs,
    ):
        self.num_actions = num_actions
        self.width = width
        self.depth = depth
        self.dtype = dtype

        input_dim = math.prod(observation_shape)
        self.intake = nnx.Linear(input_dim, width, dtype=dtype, rngs=rngs)
        self.blocks = nnx.List(
            [
                ReZeroResidual(
                    width,
                    dtype=dtype,
                    kernel_init_mode=rezero_kernel_init,
                    rngs=rngs,
                )
                for _ in range(depth)
            ]
        )
        self.policy_head = nnx.Linear(
            width,
            num_actions,
            dtype=dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )
        self.value_head = nnx.Linear(width, 1, dtype=dtype, rngs=rngs)

    def __call__(self, x: jax.Array, *, train: bool) -> tuple[jax.Array, jax.Array]:
        del train
        x = x.astype(self.dtype)
        x = x.reshape((x.shape[0], -1))
        x = self.intake(x)
        for block in self.blocks:
            x = block(x)
        logits = self.policy_head(x)
        value = jnp.tanh(self.value_head(x)).reshape((-1,))
        return logits.astype(jnp.float32), value.astype(jnp.float32)


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
        dtype=jnp.float32,
        dirichlet_concentration_clip: float | None = 8.0,
        legacy_dirichlet_head_init: bool = False,
        rezero_kernel_init: str = "variance_scaling",
        rngs: nnx.Rngs,
    ):
        self.num_actions = num_actions
        self.num_outcomes = num_outcomes
        self.width = width
        self.depth = depth
        self.dtype = dtype
        self.dirichlet_concentration_clip = dirichlet_concentration_clip

        input_dim = math.prod(observation_shape)
        self.intake = nnx.Linear(input_dim, width, dtype=dtype, rngs=rngs)
        self.blocks = nnx.List([ReZeroResidual(width, dtype=dtype, kernel_init_mode=rezero_kernel_init, rngs=rngs) for _ in range(depth)])
        
        if legacy_dirichlet_head_init:
            self.policy_head = nnx.Linear(width, num_actions, dtype=dtype, rngs=rngs)
            self.value_dir_head = nnx.Linear(width, num_outcomes, dtype=dtype, rngs=rngs)
            self.value_conc_head = nnx.Linear(width, 1, dtype=dtype, rngs=rngs)
            self.q_dir_head = nnx.Linear(width, num_actions * num_outcomes, dtype=dtype, rngs=rngs)
            self.q_conc_head = nnx.Linear(width, num_actions, dtype=dtype, rngs=rngs)
        else:
            self.policy_head = nnx.Linear(
                width,
                num_actions,
                dtype=dtype,
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
                rngs=rngs,
            )
            self.value_dir_head = nnx.Linear(
                width,
                num_outcomes,
                dtype=dtype,
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
                dtype=dtype,
                kernel_init=jax.nn.initializers.zeros,
                bias_init=concentration_bias_init,
                rngs=rngs,
            )
            self.q_dir_head = nnx.Linear(
                width,
                num_actions * num_outcomes,
                dtype=dtype,
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
                rngs=rngs,
            )
            self.q_conc_head = nnx.Linear(
                width,
                num_actions,
                dtype=dtype,
                kernel_init=jax.nn.initializers.zeros,
                bias_init=concentration_bias_init,
                rngs=rngs,
            )

    def __call__(self, x: jax.Array, *, train: bool) -> tuple[jax.Array, jax.Array, jax.Array]:
        del train
        x = x.astype(self.dtype)
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
        return logits.astype(jnp.float32), alpha_v, alpha_q


def build_model(
    config: Config,
    *,
    num_actions: int,
    observation_shape: Sequence[int],
    rngs: nnx.Rngs,
) -> nnx.Module:
    compute_dtype = _dtype_from_name(config.model.compute_dtype)

    def dirichlet_num_outcomes() -> int:
        num_outcomes = config.env.num_outcomes
        if num_outcomes is None:
            return 2 if config.env.id == "hex" else 3
        return num_outcomes

    if config.model.network == "aznet":
        return AZNet(
            num_actions=num_actions,
            observation_shape=observation_shape,
            num_channels=config.model.num_channels,
            num_blocks=config.model.num_layers,
            resnet_v2=config.model.resnet_v2,
            dtype=compute_dtype,
            rngs=rngs,
        )
    if config.model.network == "aznet_dirichlet":
        return AZDirichletNet(
            num_actions=num_actions,
            observation_shape=observation_shape,
            num_outcomes=dirichlet_num_outcomes(),
            num_channels=config.model.num_channels,
            num_blocks=config.model.num_layers,
            resnet_v2=config.model.resnet_v2,
            dtype=compute_dtype,
            dirichlet_concentration_clip=(
                config.training.regularization.dirichlet_concentration_clip
            ),
            rngs=rngs,
        )
    if config.model.network == "boardlaw":
        return BoardlawNet(
            num_actions=num_actions,
            observation_shape=observation_shape,
            width=config.model.num_channels,
            depth=config.model.num_layers,
            dtype=compute_dtype,
            rezero_kernel_init=config.model.rezero_kernel_init,
            rngs=rngs,
        )
    if config.model.network == "boardlaw_dirichlet":
        return BoardlawDirichletNet(
            num_actions=num_actions,
            observation_shape=observation_shape,
            num_outcomes=dirichlet_num_outcomes(),
            width=config.model.num_channels,
            depth=config.model.num_layers,
            dtype=compute_dtype,
            dirichlet_concentration_clip=(
                config.training.regularization.dirichlet_concentration_clip
            ),
            legacy_dirichlet_head_init=config.model.legacy_dirichlet_head_init,
            rezero_kernel_init=config.model.rezero_kernel_init,
            rngs=rngs,
        )
    raise ValueError(f"unknown network: {config.model.network!r}")
