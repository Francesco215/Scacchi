from collections.abc import Sequence

import jax
import jax.numpy as jnp
from flax import nnx


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
    """AlphaZero NN architecture."""

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
        self.policy_linear = nnx.Linear(height * width * 2, num_actions, rngs=rngs)

        # State Dirichlet value head: outputs (r_V [B,3], c_V [B]) → α_V = 1 + softplus(c_V) * softmax(r_V)
        self.value_conv = nnx.Conv(num_channels, 1, kernel_size=1, padding="SAME", rngs=rngs)
        self.value_bn = nnx.BatchNorm(1, momentum=0.9, rngs=rngs)
        self.value_linear = nnx.Linear(height * width, num_channels, rngs=rngs)
        self.value_dir_out = nnx.Linear(num_channels, 3, rngs=rngs)
        self.value_conc_out = nnx.Linear(num_channels, 1, rngs=rngs)

        # Action Dirichlet Q head: outputs (r_Q [B,A,3], c_Q [B,A]) → α_Q = 1 + softplus(c_Q) * softmax(r_Q)
        self.q_conv = nnx.Conv(num_channels, 2, kernel_size=1, padding="SAME", rngs=rngs)
        self.q_bn = nnx.BatchNorm(2, momentum=0.9, rngs=rngs)
        self.q_dir_linear = nnx.Linear(height * width * 2, num_actions * 3, rngs=rngs)
        self.q_conc_linear = nnx.Linear(height * width * 2, num_actions, rngs=rngs)

    def __call__(self, x: jax.Array, *, train: bool) -> tuple[jax.Array, jax.Array, jax.Array]:
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

        v = self.value_conv(x)
        v = self.value_bn(v, use_running_average=not train)
        v = jax.nn.relu(v)
        v = v.reshape((v.shape[0], -1))
        v = self.value_linear(v)
        v = jax.nn.relu(v)
        r_V = self.value_dir_out(v)
        c_V = self.value_conc_out(v).reshape((-1,))
        alpha_V = 1.0 + jax.nn.softplus(c_V)[..., None] * jax.nn.softmax(r_V, axis=-1)

        q = self.q_conv(x)
        q = self.q_bn(q, use_running_average=not train)
        q = jax.nn.relu(q)
        q = q.reshape((q.shape[0], -1))
        r_Q = self.q_dir_linear(q).reshape((-1, self.num_actions, 3))
        c_Q = self.q_conc_linear(q)
        alpha_Q = 1.0 + jax.nn.softplus(c_Q)[..., None] * jax.nn.softmax(r_Q, axis=-1)

        return logits, alpha_V, alpha_Q
