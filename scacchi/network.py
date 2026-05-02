from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx


def _mish(x: jax.Array) -> jax.Array:
    return x * jnp.tanh(jax.nn.softplus(x))


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
    """AlphaZero-style ResNet policy/value model."""

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


class RelativePositionAttention(nnx.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        board_shape: tuple[int, int],
        rngs: nnx.Rngs,
    ):
        if embed_dim % num_heads != 0:
            msg = "embed_dim must be divisible by num_heads."
            raise ValueError(msg)

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.board_shape = board_shape
        self.q_proj = nnx.Linear(embed_dim, embed_dim, use_bias=False, rngs=rngs)
        self.k_proj = nnx.Linear(embed_dim, embed_dim, use_bias=False, rngs=rngs)
        self.v_proj = nnx.Linear(embed_dim, embed_dim, use_bias=False, rngs=rngs)
        self.out_proj = nnx.Linear(embed_dim, embed_dim, rngs=rngs)

        rel_shape = (self._num_relative_positions(), num_heads, self.head_dim)
        scale = 1.0 / jnp.sqrt(float(embed_dim))
        normal = jax.nn.initializers.normal(stddev=scale)
        q_key, k_key, v_key = jax.random.split(rngs.params(), 3)
        self.rel_q = nnx.Param(normal(q_key, rel_shape, jnp.float32))
        self.rel_k = nnx.Param(normal(k_key, rel_shape, jnp.float32))
        self.rel_v = nnx.Param(normal(v_key, rel_shape, jnp.float32))
        self.rel_index = self._relative_index()

    def _num_relative_positions(self) -> int:
        height, width = self.board_shape
        return (2 * height - 1) * (2 * width - 1)

    def _relative_index(self) -> jax.Array:
        height, width = self.board_shape
        coords = jnp.stack(
            jnp.meshgrid(jnp.arange(height), jnp.arange(width), indexing="ij"),
            axis=-1,
        ).reshape((-1, 2))
        delta = coords[:, None, :] - coords[None, :, :]
        row = delta[..., 0] + (height - 1)
        col = delta[..., 1] + (width - 1)
        return row * (2 * width - 1) + col

    def __call__(self, x: jax.Array) -> jax.Array:
        batch_size, num_tokens, _ = x.shape
        q = self.q_proj(x).reshape(batch_size, num_tokens, self.num_heads, self.head_dim)
        k = self.k_proj(x).reshape(batch_size, num_tokens, self.num_heads, self.head_dim)
        v = self.v_proj(x).reshape(batch_size, num_tokens, self.num_heads, self.head_dim)
        q = jnp.transpose(q, (0, 2, 1, 3))
        k = jnp.transpose(k, (0, 2, 1, 3))
        v = jnp.transpose(v, (0, 2, 1, 3))

        rel_q = self.rel_q[self.rel_index]
        rel_k = self.rel_k[self.rel_index]
        rel_v = self.rel_v[self.rel_index]

        logits = jnp.einsum("bhtd,bhsd->bhts", q, k)
        logits = logits + jnp.einsum("bhtd,tshd->bhts", q, rel_k)
        logits = logits + jnp.einsum("bhsd,tshd->bhts", k, rel_q)
        logits = logits / jnp.sqrt(float(self.head_dim))

        attn = jax.nn.softmax(logits, axis=-1)
        output = jnp.einsum("bhts,bhsd->bhtd", attn, v)
        output = output + jnp.einsum("bhts,tshd->bhtd", attn, rel_v)
        output = jnp.transpose(output, (0, 2, 1, 3)).reshape(batch_size, num_tokens, self.embed_dim)
        return self.out_proj(output)


class BasicSelfAttention(nnx.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        rngs: nnx.Rngs,
    ):
        self.attn = nnx.MultiHeadAttention(
            num_heads=num_heads,
            in_features=embed_dim,
            qkv_features=embed_dim,
            out_features=embed_dim,
            dropout_rate=0.0,
            use_bias=False,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        return self.attn(x)


class TransformerBlock(nnx.Module):
    def __init__(
        self,
        *,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        board_shape: tuple[int, int],
        attention_type: str,
        rngs: nnx.Rngs,
    ):
        self.attn_norm = nnx.LayerNorm(embed_dim, use_bias=False, rngs=rngs)
        self.ffn_norm = nnx.LayerNorm(embed_dim, use_bias=False, rngs=rngs)
        if attention_type == "relative_2d":
            self.attn = RelativePositionAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                board_shape=board_shape,
                rngs=rngs,
            )
        elif attention_type == "basic_learned":
            self.attn = BasicSelfAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                rngs=rngs,
            )
        else:
            msg = f"Unknown attention_type '{attention_type}'."
            raise ValueError(msg)
        self.ffn_up = nnx.Linear(embed_dim, mlp_dim, rngs=rngs)
        self.ffn_down = nnx.Linear(mlp_dim, embed_dim, rngs=rngs)

    def __call__(self, x: jax.Array) -> jax.Array:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn_down(_mish(self.ffn_up(self.ffn_norm(x))))
        return x


class TransformerNet(nnx.Module):
    """Board transformer with 2D relative square encodings."""

    def __init__(
        self,
        num_actions: int,
        observation_shape: Sequence[int],
        *,
        num_blocks: int,
        embed_dim: int,
        num_heads: int,
        mlp_dim: int,
        value_hidden_dim: int,
        position_encoding: str,
        use_absolute_positions: bool,
        rngs: nnx.Rngs,
    ):
        height, width, input_channels = observation_shape
        self.num_actions = num_actions
        self.observation_shape = tuple(observation_shape)
        self.board_shape = (height, width)
        self.num_tokens = height * width
        self.embed_dim = embed_dim
        self.position_encoding = position_encoding
        self.use_absolute_positions = use_absolute_positions
        self.input_proj = nnx.Linear(input_channels, embed_dim, rngs=rngs)
        self.blocks = nnx.List(
            [
                TransformerBlock(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_dim=mlp_dim,
                    board_shape=self.board_shape,
                    attention_type=position_encoding,
                    rngs=rngs,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = nnx.LayerNorm(embed_dim, use_bias=False, rngs=rngs)
        self.policy_head = nnx.Linear(self.num_tokens * embed_dim, num_actions, rngs=rngs)
        self.value_hidden = nnx.Linear(self.num_tokens * embed_dim, value_hidden_dim, rngs=rngs)
        self.value_head = nnx.Linear(value_hidden_dim, 1, rngs=rngs)

        abs_shape = (self.num_tokens, embed_dim)
        scale = 1.0 / jnp.sqrt(float(embed_dim))
        normal = jax.nn.initializers.normal(stddev=scale)
        add_key, mul_key = jax.random.split(rngs.params())
        self.abs_add = nnx.Param(normal(add_key, abs_shape, jnp.float32))
        self.abs_mul = nnx.Param(normal(mul_key, abs_shape, jnp.float32))

    def __call__(self, x: jax.Array, *, train: bool) -> tuple[jax.Array, jax.Array]:
        del train
        x = x.astype(jnp.float32)
        x = x.reshape((x.shape[0], self.num_tokens, self.observation_shape[-1]))
        x = self.input_proj(x)

        use_absolute = self.use_absolute_positions or self.position_encoding == "basic_learned"
        if use_absolute:
            x = x * (1.0 + self.abs_mul[...]) + self.abs_add[...]

        for block in self.blocks:
            x = block(x)

        x = self.final_norm(x)
        flat = x.reshape((x.shape[0], self.num_tokens * self.embed_dim))
        logits = self.policy_head(flat)
        value = jnp.tanh(self.value_head(jax.nn.relu(self.value_hidden(flat))))
        value = value.reshape((-1,))
        return logits, value


def build_model(
    config: Any,
    *,
    num_actions: int,
    observation_shape: Sequence[int],
    rngs: nnx.Rngs,
) -> nnx.Module:
    model_name = str(getattr(config, "model_name", "resnet")).lower()
    if model_name == "resnet":
        return AZNet(
            num_actions=num_actions,
            observation_shape=observation_shape,
            num_channels=int(getattr(config, "num_channels", 128)),
            num_blocks=int(getattr(config, "num_layers", 6)),
            resnet_v2=bool(getattr(config, "resnet_v2", True)),
            rngs=rngs,
        )
    if model_name == "transformer":
        return TransformerNet(
            num_actions=num_actions,
            observation_shape=observation_shape,
            num_blocks=int(getattr(config, "num_layers", 6)),
            embed_dim=int(getattr(config, "embed_dim", 128)),
            num_heads=int(getattr(config, "num_heads", 8)),
            mlp_dim=int(getattr(config, "mlp_dim", 512)),
            value_hidden_dim=int(getattr(config, "value_hidden_dim", 64)),
            position_encoding=str(getattr(config, "position_encoding", "relative_2d")),
            use_absolute_positions=bool(getattr(config, "use_absolute_positions", True)),
            rngs=rngs,
        )
    msg = f"Unknown model_name '{model_name}'."
    raise ValueError(msg)
