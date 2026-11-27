from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from flax import nnx

from .dirichlet_mctx import outcomes
from . import binned

if TYPE_CHECKING:
    from .types import BinConfig, Config


def _dtype_from_name(name: str):
    if name in {"float32", "fp32", "f32"}:
        return jnp.float32
    if name in {"bfloat16", "bf16"}:
        return jnp.bfloat16
    if name in {"float16", "fp16", "f16"}:
        return jnp.float16
    raise ValueError(f"unknown model.compute_dtype: {name!r}")


def _squared_softplus_concentration_logit(concentration: float) -> float:
    """Legacy logit whose squared-softplus transform has the requested mass."""

    if concentration <= 0.0:
        raise ValueError(f"concentration must be > 0, got {concentration}")
    return math.log(math.expm1(math.sqrt(concentration)))


_DIRICHLET_INITIAL_EXCESS_CONCENTRATION = 0.1
_LOG_CONCENTRATION_MIN = -80.0
_LOG_CONCENTRATION_MAX = 80.0


@jax.custom_jvp
def _guard_log_concentration(log_concentration: jax.Array) -> jax.Array:
    """Bound the forward value while retaining an identity recovery gradient."""

    return jnp.clip(
        log_concentration,
        _LOG_CONCENTRATION_MIN,
        _LOG_CONCENTRATION_MAX,
    )


@_guard_log_concentration.defjvp
def _guard_log_concentration_jvp(
    primals: tuple[jax.Array],
    tangents: tuple[jax.Array],
) -> tuple[jax.Array, jax.Array]:
    (log_concentration,) = primals
    (log_concentration_tangent,) = tangents
    guarded = jnp.clip(
        log_concentration,
        _LOG_CONCENTRATION_MIN,
        _LOG_CONCENTRATION_MAX,
    )
    return guarded, log_concentration_tangent


def _direct_log_concentration(concentration: float) -> float:
    """Raw direct-log head value for a requested initial concentration."""

    if not math.isfinite(concentration) or concentration <= 0.0:
        raise ValueError(f"concentration must be finite and > 0, got {concentration}")
    return math.log(concentration)


def _smooth_dirichlet_concentration_logit(
    concentration_floor: float,
    concentration_clip: float | None,
    *,
    initial_excess: float = _DIRICHLET_INITIAL_EXCESS_CONCENTRATION,
) -> float:
    """Initialize the bounded concentration just above its dumb-prior floor."""

    if initial_excess <= 0:
        raise ValueError(f"initial_excess must be > 0, got {initial_excess}")
    if concentration_clip is None:
        return math.log(math.expm1(initial_excess))
    concentration_range = concentration_clip - concentration_floor
    if concentration_range <= initial_excess:
        raise ValueError(
            "dirichlet_concentration_clip must exceed the dumb-prior floor "
            f"by more than {initial_excess}; got floor={concentration_floor}, "
            f"clip={concentration_clip}"
        )
    fraction = initial_excess / concentration_range
    return math.log(fraction / (1.0 - fraction))


def _initial_dirichlet_concentration_logit(
    parameterization: str,
    *,
    concentration_floor: float | None,
    concentration_clip: float | None,
    initial_concentration: float | None,
    default_concentration: float,
) -> float:
    """Resolve the shared head bias while preserving each network's default."""
    if parameterization == "log_concentration":
        if concentration_floor is not None:
            raise ValueError(
                "direct log-concentration heads do not support a "
                "concentration floor"
            )
    elif parameterization != "legacy":
        raise ValueError(
            "unknown Dirichlet concentration parameterization: "
            f"{parameterization!r}"
        )

    if concentration_floor is None:
        concentration = (
            default_concentration
            if initial_concentration is None
            else float(initial_concentration)
        )
        if parameterization == "log_concentration":
            return _direct_log_concentration(concentration)
        if concentration_clip is not None and concentration >= float(concentration_clip):
            raise ValueError(
                "dirichlet_concentration_clip must exceed "
                "dirichlet_initial_concentration; got "
                f"initial={concentration}, clip={concentration_clip}"
            )
        return _squared_softplus_concentration_logit(concentration)

    initial_excess = _DIRICHLET_INITIAL_EXCESS_CONCENTRATION
    if initial_concentration is not None:
        initial_excess = float(initial_concentration) - concentration_floor
        if initial_excess <= 0.0:
            raise ValueError(
                "dirichlet_initial_concentration must exceed the "
                f"configured floor {concentration_floor}; got "
                f"{initial_concentration}"
            )
    return _smooth_dirichlet_concentration_logit(
        concentration_floor,
        concentration_clip,
        initial_excess=initial_excess,
    )


def dirichlet_from_logits(
    mean_logits: jax.Array,
    concentration_logit: jax.Array,
    *,
    parameterization: str = "log_concentration",
    concentration_floor: float | None = None,
    concentration_clip: float | None = None,
) -> jax.Array:
    mean_logits = mean_logits.astype(jnp.float32)
    concentration_logit = concentration_logit.astype(jnp.float32)
    if parameterization == "log_concentration":
        if concentration_floor is not None:
            raise ValueError(
                "concentration_floor is incompatible with direct "
                "log-concentration heads"
            )
        # This is a floating-point guard, not a trainable floor or confidence
        # ceiling. Its straight-through derivative preserves the radial
        # recovery signal even if a raw parameter temporarily leaves the safe
        # exponent range.
        safe_log_concentration = _guard_log_concentration(concentration_logit)
        concentration = jnp.exp(safe_log_concentration)
    elif parameterization == "legacy":
        if concentration_floor is None:
            concentration = jax.nn.softplus(concentration_logit) ** 2
            if concentration_clip is not None:
                concentration = jnp.minimum(
                    concentration,
                    jnp.asarray(
                        concentration_clip,
                        dtype=concentration.dtype,
                    ),
                )
        else:
            floor = jnp.asarray(
                concentration_floor,
                dtype=concentration_logit.dtype,
            )
            if concentration_clip is None:
                concentration = floor + jax.nn.softplus(
                    concentration_logit
                )
            else:
                if concentration_clip <= concentration_floor:
                    raise ValueError(
                        "concentration_clip must be greater than "
                        "concentration_floor; got "
                        f"floor={concentration_floor}, "
                        f"clip={concentration_clip}"
                    )
                concentration_range = jnp.asarray(
                    concentration_clip - concentration_floor,
                    dtype=concentration_logit.dtype,
                )
                concentration = floor + concentration_range * jax.nn.sigmoid(
                    concentration_logit
                )
    else:
        raise ValueError(
            "unknown Dirichlet concentration parameterization: "
            f"{parameterization!r}"
        )
    return concentration[..., None] * jax.nn.softmax(mean_logits, axis=-1)


def policy_value_from_output(output):
    if isinstance(output, binned.BinnedOutput):
        return output.logits, 2.0 * binned.mean(output.value_probs, output.support) - 1.0
    if len(output) == 2:
        return output
    logits, alpha_v, _alpha_q = output
    return logits, outcomes.outcome_utility(outcomes.outcome_mean(alpha_v))


class BlockV1(nnx.Module):
    def __init__(self, num_channels: int, *, dtype=jnp.float32, rngs: nnx.Rngs):
        self.conv1 = nnx.Conv(num_channels, num_channels, kernel_size=(3, 3), padding="SAME", dtype=dtype, rngs=rngs)
        self.bn1 = nnx.BatchNorm(num_channels, momentum=0.9, dtype=dtype, rngs=rngs)
        self.conv2 = nnx.Conv(num_channels, num_channels, kernel_size=(3, 3), padding="SAME", dtype=dtype, rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_channels, momentum=0.9, dtype=dtype, rngs=rngs)

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        residual = x
        x = self.conv1(x)
        x = self.bn1(x, use_running_average=not train)
        x = jax.nn.relu(x)
        x = self.conv2(x)
        x = self.bn2(x, use_running_average=not train)
        return jax.nn.relu(x + residual)


_FP8_E4M3_MAX = 448.0


@jax.custom_vjp
def _fp8_conv3x3(x: jax.Array, kernel: jax.Array) -> jax.Array:
    """Dynamically-scaled FP8 (e4m3fn) 3x3 SAME convolution with scaled FP8 backward."""
    y, _ = _fp8_conv3x3_fwd(x, kernel)
    return y


def _fp8_quantize_e4m3(t: jax.Array) -> tuple[jax.Array, jax.Array]:
    # Compute amax directly in input dtype (avoids 2x memory bandwidth f32 cast on 33MB activations).
    amax = jnp.max(jnp.abs(t)).astype(jnp.float32)
    scale = jnp.where(amax > 0.0, amax / _FP8_E4M3_MAX, 1.0)
    inv_scale = jnp.where(amax > 0.0, _FP8_E4M3_MAX / amax, 1.0).astype(t.dtype)
    q = jnp.clip(t * inv_scale, -_FP8_E4M3_MAX, _FP8_E4M3_MAX).astype(jnp.float8_e4m3fn)
    return q, scale


@jax.custom_vjp
def _fp8_linear(x: jax.Array, kernel: jax.Array) -> jax.Array:
    """Dynamically-scaled FP8 (e4m3fn) dense linear matmul (fwd, dx, and dw all on FP8 Tensor Cores)."""
    y, _ = _fp8_linear_fwd(x, kernel)
    return y


def _fp8_linear_fwd(x: jax.Array, kernel: jax.Array):
    x_q, x_scale = _fp8_quantize_e4m3(x)
    w_q, w_scale = _fp8_quantize_e4m3(kernel)
    y_f32 = jax.lax.dot_general(
        x_q,
        w_q,
        dimension_numbers=(((x_q.ndim - 1,), (0,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    y = (y_f32 * (x_scale * w_scale)).astype(x.dtype)
    return y, (x_q, x_scale, w_q, w_scale)


def _fp8_linear_bwd(res, dy: jax.Array):
    x_q, x_scale, w_q, w_scale = res
    dy_q, dy_scale = _fp8_quantize_e4m3(dy)
    dx_f32 = jax.lax.dot_general(
        dy_q,
        w_q,
        dimension_numbers=(((dy_q.ndim - 1,), (1,)), ((), ())),
        preferred_element_type=jnp.float32,
    )
    dx = (dx_f32 * (dy_scale * w_scale)).astype(dy.dtype)
    batch_axes = tuple(range(x_q.ndim - 1))
    dw_f32 = jax.lax.dot_general(
        x_q,
        dy_q,
        dimension_numbers=((batch_axes, batch_axes), ((), ())),
        preferred_element_type=jnp.float32,
    )
    dw = (dw_f32 * (x_scale * dy_scale)).astype(dy.dtype)
    return dx, dw


_fp8_linear.defvjp(_fp8_linear_fwd, _fp8_linear_bwd)


def _fp8_conv3x3_fwd(x: jax.Array, kernel: jax.Array):
    x_q, x_scale = _fp8_quantize_e4m3(x)
    w_q, w_scale = _fp8_quantize_e4m3(kernel)
    y_f32 = jax.lax.conv_general_dilated(
        x_q,
        w_q,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        preferred_element_type=jnp.float32,
    )
    y = (y_f32 * (x_scale * w_scale)).astype(jnp.bfloat16)
    return y, (x_q, x_scale, w_q, w_scale)


def _fp8_conv3x3_bwd(res, dy: jax.Array):
    x_q, x_scale, w_q, w_scale = res
    dy_q, dy_scale = _fp8_quantize_e4m3(dy)
    w_q_rev = jnp.swapaxes(w_q[::-1, ::-1, :, :], 2, 3)
    dx_f32 = jax.lax.conv_general_dilated(
        dy_q,
        w_q_rev,
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
        preferred_element_type=jnp.float32,
    )
    dx = (dx_f32 * (dy_scale * w_scale)).astype(jnp.bfloat16)
    # Weight gradient from saved 1-byte x_q and dy_q via bf16 tensor cores (cuDNN optimal wgrad).
    x_bf16 = x_q.astype(jnp.bfloat16)
    dy_bf16 = dy_q.astype(jnp.bfloat16)
    _, vjp_w = jax.vjp(
        lambda w_: jax.lax.conv_general_dilated(
            x_bf16, w_, window_strides=(1, 1), padding="SAME", dimension_numbers=("NHWC", "HWIO", "NHWC")
        ),
        jnp.zeros(w_q.shape, dtype=jnp.bfloat16),
    )
    (dw_bf16,) = vjp_w(dy_bf16)
    dw = (dw_bf16.astype(jnp.float32) * (x_scale * dy_scale)).astype(jnp.bfloat16)
    return dx, dw


_fp8_conv3x3.defvjp(_fp8_conv3x3_fwd, _fp8_conv3x3_bwd)


class BlockV2(nnx.Module):
    def __init__(self, num_channels: int, *, dtype=jnp.float32, fp8_training: bool = False, rngs: nnx.Rngs):
        self.fp8_training = fp8_training
        self.bn1 = nnx.BatchNorm(num_channels, momentum=0.9, dtype=dtype, rngs=rngs)
        self.conv1 = nnx.Conv(num_channels, num_channels, kernel_size=(3, 3), padding="SAME", dtype=dtype, rngs=rngs)
        self.bn2 = nnx.BatchNorm(num_channels, momentum=0.9, dtype=dtype, rngs=rngs)
        self.conv2 = nnx.Conv(num_channels, num_channels, kernel_size=(3, 3), padding="SAME", dtype=dtype, rngs=rngs)

    def _apply_conv(self, conv: nnx.Conv, x: jax.Array, *, train: bool) -> jax.Array:
        if self.fp8_training and train and x.dtype == jnp.bfloat16:
            y = _fp8_conv3x3(x, conv.kernel[...].astype(jnp.bfloat16))
            if conv.bias is not None:
                y = y + conv.bias[...].astype(x.dtype)
            return y
        return conv(x)

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        residual = x
        x = self.bn1(x, use_running_average=not train)
        x = jax.nn.relu(x)
        x = self._apply_conv(self.conv1, x, train=train)
        x = self.bn2(x, use_running_average=not train)
        x = jax.nn.relu(x)
        x = self._apply_conv(self.conv2, x, train=train)
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

        self.conv = nnx.Conv(input_channels, num_channels, kernel_size=(3, 3), padding="SAME", dtype=dtype, rngs=rngs)
        if not resnet_v2:
            self.bn = nnx.BatchNorm(num_channels, momentum=0.9, dtype=dtype, rngs=rngs)

        if resnet_v2:
            self.blocks = nnx.List(
                [BlockV2(num_channels, dtype=dtype, rngs=rngs) for _ in range(num_blocks)]
            )
        else:
            self.blocks = nnx.List([BlockV1(num_channels, dtype=dtype, rngs=rngs) for _ in range(num_blocks)])

        if resnet_v2:
            self.bn = nnx.BatchNorm(num_channels, momentum=0.9, dtype=dtype, rngs=rngs)

        self.policy_conv = nnx.Conv(num_channels, 2, kernel_size=(1, 1), padding="SAME", dtype=dtype, rngs=rngs)
        self.policy_bn = nnx.BatchNorm(2, momentum=0.9, dtype=dtype, rngs=rngs)
        self.policy_linear = nnx.Linear(
            height * width * 2,
            num_actions,
            dtype=dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )

        self.value_conv = nnx.Conv(num_channels, 1, kernel_size=(1, 1), padding="SAME", dtype=dtype, rngs=rngs)
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


class AZBeliefNet(nnx.Module):
    """Shared AlphaZero features with Dirichlet or binned V/Q projections.

    Existing Dirichlet parameter names/shapes stay checkpoint-compatible.
    Binned heads keep feature widths fixed as the output resolution changes.
    """

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
        dirichlet_concentration_floor: float | None = None,
        dirichlet_initial_concentration: float | None = None,
        dirichlet_head_parameterization: str = "log_concentration",
        bins: BinConfig | None = None,
        fp8_training: bool = False,
        rngs: nnx.Rngs,
    ):
        height, width, input_channels = observation_shape
        self.num_actions = num_actions
        self.num_outcomes = num_outcomes
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.resnet_v2 = resnet_v2
        self.dtype = dtype
        self.bins = bins
        self.fp8_training = fp8_training
        output_size = num_outcomes if bins is None else bins.count
        self.dirichlet_head_parameterization = str(
            dirichlet_head_parameterization
        )
        self.dirichlet_concentration_floor = (
            None
            if dirichlet_concentration_floor is None
            else float(dirichlet_concentration_floor)
        )
        self.dirichlet_concentration_clip = dirichlet_concentration_clip
        self.dirichlet_initial_concentration = dirichlet_initial_concentration

        concentration_logit = _initial_dirichlet_concentration_logit(
            self.dirichlet_head_parameterization,
            concentration_floor=self.dirichlet_concentration_floor,
            concentration_clip=self.dirichlet_concentration_clip,
            initial_concentration=dirichlet_initial_concentration,
            default_concentration=float(num_outcomes),
        )
        if bins is None:
            belief_bias_init = jax.nn.initializers.zeros
        else:
            # Start every belief as a uniform density over win probability.
            initial_bias = binned.initial_logits(bins)
            def belief_bias_init(key, shape, dtype=jnp.float32, **kwargs):
                del key, kwargs
                return jnp.broadcast_to(jnp.asarray(initial_bias, dtype=dtype), (shape[0] // output_size, output_size)).reshape(shape)

        self.conv = nnx.Conv(input_channels, num_channels, kernel_size=(3, 3), padding="SAME", dtype=dtype, rngs=rngs)
        if not resnet_v2:
            self.bn = nnx.BatchNorm(num_channels, momentum=0.9, dtype=dtype, rngs=rngs)

        if resnet_v2:
            self.blocks = nnx.List(
                [BlockV2(num_channels, dtype=dtype, fp8_training=fp8_training, rngs=rngs) for _ in range(num_blocks)]
            )
        else:
            self.blocks = nnx.List([BlockV1(num_channels, dtype=dtype, rngs=rngs) for _ in range(num_blocks)])

        if resnet_v2:
            self.bn = nnx.BatchNorm(num_channels, momentum=0.9, dtype=dtype, rngs=rngs)

        self.policy_conv = nnx.Conv(num_channels, 2, kernel_size=(1, 1), padding="SAME", dtype=dtype, rngs=rngs)
        self.policy_bn = nnx.BatchNorm(2, momentum=0.9, dtype=dtype, rngs=rngs)
        self.policy_linear = nnx.Linear(
            height * width * 2,
            num_actions,
            dtype=dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )

        self.value_conv = nnx.Conv(num_channels, 1, kernel_size=(1, 1), padding="SAME", dtype=dtype, rngs=rngs)
        self.value_bn = nnx.BatchNorm(1, momentum=0.9, dtype=dtype, rngs=rngs)
        self.value_linear = nnx.Linear(height * width, num_channels, dtype=dtype, rngs=rngs)
        self.value_dir_out = nnx.Linear(
            num_channels,
            output_size,
            dtype=jnp.float32 if bins is not None else dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=belief_bias_init,
            rngs=rngs,
        )
        concentration_bias_init = jax.nn.initializers.constant(
            concentration_logit
        )
        self.value_conc_out = None if bins is not None else nnx.Linear(
            num_channels,
            1,
            dtype=dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=concentration_bias_init,
            rngs=rngs,
        )

        self.q_dir_conv = nnx.Conv(num_channels, num_outcomes, kernel_size=(1, 1), padding="SAME", dtype=dtype, rngs=rngs)
        self.q_dir_bn = nnx.BatchNorm(num_outcomes, momentum=0.9, dtype=dtype, rngs=rngs)
        self.q_dir_linear = nnx.Linear(
            height * width * num_outcomes,
            num_channels,
            dtype=dtype,
            rngs=rngs,
        )
        self.q_dir_out = nnx.Linear(
            num_channels,
            num_actions * output_size,
            dtype=jnp.float32 if bins is not None else dtype,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=belief_bias_init,
            rngs=rngs,
        )
        self.q_conc_conv = None if bins is not None else nnx.Conv(num_channels, 1, kernel_size=(1, 1), padding="SAME", dtype=dtype, rngs=rngs)
        self.q_conc_bn = None if bins is not None else nnx.BatchNorm(1, momentum=0.9, dtype=dtype, rngs=rngs)
        self.q_conc_linear = None if bins is not None else nnx.Linear(
            height * width,
            num_channels,
            dtype=dtype,
            rngs=rngs,
        )
        self.q_conc_out = None if bins is not None else nnx.Linear(
            num_channels,
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

    def __call__(self, x: jax.Array, *, train: bool) -> tuple[jax.Array, jax.Array, jax.Array] | binned.BinnedOutput:
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
        value_logits = self.value_dir_out(value_features)

        q_mean_logits = self.q_dir_conv(x)
        q_mean_logits = self.q_dir_bn(q_mean_logits, use_running_average=not train)
        q_mean_logits = jax.nn.relu(q_mean_logits)
        q_mean_logits = q_mean_logits.reshape((q_mean_logits.shape[0], -1))
        q_mean_logits = self.q_dir_linear(q_mean_logits)
        q_mean_logits = jax.nn.relu(q_mean_logits)
        q_mean_logits = self.q_dir_out(q_mean_logits).reshape(
            (x.shape[0], self.num_actions, -1)
        )

        if self.bins is not None:
            return binned.BinnedOutput(
                logits.astype(jnp.float32),
                binned.from_logits(value_logits),
                binned.from_logits(q_mean_logits),
                jnp.asarray(binned.support(self.bins)),
            )

        alpha_v = dirichlet_from_logits(
            value_logits,
            self.value_conc_out(value_features).reshape((value_features.shape[0],)),
            parameterization=self.dirichlet_head_parameterization,
            concentration_floor=self.dirichlet_concentration_floor,
            concentration_clip=self.dirichlet_concentration_clip,
        )

        q_concentration_logit = self.q_conc_conv(x)
        q_concentration_logit = self.q_conc_bn(q_concentration_logit, use_running_average=not train)
        q_concentration_logit = jax.nn.relu(q_concentration_logit)
        q_concentration_logit = q_concentration_logit.reshape((q_concentration_logit.shape[0], -1))
        q_concentration_logit = self.q_conc_linear(q_concentration_logit)
        q_concentration_logit = jax.nn.relu(q_concentration_logit)
        q_concentration_logit = self.q_conc_out(q_concentration_logit)
        alpha_q = dirichlet_from_logits(
            q_mean_logits,
            q_concentration_logit,
            parameterization=self.dirichlet_head_parameterization,
            concentration_floor=self.dirichlet_concentration_floor,
            concentration_clip=self.dirichlet_concentration_clip,
        )
        return logits.astype(jnp.float32), alpha_v, alpha_q


class AZDirichletNet(AZBeliefNet):
    """AlphaZero features with positive Dirichlet V/Q heads."""


class AZBinnedNet(AZBeliefNet):
    """AlphaZero features with categorical beliefs over win probability."""

    def __init__(self, *args, bins: BinConfig, **kwargs):
        if kwargs.get("num_outcomes", 2) != 2:
            raise ValueError("AZBinnedNet requires binary outcomes")
        kwargs["num_outcomes"] = 2
        super().__init__(*args, bins=bins, **kwargs)


class RMSNorm(nnx.Module):
    """Channelwise RMSNorm with affine scale+bias computed in float32 (no batch stats)."""

    def __init__(self, num_channels: int, *, eps: float = 1e-5):
        self.eps = float(eps)
        self.scale = nnx.Param(jnp.ones((num_channels,), dtype=jnp.float32))
        self.bias = nnx.Param(jnp.zeros((num_channels,), dtype=jnp.float32))

    def __call__(self, x: jax.Array) -> jax.Array:
        xf = x.astype(jnp.float32)
        rms = jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + self.eps)
        return (xf * rms * self.scale[...] + self.bias[...]).astype(x.dtype)


def _belief_bias_initializer(bins: BinConfig):
    initial_bias = binned.initial_logits(bins)

    def init_fn(key, shape, dtype=jnp.float32, **kwargs):
        del key, kwargs
        return jnp.broadcast_to(jnp.asarray(initial_bias, dtype=dtype), shape)

    return init_fn


def _canonicalize_hex_obs(
    x: jax.Array,
    *,
    enabled: bool,
    num_actions: int,
) -> tuple[jax.Array, jax.Array | None]:
    """Canonicalize Hex observations by transposing White-to-move boards to Left-Right.

    Always appends an `on_board = 1.0` channel (matching KataGo's `bin_input[:, 0]`) so:
      1. `3x3` SAME convolutions detect exact board edges/corners from layer 0, and
      2. empty board patches never produce exact-zero activations into RMSNorm.
    """
    on_board = jnp.ones_like(x[..., :1])
    if (
        enabled
        and x.ndim == 4
        and x.shape[1] == x.shape[2]
        and x.shape[3] == 4
        and num_actions == x.shape[1] * x.shape[2] + 1
    ):
        is_white = x[:, 0, 0, 2] > 0.5
        stones = x[..., :2]
        stones = jnp.where(
            is_white[:, None, None, None],
            jnp.swapaxes(stones, 1, 2),
            stones,
        )
        return jnp.concatenate([stones, on_board], axis=-1), is_white
    return jnp.concatenate([x, on_board], axis=-1), None


def _global_pool_mean_max(x: jax.Array) -> jax.Array:
    """Concatenate spatial mean and max over (H, W) -> [B, 2C]."""
    mean = jnp.mean(x, axis=(1, 2))
    mx = jnp.max(x, axis=(1, 2))
    return jnp.concatenate([mean, mx], axis=-1)


class KataPolicyQHead(nnx.Module):
    """KataGo v16/v17 local + global-pooled 1x1 spatial Policy and Q belief head."""

    def __init__(
        self,
        num_channels: int,
        num_actions: int,
        height: int,
        width: int,
        bins: BinConfig,
        *,
        p_channels: int = 48,
        g_channels: int = 48,
        dtype=jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.num_actions = int(num_actions)
        self.height = int(height)
        self.width = int(width)
        self.cells = self.height * self.width
        self.has_pass = self.num_actions == self.cells + 1
        if self.num_actions not in {self.cells, self.cells + 1}:
            raise ValueError(
                f"KataPolicyQHead expects num_actions in {{{self.cells}, {self.cells + 1}}}, got {num_actions}"
            )
        self.bins = bins
        self.dtype = dtype

        self.conv1p = nnx.Conv(num_channels, p_channels, kernel_size=(1, 1), use_bias=False, dtype=dtype, rngs=rngs)
        self.conv1g = nnx.Conv(num_channels, g_channels, kernel_size=(1, 1), use_bias=False, dtype=dtype, rngs=rngs)
        self.norm_g = RMSNorm(g_channels)
        self.linear_g = nnx.Linear(2 * g_channels, p_channels, use_bias=False, dtype=dtype, rngs=rngs)
        self.norm_p = RMSNorm(p_channels)

        belief_init = _belief_bias_initializer(bins)
        self.policy_conv = nnx.Conv(
            p_channels,
            1,
            kernel_size=(1, 1),
            dtype=jnp.float32,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )
        self.q_conv = nnx.Conv(
            p_channels,
            bins.count,
            kernel_size=(1, 1),
            dtype=jnp.float32,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=belief_init,
            rngs=rngs,
        )
        if self.has_pass:
            self.pass_policy = nnx.Linear(
                2 * g_channels,
                1,
                dtype=jnp.float32,
                kernel_init=jax.nn.initializers.zeros,
                bias_init=jax.nn.initializers.zeros,
                rngs=rngs,
            )
            self.pass_q = nnx.Linear(
                2 * g_channels,
                bins.count,
                dtype=jnp.float32,
                kernel_init=jax.nn.initializers.zeros,
                bias_init=belief_init,
                rngs=rngs,
            )
        else:
            self.pass_policy = None
            self.pass_q = None

    def __call__(self, x: jax.Array, *, is_white: jax.Array | None = None) -> tuple[jax.Array, jax.Array]:
        # Fuse 1x1 conv1p and conv1g into a single matmul (reads x once; preserves param names).
        p_ch = int(self.conv1p.kernel[...].shape[-1])
        w_pg = jnp.concatenate(
            [self.conv1p.kernel[...][0, 0], self.conv1g.kernel[...][0, 0]],
            axis=-1,
        ).astype(x.dtype)
        pg = jnp.matmul(x, w_pg, preferred_element_type=x.dtype)
        outp, outg_raw = pg[..., :p_ch], pg[..., p_ch:]
        outg = jax.nn.silu(self.norm_g(outg_raw))
        pooled_g = _global_pool_mean_max(outg)
        g_bias = self.linear_g(pooled_g)[:, None, None, :]
        h = jax.nn.silu(self.norm_p(outp + g_bias))

        h_f32 = h.astype(jnp.float32)
        pol_grid = self.policy_conv(h_f32)[..., 0]  # [B, H, W]
        q_grid = self.q_conv(h_f32)                 # [B, H, W, K]
        if is_white is not None:
            pol_grid = jnp.where(is_white[:, None, None], jnp.swapaxes(pol_grid, 1, 2), pol_grid)
            q_grid = jnp.where(is_white[:, None, None, None], jnp.swapaxes(q_grid, 1, 2), q_grid)

        batch = x.shape[0]
        logits = pol_grid.reshape((batch, self.cells))
        q_logits = q_grid.reshape((batch, self.cells, self.bins.count))
        if self.has_pass:
            assert self.pass_policy is not None and self.pass_q is not None
            pooled_f32 = pooled_g.astype(jnp.float32)
            logits = jnp.concatenate([logits, self.pass_policy(pooled_f32)], axis=-1)
            q_logits = jnp.concatenate([q_logits, self.pass_q(pooled_f32)[:, None, :]], axis=1)
        return logits, q_logits


class KataValueHead(nnx.Module):
    """KataGo global-pooled Value belief head (board-size and rot180 invariant)."""

    def __init__(
        self,
        num_channels: int,
        bins: BinConfig,
        *,
        v_channels: int = 48,
        hidden_dim: int = 128,
        dtype=jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.conv1 = nnx.Conv(num_channels, v_channels, kernel_size=(1, 1), use_bias=False, dtype=dtype, rngs=rngs)
        self.norm1 = RMSNorm(v_channels)
        self.linear1 = nnx.Linear(2 * v_channels, hidden_dim, dtype=dtype, rngs=rngs)
        self.out = nnx.Linear(
            hidden_dim,
            bins.count,
            dtype=jnp.float32,
            kernel_init=jax.nn.initializers.zeros,
            bias_init=_belief_bias_initializer(bins),
            rngs=rngs,
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        v = jax.nn.silu(self.norm1(self.conv1(x)))
        pooled = _global_pool_mean_max(v)
        h = jax.nn.silu(self.linear1(pooled))
        return self.out(h.astype(jnp.float32))


class KataConvBlock(nnx.Module):
    """Pre-RMSNorm SiLU ResNet-v2 block with optional KataGo Global Pooling Bias."""

    def __init__(
        self,
        num_channels: int,
        *,
        gpool_channels: int = 0,
        residual_scale: float = 1.0,
        dtype=jnp.bfloat16,
        fp8_training: bool = False,
        rngs: nnx.Rngs,
    ):
        self.gpool_channels = int(gpool_channels)
        self.main_channels = int(num_channels) - self.gpool_channels
        self.fp8_training = bool(fp8_training)
        self.norm1 = RMSNorm(num_channels)
        self.conv1 = nnx.Conv(
            num_channels,
            self.main_channels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            dtype=dtype,
            rngs=rngs,
        )
        if self.gpool_channels > 0:
            self.conv1g = nnx.Conv(
                num_channels,
                self.gpool_channels,
                kernel_size=(3, 3),
                padding="SAME",
                use_bias=False,
                dtype=dtype,
                rngs=rngs,
            )
            self.norm_g = RMSNorm(self.gpool_channels)
            self.linear_g = nnx.Linear(
                2 * self.gpool_channels,
                self.main_channels,
                use_bias=False,
                dtype=dtype,
                rngs=rngs,
            )
        else:
            self.conv1g = None
            self.norm_g = None
            self.linear_g = None
        self.norm2 = RMSNorm(self.main_channels)
        out_init = jax.nn.initializers.variance_scaling(
            scale=float(residual_scale) ** 2,
            mode="fan_in",
            distribution="truncated_normal",
        )
        self.conv2 = nnx.Conv(
            self.main_channels,
            num_channels,
            kernel_size=(3, 3),
            padding="SAME",
            use_bias=False,
            dtype=dtype,
            kernel_init=out_init,
            rngs=rngs,
        )

    def _apply_conv3x3(self, conv: nnx.Conv, x: jax.Array, *, train: bool) -> jax.Array:
        if self.fp8_training and train and x.dtype == jnp.bfloat16:
            return _fp8_conv3x3(x, conv.kernel[...].astype(jnp.bfloat16))
        return conv(x)

    def __call__(self, x: jax.Array, *, train: bool) -> jax.Array:
        h = jax.nn.silu(self.norm1(x))
        main = self._apply_conv3x3(self.conv1, h, train=train)
        if self.gpool_channels > 0:
            assert self.conv1g is not None and self.norm_g is not None and self.linear_g is not None
            g = jax.nn.silu(self.norm_g(self._apply_conv3x3(self.conv1g, h, train=train)))
            main = main + self.linear_g(_global_pool_mean_max(g))[:, None, None, :]
        main = jax.nn.silu(self.norm2(main))
        return x + self._apply_conv3x3(self.conv2, main, train=train)


class AZKataConvBinnedNet(nnx.Module):
    """Modernized KataGo ConvNet (RMSNorm, SiLU, Global Pooling, spatial Q/Policy heads)."""

    def __init__(
        self,
        num_actions: int,
        observation_shape: Sequence[int],
        *,
        num_channels: int = 128,
        num_blocks: int = 6,
        bins: BinConfig,
        dtype=jnp.bfloat16,
        fp8_training: bool = False,
        canonicalize_hex_color: bool = True,
        rngs: nnx.Rngs,
    ):
        height, width, input_channels = observation_shape
        self.num_actions = int(num_actions)
        self.num_channels = int(num_channels)
        self.num_blocks = int(num_blocks)
        self.bins = bins
        self.dtype = dtype
        self.fp8_training = bool(fp8_training)
        self.canonicalize_hex_color = bool(canonicalize_hex_color)
        use_canon = (
            self.canonicalize_hex_color
            and height == width
            and input_channels == 4
            and num_actions == height * width + 1
        )
        stem_in = 3 if use_canon else (input_channels + 1)

        self.conv = nnx.Conv(
            stem_in, num_channels, kernel_size=(3, 3), padding="SAME", use_bias=True, dtype=dtype, rngs=rngs
        )
        gpool_ch = max(16, num_channels // 4)
        # Place global pooling blocks evenly across the trunk (~1/3 of blocks, matching KataGo b6c96).
        gpool_indices = {i for i in range(num_blocks) if (i % 3 == 1)} if num_blocks >= 2 else set()
        res_scale = 1.0 / math.sqrt(max(num_blocks, 1))
        self.blocks = nnx.List(
            [
                KataConvBlock(
                    num_channels,
                    gpool_channels=gpool_ch if i in gpool_indices else 0,
                    residual_scale=res_scale,
                    dtype=dtype,
                    fp8_training=fp8_training,
                    rngs=rngs,
                )
                for i in range(num_blocks)
            ]
        )
        self.final_norm = RMSNorm(num_channels)
        self.policy_q_head = KataPolicyQHead(
            num_channels, num_actions, height, width, bins, p_channels=48, g_channels=48, dtype=dtype, rngs=rngs
        )
        self.value_head = KataValueHead(
            num_channels, bins, v_channels=48, hidden_dim=128, dtype=dtype, rngs=rngs
        )

    def __call__(self, x: jax.Array, *, train: bool) -> binned.BinnedOutput:
        x_in, is_white = _canonicalize_hex_obs(
            x, enabled=self.canonicalize_hex_color, num_actions=self.num_actions
        )
        h = self.conv(x_in.astype(self.dtype))
        for block in self.blocks:
            h = block(h, train=train)
        h = jax.nn.silu(self.final_norm(h))
        logits, q_logits = self.policy_q_head(h, is_white=is_white)
        v_logits = self.value_head(h)
        return binned.BinnedOutput(
            logits.astype(jnp.float32),
            binned.from_logits(v_logits),
            binned.from_logits(q_logits),
            jnp.asarray(binned.support(self.bins)),
        )


class KataRoPEAttention(nnx.Module):
    """2026 KataGo Multi-Head Self-Attention with QK-Norm and per-head 2D Learnable RoPE."""

    def __init__(
        self,
        channels: int,
        num_heads: int,
        height: int,
        width: int,
        *,
        qk_norm: bool = True,
        residual_scale: float = 1.0,
        dtype=jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        if channels % num_heads != 0:
            raise ValueError(f"channels={channels} must be divisible by num_heads={num_heads}")
        head_dim = channels // num_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim={head_dim} must be even for 2D learnable RoPE")
        self.channels = int(channels)
        self.num_heads = int(num_heads)
        self.head_dim = int(head_dim)
        self.height = int(height)
        self.width = int(width)
        self.qk_norm = bool(qk_norm)
        self.dtype = dtype

        self.norm = RMSNorm(channels)
        # Keep Q, K, V separate so Muon orthogonalizes each projection independently.
        self.q_proj = nnx.Linear(channels, channels, use_bias=False, dtype=dtype, rngs=rngs)
        self.k_proj = nnx.Linear(channels, channels, use_bias=False, dtype=dtype, rngs=rngs)
        self.v_proj = nnx.Linear(channels, channels, use_bias=False, dtype=dtype, rngs=rngs)
        if self.qk_norm:
            self.q_norm = RMSNorm(head_dim)
            self.k_norm = RMSNorm(head_dim)
        else:
            self.q_norm = None
            self.k_norm = None
        out_init = jax.nn.initializers.variance_scaling(
            scale=float(residual_scale) ** 2,
            mode="fan_in",
            distribution="truncated_normal",
        )
        self.out_proj = nnx.Linear(
            channels, channels, use_bias=False, dtype=dtype, kernel_init=out_init, rngs=rngs
        )

        # Learnable 2D RoPE frequencies: log-uniform in [1/50, 1] rad/cell with random sign
        # (matches lightvector/KataGo model_pytorch.py).
        num_pairs = head_dim // 2
        k1, k2 = jax.random.split(rngs.params())
        mags = jnp.exp(
            jax.random.uniform(
                k1,
                (num_heads, num_pairs, 2),
                minval=math.log(1.0 / 50.0),
                maxval=math.log(1.0),
                dtype=jnp.float32,
            )
        )
        signs = jax.random.choice(
            k2, jnp.array([-1.0, 1.0], dtype=jnp.float32), shape=(num_heads, num_pairs, 2)
        )
        self.rope_freqs = nnx.Param(mags * signs)

    def _apply_rope_2d(self, t: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
        half = self.head_dim // 2
        t0, t1 = t[..., :half], t[..., half:]
        return jnp.concatenate([t0 * cos - t1 * sin, t0 * sin + t1 * cos], axis=-1)

    def __call__(self, x: jax.Array, s_x: jax.Array, s_y: jax.Array) -> jax.Array:
        b, s, c = x.shape
        xn = self.norm(x)
        # Fuse Q, K, V projections into a single matmul while keeping separate params for Muon.
        w_qkv = jnp.concatenate(
            [self.q_proj.kernel[...], self.k_proj.kernel[...], self.v_proj.kernel[...]],
            axis=-1,
        ).astype(xn.dtype)
        qkv = jnp.matmul(xn, w_qkv, preferred_element_type=xn.dtype)
        q, k, v = jnp.split(qkv, 3, axis=-1)
        q = q.reshape((b, s, self.num_heads, self.head_dim))
        k = k.reshape((b, s, self.num_heads, self.head_dim))
        v = v.reshape((b, s, self.num_heads, self.head_dim))
        if self.qk_norm:
            assert self.q_norm is not None and self.k_norm is not None
            q = self.q_norm(q)
            k = self.k_norm(k)

        freqs = self.rope_freqs[...]
        angle = s_x[:, None, None] * freqs[None, :, :, 0] + s_y[:, None, None] * freqs[None, :, :, 1]
        cos = jnp.cos(angle)[None, :, :, :].astype(self.dtype)
        sin = jnp.sin(angle)[None, :, :, :].astype(self.dtype)
        q = self._apply_rope_2d(q, cos, sin)
        k = self._apply_rope_2d(k, cos, sin)

        use_cudnn = (
            jax.default_backend() == "gpu"
            and q.dtype in (jnp.bfloat16, jnp.float16)
            and self.head_dim <= 128
            and self.head_dim % 8 == 0
            and s % 2 == 0
        )
        attn_out = jax.nn.dot_product_attention(
            q, k, v, implementation="cudnn" if use_cudnn else None
        )
        return self.out_proj(attn_out.reshape((b, s, c)))


class KataSwiGLU(nnx.Module):
    """Pre-RMSNorm SwiGLU feed-forward sublayer."""

    def __init__(
        self,
        channels: int,
        ffn_channels: int,
        *,
        residual_scale: float = 1.0,
        dtype=jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.norm = RMSNorm(channels)
        self.w_gate = nnx.Linear(channels, ffn_channels, use_bias=False, dtype=dtype, rngs=rngs)
        self.w_up = nnx.Linear(channels, ffn_channels, use_bias=False, dtype=dtype, rngs=rngs)
        out_init = jax.nn.initializers.variance_scaling(
            scale=float(residual_scale) ** 2,
            mode="fan_in",
            distribution="truncated_normal",
        )
        self.w_down = nnx.Linear(
            ffn_channels, channels, use_bias=False, dtype=dtype, kernel_init=out_init, rngs=rngs
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        xn = self.norm(x)
        # Fuse W_gate and W_up into a single matmul while keeping separate params for Muon.
        w_gu = jnp.concatenate(
            [self.w_gate.kernel[...], self.w_up.kernel[...]],
            axis=-1,
        ).astype(xn.dtype)
        gu = jnp.matmul(xn, w_gu, preferred_element_type=xn.dtype)
        g, u = jnp.split(gu, 2, axis=-1)
        return self.w_down(jax.nn.silu(g) * u)


class KataNestedBottleneckTransformerBlock(nnx.Module):
    """2026 KataGo Transformer block: flat when `inner_blocks==1 and mid==trunk`, else nested bottleneck."""

    def __init__(
        self,
        trunk_channels: int,
        mid_channels: int,
        ffn_channels: int,
        num_heads: int,
        height: int,
        width: int,
        *,
        inner_blocks: int = 2,
        qk_norm: bool = True,
        outer_residual_scale: float = 1.0,
        dtype=jnp.bfloat16,
        rngs: nnx.Rngs,
    ):
        self.use_bottleneck = not (int(inner_blocks) == 1 and int(mid_channels) == int(trunk_channels))
        if self.use_bottleneck:
            self.norm_in = RMSNorm(trunk_channels)
            self.proj_in = nnx.Linear(trunk_channels, mid_channels, use_bias=False, dtype=dtype, rngs=rngs)
            inner_scale = 1.0 / math.sqrt(max(2 * inner_blocks, 1))
        else:
            self.norm_in = None
            self.proj_in = None
            inner_scale = float(outer_residual_scale) / math.sqrt(2.0)

        self.attns = nnx.List(
            [
                KataRoPEAttention(
                    mid_channels,
                    num_heads,
                    height,
                    width,
                    qk_norm=qk_norm,
                    residual_scale=inner_scale,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(inner_blocks)
            ]
        )
        self.ffns = nnx.List(
            [
                KataSwiGLU(
                    mid_channels,
                    ffn_channels,
                    residual_scale=inner_scale,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(inner_blocks)
            ]
        )

        if self.use_bottleneck:
            self.norm_out = RMSNorm(mid_channels)
            out_init = jax.nn.initializers.variance_scaling(
                scale=float(outer_residual_scale) ** 2,
                mode="fan_in",
                distribution="truncated_normal",
            )
            self.proj_out = nnx.Linear(
                mid_channels, trunk_channels, use_bias=False, dtype=dtype, kernel_init=out_init, rngs=rngs
            )
        else:
            self.norm_out = None
            self.proj_out = None

    def __call__(self, x: jax.Array, s_x: jax.Array, s_y: jax.Array) -> jax.Array:
        if not self.use_bottleneck:
            h = x
            for attn, ffn in zip(self.attns, self.ffns):
                h = h + attn(h, s_x, s_y)
                h = h + ffn(h)
            return h
        assert self.norm_in is not None and self.proj_in is not None
        assert self.norm_out is not None and self.proj_out is not None
        h = self.proj_in(jax.nn.silu(self.norm_in(x)))
        for attn, ffn in zip(self.attns, self.ffns):
            h = h + attn(h, s_x, s_y)
            h = h + ffn(h)
        return x + self.proj_out(jax.nn.silu(self.norm_out(h)))


class AZKataTransformerBinnedNet(nnx.Module):
    """2026 KataGo Transformer (3x3 stem + QK-Norm 2D-RoPE SwiGLU blocks + spatial Policy/Q)."""

    def __init__(
        self,
        num_actions: int,
        observation_shape: Sequence[int],
        *,
        num_channels: int = 192,
        num_blocks: int = 4,
        mid_channels: int | None = None,
        ffn_channels: int | None = None,
        num_heads: int = 6,
        inner_blocks: int = 1,
        qk_norm: bool = True,
        infer_chunk_size: int = 512,
        bins: BinConfig,
        dtype=jnp.bfloat16,
        fp8_training: bool = False,
        canonicalize_hex_color: bool = True,
        rngs: nnx.Rngs,
    ):
        height, width, input_channels = observation_shape
        self.height = int(height)
        self.width = int(width)
        self.num_actions = int(num_actions)
        self.num_channels = int(num_channels)
        self.num_blocks = int(num_blocks)
        self.infer_chunk_size = int(infer_chunk_size)
        self.bins = bins
        self.dtype = dtype
        self.fp8_training = bool(fp8_training)
        self.canonicalize_hex_color = bool(canonicalize_hex_color)
        use_canon = (
            self.canonicalize_hex_color
            and height == width
            and input_channels == 4
            and num_actions == height * width + 1
        )
        stem_in = 3 if use_canon else (input_channels + 1)

        if mid_channels is None:
            mid_ch = int(num_channels) if int(inner_blocks) == 1 else int(num_channels) // 2
        else:
            mid_ch = int(mid_channels)
        if ffn_channels is None:
            ffn_ch = int(round(mid_ch * 2.5)) if int(inner_blocks) == 1 else int(round(num_channels * 1.25))
        else:
            ffn_ch = int(ffn_channels)

        self.conv = nnx.Conv(
            stem_in, num_channels, kernel_size=(3, 3), padding="SAME", use_bias=True, dtype=dtype, rngs=rngs
        )
        outer_scale = 1.0 / math.sqrt(max(num_blocks, 1))
        self.blocks = nnx.List(
            [
                KataNestedBottleneckTransformerBlock(
                    num_channels,
                    mid_ch,
                    ffn_ch,
                    num_heads,
                    height,
                    width,
                    inner_blocks=inner_blocks,
                    qk_norm=qk_norm,
                    outer_residual_scale=outer_scale,
                    dtype=dtype,
                    rngs=rngs,
                )
                for _ in range(num_blocks)
            ]
        )
        self.final_norm = RMSNorm(num_channels)
        self.policy_q_head = KataPolicyQHead(
            num_channels, num_actions, height, width, bins, p_channels=48, g_channels=48, dtype=dtype, rngs=rngs
        )
        self.value_head = KataValueHead(
            num_channels, bins, v_channels=48, hidden_dim=128, dtype=dtype, rngs=rngs
        )

    def _forward_chunk(
        self, x: jax.Array, s_x: jax.Array, s_y: jax.Array, *, train: bool = False
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        del train
        x_in, is_white = _canonicalize_hex_obs(
            x, enabled=self.canonicalize_hex_color, num_actions=self.num_actions
        )
        h = self.conv(x_in.astype(self.dtype))
        b, hh, ww, c = h.shape
        seq = h.reshape((b, hh * ww, c))
        for block in self.blocks:
            seq = block(seq, s_x, s_y)
        h = jax.nn.silu(self.final_norm(seq)).reshape((b, hh, ww, c))
        logits, q_logits = self.policy_q_head(h, is_white=is_white)
        v_logits = self.value_head(h)
        return logits, v_logits, q_logits

    def __call__(self, x: jax.Array, *, train: bool) -> binned.BinnedOutput:
        hh, ww = x.shape[1], x.shape[2]
        idx = jnp.arange(hh * ww, dtype=jnp.int32)
        s_y = (idx // ww).astype(jnp.float32) - 0.5 * (hh - 1)
        s_x = (idx % ww).astype(jnp.float32) - 0.5 * (ww - 1)

        chunk = self.infer_chunk_size
        batch = x.shape[0]
        if (not train) and chunk > 0 and batch > chunk and (batch % chunk == 0):
            n_chunks = batch // chunk
            x_chunks = x.reshape((n_chunks, chunk) + x.shape[1:])
            logits_c, v_c, q_c = jax.lax.map(
                lambda xc: self._forward_chunk(xc, s_x, s_y, train=False), x_chunks
            )
            logits = logits_c.reshape((batch, self.num_actions))
            v_logits = v_c.reshape((batch, self.bins.count))
            q_logits = q_c.reshape((batch, self.num_actions, self.bins.count))
        else:
            logits, v_logits, q_logits = self._forward_chunk(x, s_x, s_y, train=train)

        return binned.BinnedOutput(
            logits.astype(jnp.float32),
            binned.from_logits(v_logits),
            binned.from_logits(q_logits),
            jnp.asarray(binned.support(self.bins)),
        )


def _unit_rms(x: jax.Array, eps: float = 1e-5) -> jax.Array:
    """Normalize vectors along the last axis to unit RMS (computed in float32)."""
    xf = x.astype(jnp.float32)
    return (xf * jax.lax.rsqrt(jnp.mean(jnp.square(xf), axis=-1, keepdims=True) + eps)).astype(x.dtype)


def _clip_heads(x: jax.Array, threshold: float) -> jax.Array:
    """Inkling-2 head L2 norm cap (shrinks large vectors, leaves small vectors unchanged)."""
    xf = x.astype(jnp.float32)
    sq = jnp.sum(jnp.square(xf), axis=-1, keepdims=True)
    norm = jnp.sqrt(jnp.maximum(sq, 1e-24))
    return (xf * jnp.minimum(1.0, float(threshold) / (norm + 1e-6))).astype(x.dtype)


def _read_depth_nostack(queries: jax.Array, norm_sources: tuple[jax.Array, ...]) -> jax.Array:
    """Attend across depth over pre-normalized unit-RMS sources without stacking full tensors.

    Args:
        queries: [Q, C] float32 learned query vectors.
        norm_sources: tuple of N pre-normalized [B, S, C] arrays.
    Returns:
        [B, S, Q, C] weighted combinations in the source dtype.
    """
    dtype = norm_sources[0].dtype
    scale = jnp.asarray(queries.shape[-1] ** -0.5, dtype=jnp.float32)
    if len(norm_sources) == 1:
        return jnp.broadcast_to(
            norm_sources[0][:, :, None, :],
            norm_sources[0].shape[:2] + (queries.shape[0], queries.shape[1]),
        )
    if queries.shape[0] == 1:
        # Avoid launching N tiny Q=1 cuBLAS GEMMs; elementwise reduction fuses cleanly in XLA.
        q0 = queries[0]  # [C] f32
        logits = jnp.stack(
            [jnp.sum(src.astype(jnp.float32) * q0, axis=-1) * scale for src in norm_sources],
            axis=-1,
        )  # [B, S, N]
        weights = jax.nn.softmax(logits, axis=-1).astype(dtype)
        out = weights[..., 0:1] * norm_sources[0]
        for idx in range(1, len(norm_sources)):
            out = out + weights[..., idx : idx + 1] * norm_sources[idx]
        return out[:, :, None, :]
    q = queries.astype(dtype)
    logits_list = [
        jnp.einsum("bsc,qc->bsq", src, q, preferred_element_type=jnp.float32) * scale
        for src in norm_sources
    ]
    weights = jax.nn.softmax(jnp.stack(logits_list, axis=-1), axis=-1).astype(dtype)  # [B, S, Q, N]
    out = weights[..., 0:1] * norm_sources[0][:, :, None, :]
    for idx in range(1, len(norm_sources)):
        out = out + weights[..., idx : idx + 1] * norm_sources[idx][:, :, None, :]
    return out


class InklingShortConv2D(nnx.Module):
    """2D Hex-board spatial analog of Inkling-2's residual depthwise ShortConv.

    Parameter is named `dw_weight` (not `kernel`) so Muon skips depthwise filters
    and routes them to Adam.
    """

    def __init__(self, width: int, height: int, width_board: int, *, dtype=jnp.bfloat16, rngs: nnx.Rngs):
        self.width = int(width)
        self.height = int(height)
        self.width_board = int(width_board)
        self.dtype = dtype
        self.dw_weight = nnx.Param(
            jax.random.normal(rngs.params(), (3, 3, 1, self.width), dtype=jnp.float32) * 0.02
        )

    def __call__(self, x: jax.Array) -> jax.Array:
        b, s, c = x.shape
        img = x.reshape((b, self.height, self.width_board, c))
        y = jax.lax.conv_general_dilated(
            img,
            self.dw_weight[...].astype(x.dtype),
            window_strides=(1, 1),
            padding="SAME",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
            feature_group_count=self.width,
        )
        inv_s = jnp.asarray(1.0 / math.sqrt(2.0), dtype=x.dtype)
        return (x + y.reshape((b, s, c))) * inv_s


class InklingMessageGate(nnx.Module):
    """Low-rank recurrent loop gate initialized to exact identity on x."""

    def __init__(self, width: int, rank: int, *, dtype=jnp.bfloat16, rngs: nnx.Rngs):
        self.down = nnx.Linear(
            2 * width,
            rank,
            use_bias=False,
            dtype=dtype,
            kernel_init=jax.nn.initializers.normal(0.02),
            rngs=rngs,
        )
        self.forget = nnx.Linear(
            rank,
            width,
            use_bias=False,
            dtype=dtype,
            kernel_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )
        self.gain = nnx.Linear(
            rank,
            width,
            use_bias=False,
            dtype=dtype,
            kernel_init=jax.nn.initializers.zeros,
            rngs=rngs,
        )

    def __call__(self, x: jax.Array, msg: jax.Array | None) -> jax.Array:
        if msg is None:
            return x
        # Both x (from DepthRead) and msg (from prev_core DepthRead) are unit-RMS-scaled.
        feat = self.down(jnp.concatenate([_unit_rms(x), _unit_rms(msg)], axis=-1))
        one = jnp.asarray(1.0, dtype=x.dtype)
        return (one - jnp.tanh(self.forget(feat))) * x + jnp.tanh(self.gain(feat)) * msg


class InklingDenseLayer(nnx.Module):
    """Dense (no-MoE) Inkling-2 Mini decoder layer adapted for 2D bidirectional Hex boards.

    Combines:
      - Cached-norm block-wise DepthRead (separate Q/K/V reads on block-final full layers)
      - Looped-core MessageGate (for attention and MLP streams)
      - Native GQA + head-norm QK-clip + unit-RMS V + 2D ShortConv on K/V/post-attn/post-MLP
      - Per-head 2D Learnable RoPE + 2 zero-value learned attention sink tokens (cuDNN FlashAttn)
      - Sigmoid attention output gate
      - Dense Inkling-2 Capped-SwiGLU MLP: (4*tanh(g/4))*sigmoid(g)*(25*tanh(u/25))
    """

    def __init__(
        self,
        index: int,
        layers_per_block: int,
        width: int,
        ffn_channels: int,
        num_heads: int,
        kv_heads: int,
        gate_rank: int,
        height: int,
        width_board: int,
        *,
        in_core: bool = False,
        full: bool = False,
        dtype=jnp.bfloat16,
        fp8_training: bool = False,
        rngs: nnx.Rngs,
    ):
        if width % num_heads != 0:
            raise ValueError(f"width={width} must be divisible by num_heads={num_heads}")
        if num_heads % kv_heads != 0:
            raise ValueError(f"num_heads={num_heads} must be divisible by kv_heads={kv_heads}")
        head_dim = width // num_heads
        if head_dim % 2 != 0:
            raise ValueError(f"head_dim={head_dim} must be even for 2D RoPE")

        self.index = int(index)
        self.layers_per_block = int(layers_per_block)
        self.full = bool(full)
        self.seals_bank = (self.index % self.layers_per_block) == 0
        self.in_core = bool(in_core)
        self.width = int(width)
        self.num_heads = int(num_heads)
        self.kv_heads = int(kv_heads)
        self.head_dim = int(head_dim)
        self.qk_clip = 2.0 * math.sqrt(head_dim)
        self.dtype = dtype
        self.fp8_training = bool(fp8_training)

        if self.index > 0:
            self.attn_read_q = nnx.Param(jnp.zeros((3 if self.full else 1, width), dtype=jnp.float32))
        else:
            self.attn_read_q = None
        self.mlp_read_q = nnx.Param(jnp.zeros((1, width), dtype=jnp.float32))

        if self.in_core:
            # 4 message queries on full core layers (q, k, v, mlp), else 2 (shared attn, mlp).
            self.num_msg = 4 if self.full else 2
            self.message_queries = nnx.Param(jnp.zeros((self.num_msg, width), dtype=jnp.float32))
            self.attn_gates = nnx.List(
                [InklingMessageGate(width, gate_rank, dtype=dtype, rngs=rngs) for _ in range(3 if self.full else 1)]
            )
            self.mlp_gate = InklingMessageGate(width, gate_rank, dtype=dtype, rngs=rngs)
        else:
            self.num_msg = 0
            self.message_queries = None
            self.attn_gates = None
            self.mlp_gate = None

        if self.full:
            self.q_norm = RMSNorm(width)
            self.k_norm = RMSNorm(width)
            self.v_norm = RMSNorm(width)
            self.attn_norm = None
        else:
            self.q_norm = None
            self.k_norm = None
            self.v_norm = None
            self.attn_norm = RMSNorm(width)
        self.mlp_norm = RMSNorm(width)

        n_q = num_heads * head_dim
        n_kv = kv_heads * head_dim
        w_init = jax.nn.initializers.normal(0.02)
        self.q_proj = nnx.Linear(width, n_q, use_bias=False, dtype=dtype, kernel_init=w_init, rngs=rngs)
        self.k_proj = nnx.Linear(width, n_kv, use_bias=False, dtype=dtype, kernel_init=w_init, rngs=rngs)
        self.v_proj = nnx.Linear(width, n_kv, use_bias=False, dtype=dtype, kernel_init=w_init, rngs=rngs)
        self.q_scale = nnx.Param(jnp.ones((num_heads, head_dim), dtype=jnp.float32))
        self.k_conv = InklingShortConv2D(n_kv, height, width_board, dtype=dtype, rngs=rngs)
        self.v_conv = InklingShortConv2D(n_kv, height, width_board, dtype=dtype, rngs=rngs)

        # 2 learned attention sink keys (zero values) keep S_kv even for cuDNN FlashAttention.
        self.sink_k = nnx.Param(jnp.zeros((1, 2, kv_heads, head_dim), dtype=jnp.float32))

        # Per-head 2D learnable RoPE (shared across kv groups -> apply per head on q and per kv_head on k).
        num_pairs = head_dim // 2
        k1, k2 = jax.random.split(rngs.params())
        mags = jnp.exp(
            jax.random.uniform(
                k1,
                (kv_heads, num_pairs, 2),
                minval=math.log(1.0 / 50.0),
                maxval=math.log(1.0),
                dtype=jnp.float32,
            )
        )
        signs = jax.random.choice(
            k2, jnp.array([-1.0, 1.0], dtype=jnp.float32), shape=(kv_heads, num_pairs, 2)
        )
        self.rope_freqs = nnx.Param(mags * signs)

        self.out_gate = nnx.Linear(width, n_q, use_bias=False, dtype=dtype, kernel_init=w_init, rngs=rngs)
        self.out_proj = nnx.Linear(n_q, width, use_bias=False, dtype=dtype, kernel_init=w_init, rngs=rngs)
        self.after_attn = InklingShortConv2D(width, height, width_board, dtype=dtype, rngs=rngs)

        self.w_gate = nnx.Linear(width, ffn_channels, use_bias=False, dtype=dtype, kernel_init=w_init, rngs=rngs)
        self.w_up = nnx.Linear(width, ffn_channels, use_bias=False, dtype=dtype, kernel_init=w_init, rngs=rngs)
        self.w_down = nnx.Linear(ffn_channels, width, use_bias=False, dtype=dtype, kernel_init=w_init, rngs=rngs)
        self.after_mlp = InklingShortConv2D(width, height, width_board, dtype=dtype, rngs=rngs)

    def _apply_rope_2d(self, t: jax.Array, cos: jax.Array, sin: jax.Array) -> jax.Array:
        half = self.head_dim // 2
        t0, t1 = t[..., :half], t[..., half:]
        return jnp.concatenate([t0 * cos - t1 * sin, t0 * sin + t1 * cos], axis=-1)

    def _apply_kernel(self, kernel: jax.Array, x: jax.Array, *, train: bool) -> jax.Array:
        w = kernel.astype(x.dtype)
        if self.fp8_training and train and x.dtype == jnp.bfloat16:
            return _fp8_linear(x, w)
        return jax.lax.dot_general(
            x,
            w,
            dimension_numbers=(((x.ndim - 1,), (0,)), ((), ())),
            preferred_element_type=x.dtype,
        )

    def _apply_linear(self, linear: nnx.Linear, x: jax.Array, *, train: bool) -> jax.Array:
        if self.fp8_training and train and x.dtype == jnp.bfloat16:
            return _fp8_linear(x, linear.kernel[...].astype(jnp.bfloat16))
        return linear(x)

    def __call__(
        self,
        partial: jax.Array,
        bank_norm: tuple[jax.Array, ...],
        prev_core_norm: tuple[jax.Array, ...] | None,
        s_x: jax.Array,
        s_y: jax.Array,
        *,
        train: bool = False,
    ) -> tuple[jax.Array, tuple[jax.Array, ...]]:
        b, s, c = partial.shape
        partial_norm = _unit_rms(partial)

        if self.in_core and prev_core_norm is not None:
            assert self.message_queries is not None
            msgs = _read_depth_nostack(self.message_queries[...], prev_core_norm)  # [b, s, num_msg, c]
            if self.full:
                msg_qkv = (msgs[:, :, 0], msgs[:, :, 1], msgs[:, :, 2])
                msg_mlp = msgs[:, :, 3]
            else:
                msg_qkv = (msgs[:, :, 0],)
                msg_mlp = msgs[:, :, 1]
        else:
            msg_qkv = (None, None, None) if self.full else (None,)
            msg_mlp = None

        if self.index == 0:
            q_in = k_in = v_in = partial
        elif self.full:
            assert self.attn_read_q is not None
            reads = _read_depth_nostack(self.attn_read_q[...], bank_norm + (partial_norm,))
            q_in, k_in, v_in = reads[:, :, 0], reads[:, :, 1], reads[:, :, 2]
        else:
            assert self.attn_read_q is not None
            shared = _read_depth_nostack(self.attn_read_q[...], bank_norm + (partial_norm,))[:, :, 0]
            q_in = k_in = v_in = shared

        if self.in_core:
            assert self.attn_gates is not None
            if self.full:
                q_in = self.attn_gates[0](q_in, msg_qkv[0])
                k_in = self.attn_gates[1](k_in, msg_qkv[1])
                v_in = self.attn_gates[2](v_in, msg_qkv[2])
            else:
                shared = self.attn_gates[0](q_in, msg_qkv[0])
                q_in = k_in = v_in = shared

        if self.full:
            assert self.q_norm is not None and self.k_norm is not None and self.v_norm is not None
            q_in = self.q_norm(q_in)
            k_in = self.k_norm(k_in)
            v_in = self.v_norm(v_in)
        else:
            assert self.attn_norm is not None
            q_in = self.attn_norm(q_in)
            k_in = v_in = q_in

        n_q = self.num_heads * self.head_dim
        n_kv = self.kv_heads * self.head_dim
        if not self.full:
            # Fuse [q_proj | out_gate | k_proj | v_proj] into 1 GEMM (same parameter tree, 1/4 launches).
            w_qkvg = jnp.concatenate(
                [
                    self.q_proj.kernel[...],
                    self.out_gate.kernel[...],
                    self.k_proj.kernel[...],
                    self.v_proj.kernel[...],
                ],
                axis=-1,
            )
            qkvg = self._apply_kernel(w_qkvg, q_in, train=train)
            q_raw = qkvg[..., :n_q]
            og_raw = qkvg[..., n_q : 2 * n_q]
            k_raw = qkvg[..., 2 * n_q : 2 * n_q + n_kv]
            v_raw = qkvg[..., 2 * n_q + n_kv :]
        else:
            w_qog = jnp.concatenate([self.q_proj.kernel[...], self.out_gate.kernel[...]], axis=-1)
            qog = self._apply_kernel(w_qog, q_in, train=train)
            q_raw = qog[..., :n_q]
            og_raw = qog[..., n_q:]
            k_raw = self._apply_linear(self.k_proj, k_in, train=train)
            v_raw = self._apply_linear(self.v_proj, v_in, train=train)

        q = _clip_heads(
            q_raw.reshape((b, s, self.num_heads, self.head_dim)) * self.q_scale[...].astype(self.dtype),
            self.qk_clip,
        )
        k = _clip_heads(self.k_conv(k_raw).reshape((b, s, self.kv_heads, self.head_dim)), self.qk_clip)
        v = _unit_rms(self.v_conv(v_raw).reshape((b, s, self.kv_heads, self.head_dim)))

        # Apply matching 2D RoPE per KV-head group via broadcast (zero jnp.repeat copy).
        freqs_kv = self.rope_freqs[...]  # [kv_heads, D/2, 2]
        angle_kv = s_x[:, None, None] * freqs_kv[None, :, :, 0] + s_y[:, None, None] * freqs_kv[None, :, :, 1]
        cos_kv = jnp.cos(angle_kv)[None, :, :, :].astype(self.dtype)
        sin_kv = jnp.sin(angle_kv)[None, :, :, :].astype(self.dtype)
        groups = self.num_heads // self.kv_heads
        q_grp = q.reshape((b, s, self.kv_heads, groups, self.head_dim))
        q = self._apply_rope_2d(q_grp, cos_kv[:, :, :, None, :], sin_kv[:, :, :, None, :]).reshape(
            (b, s, self.num_heads, self.head_dim)
        )
        k = self._apply_rope_2d(k, cos_kv, sin_kv)

        # Append 2 learned zero-value attention sink KV tokens (keeps S_kv even for cuDNN FlashAttn).
        sk = _clip_heads(self.sink_k[...].astype(self.dtype), self.qk_clip)
        sk_b = jnp.broadcast_to(sk, (b, 2, self.kv_heads, self.head_dim))
        sv_b = jnp.zeros((b, 2, self.kv_heads, self.head_dim), dtype=self.dtype)
        k_aug = jnp.concatenate([k, sk_b], axis=1)
        v_aug = jnp.concatenate([v, sv_b], axis=1)

        use_cudnn = (
            jax.default_backend() == "gpu"
            and q.dtype in (jnp.bfloat16, jnp.float16)
            and self.head_dim <= 128
            and self.head_dim % 8 == 0
            and s % 2 == 0
        )
        attn_out = jax.nn.dot_product_attention(
            q, k_aug, v_aug, implementation="cudnn" if use_cudnn else None
        ).reshape((b, s, n_q))
        attn_out = attn_out * jax.nn.sigmoid(og_raw)
        branch = self.after_attn(self._apply_linear(self.out_proj, attn_out, train=train))

        if self.seals_bank:
            bank_norm = bank_norm + (partial_norm,)
            partial = branch
        else:
            partial = partial + branch

        xm = _read_depth_nostack(self.mlp_read_q[...], bank_norm + (_unit_rms(partial),))[:, :, 0]
        if self.in_core:
            assert self.mlp_gate is not None
            xm = self.mlp_gate(xm, msg_mlp)
        xm = self.mlp_norm(xm)

        # Fuse [w_gate | w_up] into 1 GEMM + Dense Inkling-2 Capped-SwiGLU.
        ffn = self.w_gate.kernel[...].shape[-1]
        w_gu = jnp.concatenate([self.w_gate.kernel[...], self.w_up.kernel[...]], axis=-1)
        gu = self._apply_kernel(w_gu, xm, train=train)
        gt, up = gu[..., :ffn], gu[..., ffn:]
        c4 = jnp.asarray(4.0, dtype=self.dtype)
        c25 = jnp.asarray(25.0, dtype=self.dtype)
        gated = (c4 * jnp.tanh(gt * jnp.asarray(0.25, dtype=self.dtype))) * jax.nn.sigmoid(gt) * (
            c25 * jnp.tanh(up * jnp.asarray(0.04, dtype=self.dtype))
        )
        return partial + self.after_mlp(self._apply_linear(self.w_down, gated, train=train)), bank_norm


class AZInklingBinnedNet(nnx.Module):
    """Dense (no-MoE) Inkling-2 Mini trunk with KataGo spatial Policy/Q & Value heads.

    When `passes == 1` (default): runs a straight `num_blocks`-layer Inkling-2 trunk
    (no looping, no MessageGates) keeping block-wise DepthRead, QK-clip + unit-RMS V,
    2D ShortConv, GQA + 2D RoPE + attention sinks, output gate, and Capped-SwiGLU.
    When `passes > 1`: runs the 4-block looped Prelude / Core (x passes) / Coda graph.
    """

    def __init__(
        self,
        num_actions: int,
        observation_shape: Sequence[int],
        *,
        num_channels: int = 192,
        num_blocks: int = 6,
        layers_per_block: int = 2,
        passes: int = 1,
        ffn_channels: int | None = None,
        num_heads: int = 6,
        kv_heads: int = 2,
        gate_rank: int = 32,
        infer_chunk_size: int = 512,
        bins: BinConfig,
        dtype=jnp.bfloat16,
        fp8_training: bool = False,
        canonicalize_hex_color: bool = True,
        rngs: nnx.Rngs,
    ):
        height, width, input_channels = observation_shape
        self.height = int(height)
        self.width = int(width)
        self.num_actions = int(num_actions)
        self.num_channels = int(num_channels)
        self.layers_per_block = max(1, int(layers_per_block))
        self.passes = max(1, int(passes))
        self.num_blocks = int(num_blocks) if self.passes == 1 else 4 * self.layers_per_block
        self.infer_chunk_size = int(infer_chunk_size)
        self.bins = bins
        self.dtype = dtype
        self.fp8_training = bool(fp8_training)
        self.canonicalize_hex_color = bool(canonicalize_hex_color)
        use_canon = (
            self.canonicalize_hex_color
            and height == width
            and input_channels == 4
            and num_actions == height * width + 1
        )
        stem_in = 3 if use_canon else (input_channels + 1)
        ffn_ch = int(round(num_channels * 2.0)) if ffn_channels is None else int(ffn_channels)

        self.conv = nnx.Conv(
            stem_in, num_channels, kernel_size=(3, 3), padding="SAME", use_bias=True, dtype=dtype, rngs=rngs
        )
        self.embedding_norm = RMSNorm(num_channels)

        total = self.num_blocks
        layers = []
        for i in range(total):
            if self.passes > 1:
                in_core = self.layers_per_block <= i < 3 * self.layers_per_block
                is_full = (
                    i == total - 1
                    if self.layers_per_block == 1
                    else ((i + 1) % self.layers_per_block) == 0
                )
            else:
                in_core = False
                # Split Q/K/V streams on the final layer (and every 3rd layer if deeper).
                is_full = (i == total - 1) or (total >= 6 and (i + 1) % 3 == 0)
            layers.append(
                InklingDenseLayer(
                    i,
                    self.layers_per_block,
                    num_channels,
                    ffn_ch,
                    num_heads,
                    kv_heads,
                    gate_rank,
                    height,
                    width,
                    in_core=in_core,
                    full=is_full,
                    dtype=dtype,
                    fp8_training=fp8_training,
                    rngs=rngs,
                )
            )
        self.blocks = nnx.List(layers)
        self.final_read_q = nnx.Param(jnp.zeros((1, num_channels), dtype=jnp.float32))
        self.final_norm = RMSNorm(num_channels)
        self.policy_q_head = KataPolicyQHead(
            num_channels, num_actions, height, width, bins, p_channels=48, g_channels=48, dtype=dtype, rngs=rngs
        )
        self.value_head = KataValueHead(
            num_channels, bins, v_channels=48, hidden_dim=128, dtype=dtype, rngs=rngs
        )

    def _forward_chunk(
        self, x: jax.Array, s_x: jax.Array, s_y: jax.Array, *, train: bool
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        x_in, is_white = _canonicalize_hex_obs(
            x, enabled=self.canonicalize_hex_color, num_actions=self.num_actions
        )
        h = self.conv(x_in.astype(self.dtype))
        b, hh, ww, c = h.shape
        seq = self.embedding_norm(h.reshape((b, hh * ww, c)))

        bank_norm: tuple[jax.Array, ...] = ()
        if self.passes == 1:
            for layer in self.blocks:
                seq, bank_norm = layer(seq, bank_norm, None, s_x, s_y, train=train)
        else:
            size = self.layers_per_block
            for layer in self.blocks[:size]:
                seq, bank_norm = layer(seq, bank_norm, None, s_x, s_y, train=train)
            prelude = (seq, bank_norm)

            prev_core_norm: tuple[jax.Array, ...] | None = None
            for _ in range(self.passes):
                seq, bank_norm = prelude
                for layer in self.blocks[size : 3 * size]:
                    seq, bank_norm = layer(seq, bank_norm, prev_core_norm, s_x, s_y, train=train)
                # Two completed core block summaries (unit-RMS normalized), kept in the autodiff graph.
                prev_core_norm = (bank_norm[-1], _unit_rms(seq))

            for layer in self.blocks[3 * size :]:
                seq, bank_norm = layer(seq, bank_norm, None, s_x, s_y, train=train)

        seq = _read_depth_nostack(self.final_read_q[...], bank_norm + (_unit_rms(seq),))[:, :, 0]
        h_out = jax.nn.silu(self.final_norm(seq)).reshape((b, hh, ww, c))
        logits, q_logits = self.policy_q_head(h_out, is_white=is_white)
        v_logits = self.value_head(h_out)
        return logits, v_logits, q_logits

    def __call__(self, x: jax.Array, *, train: bool) -> binned.BinnedOutput:
        hh, ww = x.shape[1], x.shape[2]
        idx = jnp.arange(hh * ww, dtype=jnp.int32)
        s_y = (idx // ww).astype(jnp.float32) - 0.5 * (hh - 1)
        s_x = (idx % ww).astype(jnp.float32) - 0.5 * (ww - 1)

        chunk = self.infer_chunk_size
        batch = x.shape[0]
        if (not train) and chunk > 0 and batch > chunk and (batch % chunk == 0):
            n_chunks = batch // chunk
            x_chunks = x.reshape((n_chunks, chunk) + x.shape[1:])
            logits_c, v_c, q_c = jax.lax.map(
                lambda xc: self._forward_chunk(xc, s_x, s_y, train=False), x_chunks
            )
            logits = logits_c.reshape((batch, self.num_actions))
            v_logits = v_c.reshape((batch, self.bins.count))
            q_logits = q_c.reshape((batch, self.num_actions, self.bins.count))
        else:
            logits, v_logits, q_logits = self._forward_chunk(x, s_x, s_y, train=train)

        return binned.BinnedOutput(
            logits.astype(jnp.float32),
            binned.from_logits(v_logits),
            binned.from_logits(q_logits),
            jnp.asarray(binned.support(self.bins)),
        )


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
        return x + self.alpha[...] * self.linear(jax.nn.relu(x))


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
        dirichlet_concentration_floor: float | None = None,
        dirichlet_initial_concentration: float | None = None,
        dirichlet_head_parameterization: str = "log_concentration",
        rezero_kernel_init: str = "variance_scaling",
        rngs: nnx.Rngs,
    ):
        self.num_actions = num_actions
        self.num_outcomes = num_outcomes
        self.width = width
        self.depth = depth
        self.dtype = dtype
        self.dirichlet_head_parameterization = str(
            dirichlet_head_parameterization
        )
        concentration_floor = (
            None
            if dirichlet_concentration_floor is None
            else float(dirichlet_concentration_floor)
        )
        self.dirichlet_concentration_floor = (
            None if legacy_dirichlet_head_init else concentration_floor
        )
        self.dirichlet_concentration_clip = dirichlet_concentration_clip
        self.dirichlet_initial_concentration = dirichlet_initial_concentration
        if (
            self.dirichlet_head_parameterization == "log_concentration"
            and concentration_floor is not None
        ):
            raise ValueError(
                "direct log-concentration heads do not support a "
                "concentration floor"
            )
        if self.dirichlet_head_parameterization not in {
            "log_concentration",
            "legacy",
        }:
            raise ValueError(
                "unknown Dirichlet concentration parameterization: "
                f"{self.dirichlet_head_parameterization!r}"
            )

        if legacy_dirichlet_head_init and (
            dirichlet_concentration_floor is not None
            or dirichlet_initial_concentration is not None
        ):
            raise ValueError(
                "dirichlet concentration floor/initialization cannot be combined "
                "with legacy_dirichlet_head_init"
            )

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
            concentration_logit = _initial_dirichlet_concentration_logit(
                self.dirichlet_head_parameterization,
                concentration_floor=concentration_floor,
                concentration_clip=self.dirichlet_concentration_clip,
                initial_concentration=dirichlet_initial_concentration,
                default_concentration=(
                    float(num_outcomes) + _DIRICHLET_INITIAL_EXCESS_CONCENTRATION
                ),
            )
            concentration_bias_init = jax.nn.initializers.constant(
                concentration_logit
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
            parameterization=self.dirichlet_head_parameterization,
            concentration_floor=self.dirichlet_concentration_floor,
            concentration_clip=self.dirichlet_concentration_clip,
        )

        q_mean_logits = self.q_dir_head(x).reshape(
            (x.shape[0], self.num_actions, self.num_outcomes)
        )
        q_concentration_logit = self.q_conc_head(x)
        alpha_q = dirichlet_from_logits(
            q_mean_logits,
            q_concentration_logit,
            parameterization=self.dirichlet_head_parameterization,
            concentration_floor=self.dirichlet_concentration_floor,
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
    head_parameterization = str(config.model.dirichlet_head_parameterization)
    legacy_concentration_clip = (
        config.training.regularization.dirichlet_concentration_clip
        if head_parameterization == "legacy"
        else None
    )

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
    if config.model.network in {"aznet_dirichlet", "aznet_binned"}:
        model_class = AZBinnedNet if config.model.network == "aznet_binned" else AZDirichletNet
        return model_class(
            num_actions=num_actions,
            observation_shape=observation_shape,
            num_outcomes=config.env.resolved_num_outcomes(),
            num_channels=config.model.num_channels,
            num_blocks=config.model.num_layers,
            resnet_v2=config.model.resnet_v2,
            dtype=compute_dtype,
            dirichlet_concentration_clip=legacy_concentration_clip,
            dirichlet_concentration_floor=(
                config.model.dirichlet_concentration_floor
            ),
            dirichlet_initial_concentration=(
                config.model.dirichlet_initial_concentration
            ),
            dirichlet_head_parameterization=head_parameterization,
            bins=config.model.bins if config.model.network == "aznet_binned" else None,
            fp8_training=bool(getattr(config.model, "fp8_training", False)),
            rngs=rngs,
        )
    if config.model.network == "aznet_kata_conv":
        return AZKataConvBinnedNet(
            num_actions=num_actions,
            observation_shape=observation_shape,
            num_channels=config.model.num_channels,
            num_blocks=config.model.num_layers,
            bins=config.model.bins,
            dtype=compute_dtype,
            fp8_training=bool(getattr(config.model, "fp8_training", False)),
            canonicalize_hex_color=bool(getattr(config.model, "canonicalize_hex_color", True)),
            rngs=rngs,
        )
    if config.model.network == "aznet_kata_transformer":
        return AZKataTransformerBinnedNet(
            num_actions=num_actions,
            observation_shape=observation_shape,
            num_channels=config.model.num_channels,
            num_blocks=config.model.num_layers,
            mid_channels=getattr(config.model, "transformer_mid_channels", None),
            ffn_channels=getattr(config.model, "transformer_ffn_channels", None),
            num_heads=int(getattr(config.model, "transformer_heads", 4)),
            inner_blocks=int(getattr(config.model, "transformer_inner_blocks", 2)),
            qk_norm=bool(getattr(config.model, "transformer_qk_norm", True)),
            infer_chunk_size=int(getattr(config.model, "transformer_infer_chunk_size", 512)),
            bins=config.model.bins,
            dtype=compute_dtype,
            fp8_training=bool(getattr(config.model, "fp8_training", False)),
            canonicalize_hex_color=bool(getattr(config.model, "canonicalize_hex_color", True)),
            rngs=rngs,
        )
    if config.model.network == "aznet_inkling":
        return AZInklingBinnedNet(
            num_actions=num_actions,
            observation_shape=observation_shape,
            num_channels=config.model.num_channels,
            num_blocks=config.model.num_layers,
            layers_per_block=int(getattr(config.model, "inkling_layers_per_block", 2)),
            passes=int(getattr(config.model, "inkling_passes", 1)),
            ffn_channels=getattr(config.model, "transformer_ffn_channels", None),
            num_heads=int(getattr(config.model, "transformer_heads", 6)),
            kv_heads=int(getattr(config.model, "inkling_kv_heads", 2)),
            gate_rank=int(getattr(config.model, "inkling_gate_rank", 32)),
            infer_chunk_size=int(getattr(config.model, "transformer_infer_chunk_size", 512)),
            bins=config.model.bins,
            dtype=compute_dtype,
            fp8_training=bool(getattr(config.model, "fp8_training", False)),
            canonicalize_hex_color=bool(getattr(config.model, "canonicalize_hex_color", True)),
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
            num_outcomes=config.env.resolved_num_outcomes(),
            width=config.model.num_channels,
            depth=config.model.num_layers,
            dtype=compute_dtype,
            dirichlet_concentration_clip=legacy_concentration_clip,
            legacy_dirichlet_head_init=config.model.legacy_dirichlet_head_init,
            dirichlet_concentration_floor=(
                config.model.dirichlet_concentration_floor
            ),
            dirichlet_initial_concentration=(
                config.model.dirichlet_initial_concentration
            ),
            dirichlet_head_parameterization=head_parameterization,
            rezero_kernel_init=config.model.rezero_kernel_init,
            rngs=rngs,
        )
    raise ValueError(f"unknown network: {config.model.network!r}")
