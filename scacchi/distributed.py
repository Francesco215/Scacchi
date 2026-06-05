from __future__ import annotations

import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import jax
from jax import numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec


@dataclass(frozen=True)
class BatchParallel:
    enabled: bool
    axis_name: str = "batch"
    mesh: Mesh | None = None

    @property
    def device_count(self) -> int:
        if self.mesh is None:
            return 1
        return int(np.asarray(self.mesh.devices).size)

    @contextmanager
    def mesh_context(self) -> Iterator[None]:
        if self.mesh is None:
            with nullcontext():
                yield
            return
        with self.mesh:
            yield

    def sharding_for(self, ndim: int, batch_axis: int = 0) -> NamedSharding | None:
        if self.mesh is None or ndim <= batch_axis:
            return None
        axes: list[str | None] = [None] * ndim
        axes[batch_axis] = self.axis_name
        return NamedSharding(self.mesh, PartitionSpec(*axes))

    def split(self, rng_key: jax.Array, num:int) -> jax.Array:
        ids = jnp.arange(num, dtype=jnp.uint32)
        sharding = self.sharding_for(ndim=1)
        if sharding is not None:
            ids = jax.lax.with_sharding_constraint(ids, sharding)
        keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(rng_key, ids)
        return keys

DISABLED_BATCH_PARALLEL = BatchParallel(enabled=False)


def initialize_distributed() -> None:
    platforms = os.environ.get("JAX_PLATFORMS")
    if platforms is not None and "tpu" not in {platform.strip() for platform in platforms.split(",")}:
        return
    if not any(Path("/dev").glob("accel*")):
        return
    if jax.distributed.is_initialized():
        return
    jax.distributed.initialize()


def make_batch_parallel(config: Any, axis_name: str = "batch") -> BatchParallel:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX reported no devices for batch-parallel training.")

    mesh = jax.make_mesh((len(devices),), (axis_name,))
    parallel = BatchParallel(enabled=True, axis_name=axis_name, mesh=mesh)
    for batch_size in [config.selfplay.batch_size, config.training.batch_size]:
        assert batch_size % len(devices) == 0, f"batch_size ({batch_size}) must be divisible by number of devices ({len(devices)})"
    return parallel


def constrain_batch_axis(
    value: Any,
    parallel: BatchParallel | None,
    *,
    batch_axis: int = 0,
) -> Any:
    if parallel is None or not parallel.enabled:
        return value

    def constrain_leaf(leaf: Any) -> Any:
        if not isinstance(leaf, jax.Array):
            return leaf
        sharding = parallel.sharding_for(leaf.ndim, batch_axis=batch_axis)
        if sharding is None:
            return leaf
        return jax.lax.with_sharding_constraint(leaf, sharding)

    return jax.tree_util.tree_map(constrain_leaf, value)
