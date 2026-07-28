from __future__ import annotations

import inspect
import os
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from flax import nnx
import jax
from jax import numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec


@dataclass(frozen=True)
class BatchParallel:
    enabled: bool
    axis_name: str = "batch"
    mesh: Mesh | None = None

    @property
    def device_count(self) -> int:
        if not self.enabled or self.mesh is None:
            return 1
        return int(np.asarray(self.mesh.devices).size)

    @contextmanager
    def mesh_context(self) -> Iterator[None]:
        if not self.enabled or self.mesh is None:
            with nullcontext():
                yield
            return
        with jax.set_mesh(self.mesh):
            yield

    def sharding_for(self, ndim: int, batch_axis: int = 0) -> NamedSharding | None:
        if not self.enabled or self.mesh is None or ndim <= batch_axis:
            return None
        axes: list[str | None] = [None] * ndim
        axes[batch_axis] = self.axis_name
        return NamedSharding(self.mesh, PartitionSpec(*axes))

    def execution_mesh(self) -> Mesh:
        """Return a mesh for code that runs one independent local batch per device.

        A disabled ``BatchParallel`` still uses a one-device mesh here.  This
        lets callers keep a single SPMD program for both local and distributed
        execution, without turning ordinary batch tensors into sharded ones.
        """

        if self.enabled and self.mesh is not None:
            return self.mesh
        return jax.make_mesh(
            (1,),
            (self.axis_name,),
            axis_types=(AxisType.Auto,),
            devices=jax.devices()[:1],
        )

    def local_batch_size(self, batch_size: int) -> int:
        """Validate and return the portion of a batch owned by one device."""

        if batch_size % self.device_count != 0:
            raise ValueError(
                "batch size must be divisible by the batch-parallel device "
                f"count ({self.device_count})."
            )
        return batch_size // self.device_count

    def split(self, rng_key: jax.Array, num:int) -> jax.Array:
        ids = jnp.arange(num, dtype=jnp.uint32)
        sharding = self.sharding_for(ndim=1)
        if sharding is not None:
            ids = jax.lax.with_sharding_constraint(ids, sharding)
        keys = jax.vmap(jax.random.fold_in, in_axes=(None, 0))(rng_key, ids)
        return keys

DISABLED_BATCH_PARALLEL = BatchParallel(enabled=False)


def local_shard_map(**kwargs):
    """Build an NNX shard map for independent per-device computations.

    Some local control-flow carries are not manually axis-typed.  They are
    valid because the mapped computation never communicates across devices,
    but JAX's optional shard-map checker rejects them before lowering.  The
    option was renamed from ``check_rep`` to ``check_vma`` across JAX versions.
    """

    parameters = inspect.signature(nnx.shard_map).parameters
    if "check_vma" in parameters:
        return nnx.shard_map(**kwargs, check_vma=False)
    if "check_rep" in parameters:
        return nnx.shard_map(**kwargs, check_rep=False)  # ty: ignore[no-matching-overload]
    return nnx.shard_map(**kwargs)


def _env_bool(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return int(value)


def _env_local_device_ids(name: str) -> list[int] | None:
    value = os.environ.get(name)
    if value is None or value.strip() == "":
        return None
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def initialize_distributed() -> None:
    if _env_bool("SCACCHI_DISABLE_DISTRIBUTED"):
        return
    platforms = os.environ.get("JAX_PLATFORMS")
    if platforms is not None and "tpu" not in {platform.strip() for platform in platforms.split(",")}:
        return
    if not any(Path("/dev").glob("accel*")):
        return
    if jax.distributed.is_initialized():
        return
    coordinator_address = os.environ.get("SCACCHI_JAX_COORDINATOR_ADDRESS")
    num_processes = _env_int("SCACCHI_JAX_NUM_PROCESSES")
    process_id = _env_int("SCACCHI_JAX_PROCESS_ID")
    local_device_ids = _env_local_device_ids("SCACCHI_JAX_LOCAL_DEVICE_IDS")
    initialization_timeout = _env_int("SCACCHI_JAX_INITIALIZATION_TIMEOUT") or 300
    coordinator_bind_address = os.environ.get("SCACCHI_JAX_COORDINATOR_BIND_ADDRESS")

    explicit_args = (coordinator_address, num_processes, process_id)
    if any(value is not None for value in explicit_args):
        if any(value is None for value in explicit_args):
            raise ValueError(
                "SCACCHI_JAX_COORDINATOR_ADDRESS, SCACCHI_JAX_NUM_PROCESSES, "
                "and SCACCHI_JAX_PROCESS_ID must be set together."
            )
        jax.distributed.initialize(
            coordinator_address=coordinator_address,
            num_processes=num_processes,
            process_id=process_id,
            local_device_ids=local_device_ids,
            cluster_detection_method="deactivate",
            initialization_timeout=initialization_timeout,
            coordinator_bind_address=coordinator_bind_address,
        )
        return
    jax.distributed.initialize()


def make_batch_parallel(config: Any, axis_name: str = "batch") -> BatchParallel:
    devices = jax.devices()
    if not devices:
        raise RuntimeError("JAX reported no devices for batch-parallel training.")

    mesh = jax.make_mesh(
        (len(devices),),
        (axis_name,),
        axis_types=(AxisType.Auto,),
        devices=devices,
    )
    parallel = BatchParallel(enabled=True, axis_name=axis_name, mesh=mesh)
    batch_sizes = {
        "selfplay.batch_size": config.selfplay.batch_size,
        "training.batch_size": config.training.batch_size,
    }
    if config.eval.interval > 0:
        batch_sizes["eval.batch_size"] = config.eval.batch_size
    for name, batch_size in batch_sizes.items():
        if batch_size % len(devices) != 0:
            raise ValueError(
                f"{name} ({batch_size}) must be divisible by number of "
                f"devices ({len(devices)})."
            )
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


def assert_batch_axis_sharded(
    value: Any,
    parallel: BatchParallel | None,
    *,
    batch_axis: int = 0,
    label: str = "value",
) -> Any:
    if parallel is None or not parallel.enabled:
        return value

    def assert_leaf(path, leaf: Any) -> Any:
        if not isinstance(leaf, jax.Array):
            return leaf
        expected = parallel.sharding_for(leaf.ndim, batch_axis=batch_axis)
        if expected is None:
            return leaf

        leaf_ndim = leaf.ndim
        leaf_label = f"{label}{jax.tree_util.keystr(path)}"

        def check_sharding(actual) -> None:
            if expected.is_equivalent_to(actual, leaf_ndim):
                return
            raise ValueError(
                f"{leaf_label} has sharding {actual!r}; expected equivalent "
                f"to {expected!r}"
            )

        jax.debug.inspect_array_sharding(leaf, callback=check_sharding)
        return leaf

    return jax.tree_util.tree_map_with_path(assert_leaf, value)
