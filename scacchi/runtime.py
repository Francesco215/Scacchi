"""Runtime helpers for mesh-based execution."""

from __future__ import annotations

import numpy as np
import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
from omegaconf import DictConfig

from scacchi.config import RuntimeConfig


def create_mesh(cfg: DictConfig | RuntimeConfig) -> Mesh:
    """Create the runtime mesh.

    The default config creates a one-device mesh. Future data parallelism can
    increase `num_devices` without changing global batch shapes.
    """

    num_devices = int(cfg.num_devices)
    devices = np.asarray(jax.devices()[:num_devices])
    if devices.size != num_devices:
        msg = f"Requested {num_devices} devices, but JAX sees {devices.size}."
        raise ValueError(msg)
    return Mesh(devices, (str(cfg.axis_name),))


def batch_sharding(mesh: Mesh, rank: int) -> NamedSharding:
    """Shard the first axis over the mesh data axis and replicate the rest."""

    if rank < 1:
        msg = "Rank must be at least one for batch sharding."
        raise ValueError(msg)
    axis_name = mesh.axis_names[0]
    return NamedSharding(mesh, P(axis_name, *([None] * (rank - 1))))


def replicated_sharding(mesh: Mesh) -> NamedSharding:
    """Replicate an array or pytree leaf on every device in the mesh."""

    return NamedSharding(mesh, P())


def validate_batch_size(name: str, batch_size: int, mesh: Mesh) -> None:
    """Require global batches to split evenly across the data mesh."""

    axis_size = int(mesh.shape[mesh.axis_names[0]])
    if int(batch_size) % axis_size:
        msg = f"{name}={batch_size} must be divisible by data mesh size {axis_size}."
        raise ValueError(msg)
