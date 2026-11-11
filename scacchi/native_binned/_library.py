"""Build and register the binned CUDA kernels as XLA FFI targets.

Opt-in and local, like ``native_hex``: nothing here runs at import time. The
first call to :func:`load_library` compiles ``backup.cu`` (the batched backup)
and ``search.cu`` (the fused search step) with nvcc into one library under
``outputs/native_binned/<fingerprint>/`` (kept between processes), and
:func:`register` / :func:`register_search` hand the handlers to JAX for the
CUDA platform.

nvcc is looked for in ``$NVCC``, ``$CUDA_HOME/bin``, ``PATH`` and the
``nvidia-cuda-nvcc`` wheel that ships with the JAX CUDA install. The FFI
headers come from ``jax.ffi.include_dir()``.
"""

from __future__ import annotations

import ctypes
from functools import lru_cache
import hashlib
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

TARGET = "scacchi_binned_backup"
SEARCH_TARGET = "scacchi_binned_search_step"
NETWORK_TARGET = "scacchi_binned_network"
HERE = Path(__file__).parent
SOURCES = [HERE / "backup.cu", HERE / "search.cu", HERE / "network.cu"]
HEADERS = [HERE / "backup_lane.cuh", HERE / "conv_mma.cuh"]
REPO = Path(__file__).resolve().parents[2]
FLAGS = ["-std=c++17", "-O3", "--shared", "-Xcompiler", "-fPIC", "-lineinfo", "-Xptxas", "-v"]


def find_nvcc() -> Path:
    candidates = [os.environ.get("NVCC")]
    if os.environ.get("CUDA_HOME"):
        candidates.append(str(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc"))
    candidates.append(shutil.which("nvcc"))
    try:
        import nvidia.cu13 as wheel  # nvidia-cuda-nvcc wheel layout

        candidates.append(str(Path(next(iter(wheel.__path__))) / "bin" / "nvcc"))
    except ImportError:
        pass
    for site in sys.path:
        for wheel in ("cu13", "cuda_nvcc"):  # CUDA 13 and CUDA 12 wheel layouts
            candidates.append(str(Path(site) / "nvidia" / wheel / "bin" / "nvcc"))
    for candidate in candidates:
        if candidate and Path(candidate).is_file():
            return Path(candidate)
    raise RuntimeError("nvcc not found; set NVCC or CUDA_HOME (looked in: " + ", ".join(c for c in candidates if c) + ")")


def cuda_include(nvcc: Path) -> Path:
    include = nvcc.parent.parent / "include"
    if not (include / "cuda_runtime.h").is_file():
        include = nvcc.parent.parent.parent / "include"
    if not (include / "cuda_runtime.h").is_file():
        raise RuntimeError(f"cuda_runtime.h not found next to {nvcc}")
    return include


def compute_capability() -> str:
    import jax

    capability = jax.devices()[0].compute_capability  # e.g. "12.0"
    return capability.replace(".", "")


def fingerprint(nvcc: Path, arch: str) -> str:
    version = subprocess.run([str(nvcc), "--version"], capture_output=True, text=True, check=True).stdout
    digest = hashlib.sha256()
    sources = [path.read_bytes() for path in (*SOURCES, *HEADERS)]
    for part in (*sources, version.encode(), arch.encode(), " ".join(FLAGS).encode(), os.environ.get("NVCC_EXTRA", "").encode()):
        digest.update(part)
    return digest.hexdigest()[:16]


def build() -> Path:
    import jax

    nvcc = find_nvcc()
    arch = compute_capability()
    directory = REPO / "outputs" / "native_binned" / fingerprint(nvcc, arch)
    library = directory / "native_binned.so"
    if library.is_file():
        return library
    directory.mkdir(parents=True, exist_ok=True)
    temporary = directory / f"native_binned.{os.getpid()}.so"
    command = [
        str(nvcc), *FLAGS, f"-gencode=arch=compute_{arch},code=sm_{arch}",
        f"-I{jax.ffi.include_dir()}", f"-I{cuda_include(nvcc)}",
        *shlex.split(os.environ.get("NVCC_EXTRA", "")),
        *(str(source) for source in SOURCES), "-o", str(temporary),
    ]
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        detail = getattr(error, "stderr", None) or str(error)
        raise RuntimeError("native_binned build failed: " + " ".join(command) + "\n" + detail) from error
    (directory / "build.log").write_text(result.stderr)
    temporary.replace(library)
    return library


@lru_cache(maxsize=1)
def load_library() -> ctypes.CDLL:
    return ctypes.cdll.LoadLibrary(str(build()))


@lru_cache(maxsize=1)
def register() -> str:
    """Register the FFI handler for the CUDA platform (once per process)."""
    import jax

    library = load_library()
    jax.ffi.register_ffi_target(TARGET, jax.ffi.pycapsule(library.ScacchiBinnedBackup), platform="CUDA", api_version=1)
    return TARGET


@lru_cache(maxsize=1)
def register_search() -> str:
    """Register the fused search-step handler for the CUDA platform (once per process)."""
    import jax

    library = load_library()
    jax.ffi.register_ffi_target(SEARCH_TARGET, jax.ffi.pycapsule(library.ScacchiBinnedSearchStep), platform="CUDA", api_version=1)
    return SEARCH_TARGET


@lru_cache(maxsize=1)
def register_network() -> str:
    """Register the network handler for the CUDA platform (once per process)."""
    import jax

    library = load_library()
    jax.ffi.register_ffi_target(NETWORK_TARGET, jax.ffi.pycapsule(library.ScacchiBinnedNetwork), platform="CUDA", api_version=1)
    return NETWORK_TARGET
