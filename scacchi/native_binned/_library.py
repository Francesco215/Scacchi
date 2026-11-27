"""Build and register the binned CUDA kernels as XLA FFI targets."""

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
FAST_SEARCH_TARGET = "scacchi_binned_search_fast_step"
NETWORK_TARGET = "scacchi_binned_network"
NETWORK_COMPACT_TARGET = "scacchi_binned_network_compact"
INKLING_NETWORK_TARGET = "scacchi_inkling_network"
INKLING_COMPACT_TARGET = "scacchi_inkling_network_compact"
HERE = Path(__file__).parent
SOURCES = [HERE / "backup.cu", HERE / "search.cu", HERE / "network.cu", HERE / "inkling_network.cu"]
HEADERS = [
    HERE / "backup_lane.cuh",
    HERE / "cache_key.cuh",
    HERE / "conv_mma.cuh",
    HERE / "conv_int8.cuh",
    HERE / "conv_board_int8.cuh",
    HERE / "inkling_mma.cuh",
]
REPO = Path(__file__).resolve().parents[2]
FLAGS = ["-std=c++17", "-O3", "--shared", "-Xcompiler", "-fPIC", "-lineinfo", "-Xptxas", "-v"]


def find_nvcc() -> Path:
    candidates = [os.environ.get("NVCC")]
    if os.environ.get("CUDA_HOME"):
        candidates.append(str(Path(os.environ["CUDA_HOME"]) / "bin" / "nvcc"))
    candidates.append(shutil.which("nvcc"))
    try:
        import nvidia.cu13 as wheel

        candidates.append(str(Path(next(iter(wheel.__path__))) / "bin" / "nvcc"))
    except ImportError:
        pass
    for site in sys.path:
        for wheel in ("cu13", "cuda_nvcc"):
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

    capability = jax.devices()[0].compute_capability
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
def device_sm_count() -> int:
    """Actual SM count of the active CUDA device (fixes hardcoded SM-count scratch checks)."""
    lib = load_library()
    fn = lib.ScacchiDeviceSMCount
    fn.restype = ctypes.c_int
    sms = int(fn())
    return sms if sms > 0 else 1


@lru_cache(maxsize=1)
def register() -> str:
    import jax

    library = load_library()
    jax.ffi.register_ffi_target(TARGET, jax.ffi.pycapsule(library.ScacchiBinnedBackup), platform="CUDA", api_version=1)
    return TARGET


@lru_cache(maxsize=1)
def register_search() -> str:
    import jax

    library = load_library()
    jax.ffi.register_ffi_target(SEARCH_TARGET, jax.ffi.pycapsule(library.ScacchiBinnedSearchStep), platform="CUDA", api_version=1)
    return SEARCH_TARGET


@lru_cache(maxsize=1)
def register_fast_search() -> str:
    import jax

    library = load_library()
    jax.ffi.register_ffi_target(FAST_SEARCH_TARGET, jax.ffi.pycapsule(library.ScacchiBinnedSearchFastStep), platform="CUDA", api_version=1)
    return FAST_SEARCH_TARGET


@lru_cache(maxsize=1)
def register_network() -> str:
    import jax

    library = load_library()
    jax.ffi.register_ffi_target(NETWORK_TARGET, jax.ffi.pycapsule(library.ScacchiBinnedNetwork), platform="CUDA", api_version=1)
    return NETWORK_TARGET


@lru_cache(maxsize=1)
def register_network_compact() -> str:
    import jax

    library = load_library()
    jax.ffi.register_ffi_target(NETWORK_COMPACT_TARGET, jax.ffi.pycapsule(library.ScacchiBinnedNetworkCompact), platform="CUDA", api_version=1)
    return NETWORK_COMPACT_TARGET


@lru_cache(maxsize=1)
def register_inkling_network() -> str:
    import jax

    library = load_library()
    jax.ffi.register_ffi_target(INKLING_NETWORK_TARGET, jax.ffi.pycapsule(library.ScacchiInklingNetwork), platform="CUDA", api_version=1)
    return INKLING_NETWORK_TARGET


@lru_cache(maxsize=1)
def register_inkling_compact() -> str:
    import jax

    library = load_library()
    jax.ffi.register_ffi_target(INKLING_COMPACT_TARGET, jax.ffi.pycapsule(library.ScacchiInklingNetworkCompact), platform="CUDA", api_version=1)
    return INKLING_COMPACT_TARGET
