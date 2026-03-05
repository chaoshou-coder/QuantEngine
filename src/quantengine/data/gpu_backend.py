from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any
import logging

import numpy as np

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cp = None

try:
    import cudf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cudf = None


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BackendInfo:
    requested: str
    active: str
    reason: str
    gpu_available: bool
    cudf_available: bool
    cupy_available: bool


@lru_cache(maxsize=8)
def get_backend_info(requested: str = "auto", use_gpu: bool = True) -> BackendInfo:
    normalized = (requested or "auto").strip().lower()
    gpu_available = cp is not None and _has_cuda_device()
    cudf_available = cudf is not None
    cupy_available = cp is not None

    if not use_gpu:
        return BackendInfo(
            requested=normalized,
            active="cpu",
            reason="use_gpu=false",
            gpu_available=gpu_available,
            cudf_available=cudf_available,
            cupy_available=cupy_available,
        )

    if normalized == "cpu":
        return BackendInfo(
            requested=normalized,
            active="cpu",
            reason="explicit cpu backend",
            gpu_available=gpu_available,
            cudf_available=cudf_available,
            cupy_available=cupy_available,
        )

    if normalized == "gpu":
        if gpu_available:
            return BackendInfo(
                requested=normalized,
                active="gpu",
                reason="explicit gpu backend",
                gpu_available=gpu_available,
                cudf_available=cudf_available,
                cupy_available=cupy_available,
            )
        return BackendInfo(
            requested=normalized,
            active="cpu",
            reason="gpu requested but unavailable",
            gpu_available=gpu_available,
            cudf_available=cudf_available,
            cupy_available=cupy_available,
        )

    if gpu_available:
        return BackendInfo(
            requested=normalized,
            active="gpu",
            reason="auto selected gpu",
            gpu_available=gpu_available,
            cudf_available=cudf_available,
            cupy_available=cupy_available,
        )
    return BackendInfo(
        requested=normalized,
        active="cpu",
        reason="auto fallback to cpu",
        gpu_available=gpu_available,
        cudf_available=cudf_available,
        cupy_available=cupy_available,
    )


def _has_cuda_device() -> bool:
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def get_xp(backend: BackendInfo):
    return cp if backend.active == "gpu" and cp is not None else np


def xp_from_array(values: Any):
    if cp is not None and isinstance(values, cp.ndarray):
        return cp
    return np


def to_numpy(array: Any) -> np.ndarray:
    if cp is not None and isinstance(array, cp.ndarray):
        return cp.asnumpy(array)
    return np.asarray(array)


def get_gpu_memory_info() -> tuple[int, int]:
    """Return (free_bytes, total_bytes) for current CUDA device.

    Falls back to (0, 0) when GPU/CUDA runtime is unavailable.
    """
    if cp is None:
        return (0, 0)
    try:
        free_bytes, total_bytes = cp.cuda.Device().mem_info
        return int(free_bytes), int(total_bytes)
    except Exception:
        return (0, 0)


def estimate_max_batch_size(
    n_bars: int,
    n_assets: int,
    reserve_ratio: float = 0.25,
    n_buffers: int = 8,
    dtype_bytes: int = 8,
) -> int:
    """Estimate safe combo batch size under current GPU memory.

    The estimate is based on memory required for main tensors:
    signal + positions + equity + returns + temporary work buffers.
    """
    n_bars = max(int(n_bars), 1)
    n_assets = max(int(n_assets), 1)
    reserve_ratio = min(max(float(reserve_ratio), 0.0), 0.95)
    n_buffers = max(int(n_buffers), 1)
    dtype_bytes = max(int(dtype_bytes), 1)

    free_bytes, total_bytes = get_gpu_memory_info()
    if free_bytes <= 0 or total_bytes <= 0:
        return 1

    usable_bytes = int(free_bytes * (1.0 - reserve_ratio))
    bytes_per_combo = n_bars * n_assets * n_buffers * dtype_bytes
    if bytes_per_combo <= 0:
        return 1

    estimate = usable_bytes // bytes_per_combo
    return max(int(estimate), 1)
