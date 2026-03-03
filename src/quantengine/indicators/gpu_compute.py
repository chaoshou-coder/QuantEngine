from __future__ import annotations

import math
from typing import Any

import numpy as np

from . import technical

try:
    import cupy as cp  # type: ignore
except Exception:  # pragma: no cover
    cp = None

try:
    from numba import cuda  # type: ignore
except Exception:  # pragma: no cover
    cuda = None


if cuda is not None:  # pragma: no cover - optional GPU path

    @cuda.jit
    def _sma_kernel(inp, out, window):
        row, col = cuda.grid(2)
        if row >= inp.shape[0] or col >= inp.shape[1]:
            return
        if row + 1 < window:
            out[row, col] = math.nan
            return
        total = 0.0
        start = row - window + 1
        for i in range(start, row + 1):
            total += inp[i, col]
        out[row, col] = total / window


def sma_gpu(values: Any, window: int) -> Any:
    if cp is None or cuda is None:
        return technical.sma(values, window)

    if not isinstance(values, cp.ndarray):
        values = cp.asarray(values)
    out = cp.empty_like(values, dtype=cp.float64)
    threads_per_block = (16, 16)
    blocks_per_grid_x = int(np.ceil(values.shape[0] / threads_per_block[0]))
    blocks_per_grid_y = int(np.ceil(values.shape[1] / threads_per_block[1]))
    _sma_kernel[(blocks_per_grid_x, blocks_per_grid_y), threads_per_block](values, out, window)
    return out


def ema_gpu(values: Any, span: int) -> Any:
    return technical.ema(values, span)


def rsi_gpu(close: Any, window: int = 14) -> Any:
    return technical.rsi(close, window)
