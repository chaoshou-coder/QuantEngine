from __future__ import annotations

from typing import Any

import numpy as np

from quantengine.data.gpu_backend import xp_from_array

SignalArray = Any


def clip_signal(values: SignalArray) -> SignalArray:
    xp = xp_from_array(values)
    if xp.__name__ == "cupy":  # pragma: no cover - GPU path
        return xp.clip(values, -1, 1)
    return np.clip(values, -1, 1)


def to_position(signal: SignalArray) -> SignalArray:
    xp = xp_from_array(signal)
    if xp.__name__ == "cupy":  # pragma: no cover - GPU path
        return xp.sign(signal)
    return np.sign(signal)
