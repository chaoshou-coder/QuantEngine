from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from quantengine.data.gpu_backend import xp_from_array


@dataclass
class TradingRules:
    limit_up_ratio: float | None = None
    limit_down_ratio: float | None = None
    margin_ratio: float = 0.1

    def apply_price_limit(self, price: float, prev_close: float) -> float:
        out = price
        if self.limit_up_ratio is not None:
            out = min(out, prev_close * (1.0 + self.limit_up_ratio))
        if self.limit_down_ratio is not None:
            out = max(out, prev_close * (1.0 - self.limit_down_ratio))
        return float(out)

    def apply_price_limit_vector(self, price: Any, prev_close: Any) -> Any:
        xp = xp_from_array(price)
        out = price
        if self.limit_up_ratio is not None:
            out = xp.minimum(out, prev_close * (1.0 + self.limit_up_ratio))
        if self.limit_down_ratio is not None:
            out = xp.maximum(out, prev_close * (1.0 - self.limit_down_ratio))
        return out

    def required_margin(self, notional: float) -> float:
        return float(max(notional, 0.0) * self.margin_ratio)
