from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from .preprocessor import ensure_ohlcv_columns

RollMethod = Literal["difference", "ratio"]


@dataclass
class ContinuousContractBuilder:
    """构建连续合约，支持价差或比例回补。"""

    roll_method: RollMethod = "difference"

    def build(
        self,
        contracts: list[pd.DataFrame],
        roll_dates: list[pd.Timestamp] | None = None,
    ) -> pd.DataFrame:
        if not contracts:
            raise ValueError("contracts 不能为空")
        normalized = [ensure_ohlcv_columns(frame) for frame in contracts]
        if len(normalized) == 1:
            return normalized[0].copy()

        prepared = [frame.set_index("datetime").sort_index() for frame in normalized]
        if roll_dates is None:
            roll_dates = self._auto_roll_dates(prepared)
        if len(roll_dates) != len(prepared) - 1:
            raise ValueError("roll_dates 数量必须是合约数量减一")

        result = prepared[0].copy()
        for idx in range(1, len(prepared)):
            next_contract = prepared[idx]
            roll_dt = pd.Timestamp(roll_dates[idx - 1], tz="UTC")

            current_slice = result[result.index < roll_dt]
            next_slice = next_contract[next_contract.index >= roll_dt]
            if next_slice.empty:
                continue

            bridge = self._compute_bridge(result, next_contract, roll_dt)
            adjusted_next = self._apply_bridge(next_slice, bridge)
            result = pd.concat([current_slice, adjusted_next], axis=0)

        result = result[~result.index.duplicated(keep="last")]
        result = result.sort_index().reset_index()
        return result

    def _auto_roll_dates(self, contracts: list[pd.DataFrame]) -> list[pd.Timestamp]:
        dates: list[pd.Timestamp] = []
        for prev_frame, next_frame in zip(contracts[:-1], contracts[1:], strict=True):
            start = next_frame.index.min()
            if pd.isna(start):
                raise ValueError("后续合约无有效起始时间")
            dates.append(pd.Timestamp(start))
        return dates

    def _compute_bridge(
        self,
        prev_contract: pd.DataFrame,
        next_contract: pd.DataFrame,
        roll_dt: pd.Timestamp,
    ) -> float:
        prev_close = prev_contract.loc[prev_contract.index < roll_dt, "close"]
        next_close = next_contract.loc[next_contract.index >= roll_dt, "close"]
        if prev_close.empty or next_close.empty:
            return 0.0
        p = float(prev_close.iloc[-1])
        n = float(next_close.iloc[0])
        if self.roll_method == "ratio":
            return 1.0 if p == 0 else n / p
        return n - p

    def _apply_bridge(self, frame: pd.DataFrame, bridge: float) -> pd.DataFrame:
        adjusted = frame.copy()
        if self.roll_method == "ratio":
            if bridge == 0:
                return adjusted
            for col in ("open", "high", "low", "close"):
                adjusted[col] = adjusted[col] / bridge
            return adjusted

        for col in ("open", "high", "low", "close"):
            adjusted[col] = adjusted[col] - bridge
        return adjusted
