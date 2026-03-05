from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

REQUIRED_COLUMNS = ("datetime", "open", "high", "low", "close", "volume")


def ensure_ohlcv_columns(frame: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for col in frame.columns:
        lowered = col.strip().lower().replace(" ", "_")
        rename_map[col] = lowered
    normalized = frame.rename(columns=rename_map).copy()

    if "index" in normalized.columns and "datetime" not in normalized.columns:
        normalized = normalized.rename(columns={"index": "datetime"})
    if "date" in normalized.columns and "datetime" not in normalized.columns:
        normalized = normalized.rename(columns={"date": "datetime"})

    missing = [col for col in REQUIRED_COLUMNS if col not in normalized.columns]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"OHLCV 数据缺少字段: {joined}")

    normalized["datetime"] = pd.to_datetime(normalized["datetime"], utc=True, errors="coerce")
    normalized = normalized.dropna(subset=["datetime"]).sort_values("datetime")
    normalized = normalized.drop_duplicates(subset=["datetime"], keep="last")
    return normalized


def align_and_fill(frames: Mapping[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    if not frames:
        return {}
    base_index = sorted(
        set().union(*[set(frame["datetime"].tolist()) for frame in frames.values()])  # type: ignore[arg-type]
    )
    aligned: dict[str, pd.DataFrame] = {}
    for symbol, frame in frames.items():
        tmp = frame.set_index("datetime").reindex(base_index).sort_index()
        tmp[["open", "high", "low", "close"]] = (
            tmp[["open", "high", "low", "close"]].ffill().bfill(limit=1)
        )
        tmp["volume"] = tmp["volume"].fillna(0.0)
        tmp = tmp.reset_index(names="datetime")
        aligned[symbol] = tmp
    return aligned


def resample_ohlcv(frame: pd.DataFrame, rule: str) -> pd.DataFrame:
    normalized = ensure_ohlcv_columns(frame)
    indexed = normalized.set_index("datetime").sort_index()
    out = indexed.resample(rule).agg(
        {
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }
    )
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out.reset_index()
    return out
