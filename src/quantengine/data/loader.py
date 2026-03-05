from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import pandas as pd
import polars as pl

from .cache import compute_cache_key, load_from_cache, save_to_cache
from .gpu_backend import BackendInfo, get_backend_info, get_xp
from .preprocessor import align_and_fill, ensure_ohlcv_columns

try:
    import cudf  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cudf = None


BASE_COLUMNS = ["datetime", "open", "high", "low", "close", "volume"]
logger = logging.getLogger(__name__)


@dataclass
class DataBundle:
    symbols: list[str]
    timestamps: np.ndarray
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    volume: np.ndarray
    backend: BackendInfo

    def as_dict(self) -> dict[str, np.ndarray]:
        return {
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }

    def __repr__(self) -> str:
        return f"DataBundle(symbols={len(self.symbols)} bars={self.timestamps.shape[0]} backend={self.backend.active})"

    def slice_by_index(self, start: int, end: int) -> "DataBundle":
        return DataBundle(
            symbols=list(self.symbols),
            timestamps=self.timestamps[start:end],
            open=self.open[start:end],
            high=self.high[start:end],
            low=self.low[start:end],
            close=self.close[start:end],
            volume=self.volume[start:end],
            backend=self.backend,
        )


class DataLoader:
    def __init__(self, backend: str = "auto", use_gpu: bool = True, cache: bool = True):
        self.backend_info = get_backend_info(backend, use_gpu)
        self._xp = get_xp(self.backend_info)
        self._cache_enabled = cache

    def load(
        self,
        path: str | Path,
        symbols: list[str] | None = None,
    ) -> DataBundle:
        source_path = Path(path)
        if not source_path.exists():
            raise FileNotFoundError(f"数据路径不存在: {source_path}")

        files = self._discover_files(source_path)
        if symbols:
            symbol_set = {s.strip() for s in symbols if s.strip()}
            files = [item for item in files if item[0] in symbol_set]
        if not files:
            raise ValueError(f"未发现可用数据文件: {source_path}")

        cache_key = compute_cache_key(files, symbols)
        if self._cache_enabled:
            cache_root = source_path.parent / ".quantengine_cache" if source_path.is_file() else source_path / ".quantengine_cache"
            cached = load_from_cache(cache_root=cache_root, cache_key=cache_key, backend_info=self.backend_info)
            if cached is not None:
                if self.backend_info.active == "gpu" and self._xp is not np:
                    cached = DataBundle(
                        symbols=cached.symbols,
                        timestamps=cached.timestamps,
                        open=self._xp.asarray(cached.open),
                        high=self._xp.asarray(cached.high),
                        low=self._xp.asarray(cached.low),
                        close=self._xp.asarray(cached.close),
                        volume=self._xp.asarray(cached.volume),
                        backend=self.backend_info,
                    )
                return cached

        grouped: dict[str, list[pd.DataFrame]] = {}
        for symbol, file_path in files:
            grouped.setdefault(symbol, []).append(self._read_file(file_path))

        merged: dict[str, pd.DataFrame] = {}
        for symbol, frame_list in grouped.items():
            frame = pd.concat(frame_list, axis=0, ignore_index=True)
            frame = ensure_ohlcv_columns(frame)
            merged[symbol] = frame

        aligned = align_and_fill(merged)
        ordered_symbols = sorted(aligned.keys())
        if not ordered_symbols:
            raise ValueError("无可用对齐数据")

        base_index = pd.Index(aligned[ordered_symbols[0]]["datetime"])
        open_arr = np.column_stack([aligned[s]["open"].to_numpy(dtype=float) for s in ordered_symbols])
        high_arr = np.column_stack([aligned[s]["high"].to_numpy(dtype=float) for s in ordered_symbols])
        low_arr = np.column_stack([aligned[s]["low"].to_numpy(dtype=float) for s in ordered_symbols])
        close_arr = np.column_stack([aligned[s]["close"].to_numpy(dtype=float) for s in ordered_symbols])
        volume_arr = np.column_stack([aligned[s]["volume"].to_numpy(dtype=float) for s in ordered_symbols])
        timestamps = base_index.to_numpy()
        if not (
            np.isfinite(open_arr).all()
            and np.isfinite(high_arr).all()
            and np.isfinite(low_arr).all()
            and np.isfinite(close_arr).all()
            and np.isfinite(volume_arr).all()
        ):
            for idx in range(open_arr.shape[1]):
                o = open_arr[:, idx]
                h = high_arr[:, idx]
                l = low_arr[:, idx]
                c = close_arr[:, idx]
                v = volume_arr[:, idx]
                finite = np.isfinite(o) & np.isfinite(h) & np.isfinite(l) & np.isfinite(c)
                if not finite.any():
                    raise ValueError("OHLCV 对齐后全部为非有效值")
                if not finite[0]:
                    first = int(np.argmax(finite))
                    o[:first] = o[first]
                    h[:first] = h[first]
                    l[:first] = l[first]
                    c[:first] = c[first]
                    v[:first] = v[first]
                for i in range(1, o.shape[0]):
                    if not np.isfinite(o[i]):
                        o[i] = o[i - 1]
                    if not np.isfinite(h[i]):
                        h[i] = h[i - 1]
                    if not np.isfinite(l[i]):
                        l[i] = l[i - 1]
                    if not np.isfinite(c[i]):
                        c[i] = c[i - 1]
                    if not np.isfinite(v[i]):
                        v[i] = v[i - 1]

        self._validate_bundle(
            symbols=ordered_symbols,
            open_arr=open_arr,
            high_arr=high_arr,
            low_arr=low_arr,
            close_arr=close_arr,
            volume_arr=volume_arr,
        )

        xp = self._xp
        if self.backend_info.active == "gpu" and xp is not np:
            open_arr = xp.asarray(open_arr)
            high_arr = xp.asarray(high_arr)
            low_arr = xp.asarray(low_arr)
            close_arr = xp.asarray(close_arr)
            volume_arr = xp.asarray(volume_arr)

        if self._cache_enabled:
            cache_root = source_path.parent / ".quantengine_cache" if source_path.is_file() else source_path / ".quantengine_cache"
            bundle = DataBundle(
                symbols=ordered_symbols,
                timestamps=timestamps,
                open=open_arr,
                high=high_arr,
                low=low_arr,
                close=close_arr,
                volume=volume_arr,
                backend=self.backend_info,
            )
            # 始终按 CPU 数据缓存，避免序列化 GPU 对象带来的兼容问题
            save_to_cache(cache_root=cache_root, cache_key=cache_key, bundle=bundle, backend_info=self.backend_info)
            return bundle

        return DataBundle(
            symbols=ordered_symbols,
            timestamps=timestamps,
            open=open_arr,
            high=high_arr,
            low=low_arr,
            close=close_arr,
            volume=volume_arr,
            backend=self.backend_info,
        )

    def _validate_bundle(
        self,
        symbols: list[str],
        open_arr: np.ndarray,
        high_arr: np.ndarray,
        low_arr: np.ndarray,
        close_arr: np.ndarray,
        volume_arr: np.ndarray,
    ) -> None:
        if not (open_arr.shape == high_arr.shape == low_arr.shape == close_arr.shape == volume_arr.shape):
            raise ValueError("OHLCV 字段形状不一致")
        if open_arr.size == 0:
            raise ValueError("OHLCV 数据为空")
        if not np.isfinite(open_arr).all() or not np.isfinite(high_arr).all() or not np.isfinite(low_arr).all() or not np.isfinite(
            close_arr
        ).all():
            raise ValueError("OHLCV 中存在非有限值(NaN/Inf)")
        if np.any(high_arr < low_arr):
            raise ValueError("high 值必须大于或等于 low 值")
        if np.any(open_arr <= 0.0) or np.any(close_arr <= 0.0) or np.any(high_arr <= 0.0):
            raise ValueError("open/high/close 中存在非正值，行情数据无效")
        if np.any(volume_arr < 0.0):
            raise ValueError("volume 存在负值")
        for idx, symbol in enumerate(symbols):
            col = np.isfinite(volume_arr[:, idx]).all()
            if not col:
                raise ValueError(f"symbol={symbol} volume 中存在非有限值")

    def _discover_files(self, source_path: Path) -> list[tuple[str, Path]]:
        if source_path.is_file():
            symbol = source_path.parent.name.upper()
            return [(symbol, source_path)]

        discovered: list[tuple[str, Path]] = []
        for file_path in source_path.rglob("*"):
            if file_path.suffix.lower() not in {".parquet", ".csv"}:
                continue
            symbol = file_path.parent.name.upper()
            discovered.append((symbol, file_path))
        discovered.sort(key=lambda item: (item[0], str(item[1])))
        return discovered

    def _read_file(self, file_path: Path) -> pd.DataFrame:
        suffix = file_path.suffix.lower()
        if suffix == ".parquet":
            return self._read_parquet(file_path)
        if suffix == ".csv":
            if self.backend_info.active == "gpu" and cudf is not None:
                try:
                    gdf = cudf.read_csv(str(file_path))
                    return gdf.to_pandas()
                except Exception as exc:
                    logger.debug("cudf 读取 CSV 失败，回退到 pandas: %s", exc)
            return pd.read_csv(file_path)
        raise ValueError(f"不支持的文件格式: {file_path.suffix}")

    def _read_parquet(self, file_path: Path) -> pd.DataFrame:
        if self.backend_info.active == "gpu" and cudf is not None:
            try:
                gdf = cudf.read_parquet(str(file_path))
                return gdf.to_pandas()
            except Exception as exc:
                logger.debug("cudf 读取 Parquet 失败，回退到 polars: %s", exc)
        # 优先使用 polars 读取 parquet，兼顾稳定性与性能
        frame = pl.read_parquet(file_path)
        return frame.to_pandas()
