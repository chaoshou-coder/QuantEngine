from __future__ import annotations

from pathlib import Path
import hashlib
import pickle
from typing import TYPE_CHECKING

import numpy as np

from quantengine.data.gpu_backend import to_numpy
from .gpu_backend import BackendInfo

if TYPE_CHECKING:
    from .loader import DataBundle


def _normalize_paths(files: list[tuple[str, Path]]) -> list[tuple[str, Path]]:
    normalized: list[tuple[str, Path]] = []
    for symbol, file_path in files:
        resolved = Path(file_path).resolve()
        normalized.append((symbol.upper(), resolved))
    normalized.sort(key=lambda item: (item[0], str(item[1])))
    return normalized


def compute_cache_key(files: list[tuple[str, Path]], symbols: list[str] | None) -> str:
    normalized = _normalize_paths(files)
    token_parts = []
    for symbol, file_path in normalized:
        stat = file_path.stat()
        token_parts.append(f"{symbol}|{file_path.as_posix()}|{int(stat.st_mtime_ns)}|{stat.st_size}")
    if symbols is None:
        token_parts.append("symbols:__all__")
    else:
        token_parts.append("symbols:" + ",".join(sorted({item.strip().upper() for item in symbols if item.strip()})))
    raw = "\n".join(token_parts)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cache_file_path(cache_root: Path, cache_key: str) -> Path:
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"{cache_key}.cache"


def save_to_cache(cache_root: Path, cache_key: str, bundle: DataBundle, backend_info: BackendInfo) -> None:
    cache_path = _cache_file_path(cache_root, cache_key)
    timestamps = np.asarray(bundle.timestamps)
    tmp_npz = cache_path.with_suffix(".npz.tmp")
    tmp_meta = cache_path.with_suffix(".meta.tmp")
    with open(tmp_npz, "wb") as f:
        np.savez_compressed(
            f,
            timestamps=to_numpy(timestamps),
            open=to_numpy(bundle.open),
            high=to_numpy(bundle.high),
            low=to_numpy(bundle.low),
            close=to_numpy(bundle.close),
            volume=to_numpy(bundle.volume),
        )
    metadata = {
        "symbols": list(bundle.symbols),
        "version": backend_info.__dict__.copy(),
        "shape": tuple(np.asarray(bundle.open).shape),
    }
    with open(tmp_meta, "wb") as f:
        pickle.dump(metadata, f)
    tmp_npz.replace(cache_path.with_suffix(".npz"))
    tmp_meta.replace(cache_path.with_suffix(".meta"))


def _load_cache_arrays(npz_path: Path) -> dict[str, np.ndarray]:
    with np.load(npz_path, allow_pickle=False) as data:
        return {
            "timestamps": np.asarray(data["timestamps"]),
            "open": np.asarray(data["open"]),
            "high": np.asarray(data["high"]),
            "low": np.asarray(data["low"]),
            "close": np.asarray(data["close"]),
            "volume": np.asarray(data["volume"]),
        }


def load_from_cache(cache_root: Path, cache_key: str, backend_info: BackendInfo) -> DataBundle | None:
    cache_path = _cache_file_path(cache_root, cache_key)
    npz_path = cache_path.with_suffix(".npz")
    meta_path = cache_path.with_suffix(".meta")
    if not npz_path.exists() or not meta_path.exists():
        return None
    try:
        with open(meta_path, "rb") as f:
            metadata = pickle.load(f)
        arrays = _load_cache_arrays(npz_path)
        # metadata 校验（兼容不同形状/缺失情况）
        symbols = metadata.get("symbols") or []
        if not isinstance(symbols, list):
            return None
        if arrays["open"].shape != tuple(metadata.get("shape", arrays["open"].shape)):
            return None
        return DataBundle(
            symbols=metadata.get("symbols", []),
            timestamps=arrays["timestamps"],
            open=arrays["open"],
            high=arrays["high"],
            low=arrays["low"],
            close=arrays["close"],
            volume=arrays["volume"],
            backend=backend_info,
        )
    except Exception:
        return None
