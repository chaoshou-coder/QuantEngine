from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

try:
    import plotly.graph_objects as go  # type: ignore
except Exception:  # pragma: no cover
    go = None


def build_equity_figure(
    timestamps: Sequence,
    equity_curve: np.ndarray,
    title: str = "Equity Curve",
):
    if go is None:
        raise RuntimeError("plotly 未安装，无法生成交互图")
    equity = np.asarray(equity_curve, dtype=float)
    running_max = np.maximum.accumulate(equity)
    drawdown = (equity - running_max) / np.maximum(running_max, 1e-12)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(timestamps), y=equity, mode="lines", name="Equity"))
    fig.add_trace(go.Scatter(x=list(timestamps), y=drawdown, mode="lines", name="Drawdown", yaxis="y2"))
    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Equity",
        yaxis2=dict(
            title="Drawdown",
            overlaying="y",
            side="right",
            tickformat=".1%",
        ),
        legend=dict(orientation="h"),
    )
    return fig


def save_equity_figure(
    timestamps: Sequence,
    equity_curve: np.ndarray,
    output_path: str | Path,
    title: str = "Equity Curve",
) -> Path:
    if go is None:
        raise RuntimeError("plotly 未安装，无法生成交互图")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig = build_equity_figure(timestamps=timestamps, equity_curve=equity_curve, title=title)
    fig.write_html(out, include_plotlyjs="cdn")
    return out
