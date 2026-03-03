from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None


def save_param_heatmap(
    x_values: list[float],
    y_values: list[float],
    z_matrix: np.ndarray,
    output_path: str | Path,
    title: str = "Parameter Heatmap",
    x_label: str = "x",
    y_label: str = "y",
) -> Path:
    if plt is None:
        raise RuntimeError("matplotlib 未安装，无法输出热力图")
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    image = ax.imshow(z_matrix, origin="lower", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(range(len(x_values)))
    ax.set_xticklabels([str(v) for v in x_values], rotation=45, ha="right")
    ax.set_yticks(range(len(y_values)))
    ax.set_yticklabels([str(v) for v in y_values])
    fig.colorbar(image, ax=ax)
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out
