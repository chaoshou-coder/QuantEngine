from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from quantengine.cli import main


def _build_sample_csv(path: Path) -> None:
    lines = ["datetime,open,high,low,close,volume"]
    for idx in range(20):
        base = 100.0 + idx
        lines.append(f"2026-03-01T09:{30 + idx:02d}:00,{base:.2f},{base+1:.2f},{base-1:.2f},{base+0.5:.2f},1000")
    path.write_text("\n".join(lines), encoding="utf-8")


def test_cli_list_strategies_ok():
    runner = CliRunner()
    result = runner.invoke(main, ["list-strategies"])
    assert result.exit_code == 0
    assert "sma_cross" in result.output
    assert "rsi_mean_reversion" in result.output


def test_cli_backtest_smoke(tmp_path: Path):
    data_file = tmp_path / "sample.csv"
    output_file = tmp_path / "result.json"
    _build_sample_csv(data_file)

    runner = CliRunner()
    result = runner.invoke(
        main,
        [
            "backtest",
            "--strategy",
            "sma_cross",
            "--data",
            str(data_file),
            "--output",
            str(output_file),
        ],
    )
    assert result.exit_code == 0
    assert output_file.exists()
