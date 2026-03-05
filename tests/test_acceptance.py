from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import json
import time
from typing import Iterator

import yaml
import numpy as np
import pytest

from click.testing import CliRunner

from quantengine.config import QuantEngineConfig
from quantengine.data.loader import DataBundle, DataLoader
from quantengine.engine.factory import build_engine
from quantengine.cli import main
from quantengine.interface.api import QuantEngineAPI
from quantengine.optimizer import GridSearchOptimizer, RandomSearchOptimizer
from quantengine.optimizer.parallel import evaluate_trials_parallel
from quantengine.engine import portfolio as portfolio_module
from quantengine.strategy import ParameterSpace, get_strategy
from quantengine.strategy.base import cartesian_from_spaces
from quantengine.visualization.reports import write_backtest_report_html, write_optimization_report_html

pytestmark = pytest.mark.slow

TEST_DATA_DIR = Path(__file__).resolve().parents[1] / "test_data"
DEFAULT_DATA_FILE = "XAUUSD_1m_20170101_20180101.csv"


def _resolve_test_data_dir() -> Path:
    candidates = [
        TEST_DATA_DIR,
        Path(__file__).resolve().parents[1] / "test_data",
        Path.cwd() / "test_data",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("未找到 test_data 目录，请确认路径为 E:\\code\\BTEngine\\test_data 或仓库根目录下有 test_data")


def _trim_leading_missing_rows(data: DataBundle) -> DataBundle:
    valid_rows = np.isfinite(data.open).all(axis=1) & np.isfinite(data.high).all(axis=1)
    valid_rows &= np.isfinite(data.low).all(axis=1) & np.isfinite(data.close).all(axis=1)
    if not valid_rows.any():
        raise AssertionError("数据全为 NaN，无法继续验收")
    start = int(np.argmax(valid_rows))
    return DataBundle(
        symbols=data.symbols,
        timestamps=data.timestamps[start:],
        open=data.open[start:],
        high=data.high[start:],
        low=data.low[start:],
        close=data.close[start:],
        volume=data.volume[start:],
        backend=data.backend,
    )


@contextmanager
def _temporary_strategy_parameters(strategy_name: str, parameters: dict[str, ParameterSpace]) -> Iterator[None]:
    strategy = get_strategy(strategy_name)
    strategy_cls = type(strategy)
    original = strategy_cls.parameters

    def _params(_self=None):
        return parameters

    # 同时注入到类级别，确保优化器每次通过 get_strategy() 创建的新实例
    # 都会拿到测试指定的参数空间定义。
    strategy_cls.parameters = _params  # type: ignore[method-assign]
    try:
        yield
    finally:
        strategy_cls.parameters = original


def _take_head(data: DataBundle, max_rows: int) -> DataBundle:
    if data.timestamps.shape[0] <= max_rows:
        return data
    return DataBundle(
        symbols=data.symbols,
        timestamps=data.timestamps[:max_rows],
        open=data.open[:max_rows],
        high=data.high[:max_rows],
        low=data.low[:max_rows],
        close=data.close[:max_rows],
        volume=data.volume[:max_rows],
        backend=data.backend,
    )


def _to_iso_timestamp(ts: object) -> str:
    try:
        dt = np.datetime64(ts)
    except (TypeError, ValueError):
        return str(ts)
    return np.datetime_as_string(dt.astype("datetime64[ns]"), unit="ns")


@pytest.fixture(scope="module")
def data_dir() -> Path:
    return _resolve_test_data_dir()


@pytest.fixture(scope="module")
def single_file(data_dir: Path) -> Path:
    return data_dir / DEFAULT_DATA_FILE


@pytest.fixture(scope="module")
def single_bundle(single_file: Path) -> DataBundle:
    loader = DataLoader(backend="cpu", use_gpu=False)
    return loader.load(single_file)


@pytest.fixture(scope="module")
def full_bundle(data_dir: Path) -> DataBundle:
    loader = DataLoader(backend="cpu", use_gpu=False)
    return loader.load(data_dir)


@pytest.fixture(scope="module")
def valid_single_bundle(single_bundle: DataBundle) -> DataBundle:
    return _trim_leading_missing_rows(single_bundle)


@pytest.fixture(scope="module")
def optimization_bundle(valid_single_bundle: DataBundle) -> DataBundle:
    return _take_head(valid_single_bundle, max_rows=120_000)


@pytest.fixture(scope="module")
def engine():
    cfg = QuantEngineConfig()
    cfg.runtime.backend = "cpu"
    cfg.runtime.use_gpu = False
    return build_engine(cfg)


@pytest.fixture(scope="module")
def cli_config_path(tmp_path_factory: pytest.TempPathFactory) -> Path:
    cfg = QuantEngineConfig()
    cfg.runtime.backend = "cpu"
    cfg.runtime.use_gpu = False
    cfg.optimize.n_trials = 3
    path = tmp_path_factory.mktemp("acceptance-config") / "quantengine_cpu.yaml"
    path.write_text(yaml.safe_dump(cfg.model_dump()), encoding="utf-8")
    return path


def test_load_single_csv(single_bundle: DataBundle):
    assert single_bundle.symbols == ["TEST_DATA"]
    assert single_bundle.open.shape[0] > 100_000
    assert len(single_bundle.timestamps) == single_bundle.open.shape[0]
    assert single_bundle.open.shape[1] == 1
    assert single_bundle.close.shape == single_bundle.open.shape


def test_load_full_directory(full_bundle: DataBundle):
    assert full_bundle.open.shape[0] > 500_000
    assert full_bundle.symbols == ["TEST_DATA"]
    tss = np.asarray(full_bundle.timestamps, dtype="datetime64[ns]")
    assert tss[0] <= np.datetime64("2017-01-01")
    assert tss[-1] >= np.datetime64("2019-12-30")


def test_nan_handling(single_bundle: DataBundle):
    valid = _trim_leading_missing_rows(single_bundle)
    assert not np.isnan(valid.open).any()
    assert not np.isnan(valid.high).any()
    assert not np.isnan(valid.low).any()
    assert not np.isnan(valid.close).any()
    assert not np.isnan(valid.volume).any()


def test_timestamps_sorted_unique(single_bundle: DataBundle):
    tss = np.asarray(single_bundle.timestamps, dtype="datetime64[ns]")
    assert np.all(np.diff(tss) > np.timedelta64(0, "ns"))
    assert len(tss) == len(np.unique(tss))


def test_sma_cross_backtest(engine, valid_single_bundle: DataBundle):
    report = engine.run(valid_single_bundle, get_strategy("sma_cross"), {"fast": 10, "slow": 30})
    assert len(report.portfolio.equity_curve) == valid_single_bundle.timestamps.shape[0]
    assert report.portfolio.equity_curve[0] == engine.initial_cash
    assert isinstance(report.performance, dict)
    assert isinstance(report.risk, dict)
    assert isinstance(report.trade_metrics, dict)


def test_rsi_backtest(engine, valid_single_bundle: DataBundle):
    report = engine.run(
        valid_single_bundle,
        get_strategy("rsi_mean_reversion"),
        {"window": 14, "lower": 30.0, "upper": 70.0},
    )
    assert report.portfolio.equity_curve.shape[0] == valid_single_bundle.timestamps.shape[0]
    assert report.portfolio.equity_curve[0] == engine.initial_cash


def test_backtest_has_trades(engine, valid_single_bundle: DataBundle):
    report = engine.run(valid_single_bundle, get_strategy("sma_cross"), {"fast": 10, "slow": 30}, record_trades=True)
    assert len(report.portfolio.trades) > 0


def test_metrics_are_finite(engine, valid_single_bundle: DataBundle):
    report = engine.run(valid_single_bundle, get_strategy("sma_cross"), {"fast": 10, "slow": 30})
    assert all(isinstance(v, (int, float)) and np.isfinite(v) for v in report.performance.values())
    assert all(isinstance(v, (int, float)) and np.isfinite(v) for v in report.risk.values())
    for value in report.trade_metrics.values():
        if isinstance(value, (int, float)):
            assert np.isfinite(value)


def test_invalid_params_raises(engine, valid_single_bundle: DataBundle):
    with pytest.raises(ValueError, match="fast 必须小于 slow"):
        engine.run(valid_single_bundle, get_strategy("sma_cross"), {"fast": 50, "slow": 30})


def test_multi_strategy_run(engine, valid_single_bundle: DataBundle):
    report = engine.run_multi_strategy(
        data=valid_single_bundle,
        strategies=[
            (get_strategy("sma_cross"), {"fast": 10, "slow": 30}, 0.5),
            (get_strategy("rsi_mean_reversion"), {"window": 14, "lower": 30.0, "upper": 70.0}, 0.5),
        ],
    )
    assert len(report.portfolio.equity_curve) == valid_single_bundle.timestamps.shape[0]
    assert report.portfolio.equity_curve[0] == engine.initial_cash


def test_multi_strategy_name(engine, valid_single_bundle: DataBundle):
    report = engine.run_multi_strategy(
        data=valid_single_bundle,
        strategies=[
            (get_strategy("sma_cross"), {"fast": 10, "slow": 30}, 1.0),
            (get_strategy("rsi_mean_reversion"), {"window": 14, "lower": 30.0, "upper": 70.0}, 1.0),
        ],
    )
    assert "sma_cross" in report.strategy
    assert "rsi_mean_reversion" in report.strategy
    assert "+" in report.strategy


def test_grid_optimizer_small(engine, optimization_bundle: DataBundle):
    spaces = {
        "fast": ParameterSpace(kind="choice", choices=[10, 15]),
        "slow": ParameterSpace(kind="choice", choices=[30, 40]),
    }
    with _temporary_strategy_parameters("sma_cross", spaces):
        optimizer = GridSearchOptimizer(
            engine=engine,
            data=optimization_bundle,
            strategy_factory=lambda: get_strategy("sma_cross"),
            metric="sharpe",
            maximize=True,
            batch_size=4,
        )
        result = optimizer.optimize()

    assert result.best_params
    assert np.isfinite(result.best_score)
    assert len(result.trials) == 4


def test_random_optimizer(engine, optimization_bundle: DataBundle):
    first = RandomSearchOptimizer(
        engine=engine,
        data=optimization_bundle,
        strategy_factory=lambda: get_strategy("sma_cross"),
        n_trials=5,
        random_seed=42,
        metric="sharpe",
        maximize=True,
    ).optimize()

    second = RandomSearchOptimizer(
        engine=engine,
        data=optimization_bundle,
        strategy_factory=lambda: get_strategy("sma_cross"),
        n_trials=5,
        random_seed=42,
        metric="sharpe",
        maximize=True,
    ).optimize()

    assert len(first.trials) == 5
    assert first.best_params == second.best_params
    assert first.best_score == second.best_score


def test_batch_optimizer_performance(engine, optimization_bundle: DataBundle):
    spaces = {
        "fast": ParameterSpace(kind="choice", choices=[10, 15]),
        "slow": ParameterSpace(kind="choice", choices=[30, 40]),
    }
    params_list = cartesian_from_spaces(spaces)

    with _temporary_strategy_parameters("sma_cross", spaces):
        batch_optimizer = GridSearchOptimizer(
            engine=engine,
            data=optimization_bundle,
            strategy_factory=lambda: get_strategy("sma_cross"),
            metric="sharpe",
            maximize=True,
            batch_size=len(params_list),
        )
        start = time.perf_counter()
        batch_trails = batch_optimizer.optimize().trials
        batch_elapsed = time.perf_counter() - start

        serial_start = time.perf_counter()
        original_njit = portfolio_module.njit
        try:
            portfolio_module.njit = None
            serial_trials = evaluate_trials_parallel(
                engine=engine,
                data=optimization_bundle,
                strategy_factory=lambda: get_strategy("sma_cross"),
                params_list=params_list,
                metric="sharpe",
                max_workers=1,
            )
        finally:
            portfolio_module.njit = original_njit
        serial_elapsed = time.perf_counter() - serial_start

    assert len(batch_trails) == len(params_list) == 4
    assert len(serial_trials) == len(params_list)
    assert serial_elapsed > 0 and batch_elapsed < serial_elapsed * 0.5


def test_optimizer_result_structure(engine, optimization_bundle: DataBundle):
    result = RandomSearchOptimizer(
        engine=engine,
        data=optimization_bundle,
        strategy_factory=lambda: get_strategy("rsi_mean_reversion"),
        n_trials=3,
        random_seed=42,
        metric="sharpe",
        maximize=True,
    ).optimize()

    assert result.method == "random"
    assert result.metric == "sharpe"
    assert result.maximize is True
    assert isinstance(result.best_params, dict)
    assert isinstance(result.best_score, float)
    assert isinstance(result.trials, list)
    assert all(hasattr(item, "params") and hasattr(item, "score") for item in result.trials)


def test_backtest_html_report(tmp_path: Path, engine, valid_single_bundle: DataBundle):
    report = engine.run(valid_single_bundle, get_strategy("sma_cross"), {"fast": 10, "slow": 30})
    output = tmp_path / "acceptance_backtest_report.html"
    write_backtest_report_html(report, timestamps=valid_single_bundle.timestamps, output_path=output)
    assert output.exists()
    assert output.stat().st_size > 0
    content = output.read_text(encoding="utf-8")
    assert "Performance" in content


def test_optimization_html_report(tmp_path: Path, engine, optimization_bundle: DataBundle):
    result = RandomSearchOptimizer(
        engine=engine,
        data=optimization_bundle,
        strategy_factory=lambda: get_strategy("sma_cross"),
        n_trials=3,
        random_seed=42,
        metric="sharpe",
    ).optimize()
    output = tmp_path / "acceptance_opt_report.html"
    write_optimization_report_html(result, output_path=output)
    assert output.exists()
    assert output.stat().st_size > 0
    content = output.read_text(encoding="utf-8")
    assert "best_score" in content


def test_backtest_json_roundtrip(valid_single_bundle: DataBundle, engine):
    report = engine.run(valid_single_bundle, get_strategy("sma_cross"), {"fast": 10, "slow": 30})
    payload = {
        "type": "backtest",
        "strategy": report.strategy,
        "params": report.params,
        "performance": report.performance,
        "risk": report.risk,
        "trade_metrics": report.trade_metrics,
        "portfolio": report.portfolio.to_dict(),
        "timestamps": [_to_iso_timestamp(ts) for ts in valid_single_bundle.timestamps],
    }
    raw = json.dumps(payload, allow_nan=False)
    loaded = json.loads(raw)
    assert loaded["type"] == "backtest"
    assert len(loaded["portfolio"]["equity_curve"]) == len(payload["portfolio"]["equity_curve"])


def test_cli_backtest_real_data(single_file: Path, tmp_path: Path, cli_config_path: Path):
    runner = CliRunner()
    output = tmp_path / "cli_backtest.json"
    result = runner.invoke(
        main,
        [
            "--config",
            str(cli_config_path),
            "backtest",
            "--strategy",
            "sma_cross",
            "--data",
            str(single_file),
            "--output",
            str(output),
            "--params",
            json.dumps({"fast": 10, "slow": 30}),
        ],
    )
    assert result.exit_code == 0
    loaded = json.loads(output.read_text(encoding="utf-8"))
    assert loaded["type"] == "backtest"
    assert "performance" in loaded
    assert "risk" in loaded


def test_cli_backtest_with_report(single_file: Path, tmp_path: Path, cli_config_path: Path):
    runner = CliRunner()
    output = tmp_path / "cli_backtest_with_report.json"
    report = tmp_path / "cli_backtest_report.html"
    result = runner.invoke(
        main,
        [
            "--config",
            str(cli_config_path),
            "backtest",
            "--strategy",
            "sma_cross",
            "--data",
            str(single_file),
            "--output",
            str(output),
            "--report",
            str(report),
            "--params",
            json.dumps({"fast": 10, "slow": 30}),
        ],
    )
    assert result.exit_code == 0
    assert report.exists()
    assert report.stat().st_size > 0


def test_cli_optimize_random(single_file: Path, tmp_path: Path, cli_config_path: Path):
    runner = CliRunner()
    output = tmp_path / "cli_optimize.json"
    result = runner.invoke(
        main,
        [
            "--config",
            str(cli_config_path),
            "optimize",
            "--strategy",
            "sma_cross",
            "--data",
            str(single_file),
            "--method",
            "random",
            "--n-trials",
            "3",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["type"] == "optimization"
    assert payload["method"] == "random"
    assert len(payload["trials"]) == 3


def test_cli_report_from_json(single_file: Path, tmp_path: Path, cli_config_path: Path):
    runner = CliRunner()
    backtest_json = tmp_path / "cli_backtest_report_src.json"
    report_html = tmp_path / "cli_report_from_json.html"
    result = runner.invoke(
        main,
        [
            "--config",
            str(cli_config_path),
            "backtest",
            "--strategy",
            "sma_cross",
            "--data",
            str(single_file),
            "--output",
            str(backtest_json),
            "--params",
            json.dumps({"fast": 10, "slow": 30}),
        ],
    )
    assert result.exit_code == 0
    result = runner.invoke(
        main,
        [
            "report",
            "--result",
            str(backtest_json),
            "--output",
            str(report_html),
        ],
    )
    assert result.exit_code == 0
    assert report_html.exists()


def test_api_backtest(single_file: Path, cli_config_path: Path):
    api = QuantEngineAPI(config_path=cli_config_path)
    report = api.backtest(
        strategy_name="sma_cross",
        data_path=single_file,
        params={"fast": 10, "slow": 30},
    )
    assert report.strategy == "sma_cross"
    assert report.portfolio.equity_curve[0] == 1_000_000.0
    assert isinstance(report.performance, dict)


def test_api_optimize(single_file: Path, cli_config_path: Path):
    api = QuantEngineAPI(config_path=cli_config_path)
    result = api.optimize(
        strategy_name="sma_cross",
        data_path=single_file,
        method="random",
        n_trials=3,
        metric="sharpe",
        maximize=None,
    )
    assert result.method == "random"
    assert result.metric == "sharpe"
    assert len(result.trials) == 3


def test_api_report_generation(single_file: Path, tmp_path: Path, cli_config_path: Path):
    api = QuantEngineAPI(config_path=cli_config_path)
    report = api.backtest(
        strategy_name="sma_cross",
        data_path=single_file,
        params={"fast": 10, "slow": 30},
    )
    output = tmp_path / "api_backtest_report.html"
    api.generate_backtest_report(report, output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_single_backtest_under_60s(single_file: Path, engine):
    start = time.perf_counter()
    bundle = _trim_leading_missing_rows(DataLoader(backend="cpu", use_gpu=False).load(single_file))
    _ = engine.run(bundle, get_strategy("sma_cross"), {"fast": 10, "slow": 30})
    elapsed = time.perf_counter() - start
    assert elapsed < 60


def test_full_data_load_under_30s(data_dir: Path):
    start = time.perf_counter()
    _ = DataLoader(backend="cpu", use_gpu=False).load(data_dir)
    elapsed = time.perf_counter() - start
    assert elapsed < 30
