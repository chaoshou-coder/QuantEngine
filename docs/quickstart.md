# 快速开始

本页目标是：从零开始在 10 分钟内完成首次回测执行。

## 1. 环境准备

1. 安装 Python 3.11+。
2. 安装依赖：

```bash
cd <project-root>
pip install -e .          # 仅核心
pip install -e .[engine]  # 含 GPU/可视化/优化器（推荐）
```

## 2. 准备配置文件

将示例配置拷贝为运行配置，并按需修改：

```bash
# Windows
copy quantengine.example.yaml quantengine.yaml
# Linux / macOS
cp quantengine.example.yaml quantengine.yaml
```

关键字段说明：

- `runtime.backend`: `auto|cpu|gpu`，默认 `auto`
- `runtime.use_gpu`: `true|false`，默认为 `true`
- `runtime.initial_cash`: 初始资金
- `runtime.periods_per_year`: 年化换算周期（当前版本 XAUUSD 与外汇常用值 `358800`）
- `runtime.preset_periods`: 可用预设 `minute_intraday|hour_intraday|daily|forex_intraday_1m`
- `optimize.metric`: 优化目标（如 `sharpe`）

`preset_periods` 会在启动时覆盖 `periods_per_year`，若同时设置，两者最终保持一致。

## 3. 准备数据文件

`DataLoader` 接受目录或单文件，支持 `.csv` / `.parquet`，默认会按 symbol 目录聚合。

示例目录：

```text
tests/fixtures/
  └─ sample_500_bars.csv      # 仓库内置样例，可直接用于首次回测
```

每个文件必须包含以下字段（会自动规范为小写）：

- `datetime`（或 `date` / `index`，会被重命名为 `datetime`）
- `open`
- `high`
- `low`
- `close`
- `volume`

`datetime` 会转为 UTC 时区并去重排序；时间未齐时会按并集对齐；缺失 `open/high/low/close` 先前向填充，再向后补一次；`volume` 缺失补 0。

首次运行时还会在数据目录创建 `.quantengine_cache`，后续重复加载将优先命中缓存；可在代码中通过 `DataLoader(cache=False)` 显式关闭。

## 4. 首次 backtest（CLI）

按以下命令可直接完成一次回测并输出 JSON：

```bash
quantengine --config quantengine.yaml backtest \
  --strategy sma_cross \
  --data tests/fixtures/sample_500_bars.csv \
  --params '{"fast":10,"slow":30}' \
  --output ./results/backtest_first.json
```

回测执行成功后可再输出 HTML 报告：

```bash
quantengine --config quantengine.yaml backtest \
  --strategy sma_cross \
  --data tests/fixtures/sample_500_bars.csv \
  --params '{"fast":10,"slow":30}' \
  --output ./results/backtest_first.json \
  --report ./reports/backtest_first.html
```

## 5. 首次 optimize（CLI）

先确认指标与优化方法：

```bash
quantengine --config quantengine.yaml optimize \
  --strategy sma_cross \
  --data tests/fixtures/sample_500_bars.csv \
  --method random \
  --n-trials 50 \
  --metric sharpe \
  --output ./results/opt_random.json \
  --report ./reports/opt_random.html
```

`--param-grid` 仅在 `grid` 方法可用，例如：

```bash
quantengine --config quantengine.yaml optimize \
  --strategy rsi_mean_reversion \
  --data tests/fixtures/sample_500_bars.csv \
  --method grid \
  --param-grid '{"window":[6,12,18], "lower":[25,30], "upper":[70,75]}' \
  --output ./results/opt_grid.json \
  --report ./reports/opt_grid.html
```

## 6. Walk-Forward 首次体验（CLI）

Walk-Forward 用于检查参数稳定性与抗过拟合：

```bash
quantengine --config quantengine.yaml walk-forward \
  --strategy sma_cross \
  --data tests/fixtures/sample_500_bars.csv \
  --n-splits 5 \
  --is-ratio 0.7 \
  --method bayesian \
  --n-trials 20 \
  --metric sharpe \
  --output ./results/wf.json \
  --report ./reports/wf.html
```

## 7. 从 JSON 复用生成报告

```bash
quantengine report \
  --result ./results/backtest_first.json \
  --output ./reports/backtest_from_json.html

quantengine report \
  --result ./results/opt_random.json \
  --output ./reports/opt_from_json.html
```

## 8. Python API 快速上手

```python
from quantengine import QuantEngineAPI

api = QuantEngineAPI(config_path="quantengine.yaml")

report = api.backtest(
    strategy_name="sma_cross",
    data_path="tests/fixtures/sample_500_bars.csv",
    params={"fast": 10, "slow": 30},
)

api.generate_backtest_report(report, output_file="./reports/api_backtest.html")

wf_result = api.walk_forward(
    strategy_name="sma_cross",
    data_path="tests/fixtures/sample_500_bars.csv",
    n_splits=5,
    in_sample_ratio=0.7,
    n_trials=20,
    show_progress=False,
)
print(wf_result.overfitting_ratio)
```

## 9. 常见自检

```bash
quantengine check-deps           # 依赖完整性检查
quantengine --config quantengine.yaml list-strategies
quantengine --help
quantengine report --help
```

完成以上 9 步，即可确认首跑链路打通。
