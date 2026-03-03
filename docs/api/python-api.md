# Python API 说明

`QuantEngineAPI` 提供面向程序化调用的主接口，适合回测流水线和服务化接入。

```python
from quantengine import QuantEngineAPI
```

## QuantEngineAPI

### 初始化

```python
api = QuantEngineAPI(config_path=None)
```

参数：

- `config_path`：可选 YAML 路径。为 `None` 时使用默认配置。

初始化流程：

1. 加载配置文件（`load_config`）。
2. 初始化内部 `QuantEngineConfig`（运行参数、滑点、手续费、优化与可视化配置）。

### 方法

#### `backtest(strategy_name, data_path, params, symbols=None)`

执行单策略回测。

参数：

- `strategy_name`（str）：已注册策略名，例如 `sma_cross`。
- `data_path`（str/Path）：数据目录或文件。
- `params`（dict）：策略参数。
- `symbols`（list[str] | None）：按 symbol 白名单过滤。

返回值：`BacktestReport`，包含以下字段：

- `strategy`
- `params`
- `portfolio`（含 `equity_curve` / `returns` / `turnover` / `trades`）
- `performance`
- `risk`
- `trade_metrics`

#### `backtest_multi(strategy_specs, data_path, symbols=None)`

执行多策略按权重聚合回测。

- `strategy_specs`：`[(strategy_name, params, weight), ...]`
- 输出同 `backtest(...)` 的 `BacktestReport`，`strategy` 显示组合名称，`params` 为 `{"portfolio": "multi_strategy"}`。

#### `optimize(strategy_name, data_path, method, n_trials=None, metric=None, maximize=None, symbols=None, show_progress=False)`

执行参数优化。  
`method` 支持 `grid`、`random`、`bayesian`、`genetic`。

参数默认采用配置中的 `optimize` 段：

- `metric` 默认 `optimize.metric`
- `maximize` 默认 `optimize.maximize`
- `n_trials` 默认 `optimize.n_trials`
- `show_progress` 控制是否显示 Rich 进度条，默认 `False`

返回值：`OptimizationResult`，包含：

- `method`
- `metric`
- `maximize`
- `best_params`
- `best_score`
- `trials`（每次评估）

#### `walk_forward(strategy_name, data_path, n_splits=5, in_sample_ratio=0.7, method='bayesian', n_trials=None, metric=None, maximize=None, symbols=None, expanding=False, show_progress=False)`

执行 Walk-Forward 走时验证。

- `n_splits`：折数
- `in_sample_ratio`：IS 区间占比（0~1）
- `method`：`grid|random|bayesian|genetic`
- `n_trials`：每折优化试验数，默认沿用 `optimize.n_trials`
- `metric`：目标指标，默认配置项 `optimize.metric`
- `maximize`：是否最大化，默认配置项 `optimize.maximize`
- `expanding`：是否采用 expanding 窗口
- `show_progress`：是否显示 Rich 进度条

返回值：`WalkForwardResult`

- `config`：`WalkForwardConfig`
- `folds`：每折结果列表，每项为 `WalkForwardFold`
- `aggregate_oos_performance`：OOS 性能指标平均值
- `aggregate_oos_risk`：OOS 风险指标平均值
- `overfitting_ratio`：IS/OOS 指标比值平均，接近 1.0 表示过拟合压力较低

#### `generate_backtest_report(report, output_file)`

将 `BacktestReport` 生成 HTML 报告。

#### `generate_optimization_report(result, output_file)`

将 `OptimizationResult` 生成 HTML 报告。

## 与 CLI 行为一致性

CLI 的 `backtest`/`optimize` 内部逻辑与 API 调用同一套 `DataLoader` 与 `BacktestEngine`，参数优先级如下：

1. CLI 显式参数
2. YAML 配置中的对应字段
3. 代码内置默认值

## 数据结构说明（新增）

### `WalkForwardConfig`

- `n_splits`：折数
- `in_sample_ratio`：IS 占比
- `expanding`：是否 expanding
- `optimization_method`：内部优化方法
- `n_trials`：每折优化试验数

### `WalkForwardFold`

- `fold_index`：折号
- `is_start`, `is_end`, `oos_start`, `oos_end`：索引区间
- `best_params`：该折最优参数
- `is_report` / `oos_report`：对应回测报告

### `WalkForwardResult`

- `folds`：`WalkForwardFold` 列表
- `aggregate_oos_performance`：OOS 平均性能
- `aggregate_oos_risk`：OOS 平均风险
- `overfitting_ratio`：过拟合比
- `config`：`WalkForwardConfig`

## 结果字段补充

### `BacktestReport.trade_metrics`

字段类型更新为 `dict[str, float | str]`，典型键包括：

- `trade_count`
- `win_rate`
- `profit_factor`
- `avg_profit`
- `avg_loss`
- `max_consecutive_losses`

### DataLoader 缓存行为

API 内部默认用 `DataLoader(backend=..., use_gpu=...)`，缓存默认开启（`cache=True`），并优先复用 `.quantengine_cache`。
如需禁用，可直接实例化 `DataLoader(..., cache=False)`。

## 与实盘接口

工程中预留了实盘对接抽象基类 `LiveAdapter`，位于 `quantengine.interface.live_adapter`。

接口方法：

- `connect()`
- `disconnect()`
- `submit_order(order: LiveOrder) -> str`
- `cancel_order(order_id: str) -> None`
- `query_fill(order_id: str) -> LiveFill | None`

数据模型：

- `LiveOrder(symbol, side, quantity, order_type="MARKET", limit_price=None)`
- `LiveFill(order_id, symbol, side, quantity, avg_price, status)`

`LiveSide` 支持 `BUY/SELL`，`order_type` 支持 `MARKET/LIMIT`。

## 推荐接入模式

- 回测主路径：`QuantEngineAPI` + 任务调度器（Airflow/Cron）
- 运维路径：JSON/HTML 产物落盘 + 监控告警
- 研究路径：策略脚本直接调用 API，输出 `BacktestReport` 做后处理
