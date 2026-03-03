# QuantEngine CLI 命令手册

## 命令总览

`quantengine` 使用 Click 实现，默认入口：

```bash
quantengine --help
```

全局参数：

- `--config <path>`：可选，加载 YAML 配置。未提供时使用内置默认配置。

子命令：

- `list-strategies`
- `backtest`
- `optimize`
- `walk-forward`
- `report`

## list-strategies

列出已注册的策略名称（例如 `sma_cross`、`rsi_mean_reversion`）。

```bash
quantengine --config quantengine.example.yaml list-strategies
```

## backtest

执行单策略或多策略回测。

```bash
quantengine [--config PATH] backtest [选项]
```

参数说明：

- `--strategy`：单策略名称；当未传 `--strategies` 时必填。
- `--strategies`：多策略名称，英文逗号分隔。示例：`sma_cross,rsi_mean_reversion`。
- `--data`（必填）：数据目录或文件路径。
- `--params`：策略参数 JSON。默认 `{}`。
  - 单策略：对象，例如 `{"fast":10,"slow":30}`。
  - 多策略：对象映射，键是策略名，例如 `{"sma_cross":{"fast":10},"rsi_mean_reversion":{"window":14}}`。
- `--symbols`：symbol 过滤，多值英文逗号分隔。未传则加载所有符号。
- `--output`：结果 JSON 输出路径。未传时默认为 `results_root/backtest_YYYYMMDD_HHMMSS.json`。
- `--report`：可选，生成 HTML 报告路径。

执行示例：

```bash
quantengine --config quantengine.yaml backtest \
  --strategy sma_cross \
  --data ./data \
  --params '{"fast":10,"slow":30}' \
  --symbols ES \
  --output ./results/backtest.json \
  --report ./reports/backtest.html
```

多策略示例：

```bash
quantengine --config quantengine.yaml backtest \
  --strategies "sma_cross,rsi_mean_reversion" \
  --params '{"sma_cross":{"fast":10,"slow":30},"rsi_mean_reversion":{"window":14}}' \
  --data ./data \
  --symbols ES
```

输出 JSON 示例字段：

- `type`: 固定为 `backtest`
- `strategy`: 策略名或 `name1+name2` 形式的组合名
- `params`: 本次参数
- `performance`: 性能指标（`total_return`、`sharpe`、`sortino` 等）
- `risk`: 风险指标（`var_95`、`cvar_95`、`ulcer_index`）
- `trade_metrics`: 交易统计（`trade_count` 等）
- `portfolio`: 含 `equity_curve`、`returns`、`turnover`、`trades`、`metadata`
- `timestamps`: 时间序列

## optimize

执行参数优化搜索，支持四种方法。

```bash
quantengine [--config PATH] optimize [选项]
```

参数说明：

- `--strategy`（必填）：策略名称
- `--data`（必填）：数据目录或文件路径
- `--method`：`grid|random|bayesian|genetic`，默认 `bayesian`
- `--n-trials`：本次搜索试验数；缺省使用 `optimize.n_trials`（默认 200）
- `--metric`：目标指标名称；缺省使用配置中的 `optimize.metric`
- `--minimize`：是否最小化目标指标（默认 false，表示最大化）
- `--symbols`：symbol 过滤
- `--param-grid`：仅 `grid` 生效，JSON 形式参数网格，例如 `{"fast":[5,10],"slow":[20,30]}`
- `--output`：优化结果 JSON 路径，默认 `results_root/opt_<method>_YYYYMMDD_HHMMSS.json`
- `--report`：可选，生成优化报告 HTML

执行示例：

```bash
quantengine --config quantengine.yaml optimize \
  --strategy sma_cross \
  --data ./data \
  --method random \
  --n-trials 100 \
  --metric sharpe \
  --symbols ES \
  --output ./results/opt_random.json \
  --report ./reports/opt_random.html
```

输出 JSON 示例字段：

- `method`: `grid|random|bayesian|genetic`
- `metric`: 目标指标
- `maximize`: 是否最大化
- `best_params`: 当前最优参数
- `best_score`: 最优分数
- `trials`: 每个试验的 `params` 和 `score` 列表

默认行为：

- CLI 的 `optimize` 默认开启进度条（`show_progress=True`），执行过程中会显示任务名、完成比例、已耗时与剩余时长。

## walk-forward

执行 Walk-Forward 防过拟合分析。每一折按时间切分 IS/OOS，先在 IS 内做参数优化，再在 OOS 上做回测验证。

```bash
quantengine [--config PATH] walk-forward [选项]
```

参数说明：

- `--strategy`（必填）：策略名称
- `--data`（必填）：数据目录或文件路径
- `--n-splits`：折数，默认 `5`
- `--is-ratio`：IS 区间占比，默认 `0.7`
- `--method`：`grid|random|bayesian|genetic`，默认 `bayesian`
- `--n-trials`：每折优化次数；缺省使用 `optimize.n_trials`
- `--metric`：目标指标；缺省使用配置中的 `optimize.metric`
- `--minimize`：是否最小化目标
- `--symbols`：symbol 过滤，英文逗号分隔
- `--expanding`：是否启用 expanding 窗口（默认 false）
- `--output`：WF 结果 JSON 路径，默认 `results_root/walk_forward_YYYYMMDD_HHMMSS.json`
- `--report`：Walk-Forward HTML 报告路径

执行示例：

```bash
quantengine --config quantengine.yaml walk-forward \
  --strategy sma_cross \
  --data tests/fixtures/sample_500_bars.csv \
  --n-splits 5 \
  --is-ratio 0.7 \
  --method bayesian \
  --n-trials 40 \
  --metric sharpe \
  --output ./results/wf.json \
  --report ./reports/wf.html
```

输出 JSON 示例字段（`type=walk_forward`）：

- `type`：固定为 `walk_forward`
- `config`：`n_splits`、`in_sample_ratio`、`expanding`、`optimization_method`、`n_trials`
- `folds`：每折明细（`fold_index`、IS/OOS 范围、`best_params`、`is_report`、`oos_report`）
- `aggregate_oos_performance`：OOS 平均性能指标
- `aggregate_oos_risk`：OOS 平均风险指标
- `overfitting_ratio`：IS/OOS 指标比值平均

## report

基于已有 backtest/optimization 结果重新生成 HTML 报告。

```bash
quantengine report \
  --result <json_path> \
  --output <html_path>
```

`result` 文件必须包含 `type` 字段：

- `backtest`：调用回测报告渲染
- `optimization`：调用优化报告渲染
- 其他值会报错 `不支持的 result type`。

Walk-Forward 报告请通过 `walk-forward --report` 直接生成，当前 `report` 命令不支持 `type=walk_forward`。

## 错误与常见提示

- JSON 参数解析失败：`--params` / `--param-grid` 请保持合法 JSON（双引号）。
- 策略名错误：检查 `list-strategies` 输出。
- 配置加载失败：`--config` 文件不存在、YAML 不合法或字段校验失败时会抛 `ConfigError`。
