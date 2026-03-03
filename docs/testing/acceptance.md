# 测试与验收标准

本页用于发布前的回归检查。目标是保证接口一致性、回测可复现性和运行稳定性。

## 1. 自动化验收（推荐）

`tests/test_acceptance.py` 为主验收集合，已使用 `@pytest.mark.slow` 标记，发布前执行：

```bash
pytest tests/test_acceptance.py -m slow -v
```

当前版本期望通过 27 个测试，覆盖 8 个类别：

- 数据加载与预处理（4）
- 单策略回测（5）
- 多策略回测（2）
- 优化器（3）
- 报告与可视化（3）
- CLI 端到端（4）
- Python API（3）
- 性能基准（2）

其中新增重点回归点：

- `test_batch_optimizer_performance`：CPU 路径下验证 `evaluate_batch` 与 `evaluate_trials_parallel(max_workers=1)` 的耗时差异（目标：后者至少快 2 倍）
- `test_load_full_directory`：验证三年数据目录加载与时间跨度覆盖

## 2. 核对清单（人工复核）

- 命令与配置：`--help`、`list-strategies`、`backtest`、`optimize`、`walk-forward`、`report`
- 策略与参数：参数错误、参数空间、注册是否符合预期
- 数据处理一致性：排序、去重、填充、缓存命中、回退策略
- 优化行为：`grid`/`random`/`bayesian`/`genetic` 行为与降级策略
- 可视化链路：backtest 与 optimize 报告可生成并可读
- 产物一致性：backtest/optimization 结果必须带 `performance`/`risk`/`trade_metrics`/`portfolio`

## 3. 快速回归命令（非强制）

可以在开发分支合并前执行一次简化回归：

```bash
pytest tests/test_acceptance.py -m slow -k "load_full_directory or load_single_csv or test_batch_optimizer_performance" -q
```

以及 CLI 冒烟命令：

```bash
quantengine --help
quantengine list-strategies
quantengine --config quantengine.example.yaml backtest --strategy sma_cross --data tests/fixtures/sample_500_bars.csv --params '{"fast":10,"slow":30}'
quantengine --config quantengine.example.yaml optimize --strategy sma_cross --data tests/fixtures/sample_500_bars.csv --method random --n-trials 5
quantengine --config quantengine.example.yaml walk-forward --strategy sma_cross --data tests/fixtures/sample_500_bars.csv --n-splits 3 --n-trials 5
```
