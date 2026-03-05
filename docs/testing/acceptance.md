# 测试与验收标准

本页用于发布前的回归检查。目标是保证接口一致性、回测可复现性和运行稳定性。

## 1. 自动化验收（推荐）

发布前执行非慢速测试（排除 `@pytest.mark.slow`）：

```bash
pytest tests/ -m "not slow" -q
```

当前版本期望通过 20 个核心测试模块（不含依赖 test_data 的验收用例），覆盖以下模块：

| 模块 | 测试文件 | 覆盖内容 |
|------|----------|----------|
| audit | `test_audit_bundle.py`, `test_audit_io.py` | 审计包构建、ZIP 读写、round-trip 校验 |
| contracts | `test_spec.py` | 合约规格解析、默认值、覆盖 |
| engine | `test_commission.py`, `test_cost_scenarios.py`, `test_portfolio.py`, `test_rules.py`, `test_slippage.py` | 手续费、成本情景、组合仿真、风控、滑点 |
| indicators | `test_registry.py`, `test_technical.py` | 指标注册、技术指标 |
| metrics | `test_performance.py`, `test_stability_extras.py` | 绩效、稳定性 |
| strategy | `test_dsl.py` | 策略 DSL 解析、V4 字段 |
| 验收 | （依赖 test_data 的用例已排除） | — |

若需排除特定测试，可使用 `-k "not test_xxx"`。

### 模块覆盖矩阵

| 模块 | 测试数 | 覆盖要点 |
|------|--------|----------|
| audit | 15+ | 审计包构建、ZIP 读写、round-trip、replay |
| contracts | 5+ | ContractSpec 解析、默认值、覆盖 |
| engine | 50+ | 风控、成本情景、组合仿真、滑点、手续费 |
| indicators | 20+ | 技术指标、注册表、版本号 |
| metrics | 15+ | 绩效、稳定性、批量指标 |
| strategy | 15+ | DSL 解析、V4 字段、注册 |
| 验收 | — | 依赖 test_data 的用例不发布 |

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
pytest tests/ -m "not slow" -q
```

以及 CLI 冒烟命令：

```bash
quantengine --help
quantengine list-strategies
quantengine --config quantengine.example.yaml backtest --strategy sma_cross --data tests/fixtures/sample_500_bars.csv --params '{"fast":10,"slow":30}'
quantengine --config quantengine.example.yaml optimize --strategy sma_cross --data tests/fixtures/sample_500_bars.csv --method random --n-trials 5
quantengine --config quantengine.example.yaml walk-forward --strategy sma_cross --data tests/fixtures/sample_500_bars.csv --n-splits 3 --n-trials 5
```
