# 变更日志

本文件遵循 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/) 格式。

## [0.2.0] - 2026-03-05

### 新增

#### Sprint 0：Agentic 基建 + GPU 混合

- GPU 混合架构：CPU 信号生成 + GPU 批量仿真（`simulate_portfolio_batch`、`evaluate_signal_tensor`）
- 无 CUDA 时自动回退 CPU，保证功能完整性
- Numba JIT 加速关键路径（`sma`、`ema`、`wilder_smooth`、`simulate_portfolio`）

#### Sprint 1：风控 + 成本

- 风控引擎扩展：`TradingRules` 新增 7 字段（`max_risk_per_trade`、`max_daily_loss`、`max_weekly_loss`、`max_drawdown_limit`、`max_drawdown_action`、`max_position`、`max_addon_count`）
- Numba 集成：风控检查在 `simulate_portfolio` 循环内执行，输出 `risk_events`
- 成本多情景：`CostScenario` + `run_cost_scenarios`，支持低/中/高三档滑点与手续费批量评估

#### Sprint 2：审计 + 报告

- 审计包：`quantengine.audit` 模块，`AuditBundle` ZIP 结构（config、trades、equity_curve、risk_events 等）
- round-trip 校验：`verify_audit_bundle`、`replay_from_bundle` 支持 bit-identical 复现
- 报告系统：8 区块（绩效/风险/交易/权益/敏感度等）、BH 校正、参数敏感度、可选 PDF

#### Sprint 3：DSL + 合约

- 策略 DSL：YAML 格式，`strategy.name/framework/risk_mode/params/weight`，V4 全覆盖
- 合约规格：`quantengine.contracts` 模块，`ContractSpec` 抽象（multiplier、min_tick、trading_sessions）
- 指标注册表支持版本号

### 变更

- `BacktestReport` 新增 `audit_bundle` 可选字段
- `PortfolioResult` 新增 `risk_events` 字段
- CLI `backtest` 支持 `--audit-bundle` 输出审计包

### 文档

- 新增 `docs/features/` 功能文档（风控、成本情景、审计包、报告、DSL、GPU）
- 更新 `docs/architecture/overview.md` 涵盖 audit、contracts、DSL、风控、成本、GPU
- 更新 `docs/testing/acceptance.md` 反映 20 核心测试模块、130+ 用例

---

## [0.1.0] - 2026-03-03

### 新增

- 回测引擎与优化器完成批量化重构：新增 `simulate_portfolio_batch`（`portfolio.py`）实现参数组合批量仿真。
- 新增 `metrics/batch.py`，提供 `batch_score` 和 `sharpe/sortino/max_drawdown/total_return/annualized_return/calmar` 的向量化批量指标计算。
- 优化器集成 `evaluate_batch`（`parallel.py`），`grid/random` CPU 路径默认走批量链路以降低重复回测开销。
- 新增/修复 `sma`/`ema`/`wilder_smooth` 的 GPU + 无 Numba 场景处理，`to_numpy` + CPU 回退后再返回原 backend。
- 指标层完成 GPU 感知改造：`performance.py` 与 `risk.py` 全面改为 `xp_from_array`，避免无意识的 CPU 拷贝。

### 测试与验收

- 新增并完善慢速验收用例（`@pytest.mark.slow`）：`tests/test_acceptance.py` 覆盖 27 个场景，包含性能基准回归测试。
- 补充 `test_batch_optimizer_performance`，约束 `evaluate_batch` 相对串行基线的性能优势。
- 修复加载异常导致的 NaN 与缓存写入问题，确保真实数据可连续运行。

### 文档

- 完成“性能短板补齐计划验收 + 文档完善”说明文档，梳理优化链路与验收动作。
- 完善模块映射、优化器并行说明、架构优化链路、验收说明与 quickstart 跨平台指引。

