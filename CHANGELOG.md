# 变更日志

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

