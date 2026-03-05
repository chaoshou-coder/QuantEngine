# 架构总览

QuantEngine 采用分层设计：配置驱动、数据层、策略层、引擎层、优化层、可视化/运维输出层。

## 运行链路（回测）

1. `CLI` 或 `QuantEngineAPI` 创建配置。
2. `DataLoader` 按目录读取 CSV/Parquet，先计算缓存键并尝试从 `.quantengine_cache` 命中缓存；未命中则读取文件并标准化 OHLCV。
3. `align_and_fill` 对齐多 symbol 时间线，补齐缺失值。
4. 按策略调用 `generate_signals` 生成信号矩阵。
5. `BacktestEngine` 执行 `simulate_portfolio` 进行资产收益与交易计算。
6. 计算绩效/风险/交易统计。
7. 输出 JSON 与可选 HTML 报告。若命中或构建成功会复用数据缓存。

## 运行链路（优化）

1. 选择策略参数空间（`BaseStrategy.parameters`）。
2. 根据方法（grid/random/bayesian/genetic）生成参数集合。
3. 对参数集合按 batch 切片，按 backend 路线评估：
   - CPU：`simulate_portfolio_batch` + `batch_score`（一次评估一组参数张量）
   - GPU：`signal tensor` + `evaluate_signal_tensor`（参数批次张量化）
4. 选出最佳参数，输出 `OptimizationResult`。
5. 可生成优化报告和 topN 参数列表。
6. 采用 `WalkForwardAnalyzer` 时，会按折分割样本为 IS/OOS，执行“内样本优化 + 外样本验证”，再聚合 OOS 指标和 `overfitting_ratio`。

## 分层职责

- 配置层：`src/quantengine/config.py`
  - 定义运行参数、滑点、手续费、优化与图形配置，支持 YAML 入参。
- 数据层：`src/quantengine/data/*`
  - GPU 后端探测、文件发现、CSV/Parquet 读取、时间对齐、预处理与连续合约工具。
- 策略与指标层：`src/quantengine/strategy/*`、`src/quantengine/indicators/*`
  - 策略注册、参数空间、信号生成和技术指标。
  - `strategy/dsl.py`：策略 YAML DSL，支持 `framework`/`risk_mode`，V4 全覆盖。
- 引擎层：`src/quantengine/engine/*`
  - 滑点、手续费、交易规则、交易执行、投资组合仿真和总线回测引擎封装。
  - **风控扩展**：`TradingRules` 7 字段（单笔风险、日内/周亏损、回撤阈值、最大持仓、加仓次数），Numba 循环内检查，输出 `risk_events`。
  - **成本多情景**：`CostScenario` + `run_cost_scenarios`，低/中/高三档滑点与手续费批量评估。
- 审计层：`src/quantengine/audit/*`
  - `bundle.py`：`AuditBundle` 生成、ZIP 结构（manifest、trade_log、equity_curve、risk_events）。
  - `io.py`：`save_audit_bundle`、`verify_audit_bundle`。
  - `replay.py`：`replay_from_bundle`，bit-identical 复现校验。
- 合约层：`src/quantengine/contracts/*`
  - `spec.py`：`ContractSpec` 抽象（multiplier、min_tick、trading_sessions），支持配置覆盖。
- 优化层：`src/quantengine/optimizer/*`
  - grid/random/bayesian/genetic 优化器与并行/批处理工具。
  - `walk_forward.py`：Walk-Forward 分析器与折叠结果聚合（含 `WalkForwardResult`）。
- 报表与可视化：`src/quantengine/visualization/*`
  - HTML 报告、Plotly 曲线、热力图。
- 接口层：`src/quantengine/interface/*`
  - `QuantEngineAPI` 与 `LiveAdapter` 抽象。

## GPU 混合架构

- **信号生成**：CPU 路径，策略 `generate_signals` 输出 NumPy/CuPy 数组。
- **批量仿真**：`simulate_portfolio_batch` 在 GPU 上并行评估多组参数；`margin_ratio=0` 时走零时间循环的 `_batch_sim_no_margin`。
- **优化器**：`evaluate_signal_tensor` 接收信号张量，主进程 GPU 批量仿真；无 CUDA 时自动回退 CPU。

## 数据流视图

```text
Config
  └─> DataLoader ──┬─> [Cache Hit]
                   │                └─> DataBundle
                   └─> Miss -> Read Files -> align_and_fill -> Validate
                                      └─> DataBundle -> Write Cache
                              └─> DataBundle

Optimizer: 
- CPU: Strategy + DataBundle -> build_signal_tensor -> simulate_portfolio_batch -> batch_score -> OptimizationResult -> Report
- GPU: Strategy + DataBundle -> build_signal_tensor -> evaluate_signal_tensor -> OptimizationResult -> Report
- Bayesian/Genetic: Strategy + DataBundle -> sequential Sampling -> BacktestEngine -> OptimizationResult -> Report
Walk-Forward: Strategy + DataBundle -> WalkForwardAnalyzer -> Fold List
                                      └─> OOS 聚合 -> WalkForwardResult
```

## 关键设计约束

- 信号与价格矩阵均采用数组化（NumPy/CuPy）计算，减少逐笔 Python 循环。
- `DataLoader` 与 `simulate_portfolio` 对 CPU/GPU 后端保持透明。
- 所有 I/O 默认落盘为 JSON + HTML，便于离线审计和复现。
- 当缺失 GPU 时自动回退 CPU，不影响功能完整性。
- 关键指标和部分技术指标路径使用 Numba JIT 加速（如 `sma`、`ema`、`wilder_smooth`、`simulate_portfolio`）。
