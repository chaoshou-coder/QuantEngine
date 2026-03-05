# 模块映射与职责边界

本页按代码路径给出模块职责，便于快速定位实现与扩展点。

## 配置与启动

- `src/quantengine/config.py`
  - `QuantEngineConfig` 及子配置模型
  - 配置加载与校验 (`load_config`)
- `src/quantengine/__main__.py`
  - 模块脚本入口

## CLI 入口

- `src/quantengine/cli.py`
  - Click 命令组与 `check-deps/list-strategies/backtest/optimize/walk-forward/report` 实现
  - 输出 JSON / HTML

## 数据层

- `src/quantengine/data/gpu_backend.py`
  - GPU 可用性探测、backend 信息、`get_xp`、`to_numpy`
- `src/quantengine/data/loader.py`
  - 文件扫描、CSV/Parquet 读取、按 symbol 对齐后输出 `DataBundle`
- `src/quantengine/data/cache.py`
  - 缓存 key 计算、`.quantengine_cache` 读写、`.npz` 与 `.meta` 格式
- `src/quantengine/data/preprocessor.py`
  - 列名规范化、时间齐线、重采样
- `src/quantengine/data/continuous.py`
  - 连续合约构建（difference / ratio）

## 策略与指标

- `src/quantengine/strategy/base.py`
  - `BaseStrategy`、`ParameterSpace`、网格展开
- `src/quantengine/strategy/registry.py`
  - 策略注册与按名获取
- `src/quantengine/strategy/examples/sma_cross.py`
  - 示例策略 1（均线交叉）
- `src/quantengine/strategy/examples/rsi_mean_reversion.py`
  - 示例策略 2（RSI）
- `src/quantengine/indicators/base.py`
  - 指标结果封装与基类
- `src/quantengine/indicators/technical.py`
  - SMA/EMA/RSI/MACD/ATR（Numba + 向量化）
  - EMA/Wilder 在 GPU 无 Numba 时回退至 NumPy CPU 实现后转回后端
- `src/quantengine/indicators/gpu_compute.py`
  - GPU 路径（Numba/CuPy）尝试与降级

## 回测执行

- `src/quantengine/engine/slippage.py`
  - Fixed/Percent/Volume 滑点模型
- `src/quantengine/engine/commission.py`
  - Fixed/Percent/Tiered 手续费模型
- `src/quantengine/engine/rules.py`
  - 涨跌停规则与保证金约束
- `src/quantengine/engine/execution.py`
  - 市价 / 限价执行器（当前主要逻辑在 `simulate_portfolio` 中使用滑点）
- `src/quantengine/engine/portfolio.py`
  - 仿真核心、资金曲线与交易明细
- `src/quantengine/engine/factory.py`
  - 引擎与规则/模型对象构造工厂（CLI/API 共享）
- `src/quantengine/engine/backtest.py`
  - 单策略与多策略回测编排

## 指标与评估

- `src/quantengine/metrics/performance.py`
  - total_return/annualized_return/sharpe/sortino/max_drawdown/calmar
- `src/quantengine/metrics/risk.py`
  - var_95 / cvar_95 / ulcer_index / downside_deviation
- `src/quantengine/metrics/batch.py`
  - 评估指标批量评分接口（`batch_score`）与关键指标向量化实现
- `src/quantengine/metrics/trade_analysis.py`
  - 交易统计

## 优化层

- `src/quantengine/optimizer/base.py`
  - 优化器基类与结果结构
- `src/quantengine/optimizer/parallel.py`
  - 并行评估、信号张量构建与批量评估
- `src/quantengine/optimizer/grid.py`
  - 笛卡尔网格搜索
- `src/quantengine/optimizer/random_search.py`
  - 随机搜索
- `src/quantengine/optimizer/bayesian.py`
  - Optuna 贝叶斯优化（缺失时自动回退随机）
- `src/quantengine/optimizer/genetic.py`
  - Nevergrad/DEAP 遗传优化（缺失时回退随机）
- `src/quantengine/optimizer/walk_forward.py`
  - Walk-Forward 折叠构建与 `WalkForwardResult` 生成

## 可视化与报告

- `src/quantengine/visualization/plots.py`
  - Plotly 的权益和回撤图
- `src/quantengine/visualization/heatmaps.py`
  - Matplotlib 热力图
- `src/quantengine/visualization/reports.py`
  - `write_walk_forward_report_html` 输出 WF 报告
  - HTML 报告写入

## 对外接口

- `src/quantengine/interface/api.py`
  - 程序化 `QuantEngineAPI`
- `src/quantengine/interface/live_adapter.py`
  - 实盘适配器接口

## 公开 API 汇总

- 包导出见 `src/quantengine/*/__init__.py`
- CLI 依赖主要入口：`quantengine`（`project.scripts`）
