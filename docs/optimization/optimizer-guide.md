# 参数优化与并行/GPU 指南

本页说明 QuantEngine 的参数搜索方案、复杂度与建议使用场景。

## 优化器总览

`src/quantengine/optimizer` 提供四类方法：

- `grid`：穷举网格
- `random`：随机采样
- `bayesian`：Optuna 贝叶斯优化
- `genetic`：遗传算法（Nevergrad/DEAP）

返回统一 `OptimizationResult`：

- `method`
- `metric`
- `maximize`
- `best_params`
- `best_score`
- `trials`（每个参数集合及评分）

额外：若启用 Walk-Forward，可在每折内复用同样的优化器返回 `WalkForwardResult`。

## 1. Grid Search

适用于：
- 搜索空间较小
- 希望完整覆盖每个离散组合

实现：
- 通过 `ParameterSpace.grid_values()` 生成笛卡尔积
- 按 `batch_size` 分批拼接参数组合
- CPU 路径：`evaluate_batch` + `simulate_portfolio_batch`，一次性向量化评估一个 batch 的参数组合
- GPU 且可用张量路径时：`build_signal_tensor` + `evaluate_signal_tensor`

命令示例：

```bash
quantengine --config quantengine.yaml optimize \
  --strategy sma_cross \
  --data ./data \
  --method grid \
  --param-grid '{"fast":[5,10], "slow":[20,30,40]}' \
  --symbols ES \
  --output ./results/opt_grid.json
```

## 2. Random Search

适用于：
- 参数范围大但对全局最优容忍度较高
- 预算受限时先行探测

实现：
- 每次从 `ParameterSpace` 采样参数
- 按 batch 聚合后复用 `evaluate_batch`（CPU）或 `evaluate_signal_tensor`（GPU）进行批量评估

命令示例：

```bash
quantengine --config quantengine.yaml optimize \
  --strategy rsi_mean_reversion \
  --data ./data \
  --method random \
  --n-trials 500 \
  --metric sharpe \
  --symbols ES \
  --output ./results/opt_random.json
```

## 3. Bayesian（Optuna）

适用于：
- 想更快收敛到高质量参数
- 参数维度中等并能承受外部依赖

实现：
- 若未安装 `optuna`，自动回退 `random`。
- 支持 `int`、`float`、`choice` 参数 suggestion。
- 未装时 `quantengine` 会记录告警并进入随机流程。

命令示例：

```bash
quantengine optimize \
  --strategy sma_cross \
  --data ./data \
  --method bayesian \
  --n-trials 200 \
  --metric sharpe
```

## 4. Genetic（遗传）

适用于：
- 存在非凸/离散混合参数空间
- 希望快速探索更广范围

实现优先级：
1. 若安装 `nevergrad`：使用连续/混合参数器；
2. 若有 `deap`：使用遗传算法流程；
3. 否则回退 `random`。

命令示例：

```bash
quantengine optimize \
  --strategy sma_cross \
  --data ./data \
  --method genetic \
  --output ./results/opt_genetic.json
```

## 5. 指标与方向

- CLI `--metric` 覆盖 `performance`/`risk`/`trade_metrics` 中可查询键。
- `--minimize` 时将目标翻转为最小化；CLI 内部最终调用 `OptimizationResult` 的 maximize 字段。

常见 metric：

- `sharpe`, `sortino`, `calmar`, `max_drawdown`, `total_return`
- `var_95`, `ulcer_index`, `trade_count`

## 6. 并行与 batch

路径选择：

- CPU：`evaluate_batch`（默认）
  - 输入参数 -> `build_signal_tensor` -> `simulate_portfolio_batch` -> `batch_score`
  - 优点：减少 `n_trials` 次重复引擎初始化与 Python 循环
- GPU：`evaluate_signal_tensor`
  - 在 `backend.active == "gpu"` 且环境就绪时使用 `signal tensor` 张量评估
  - 批量大小通过 `batch_size` 控制，避免显存峰值
- Bayesian / Genetic：保持顺序采样流程
  - `optimize.max_workers` 和批量接口不参与自适应优化器
- `ProcessPoolExecutor` 仅在不可向量化策略对象下作为兼容 fallback 使用
- `random/grid` 的 `max_workers` 会影响报告进度显示，不改变 batch 评估核心路径

性能建议：

- `batch_size` 决定一次处理的参数数量（默认 128），建议先从 64~256 试调
- 过大 `batch_size` 会抬高显存占用，适当下调可换取稳定性

## 7. Walk-Forward 分析

Walk-Forward 用于抑制回测过拟合风险，流程：

1. 按时间将样本切分为多个 `fold`
2. 每个 fold 划分 IS（训练）与 OOS（验证）
3. IS 上进行优化
4. 用 OOS 验证该折最优参数
5. 汇总 OOS 表现并计算过拟合指标

核心指标：

- `overfitting_ratio`：IS 与 OOS 性能比值平均
  - 越接近 `1.0` 越健康
  - 显著低于 `1.0` 说明 OOS 大幅退化

CLI 示例：

```bash
quantengine --config quantengine.yaml walk-forward \
  --strategy sma_cross \
  --data tests/fixtures/sample_500_bars.csv \
  --n-splits 5 \
  --is-ratio 0.7 \
  --method bayesian \
  --n-trials 40 \
  --metric sharpe \
  --expanding \
  --output ./results/wf.json \
  --report ./reports/wf.html
```

API 示例：

```python
result = api.walk_forward(
    strategy_name="sma_cross",
    data_path="tests/fixtures/sample_500_bars.csv",
    n_splits=5,
    in_sample_ratio=0.7,
    method="bayesian",
    n_trials=40,
    metric="sharpe",
    show_progress=False,
)
```

返回字段：

- `config`
- `folds`
- `aggregate_oos_performance`
- `aggregate_oos_risk`
- `overfitting_ratio`

## 8. 进度条

QuantEngine 在优化与 Walk-Forward 里提供 Rich 进度条：

- CLI：
  - `optimize` 和 `walk-forward` 默认显示进度条（无需参数）。
  - CLI 优先向终端用户展示实时占比和耗时。
- API：
  - `QuantEngineAPI.optimize(..., show_progress=False)` 默认关闭。
  - `QuantEngineAPI.walk_forward(..., show_progress=False)` 默认关闭。
  - 如需 CI/日志场景可显式开启。

进度条同样适用于 `grid/random/bayesian/genetic` 与 Walk-Forward 的内部循环。

## 性能加速与 Numba

- `indicators/technical.py` 中关键指标（`sma`、`ema`、`wilder_smooth`）与 `engine/portfolio.py` 的核心仿真支持 Numba JIT 或纯向量化。
- 批量优化链路中，`evaluate_batch` + `batch_score` 避免 `N_trials` 次重复回测函数调用，显著压缩 CPU/GPU 两端的 Python 开销。
- 在无 Numba 环境下会自动退回 NumPy/CuPy 向量化实现，功能不变。
