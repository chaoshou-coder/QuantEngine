# GPU 加速

QuantEngine 采用 CPU 信号生成 + GPU 批量仿真的混合架构，在具备 CUDA 时显著加速参数搜索与 Walk-Forward 评估。

## 混合架构

- **信号生成**：策略 `generate_signals` 在 CPU 上执行（支持 Numba JIT），输出 NumPy/CuPy 数组
- **批量仿真**：`simulate_portfolio_batch` 在 GPU 上并行评估多组参数
- **优化器**：`evaluate_signal_tensor` 接收信号张量，主进程调用 GPU 批量仿真

## batch_sim 与 margin

- `margin_ratio=0` 时：走 `_batch_sim_no_margin`，零时间循环，完全 GPU 并行
- `margin_ratio>0` 时：保证金仿真有逐 bar 依赖，当前退化为 Python 时间循环（CPU）

## 自动回退

- 未检测到 CuPy/CUDA 时，自动回退 CPU 路径
- 功能完整性不受影响，仅性能下降

## 依赖

```bash
pip install -e .[engine]
# 含 cupy-cuda12x, cudf-cu12, numba
```

## 使用

配置 `runtime.backend: "auto"` 与 `runtime.use_gpu: true` 时，引擎自动探测并优先使用 GPU。无需修改策略代码。
