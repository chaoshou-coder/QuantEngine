# 001: V4 信号生成架构（Numba 状态机 vs 替代方案）

日期: 2026-03-05
状态: research-needed

## 背景

V4 策略使用 Numba @njit 逐 bar 状态机生成信号（SL/TP/trailing/addon 等状态依赖前一 bar）。
这导致信号生成只能走 CPU，无法 GPU 并行。
用户要求：最强性能，用满硬件。

## 待研究

- 是否存在将有状态交易逻辑 GPU 化的方案（如 CUDA kernel / CuPy scan / Numba CUDA）
- 部分逻辑（纯指标计算）已向量化，瓶颈是否仅在状态机部分
- 行业中其他引擎如何处理（vnpy/zipline/backtrader 的信号生成架构）

## 选项

待研究后填写。

## 决定

待定。

## 后果

待定。
