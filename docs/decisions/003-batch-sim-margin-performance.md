# 003: 批量仿真 margin>0 时的性能瓶颈

日期: 2026-03-05
状态: research-needed

## 背景

`_batch_sim_no_margin` 实现了零时间循环的 GPU 批量仿真，但仅在 margin_ratio=0 时可用。
margin>0 时退化为 Python 时间循环（`for t in range(1, n_bars)`），无法利用 GPU。
用户要求：想办法解决这个性能瓶颈。

## 待研究

- margin>0 的逐 bar 依赖关系是否可以通过 scan/prefix-sum 类算法并行化
- Numba CUDA 或 CuPy custom kernel 是否能处理带状态的保证金仿真
- 是否可以用近似方法（如分段线性化）打破时间依赖

## 选项

待研究后填写。

## 决定

待定。

## 后果

待定。
