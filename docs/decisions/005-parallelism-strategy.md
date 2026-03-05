# 005: 并行化策略选择

日期: 2026-03-05
状态: research-needed

## 背景

当前使用 ProcessPoolExecutor 做 CPU 并行。用户要求最强性能，用满硬件。
需要研究是否有更优的并行方案。

## 待研究

- ProcessPoolExecutor vs multiprocessing.Pool vs Ray vs Dask 的性能对比
- Numba parallel=True / prange 在信号生成中的可行性
- 异步 I/O (asyncio) 对数据传输的帮助
- CPU 并行 + GPU 批量仿真的最优协调模式
- 8 核 16 线程 + 12GB GPU 的具体硬件下最优配置

## 选项

待研究后填写。

## 决定

待定。

## 后果

待定。
