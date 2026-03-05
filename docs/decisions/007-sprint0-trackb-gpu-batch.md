# 007: Sprint 0 Track B GPU 混合重构（Step1/Step4 + fallback）
日期: 2026-03-05
状态: accepted

## 背景
Sprint 0 Track B 的目标要求将 Step1 与 Step4 改造为“信号并行、主进程 GPU 批量仿真”，并要求在未检测到 CuPy/GPU 时按 raise 语义失效。

## 选项
- 方案 A：仅在 `run_v4_sharpe_experiment.py` 内改造 Step1/Step4
  - 优点：改动面小、最先见效快
  - 缺点：无法覆盖胜率脚本 M1/M5 的同类流程
- 方案 B：在 `run_v4_sharpe_experiment.py` 与 `run_v4_winrate_experiment.py` 两条路径同步改造 Step1/Step4
  - 优点：满足 M1/M5 基线记录和两条实验线一致性目标
  - 缺点：改动文件更多、单次 refactor 周期变长

## 决定
选择方案 B，在 Step1 使用进程池仅做信号生成，主进程接收信号张量后调用 `walk_forward_evaluate_batched` + `simulate_portfolio_batch` 完成 GPU 批量评估；在 Step4 的 Optuna 搜索中复用同一批量评估路径；CuPy 不可用时立即抛出 `RuntimeError`（raise 策略）。

## 后果
正向：
- 步骤 1 与 4 的执行路径从 CPU 单组合切换到 GPU 批量计算，减少 Python 层循环
- 通过共享基线文件 `benchmarks/sprint0_baseline.json` 记录各阶段耗时与显存使用

负向：
- 实验脚本间新增耦合，尤其是 GPU 依赖；本地环境需保证 CuPy 与 CUDA 可用才能执行 Track B 路径
- 由于未引入 fallback CPU 执行分支，环境缺失 GPU 时实验会显式失败
