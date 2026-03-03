# QuantEngine

![Python](https://img.shields.io/badge/python-3.11+-blue)
![CLI](https://img.shields.io/badge/cli-quantengine-orange)
![License](https://img.shields.io/badge/license-MIT-green)

QuantEngine 是一个面向 1 分钟线回测的量化引擎，提供 CLI 与 Python API 两套入口，支持多策略、多参数搜索与 GPU 加速。

## 快速上手

```bash
pip install -e .             # 开发环境
pip install -e .[engine]     # 含 GPU/可视化/优化器可选依赖

quantengine --help
```

```bash
quantengine --config quantengine.example.yaml backtest \
  --strategy sma_cross \
  --data ./data \
  --params '{"fast":10,"slow":30}'
```

## 核心特性

- 向量化引擎：信号与资产矩阵并行计算
- Walk-Forward 防过拟合分析：IS/OOS 折叠评估、`overfitting_ratio`
- 数据缓存：`DataLoader` 命中 `.quantengine_cache` 时加速重复加载
- 多策略与参数优化：`grid`、`random`、`bayesian`、`genetic`
- Numba/CuPy 加速：技术指标与仓位仿真关键路径走 Numba JIT
- 进度条体验：优化/Walk-Forward 支持 Rich 进度条
- GPU/CPU 双路径：自动探测 CUDA，缺失时回退 CPU
- 成交建模：滑点、手续费、保证金、涨跌停
- 报告输出：JSON 与 HTML（可视化）
- 交易配对分析增强：`win_rate`、`profit_factor`、`max_consecutive_losses` 等

## 文档

- 文档入口：`docs/index.md`
- 快速开始：`docs/quickstart.md`
- CLI 指南：`docs/api/cli.md`
- API 说明：`docs/api/python-api.md`
- Walk-Forward：`docs/optimization/optimizer-guide.md`

## 许可

MIT
