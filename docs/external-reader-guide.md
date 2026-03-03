# 外部阅读者精简导航（评审/对接版）

这份文档帮助不直接开发代码的读者（评审、合作方、外部用户）在短时间内理解 QuantEngine 的能力边界与使用方式。

## 先看这三点（30 秒）

1. 这是一个 **CLI + Python API 驱动**的回测引擎，默认不做事件驱动撮合。
2. 核心设计是 **向量化矩阵回测**：数据、信号、持仓、净值都按数组批处理。
3. 支持 GPU 自动回退：有 CUDA 时加速，没有时自动降级 CPU。

## 推荐阅读顺序（3 步）

- 第一步：`quickstart.md`  
  了解安装、数据要求、第一条 backtest/optimize 的完整命令。
- 第二步：`api/cli.md`  
  快速确认 CLI 参数、输出文件、常用命令。
- 第三步：`architecture/overview.md`  
  看懂系统模块边界与数据流。

## 外部对接建议路径

- 只关心“能否跑起来”：`quickstart.md`
- 只关心“如何调用”：`api/cli.md` + `api/python-api.md`
- 只关心“防过拟合与稳定性”：`optimization/optimizer-guide.md` + `docs/architecture/overview.md`
- 只关心“是否可复现”：`testing/acceptance.md`

## 一页能力清单

- 已覆盖能力
  - 多策略 + 多参数组合回测
  - 参数优化：`grid/random/bayesian/genetic`
  - Walk-Forward 防过拟合分析（IS/OOS + overfitting_ratio）
  - 数据缓存：`.quantengine_cache` 命中加速
  - Numba JIT：关键指标与仿真路径加速
  - 报告输出：`JSON` + `HTML`（plotly/matplotlib）
  - 数据接入：CSV/Parquet，一分钟线优先
  - 连续合约工具、滑点/手续费/涨跌停/保证金相关模型
- 不承担的能力
  - 本版本无图形化策略编辑器
  - 本版本默认不提供逐笔事件驱动撮合

## 一步到位的验证清单

```bash
quantengine list-strategies
quantengine --config quantengine.example.yaml backtest --help
quantengine --config quantengine.example.yaml optimize --help
quantengine --config quantengine.example.yaml walk-forward --help
```

若以上命令均可运行，即可判断 CLI 与运行环境可用；再按 `quickstart.md` 跑一条样例任务进行结果核验。

## 快速词汇（外部读者版）

- `strategy`：策略名，由注册机制统一管理。
- `method`：优化方法（grid/random/bayesian/genetic）。
- `trial`：一次参数评估，包含一组 `params` 与评分 `score`。
- `report`：可复用 JSON 产物生成的 HTML 报告。
- `symbol`：标的目录名，`--symbols` 为大写匹配。
