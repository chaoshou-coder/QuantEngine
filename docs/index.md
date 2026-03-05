# QuantEngine 文档中心

本目录承载 QuantEngine 的文档集合，覆盖策略研发、引擎研发、运维、分析与对外使用者。

## 受众与阅读路径

- 策略研发
  - 快速起步：`quickstart.md`
  - 策略开发：`strategy/development-guide.md`
  - 回测运行：`api/cli.md`
  - 结果验收：`testing/acceptance.md`
  - Walk-Forward：`optimization/optimizer-guide.md`
- 引擎研发
  - 架构总览：`architecture/overview.md`
  - 模块映射：`architecture/module-map.md`
  - 优化器与 GPU：`optimization/optimizer-guide.md`、`optimization/gpu-setup.md`
  - API：`api/python-api.md`
- 运维与支持
  - 排障：`operations/troubleshooting.md`
  - 版本验收：`testing/acceptance.md`
- 数据与分析
  - 指标和报告解读：结合 `api/python-api.md` 和 `quickstart.md` 的输出示例
  - Walk-Forward 结果解读：`optimization/optimizer-guide.md`
- 外部读者/评审
  - 先读：`quickstart.md`
  - 再看：`architecture/overview.md`
  - 最后：`api/cli.md`
  - 精简版导航（建议）：`external-reader-guide.md`

## 文档结构

- 快速开始与命令
  - `quickstart.md`
  - `api/cli.md`
- 架构与设计
  - `architecture/overview.md`
  - `architecture/module-map.md`
  - `api/python-api.md`
  - `strategy/development-guide.md`
- 功能说明（Sprint 0-3）
  - `features/risk-management.md` — 风控引擎
  - `features/cost-scenarios.md` — 成本多情景
  - `features/audit-bundle.md` — 审计包
  - `features/report-system.md` — 报告系统
  - `features/strategy-dsl.md` — 策略 DSL
  - `features/gpu-acceleration.md` — GPU 加速
- 外部阅读者精简版：`external-reader-guide.md`
- 优化与性能
  - `optimization/optimizer-guide.md`
  - `optimization/gpu-setup.md`
- 运维与质量
  - `operations/troubleshooting.md`
  - `testing/acceptance.md`

## 术语约定

- `symbol`：标的代码，读取时按目录名做标识，并在 `--symbols` 中统一按大写匹配。
- `DataBundle`：加载后的对齐后的多标的矩阵数据对象，`shape = (bars, symbols)`。
- `signal`：策略输出信号矩阵，需是 `n_bars × n_assets`，推荐范围 `[-1, 1]`。
- `trial`：一次参数评估，包含一组 `params` 与评分 `score`。
- `method`：优化器类型，取值 `grid|random|bayesian|genetic`。
- `metric`：可选于性能、风控或交易统计字典中的目标。
- `WF`/`Walk-Forward`：时间滚动验证框架，输出 `WalkForwardResult`，含 `overfitting_ratio`。
- `report`：CLI `report` 命令可基于 `backtest` / `optimization` JSON 重放输出 HTML；Walk-Forward 建议使用 `walk-forward --report`。

## 建议使用方式

1. 新用户按 `quickstart.md` 先跑通一条 `backtest`。
2. 策略研发按 `strategy/development-guide.md` 自定义策略并注册。
3. 引擎研发按 `architecture/*` 与 `optimization/optimizer-guide.md` 进行扩展。
4. 发布前按 `testing/acceptance.md` 完成验收清单。

## 变更记录

文档按 `docs-skeleton`、`docs-onboarding`、`docs-core-design`、`docs-data-opt`、`docs-ops-qa`、`docs-consistency-pass`
的顺序逐步补齐，默认与代码实现保持一致。
