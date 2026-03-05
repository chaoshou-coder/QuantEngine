# 006: Beta 策略优化目标函数设计

日期: 2026-03-05
状态: research-needed

## 背景

当前目标函数 `_winrate_objective` 由 AI 自主设计：
`mean_wr - max(0, 0.48 - worst_wr) * 2.0 + sharpe_guard`
用于 Beta 策略（追踪市场方向）的优化。
需要验证这是否是行业最佳实践。

## 待研究

- 量化交易中 Beta 策略的标准评估指标
- 胜率 vs 盈亏比 vs 期望收益的关系
- 多目标优化（Pareto front）vs 单一复合目标
- 是否应该使用 profit factor / expectancy 替代或补充 win rate
- 行业中成熟的策略评估框架（如 QuantConnect 的评分体系）

## 选项

- **选项 A**：Expectancy = WR × AvgWin - (1-WR) × AvgLoss
  - 优点：综合胜率和盈亏比
  - 缺点：需要 trade-level 数据（而非 bar-level returns）
- **选项 B**：Profit Factor = GrossProfit / GrossLoss
  - 优点：行业标准，直观，>1.5 为公认可交易门槛
  - 缺点：不区分波动大小
- **选项 C**：多目标 Pareto (NSGA-II)
  - 优点：学术最优，不需要预设权重
  - 缺点：实现复杂，输出为 Pareto 前沿而非单一最优
- **选项 D**：加权复合 w1×WR + w2×PF + w3×Sharpe + w4×(1-MaxDD)
  - 优点：灵活
  - 缺点：权重选择主观

## 决定

采用选项 B：Profit Factor = GrossProfit / GrossLoss 作为主要优化目标。

理由：
- 行业标准指标，客户容易理解
- PF > 1.5 为公认可交易门槛
- 兼顾胜率和盈亏比（PF = WR×AvgWin / (1-WR)×AvgLoss）
- 计算简单，可从 bar returns 直接推导（正收益总和 / 负收益绝对值总和）
- 报告中同时展示 WR/Sharpe/MaxDD 作为辅助指标

研究来源：
- algostrategyanalyzer.com: PF 1.5-2.0 为"good"，2.0-3.0 为"excellent"
- horizontrading.ai: WR 单独使用有误导性，需配合 PF 或 expectancy
- Springer 2025: 多目标优化（NSGA-II）优于单目标，但对 MVP 阶段复杂度过高

## 后果

- 需要修改 walk_forward_evaluate 使其返回 profit factor
- 需要在 metrics/performance.py 中确认 profit factor 的计算
- 实验脚本中 _winrate_objective 替换为 profit factor 为核心的目标函数
- 后续可升级为多目标 Pareto 优化
