# 风控引擎

QuantEngine 的风控能力通过 `TradingRules` 与 `simulate_portfolio` 的 Numba 集成实现，在逐 bar 仿真循环内执行检查，并输出 `risk_events` 供审计与报告使用。

## TradingRules 字段

| 字段 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `limit_up_ratio` | float \| None | None | 涨停比例 |
| `limit_down_ratio` | float \| None | None | 跌停比例 |
| `margin_ratio` | float | 0.1 | 保证金比例 |
| `max_risk_per_trade` | float | 0.0 | 单笔最大风险占比（0=不限制） |
| `max_daily_loss` | float | 0.0 | 日内最大亏损比例（0=不限制） |
| `max_weekly_loss` | float | 0.0 | 周级最大亏损比例（0=不限制） |
| `max_drawdown_limit` | float | 0.0 | 最大回撤阈值（0=不限制） |
| `max_drawdown_action` | str | "stop" | 触发动作：`stop` \| `reduce` \| `alert` |
| `max_position` | float | 0.0 | 最大持仓量（0=不限制） |
| `max_addon_count` | int | 0 | 最大加仓次数（0=不限制） |

## Numba 集成

风控检查在 `simulate_portfolio` 的 Numba `@njit` 循环内执行，保证：

- 与回测主循环同路径，无额外 Python 调用
- 日/周亏损、回撤、持仓、加仓等约束在每 bar 更新后立即生效

## risk_events 输出

`PortfolioResult.risk_events` 记录所有风控触发事件，每项包含：

- `bar`：触发 bar 索引
- `type`：`daily_loss_breach` \| `weekly_loss_breach` \| `drawdown_breach` \| `position_limit`
- `detail`：描述文本

审计包与报告系统会序列化 `risk_events`，便于事后分析与复现。

## 配置示例

```yaml
engine:
  rules:
    max_risk_per_trade: 0.02
    max_daily_loss: 0.05
    max_weekly_loss: 0.10
    max_drawdown_limit: 0.15
    max_drawdown_action: "stop"
    max_position: 1.0
    max_addon_count: 2
```
