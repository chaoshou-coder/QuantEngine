# 成本多情景

QuantEngine 支持在同一组信号上，用多组滑点与手续费配置批量评估，输出各情景下的绩效与风险，用于稳健性分析。

## CostScenario

```python
@dataclass
class CostScenario:
    name: str
    slippage_model: str      # "percent" | "fixed" | "volume"
    slippage_value: float
    commission_model: str    # "percent" | "fixed" | "tiered"
    commission_value: float
    slippage_impact: float = 0.0
    commission_tiers: list | None = None
```

## 典型三档

| 情景 | 滑点 | 手续费 | 用途 |
|------|------|--------|------|
| 低 | 0.00005 | 0.0001 | 理想条件 |
| 中 | 0.0001 | 0.0002 | 常规条件 |
| 高 | 0.0002 | 0.0004 | 保守/压力测试 |

## run_cost_scenarios

`BacktestEngine.run_cost_scenarios()` 接收：

- `data`：DataBundle
- `strategy`：策略实例
- `params`：策略参数
- `scenarios`：`list[CostScenario]`

返回 `list[BacktestReport]`，与 `scenarios` 一一对应。

## 使用示例

```python
from quantengine.engine.backtest import BacktestEngine, CostScenario

scenarios = [
    CostScenario("low", "percent", 0.00005, "percent", 0.0001),
    CostScenario("mid", "percent", 0.0001, "percent", 0.0002),
    CostScenario("high", "percent", 0.0002, "percent", 0.0004),
]
reports = engine.run_cost_scenarios(data, strategy, params, scenarios)
for r in reports:
    print(r.portfolio.metadata.get("cost_scenario"), r.performance["sharpe"])
```
