# 策略 DSL

QuantEngine 支持用 YAML DSL 描述策略，避免硬编码参数组合，并覆盖 V4 策略的 `framework` 与 `risk_mode` 正交组合。

## YAML 格式

```yaml
strategy:
  name: psar_trade_assist_v4
  framework: F1      # F1 | F2 | F3 | F4 | F5
  risk_mode: standard  # baseline | conservative | standard | aggressive
  params:
    af_start: 0.02
    af_increment: 0.02
    af_max: 0.2
  weight: 1.0
```

## 解析与实例化

```python
from quantengine.strategy.dsl import load_strategy_from_dsl

strategy, params, weight = load_strategy_from_dsl("strategy.yaml")
# 或
strategy, params, weight = load_strategy_from_dsl("""
strategy:
  name: sma_cross
  params: { fast: 10, slow: 30 }
""")
```

## V4 全覆盖

- `V4_FRAMEWORKS`：F1–F5
- `V4_RISK_MODES`：baseline、conservative、standard、aggressive
- `framework` 与 `risk_mode` 会写入 `params`，供策略内部使用

## StrategyDSLSpec

```python
@dataclass(frozen=True)
class StrategyDSLSpec:
    name: str
    params: dict[str, Any]
    weight: float
```

`load_strategy_dsl()` 返回 `StrategyDSLSpec`，`build_strategy_from_dsl()` 返回 `(BaseStrategy, params, weight)`。
