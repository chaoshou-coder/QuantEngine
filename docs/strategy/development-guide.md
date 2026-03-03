# 策略开发指南

本文档说明如何在 QuantEngine 中定义、测试和集成自定义策略。

## 策略接口

核心抽象位于 `src/quantengine/strategy/base.py`：

- `BaseStrategy.parameters()`：声明可优化参数空间（`ParameterSpace`）。
- `BaseStrategy.generate_signals(data, params)`：基于数据和参数输出信号矩阵。
- `BaseStrategy.on_init(engine)`：可选初始化钩子，用于运行期信息获取（默认空实现）。

### 参数空间 `ParameterSpace`

字段：

- `kind`: `"int" | "float" | "choice"`
- `low / high / step`: 用于数值区间枚举（`int` / `float`）
- `choices`: 用于离散枚举（`choice`）

`cartesian_from_spaces` 会将参数空间展开为网格点（用于 grid 优化）。

## 信号要求

`generate_signals` 需要返回 shape 为 `(n_bars, n_assets)` 的矩阵：

- 行：时间轴
- 列：symbol 维

引擎会在 `simulate_portfolio` 中自动将信号约束到 `[-1, 1]` 并用前一时刻信号作为当前持仓执行。

信号约定：

- `1`：做多
- `-1`：做空
- `0`：空仓
- 中间值：允许放大比例（杠杆衰减），仍按连续信号处理

建议输出前用 `np.nan_to_num` 或 `cp.nan_to_num` 清理 NaN。

## 注册策略

系统通过装饰器注册策略：

```python
from quantengine.strategy.base import BaseStrategy, ParameterSpace
from quantengine.strategy.registry import register_strategy

@register_strategy("my_strategy")
class MyStrategy(BaseStrategy):
    def parameters(self):
        return {
            "lookback": ParameterSpace(kind="int", low=5, high=60, step=5),
        }

    def generate_signals(self, data, params):
        ...
```

注册后，名称会被标准化为小写，可通过 `list_strategies()` 查询。

## 示例策略结构

- `src/quantengine/strategy/examples/sma_cross.py`
- `src/quantengine/strategy/examples/rsi_mean_reversion.py`

示例策略会在 `quantengine.strategy.__init__` 中导入，从而自动注册。

### 示例约定与约束

- 参数异常（例如 `fast >= slow`）应在策略内部抛出明确错误，便于快速定位。
- 输出的信号必须兼容 NumPy/CuPy 两种数组实现。
- 对单标的与多标的逻辑保持一致，不要手工依赖固定列索引。

## 常见开发模板（基于技术指标）

```python
import numpy as np
from quantengine.indicators.technical import sma
from quantengine.strategy.base import BaseStrategy, ParameterSpace
from quantengine.strategy.registry import register_strategy

@register_strategy("ma_momentum")
class MOMA(BaseStrategy):
    def parameters(self):
        return {
            "window": ParameterSpace(kind="int", low=10, high=60, step=5),
        }

    def generate_signals(self, data, params):
        w = int(params.get("window", 20))
        ma = sma(data.close, w)
        score = (data.close - ma) / np.maximum(ma, 1e-12)
        # 以涨跌方向映射为 -1~1 区间
        return np.clip(score / (np.nanmax(np.abs(score)) + 1e-12), -1.0, 1.0)
```

## 组合策略运行

CLI `backtest` 支持 `--strategies` 与权重。权重来自 `--strategies` 中策略列表外部映射，不在策略对象内配置。

在 API 中也可通过 `backtest_multi(strategy_specs=...)` 传入：

- 每项是 `(strategy_name, params, weight)`。
- 引擎会按权重聚合信号并归一化后再执行撮合。

## 调试建议

1. 先在小样本数据上验证 `generate_signals` 形状。
2. 检查返回矩阵是否包含 `NaN`，并确认是否已填充。
3. 使用 `--symbols` 限定单标的做行为差异排查。
4. 通过 `optimize` 的 `random` 方法快速验证参数空间合理性后再切换 grid。
