# 报告系统

QuantEngine 的报告系统输出 8 区块结构，支持 BH 校正、参数敏感度分析与可选 PDF 导出。

## 8 区块

1. **绩效**：sharpe、sortino、total_return、annualized_return、calmar 等
2. **风险**：max_drawdown、volatility、var 等
3. **交易统计**：win_rate、profit_factor、max_consecutive_losses、trade_count 等
4. **权益曲线**：equity_curve 时序
5. **交易明细**：trades 列表（可选）
6. **风控事件**：risk_events（若启用风控）
7. **参数敏感度**：各参数对目标指标的影响
8. **元数据**：策略、参数、数据范围、运行时间

## BH 校正

多重比较时使用 Benjamini-Hochberg 校正，控制假阳性率，适用于 Walk-Forward 或参数筛选场景。

## 输出格式

- **JSON**：`backtest` / `optimize` 命令默认输出
- **HTML**：`report` 命令基于 JSON 生成可视化
- **PDF**：可选，需配置 `visualization` 与相应依赖

## CLI 使用

```bash
quantengine backtest ... -o result.json
quantengine report result.json -o report.html
```
