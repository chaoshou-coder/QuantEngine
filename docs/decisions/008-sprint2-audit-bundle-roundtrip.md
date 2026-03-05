# 008: Sprint 2 Track A 审计包与可复现回放
日期: 2026-03-05
状态: accepted

## 背景
Sprint 2 Track A 要求回测结果具备审计追踪与可复现能力：每次回测需携带可落盘的审计包，并支持校验与回放，确保“生成 -> 校验 -> 重放”链路闭环。

## 选项
- 方案 A：仅保存摘要信息（配置/环境/指标），不保存可比对的时序结果
  - 优点：文件体积小，写盘速度快
  - 缺点：无法做 bit-identical 级别回放校验
- 方案 B：保存审计元数据 + 关键时序输出（equity/returns）+ 交易与风控事件，并提供统一 save/verify/replay API
  - 优点：可做完整校验与重放一致性验证，满足审计闭环
  - 缺点：实现复杂度与 ZIP 体积上升

## 决定
选择方案 B。新增 `quantengine.audit` 模块，定义 `AuditBundle`，并提供 `save_audit_bundle()`、`verify_audit_bundle()`、`replay_from_bundle()`。在 `BacktestEngine.run()` 中自动构建并挂载 `BacktestReport.audit_bundle`，默认输出审计上下文。

## 后果
正向：
- 回测结果具备可追溯性（数据哈希、环境、配置、交易日志、风控事件）
- 提供标准化 ZIP 结构（含 `sha256.json` / `config.json` / `env.json` / `trades.csv` / `risk_events.csv`）
- 回放支持对 `equity_curve` 与 `returns` 做 bit-identical 校验

负向：
- `BacktestReport` 结构扩展为包含 `audit_bundle`（可选字段）
- 额外 I/O 与序列化开销，需要在大规模实验中关注审计包体积与保存频率
