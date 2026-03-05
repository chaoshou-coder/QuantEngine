# 审计包

审计包（Audit Bundle）是 QuantEngine 的可复现回测产物，以 ZIP 形式保存数据哈希、配置、环境、交易日志、权益曲线与风控事件，支持 round-trip 校验与回放。

## ZIP 结构

```
audit.zip
├── sha256.json        # 各文件 SHA256
├── config.json        # 策略、引擎、运行配置
├── env.json           # Python/平台/包版本
├── metadata.json      # 创建时间、数据哈希
├── trades.csv         # 交易明细
├── risk_events.csv    # 风控事件
├── equity_curve.csv   # 权益曲线
├── returns.csv        # 收益率序列
├── performance.json   # 绩效指标
├── risk.json          # 风险指标
└── trade_metrics.json # 交易统计
```

## 生成审计包

CLI：

```bash
quantengine --config quantengine.example.yaml backtest \
  --strategy sma_cross \
  --data ./data \
  --params '{"fast":10,"slow":30}' \
  --audit-bundle ./audit.zip
```

Python API：

```python
report = engine.run(data, strategy, params, record_trades=True)
if report.audit_bundle:
    from quantengine.audit.io import save_audit_bundle
    save_audit_bundle(report.audit_bundle, Path("./audit.zip"))
```

## round-trip 校验

`verify_audit_bundle(bundle_path, data=...)` 返回：

- `integrity_ok`：ZIP 内文件完整且哈希一致
- `data_hash_match`：数据哈希与当前 DataBundle 一致
- `env_match`：环境信息一致（可选严格模式）

## 回放

`replay_from_bundle(bundle_path, data=...)` 使用审计包内配置重新执行回测，并校验 `equity_curve` 与 `returns` 与原始结果 bit-identical。用于验证可复现性。
