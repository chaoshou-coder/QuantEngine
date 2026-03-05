# QuantEngine

![Python](https://img.shields.io/badge/python-3.11+-blue)
![CLI](https://img.shields.io/badge/cli-quantengine-orange)
![License](https://img.shields.io/badge/license-MIT-green)

QuantEngine 是一个向量化量化回测引擎，支持多周期（分钟/小时/日线）OHLCV 数据与可选 GPU 加速，提供 CLI 与 Python API 两套入口，支持多策略、多参数搜索、风控扩展、成本多情景、审计包与策略 DSL。

## 核心价值

- **可复现**：审计包（ZIP）完整记录数据哈希、配置、交易日志、权益曲线，支持 round-trip 校验与回放
- **可扩展**：风控 7 字段、成本多情景、合约规格抽象、策略 YAML DSL
- **高性能**：CPU 信号生成 + GPU 批量仿真混合架构，Numba JIT 关键路径

## 快速上手

```bash
pip install -e .             # 仅核心依赖
pip install -e .[engine]     # 含 GPU/可视化/优化器可选依赖

quantengine check-deps       # 可选：检查依赖完整性
quantengine --help
```

```bash
quantengine --config quantengine.example.yaml backtest \
  --strategy sma_cross \
  --data ./data \
  --params '{"fast":10,"slow":30}'
```

生成审计包（可复现回测）：

```bash
quantengine --config quantengine.example.yaml backtest \
  --strategy sma_cross \
  --data ./data \
  --params '{"fast":10,"slow":30}' \
  --audit-bundle ./audit.zip
```

## 功能矩阵

| 能力 | 说明 |
|------|------|
| 风控引擎 | `TradingRules` 7 字段：单笔风险、日内/周亏损、回撤阈值、最大持仓、加仓次数；Numba 集成；`risk_events` 输出 |
| 成本多情景 | `CostScenario` + `run_cost_scenarios`：低/中/高三档滑点与手续费，批量评估稳健性 |
| 审计包 | ZIP 结构含 `config.json`、`trades.csv`、`equity_curve.csv`、`risk_events.csv` 等；round-trip 校验；`replay` 复现 |
| 报告系统 | 8 区块（绩效/风险/交易/权益/敏感度等）；BH 校正；参数敏感度；可选 PDF |
| 策略 DSL | YAML 格式；`framework`/`risk_mode`；V4 策略全覆盖 |
| GPU 加速 | `batch_sim` + `evaluate_signal_tensor`；CPU 信号 + GPU 仿真混合；无 CUDA 时自动回退 |

## 项目结构概览

```
src/quantengine/
├── audit/           # 审计包生成、校验、复现
├── contracts/       # 合约规格抽象（multiplier、min_tick、trading_sessions）
├── data/            # 数据加载、缓存、GPU 后端、预处理
├── engine/          # 回测引擎、风控、成本情景、组合仿真
├── indicators/      # 技术指标（Numba/CuPy 感知）
├── metrics/         # 绩效、风险、批量指标
├── optimizer/       # grid/random/bayesian/genetic + walk-forward
├── strategy/        # 策略基类、DSL、注册表
└── interface/       # API 与 LiveAdapter
```

## 测试与质量

- **20 个核心测试模块、130+ 用例**：覆盖引擎、审计、合约、DSL、风控、成本情景、指标、策略、CLI、check-deps
- **ruff**：格式化与 lint
- **ADR**：架构决策记录见 `docs/decisions/`

```bash
ruff check src/
pytest tests/ -m "not slow"
```

## 配置说明

主配置 `quantengine.example.yaml` 支持：

- `runtime`：backend、initial_cash、periods_per_year、timezone
- `slippage` / `commission`：模型与档位
- `optimize`：metric、n_trials、batch_size

环境变量可覆盖部分配置，详见 `docs/quickstart.md`。

## 文档导航

- 快速开始：`docs/quickstart.md`
- 架构总览：`docs/architecture/overview.md`
- 功能说明：`docs/features/`（风控、成本情景、审计包、报告、DSL、GPU）
- CLI 与 API：`docs/api/cli.md`、`docs/api/python-api.md`
- 测试验收：`docs/testing/acceptance.md`

## 许可

MIT License，见 [LICENSE](LICENSE)。
