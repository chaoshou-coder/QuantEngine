# 运维排障手册

本页覆盖常见故障、排查步骤与建议修复方案。

## 1. 配置相关

### 现象：`ConfigError: 配置文件不存在`

- 原因：`--config` 指向路径不存在
- 处理：
  - 检查路径（相对路径基于当前工作目录）
  - 临时去掉 `--config`，验证默认配置能否运行

### 现象：`ConfigError: 读取配置失败`

- 原因：YAML 语法错误或字段类型不匹配
- 处理：
  - 用 `yaml` 工具验证文件格式
  - 检查字段默认值：`runtime.backend`、`optimize.n_trials`、`periods_per_year` 等

## 2. 数据相关

### 现象：`未发现可用数据文件`

- 原因：目录不存在或无 `.csv/.parquet`
- 处理：
  - `--data` 指向正确目录/文件
  - 检查是否递归子目录中存在文件

### 现象：`OHLCV 数据缺少字段`

- 原因：列名不匹配/缺列
- 处理：
  - 列名必须包含 `open/high/low/close/volume`（会自动转小写）
  - 时间列可为 `datetime/date/index`

### 现象：回测结果异常或收益全 0

- 原因：时间列无法对齐、前向填充过多、信号全部 0
- 处理：
  - 先用较小样本复现
  - 检查 `quantengine report` 输出中 `trades` 数量和 `Trade Metrics`

## 3. 运行与兼容相关

### 现象：GPU 未启用

- 现象：`backend.active` 为 `cpu`
- 处理：
  - 检查 `runtime.use_gpu`（建议 true）
  - 检查 CUDA 驱动与 cuPy/cudf 安装
  - `backend` 强制为 `auto/cpu` 时会按配置降级，不是错误

### 现象：`参数空间为空` / `未知指标`

- 现象：
  - Grid 搜索提示参数空间为空
  - 优化报 `未知指标: X`
- 处理：
  - 检查 `BaseStrategy.parameters` 是否正确返回
  - `--metric` 是否在 report 的 performance/risk/trade_metrics 字段里

### 现象：`未知策略`（`KeyError`）

- 原因：策略未注册或 import 未触发
- 处理：
  - 运行 `list-strategies` 确认名称
  - 确保策略文件被 import（当前内置在 `strategy/__init__.py`）

## 4. CLI/输出相关

### 现象：`JSON 参数解析失败`

- 原因：`--params` 或 `--param-grid` 不是合法 JSON
- 处理：
  - 使用双引号并避免尾逗号
  - 参数示例见 `quickstart.md`

### 现象：`report` 命令报不支持的 `result type`

- 原因：`--result` JSON 缺少 `type` 或 `type` 不是 `backtest/optimization`
- 处理：
  - 重新从 CLI 重新落盘 JSON，或补齐 payload 字段

## 5. 缓存相关

### 现象：缓存未命中预期

- 原因：`cache` 文件不存在、目录不一致或输入文件 `mtime/size` 已变化
- 处理：
  - 检查 `.quantengine_cache` 是否在数据目录下（或文件同级目录）
  - 确认同一数据路径下 `.npz` 与 `.meta` 成对出现
  - 若仍异常，临时禁用缓存 `DataLoader(cache=False)` 验证

### 现象：缓存损坏或报错

- 现象：缓存文件读写异常、`json`/`pickle` 反序列化失败
- 处理：
  - 删除对应 `.quantengine_cache` 目录后重试（重新建缓存）
  - 检查磁盘空间与权限

## 6. Walk-Forward 相关

### 现象：`样本长度不足，无法构建 walk-forward 折叠`

- 原因：样本规模太小，按 `n_splits` 切分后 IS/OOS 都不满足最小长度
- 处理：
  - 提高数据时段长度
  - 降低 `n-splits` 或提升 `is-ratio`
  - 检查 `n_bars` 是否过少

### 现象：`Walk-Forward 结果过于异常（overfitting_ratio 极低）`

- 原因：IS 过拟合，OOS 表现显著退化
- 处理：
  - 增加折数，缩小每折学习窗口
  - 在 `WalkForward` 中降低 `n_trials` 或改用更稳健的 `method`
  - 结合外部时间段独立验证

### 现象：`overfitting_ratio` 为 0 或 NaN`

- 原因：部分折里 OOS 性能无法计算或 IS 指标为 0
- 处理：
  - 检查策略是否在某些折不交易
  - 检查输入收益序列是否过短/无效

## 7. 应急操作清单

1. `quantengine --help`
2. `quantengine --config quantengine.yaml list-strategies`
3. 随机抽样 1 个小文件运行 `backtest`
4. 检查 `runtime`、`backend`、`DataBundle.shape`
5. 若 GPU 不稳定，`runtime.use_gpu=false` 强制 CPU
