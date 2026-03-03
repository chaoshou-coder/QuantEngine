# GPU 环境与兼容性

QuantEngine 在 `runtime.backend=auto` 且 `runtime.use_gpu=true` 时会尝试启用 GPU。
本页说明安装、检测和常见兼容问题。

## 1. 安装依赖

基础版（无 GPU）：

```bash
pip install -e .
```

完整加速版：

```bash
pip install -e .[engine]
```

`[engine]` 同步安装：
- cupy-cuda12x
- cudf-cu12
- optuna
- nevergrad / deap
- plotly / matplotlib / torch / numba

## 2. 兼容性要求（建议）

- 已安装 NVIDIA 驱动与 CUDA 可见设备
- Python 版本与 CUDA wheel 兼容
- cuDF 与 CuPy 版本兼容

若任一组件缺失，系统会：
- 回退到 `auto` 到 CPU（若未强制 `gpu`）
- 或在强制 `backend=gpu` 时打印原因并转为 CPU 执行

## 3. 运行时后端检查

在 Python 中可快速确认当前后端：

```python
from quantengine.data.gpu_backend import get_backend_info
print(get_backend_info("auto", True))
```

输出字段说明：

- `requested`：请求的 backend（auto/cpu/gpu）
- `active`：实际使用的 backend（auto->gpu 或 cpu）
- `reason`：切换/回退原因
- `gpu_available`：是否有可用 CUDA
- `cudf_available`/`cupy_available`

## 4. 强制策略

- `runtime.backend: "cpu"`：即使有 GPU 也不使用
- `runtime.backend: "gpu"`：优先 GPU，不可用时自动回退 CPU
- `runtime.use_gpu: false`：明确关闭 GPU

## 5. 常见问题

### 5.1 GPU 安装成功但未生效

- 检查 `get_backend_info` 返回 `active` 是否为 `gpu`
- 检查 `runtime.use_gpu` 是否 true
- 检查是否在 Python 环境里 import cupy 失败

### 5.2 cuDF/CuPy 版本冲突

- 统一使用同一 CUDA 版本对应 wheel（例如本配置采用 `cuda12x/cuda12` 前缀）
- 若升级/降级后不稳定，先回退 CPU 验证逻辑正确，再调整依赖

### 5.3 内存溢出

- 降低 `optimization.batch_size`
- 减小待评估参数量
- 切换到 `random` 或 `bayesian`（依赖版本）替代大规模 grid
