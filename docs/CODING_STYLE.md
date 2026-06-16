# Coding Style & Quality Standards

> 本文档为 Diffusion RAG 项目的代码规范和质量要求。AI 实现时必须严格遵循。

---

## 1. 代码风格

### 1.1 格式化

- 遵循 **PEP 8**，行长度 ≤ **100 字符**
- 使用 **black** 格式化（`black . --line-length 100`）
- 使用 **isort** 管理 import 顺序（`isort .`）
- 所有 `.py` 文件头部统一：`#!/usr/bin/env python3`（仅 CLI 入口文件）

### 1.2 类型注解

- **所有公共函数**必须包含完整类型注解
- 内部函数（`_` 开头）鼓励但非强制
- 使用 `mypy --strict` 检查（可暂时为第三方库缺失加 `# type: ignore`）

```python
def encode(text: str) -> np.ndarray:
    ...
```

### 1.3 命名约定

| 类型 | 风格 | 示例 |
|------|------|------|
| 模块/文件名 | snake_case | `diffusion.py` |
| 类 | PascalCase | `ELFPipeline` |
| 函数/方法 | snake_case | `add_noise()` |
| 常量 | UPPER_SNAKE | `DEFAULT_CFG_SCALE` |
| 非公开变量/方法 | `_leading_underscore` | `_internal_cache` |
| 私有（不对外暴露） | `__double_underscore` | `__internal_state` |

---

## 2. 文档字符串

所有公共函数/类必须使用 **Google 风格 docstring**。

### 2.1 函数 docstring 模板

```python
def encode(text: str) -> np.ndarray:
    """将文本编码为 768-dim 向量。

    Args:
        text: 输入文本，长度不限。

    Returns:
        L2 归一化的浮点向量，shape (768,)。

    Raises:
        ValueError: 当 text 为空字符串时。
    """
```

### 2.2 类 docstring 模板

```python
class ELFPipeline:
    """ELF 扩散增强检索管线。

    封装 encode → add_noise → denoise → cfg_guide → final_norm 完整流程。

    Attributes:
        encoder: 底层 ELF 编码器实例。
        device: 当前计算设备。
    """
```

### 2.3 模块 docstring

每个 `.py` 文件顶部应有模块级 docstring，简要说明模块职责：

```python
"""ELF 扩散模型封装。

提供 encode / add_noise / denoise / cfg_guide 四个核心接口，
以及 ELFPipeline 完整修复链路。
"""
```

---

## 3. 错误处理

### 3.1 基本原则

- **所有外部调用**（网络下载、模型推理、文件 IO）必须捕获异常并给出可操作错误信息
- **禁止**裸露的 `except:` — 必须指定异常类型
- 关键配置缺失时应 `raise`，而不是静默使用默认值
- 使用 `logging` 记录异常堆栈，再决定是 `raise` 还是 fallback

### 3.2 推荐模式

```python
try:
    result = model.forward(inputs)
except RuntimeError as e:
    logger.error(f"ELF 模型推理失败: {e}")
    raise RuntimeError(f"模型推理失败: {e}") from e
```

### 3.3 空召回处理

检索未命中任何文档时：
- 指标返回 `Recall=0, Precision=0`
- 记录 `WARNING` 日志，包含 query_id

---

## 4. 日志规范

### 4.1 统一入口

- 使用 Python `logging` 模块，**禁止**直接 `print()`
- `src/utils/logger.py` 提供工厂函数：

```python
import logging

def get_logger(name: str) -> logging.Logger:
    """获取模块级 logger。"""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S"
        ))
        logger.addHandler(handler)
    return logger
```

### 4.2 日志级别

| 级别 | 适用场景 |
|------|---------|
| `DEBUG` | 详细的中间向量值、张量 shape（默认关闭） |
| `INFO` | 关键步骤：数据集加载、编码进度、参数组开始/结束 |
| `WARNING` | 降级行为：GPU 不可用 fallback CPU、空召回 |
| `ERROR` | 异常堆栈：模型加载失败、文件写入失败 |

### 4.3 模块使用

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("NFCorpus 数据集加载完成, %d 条文档", n_docs)
```

---

## 5. 测试规范

### 5.1 测试框架与结构

- 框架：**pytest**
- 文件位置：`tests/` 下，命名 `test_*.py`
- 覆盖率目标：`≥ 70%`（检查命令：`pytest --cov=src --cov-report=term-missing`）

### 5.2 编写原则

- **每个核心函数**至少对应一个测试用例（正常路径 + 边界条件）
- 所有测试必须**可重复运行**，不依赖外部网络（除专门的数据下载测试）
- 对涉及随机性的函数（如 `add_noise`），固定 `np.random.seed(42)` 后断言
- 评测链路测试（`test_orchestrator.py`）必须在 **10 秒内**完成（用 NFCorpus 子集）

### 5.3 Mock 策略

- 外部依赖（HuggingFace 下载、ELF 模型 forward）使用 `unittest.mock` 打桩
- `tests/fixtures/` 下放人工构造的 10 条文档 + 5 条查询的小库，用于快速测试

### 5.4 测试目录结构

```
tests/
├── conftest.py              # pytest fixtures (device, small_dataset, mock_elf)
├── test_elf.py              # ELF 接口单元测试 (≥8 用例)
├── test_baseline.py         # Baseline 编码器 + 指标测试
├── test_orchestrator.py     # 评测编排器集成测试 (NFCorpus 子集, <10s)
└── fixtures/
    ├── small_corpus.json    # 人工构建：10 条文档
    └── small_queries.json   # 人工构建：5 条查询 + qrels
```

---

## 6. 随机种子管理

### 6.1 全局固定

所有评测入口必须固定全部随机源：

```python
# src/utils/seed.py
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    """固定所有随机源，保证结果可复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 6.2 使用

在评测脚本入口、`conftest.py` 和 Notebook 第 1 Cell 调用 `set_seed(42)`。

---

## 7. 设备管理

### 7.1 统一获取设备

```python
from src.utils.device import get_device
device = get_device()   # str: "cuda" or "cpu"
```

### 7.2 显式移动张量

- **所有模型 + 张量**必须显式调用 `.to(device)`，不依赖默认设备
- 推理代码中禁止省略 `.to(device)`

```python
model = ELFModel(...).to(device)
tensor = torch.randn(768).to(device)
```

---

## 8. 性能测量

### 8.1 通用方法

首次测量前需 warm-up（至少 1 次推理排除 CUDA 懒初始化）。

### 8.2 GPU 耗时（精确）

```python
start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

start_event.record()
# ... 推理 ...
end_event.record()
torch.cuda.synchronize()
latency_ms = start_event.elapsed_time(end_event)
```

### 8.3 Wall-clock 耗时（简化）

```python
import time
t0 = time.perf_counter()
# ... 推理 ...
t1 = time.perf_counter()
latency_ms = (t1 - t0) * 1000
```

> 注：Wall-clock 包含 CPU-GPU 同步误差。在报告中注明测量方式。

---

## 9. 配置与参数管理

### 9.1 配置格式

所有实验配置使用 **YAML**，存放在 `experiments/configs/`。

### 9.2 参数网格定义

**不要使用动态笛卡尔积**。12 组参数直接在 `src/config.py` 中用列表硬编码：

```python
# src/config.py
PARAM_GRID = [
    {"id": "ELF-01", "steps": 1, "t_start": 0.3, "cfg_scale": 1.0},
    {"id": "ELF-02", "steps": 1, "t_start": 0.5, "cfg_scale": 1.0},
    # ... 全部 12 组
]
```

### 9.3 YAML schema 示例

```yaml
# experiments/configs/param_grid.yaml
dataset: "nfcorpus"          # nfcorpus / msmarco / nq / fiqa
baseline:
  encoder: "BAAI/bge-base-en-v1.5"
  device: "auto"

elf:
  param_list:                # 显式列表，非笛卡尔积
    - {steps: 1, t_start: 0.3, cfg_scale: 1.0}
    - {steps: 1, t_start: 0.5, cfg_scale: 1.0}
    # ... 全部 12 组

eval:
  k_values: [5, 10, 20]
  seed: 42
  repeats: 3
  robustness_noises: ["typo", "abbr", "colloquial"]
```

---

## 10. 数据下载与缓存

### 10.1 下载脚本

`scripts/download_assets.py` 负责下载全部需要的数据和权重：

- 数据集：使用 `datasets.load_dataset(..., cache_dir=...)` 
- ELF-B 权重：从 HuggingFace 自动拉取，缓存到本地
- BGE 模型：从 HuggingFace 自动拉取

### 10.2 缓存策略

- 所有下载应有**重试机制**（最多 3 次）和**超时设置**（60s）
- 脚本启动时检查本地缓存，已存在则跳过下载
- Colab 环境下缓存指向 Google Drive 持久化目录

### 10.3 HuggingFace 环境变量

```bash
# 可选：自定义缓存路径
export HF_HOME=/path/to/cache
export HF_DATASETS_CACHE=/path/to/cache/datasets
```

---

## 11. 断点续跑机制

### 11.1 实现方式

不引入额外框架，通过文件存在性检查实现：

```python
def get_completed_configs(output_dir: str) -> set[str]:
    """扫描已有输出，返回已完成配置的 config_id 集合。"""
    summary_path = Path(output_dir) / "summary.csv"
    if not summary_path.exists():
        return set()
    df = pd.read_csv(summary_path)
    return set(df["config_id"].tolist())
```

### 11.2 行为

- 运行前扫描 `summary.csv`，跳过已完成的 `config_id`
- 支持 `--force` 标志强制重新运行所有
- 评测器在每组参数完成后立即写入 `summary.csv`（而不是统一在最后写入）

---

## 12. pyproject.toml 配置

```toml
[tool.black]
line-length = 100
target-version = ["py310"]

[tool.isort]
profile = "black"
line-length = 100

[tool.mypy]
strict = true
ignore_missing_imports = true
python_version = "3.10"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=src --cov-report=term-missing"
```

---

> **AI 开发者**：每次提交前请运行以下检查：
> ```bash
> black . --line-length 100 && isort . && mypy src/ && pytest --cov=src
> ```
