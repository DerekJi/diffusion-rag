---
applyTo: "**/*.py"
---

# Python 命名规范（Diffusion RAG）

> 本文档规定本仓库 Python 代码的命名规范，目标是保持一致性、可读性和可维护性。

## 总原则

遵循 **PEP 8**，并结合当前仓库的模块划分：
- `src/baseline/`
- `src/evaluation/`
- `src/vector_store/`
- `src/utils/`

---

## 1. 包和模块名

### 规则
- Python 包名使用全小写 `snake_case`
- 模块文件名使用全小写 `snake_case`
- 目录名与 Python 包保持一致时，也使用 `snake_case`

### 示例

✅ 正确
```python
from src.baseline.encoder import BaselineEncoder
from src.vector_store.indexer import FAISSIndexer
from src.evaluation.metrics import compute_metrics
from src.utils.logger import get_logger
```

❌ 错误
```python
from src.Baseline.encoder import BaselineEncoder
from src.vectorStore.indexer import FAISSIndexer
```

---

## 2. 类名

### 规则
- 类名使用 `PascalCase`
- 类名应简洁、明确，尽量表达职责

### 常见后缀
- 编码器：`Encoder`
- 索引器：`Indexer`
- 检索器：`Retriever`
- 管理器：`Manager`
- 结果对象：`Result`
- 数据对象：`Dataset` / `Config`

### 示例

✅ 正确
```python
class BaselineEncoder:
    pass

class FAISSIndexer:
    pass

class MetricsResult:
    pass
```

❌ 错误
```python
class baseline_encoder:
    pass

class faiss_indexer:
    pass
```

---

## 3. 函数和方法名

### 规则
- 函数和方法名使用 `snake_case`
- 私有辅助函数使用单下划线前缀 `_`
- 特殊方法保持 Python 约定写法，如 `__init__`

### 示例

✅ 正确
```python
def load_dataset(name: str):
    ...

def _normalize_vector(vector):
    ...
```

❌ 错误
```python
def loadDataset(name: str):
    ...

def NormalizeVector(vector):
    ...
```

---

## 4. 常量名

### 规则
- 常量使用 `UPPER_SNAKE_CASE`
- 配置项、默认值、固定参数都应使用常量命名

### 示例

✅ 正确
```python
DEFAULT_K_VALUES = [5, 10, 20]
VECTOR_DIM = 768
DEFAULT_SEED = 42
```

❌ 错误
```python
default_k_values = [5, 10, 20]
vectorDim = 768
```

---

## 5. 变量名

### 规则
- 变量名使用 `snake_case`
- 名称应尽量具体，避免模糊缩写
- 循环索引可使用 `i`, `j`, `k`

### 示例

✅ 正确
```python
query_text = "what is retrieval"
query_ids = ["q1", "q2"]
retrieved_doc_ids = ["d1", "d2"]
```

❌ 错误
```python
qTxt = "what is retrieval"
ids = ["q1", "q2"]
```

---

## 6. CLI 命令和脚本名

### 规则
- CLI 命令使用全小写 `kebab-case`
- 脚本文件名优先使用 `snake_case`

### 示例

✅ 正确
```toml
[project.scripts]
baseline-benchmark = "src.baseline.benchmark:main"
```

❌ 错误
```toml
baselineBenchmark = "src.baseline.benchmark:main"
```

---

## 7. 目录命名

### 规则
- Python 包目录使用 `snake_case`
- 功能目录保持与模块导入一致

### 当前仓库示例
```text
src/
├── baseline/
├── evaluation/
├── vector_store/
└── utils/
```

---

## 8. 特殊约定

### 8.1 导出 API
如需要导出公共 API，建议在 `__init__.py` 中显式声明。

### 8.2 类型提示
公共函数应尽量补全类型提示。

### 8.3 文档字符串
公共类和函数应使用中文或中英结合的清晰 docstring，说明用途、参数和返回值。

---

## 9. 检查清单

提交前请确认：
- [ ] 模块/包名是 `snake_case`
- [ ] 类名是 `PascalCase`
- [ ] 函数/方法名是 `snake_case`
- [ ] 常量名是 `UPPER_SNAKE_CASE`
- [ ] 私有辅助函数使用 `_` 前缀
- [ ] CLI 命令使用 `kebab-case`
- [ ] 公共 API 有类型注解和 docstring

---

**适用范围**：本仓库全部 Python 文件
