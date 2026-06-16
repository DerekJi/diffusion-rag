---
applyTo: "**/*.py"
---

# Python Naming Conventions for VedaAgent

> 本文档规定了VedaAgent项目中Python代码的命名规范，确保代码库保持一致性和可维护性。

## 核心原则

遵循 **PEP 8 — Style Guide for Python Code**，同时考虑项目特定的架构需求。

---

## 1. 包和模块名（Package & Module Names）

### 规则

- **Python包名**：全小写，使用下划线（`snake_case`）
- **模块文件名**：全小写，使用下划线（`snake_case`）
- **目录名**：对标准库/第三方的，用下划线；对CLI命令的，用短横线

### 示例

✅ **正确**
```python
from kernel.graph.state import GraphState
from kernel.llm import build_llm
from tools.code_tools.analysis import analyze_code
from memory.retrieval import MemoryRetriever
from session.context_manager import ContextManager
```

❌ **错误**
```python
from kernel.graph.state import GraphState  # 不要：Kernel (大写)
from kernel.LLM import build_llm  # 不要：LLM (全大写模块名)
from tools.CodeTools import analyze_code  # 不要：CodeTools (驼峰式)
```

---

## 2. 类名（Class Names）

### 规则

- **类名**：CapWords（PascalCase），简洁有力
- **前缀**：
  - 业务工具类用 `Tool` 后缀：`FileEditTool`, `ShellTool`
  - 管理器类用 `Manager` 后缀：`SessionManager`, `ContextManager`
  - 提取器/检索器用特定后缀：`LongTermMemoryExtractor`, `MemoryRetriever`
  - 基类用 `Base` 前缀或作为抽象：`VedaTool`（基类）

### 示例

✅ **正确**
```python
class SessionManager:
    """Manages session lifecycle."""
    pass

class FileEditTool(VedaTool):
    """Implements file editing tool."""
    pass

class LongTermMemoryExtractor:
    """Extracts long-term memory."""
    pass
```

❌ **错误**
```python
class session_manager:  # 不要：snake_case
class File_Edit_Tool:   # 不要：混合
class extractLongTermMemory:  # 不要：驼峰式
```

---

## 3. 函数和方法名（Function & Method Names）

### 规则

- **函数/方法名**：全小写，单词间用下划线（`snake_case`）
- **约定**：
  - `_private_method()`：单下划线前缀表示私有
  - `__dunder__()`：双下划线用于特殊方法（`__init__`, `__str__`, 等）
  - `async_method()`：异步方法应在名称中标明（可选，但推荐）或通过 `async def` 清晰

### 示例

✅ **正确**
```python
def build_tool_registry():
    """Public function."""
    pass

def _parse_session_id(session_id):
    """Private helper."""
    pass

async def execute_tools():
    """Async method."""
    pass

def get_session_info(self):
    """Method name is lowercase."""
    pass
```

❌ **错误**
```python
def buildToolRegistry():  # 不要：驼峰式
def parse_session_id():   # 不要：应加 _ 表示私有
def GetSessionInfo():     # 不要：CapWords用于函数
```

---

## 4. 常量名（Constant Names）

### 规则

- **常量名**：全大写，单词间用下划线（`UPPER_SNAKE_CASE`）
- **位置**：在模块或类的顶部定义
- **约定**：
  - 配置常数、魔数、枚举值都用全大写
  - 如果是模块级常数，应在 `__all__` 中导出（可选）

### 示例

✅ **正确**
```python
# veda_agent/config.py
DEFAULT_MAX_ITERATIONS = 10
CONTEXT_MAX_TOKENS = 8192
CONTEXT_RESERVE_TOKENS = 512
MEMORY_TYPE_LABELS = {
    "short_term": "Short-term",
    "long_term": "Long-term",
}

class SessionManager:
    MAX_SESSION_LIFETIME = 3600  # 1 hour
```

❌ **错误**
```python
default_max_iterations = 10  # 不要：应为常数
DefaultMaxIterations = 10    # 不要：常数用全大写
```

---

## 5. 变量名（Variable Names）

### 规则

- **变量名**：全小写，单词间用下划线（`snake_case`），清晰有意义
- **约定**：
  - 避免单字母变量，除非是循环索引（`i`, `j`, `k`）或明确的数学符号（`n`, `m`）
  - 避免模糊缩写；使用完整单词或已建立的术语

### 示例

✅ **正确**
```python
session_id = "sess_123"
context_window_size = 8192
llm_response = model.generate(prompt)
for i in range(len(items)):
    process(items[i])
```

❌ **错误**
```python
sessionID = "sess_123"  # 不要：驼峰式
s = "sess_123"          # 不要：模糊
ctx_sz = 8192           # 不要：无意义缩写
for x in range(len(items)):  # 不要：x 不清晰
```

---

## 6. CLI 命令和脚本名（CLI Command & Script Names）

### 规则

- **CLI 命令**：全小写，单词间用短横线（`kebab-case`）
- **脚本文件名**：同上或 `snake_case`（均接受）
- **目录名**：如果代表 CLI 命令组，用短横线；如果代表 Python 包，用下划线

### 示例

✅ **正确**
```toml
# pyproject.toml
[project.scripts]
veda = "kernel.cli.main:main"
kernel-web = "kernel.web.app:main"
kernel-stdio = "kernel.stdio.adapter:run_adapter"
```

```bash
# CLI usage
$ veda --help
$ kernel-web --port 8000
$ kernel-stdio start
```

❌ **错误**
```toml
veda_agent = "kernel.cli.main:main"  # 不要：CLI命令用短横线
veda-Agent = "kernel.cli.main:main"  # 不要：大小写不一
```

---

## 7. 目录结构命名（Directory Structure）

### 规则

- **Python 包目录**：下划线（`snake_case`），与模块导入一致
- **功能目录**：短横线（`kebab-case`）- 如测试数据、文档、脚本目录
- **顶级项目目录**：短横线（`kebab-case`）- 如 `veda-agent`, `tui` 等

### VedaAgent 标准结构

```
veda-agent/
├── kernel/           # Python package (underscores)
│   ├── cli/
│   ├── db/
│   ├── graph/
│   ├── llm.py
│   └── utils/
├── tools/            # Python package (underscores)
├── memory/           # Python package (underscores)
├── session/          # Python package (underscores)
├── tui/              # TypeScript frontend
├── harness/          # Test framework
├── tests/            # Test suite
├── docs/             # Documentation
├── data/             # Data files
└── pyproject.toml
```

---

## 8. 特殊约定（Special Conventions）

### 导出和 `__all__`

```python
# kernel/__init__.py
from kernel.graph import build_graph
from kernel.llm import build_llm

__all__ = [
    "build_graph",
    "build_llm",
]
```

### 类型提示（Type Hints）

```python
from typing import Optional, List, Dict

def build_context(
    session_id: str,
    messages: List[Dict[str, str]],
    max_tokens: Optional[int] = None,
) -> Dict[str, str]:
    """Type hints follow the same naming rules."""
    pass
```

### 枚举（Enums）

```python
from enum import Enum

class MemoryType(Enum):
    """Memory type enumeration."""
    SHORT_TERM = "short_term"
    LONG_TERM = "long_term"
```

---

## 9. 检查清单（Checklist）

在提交代码前，检查：

- [ ] 模块/包名都是 `snake_case`（小写字母和下划线）
- [ ] 类名都是 `CapWords`（PascalCase）
- [ ] 函数/方法名都是 `snake_case`（小写字母和下划线）
- [ ] 常量名都是 `UPPER_SNAKE_CASE`（全大写和下划线）
- [ ] CLI 命令都是 `kebab-case`（全小写和短横线）
- [ ] 没有混合使用驼峰式和下划线
- [ ] 导出的 API 在 `__all__` 中清晰列出
- [ ] 私有方法用 `_` 或 `__` 前缀

---

## 10. 参考资源

- [PEP 8 - Style Guide for Python Code](https://pep8.org/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [The Hitchhiker's Guide to Python - Code Style](https://docs.python-guide.org/writing/style/)

---

**Last Updated**: 2026-05-31  
**Applied To**: VedaAgent project repository
