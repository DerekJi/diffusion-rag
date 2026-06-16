# Copilot 指南（Diffusion RAG）

## 仓库概览

这是一个 **Python 检索与评测研究项目**，核心目标是验证扩散增强检索是否能在不改变向量数据库架构的前提下，提升语义检索的精度与鲁棒性。

当前代码库的主要模块：
- `src/baseline/`：传统基线编码与评测入口
- `src/vector_store/`：向量索引与检索
- `src/evaluation/`：数据集加载与指标计算
- `src/utils/`：日志、随机种子、设备管理

**主要技术栈：**
- Python 3.10+
- PyTorch
- FAISS
- HuggingFace `datasets` / `transformers`
- pytest

## 关键前提

- 以 `docs/` 中的需求、计划、验收标准为最高依据
- 以当前仓库已有实现、目录结构、测试风格为准
- 优先复用现有模块，避免引入无关技术栈

## 构建与测试

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行测试

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 代码检查

```bash
black . --line-length 100 --check
isort . --check-only
mypy src/ --strict
```

## 本地开发

### 运行基线评测

```bash
python -m src.baseline.benchmark --dataset nfcorpus
```

### 常见入口

- `python -m src.baseline.benchmark`：运行传统基线评测
- `python -m src.baseline.benchmark --help`：查看参数说明

## 项目结构

```text
/
├── docs/                   # 需求、计划、代码规范
├── experiments/            # 实验输出
├── models/                 # 模型权重缓存
├── src/
│   ├── baseline/
│   ├── evaluation/
│   ├── utils/
│   └── vector_store/
├── tests/
├── data/
├── README.md
├── requirements.txt
└── pyproject.toml
```

## 开发约定

### Python 代码
- 模块/文件名使用 `snake_case`
- 类名使用 `PascalCase`
- 函数/方法使用 `snake_case`
- 常量使用 `UPPER_SNAKE_CASE`
- 公共函数必须有类型注解和 docstring
- 禁止在库代码中直接使用 `print()`，统一使用 `logging`

### 测试
- 使用 `pytest`
- 测试文件命名为 `test_*.py`
- 对随机逻辑固定种子
- 对外部依赖使用 mock

## 修改代码时的原则

1. 先看 `docs/`，再看现有实现
2. 保持与 `src/baseline/`、`src/evaluation/`、`src/vector_store/`、`src/utils/` 的边界一致
3. 新增行为必须补测试
4. 优先做最小可验证改动
5. 修改后运行相关测试并确认通过

## 参考文档

- `docs/CODING_STYLE.md`
- `docs/diffusion-rag-plan.md`
- `docs/phase1-detailed-plan.md`
- `docs/diffusion-rag.md`
