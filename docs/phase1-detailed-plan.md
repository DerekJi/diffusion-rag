# Phase 1 详细计划：环境与基准链路搭建

> 对应 `docs/diffusion-rag-plan.md §3.1 — 阶段 1`  
> 执行环境：**本机**（RTX 1000 Ada 6GB）  
> 预估工时：**2-3 天**  
> 代码规范：必须遵循 `docs/CODING_STYLE.md`

---

## 一、阶段目标

建立完整的传统检索基线（Baseline）链路，可在单条命令下输出可复现的评测指标。

**阶段 1 不做的事：**
- ❌ ELF 模型加载或封装（那是 Phase 2）
- ❌ 任何扩散增强流程
- ❌ Colab 迁移

---

## 二、任务分解

### 2.1 任务 1.1：环境搭建与依赖安装

**产出物：** 可运行的 Python 环境 + 预训练权重缓存

| 步骤 | 操作 | 验证方式 |
|------|------|---------|
| 1.1.1 | 创建项目根目录结构 | `tree -L 2` 检查 |
| 1.1.2 | 创建 `pyproject.toml`（black/isort/mypy/pytest 配置） | 见 CODING_STYLE.md §12 |
| 1.1.3 | 创建 `requirements.txt` | 见下方依赖清单 |
| 1.1.4 | `pip install -r requirements.txt` | 无报错 |
| 1.1.5 | 创建 `src/` 包骨架（所有 `__init__.py`） | `python -c "import src"` 成功 |
| 1.1.6 | 创建 `.gitignore` | Git 忽略 `__pycache__/`, `.pt`, `data/`, `models/` |
| 1.1.7 | Git init + 首次 commit | `git log` 可见 |

#### 依赖清单（`requirements.txt`）

```txt
torch>=2.0.0
torchvision
faiss-cpu>=1.7.4      # 开发调试用 CPU 版
transformers>=4.30.0
datasets>=2.14.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
pandas>=2.0.0
pyyaml>=6.0
matplotlib>=3.7.0
pytest>=7.4.0
pytest-cov>=4.1.0
tqdm>=4.65.0
black>=23.0.0
isort>=5.12.0
mypy>=1.5.0
sentence-transformers>=2.2.0    # BGE 编码器
```

#### 项目目录结构（Phase 1 结束时）

```
diffusion-rag/
├── pyproject.toml
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── config.py                  # 全局配置（路径 / 默认参数 / 常量）
│   ├── baseline/
│   │   ├── __init__.py
│   │   ├── encoder.py             # BGE-Mini 编码器
│   │   └── benchmark.py           # 基线评测 CLI 入口
│   ├── vector_store/
│   │   ├── __init__.py
│   │   ├── indexer.py             # FAISS 索引构建
│   │   └── retriever.py           # 检索接口
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── dataset.py             # 数据集加载
│   │   └── metrics.py             # Recall / MRR / NDCG / HitRate
│   └── utils/
│       ├── __init__.py
│       ├── logger.py              # 日志工厂
│       ├── seed.py                # 随机种子全局固定
│       └── device.py              # 设备自动检测
├── tests/
│   ├── __init__.py
│   ├── conftest.py                # pytest fixtures
│   └── test_baseline.py           # Baseline 编码器 + 指标测试
├── data/                          # 数据集缓存（HuggingFace 自动下载，gitignored）
└── models/                        # 预训练权重（gitignored）
```

---

### 2.2 任务 1.2：BGE-Mini 基线编码器

**产出物：** `src/baseline/encoder.py`

#### 接口定义

```python
class BaselineEncoder:
    """BGE-Mini 基线编码器。

    封装 BAAI/bge-base-en-v1.5，输出 768-dim L2 归一化向量。
    与 ELF 编码器输出维度、格式完全一致。
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = "auto"):
        ...

    def encode(self, text: str) -> np.ndarray:
        """将单条文本编码为 768-dim 向量。

        Args:
            text: 输入文本。

        Returns:
            L2 归一化的 float32 向量，shape (768,)。
        """
        ...

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """批量编码。

        Returns:
            shape (len(texts), 768) float32 数组。
        """
        ...
```

#### 实现要点

| 要点 | 说明 |
|------|------|
| 模型选择 | `BAAI/bge-base-en-v1.5`（英文），768-dim |
| 归一化 | 编码后立即 `vec = vec / np.linalg.norm(vec)` |
| device 参数 | `"auto"` → 调用 `utils.device.get_device()` |
| 缓存 | HuggingFace 自动缓存到 `~/.cache/huggingface/hub/` |
| 错误处理 | 模型加载失败 → 明确异常信息（无法下载 / 磁盘空间不足） |

#### 单元测试（`tests/test_baseline.py`）

| 测试 | 断言 |
|------|------|
| `test_encode_shape` | 输出 shape == (768,) |
| `test_encode_dtype` | 输出 dtype == np.float32 |
| `test_encode_l2_norm` | `np.linalg.norm(vec) ≈ 1.0`（误差 < 1e-5） |
| `test_encode_deterministic` | 同一文本两次编码结果一致 |
| `test_encode_empty_string` | 空字符串应 raise `ValueError` |
| `test_encode_batch_shape` | batch 输出 shape == (N, 768) |

---

### 2.3 任务 1.3：FAISS 索引构建与检索接口

**产出物：** `src/vector_store/indexer.py` + `src/vector_store/retriever.py`

#### 接口定义

```python
# indexer.py
class FAISSIndexer:
    """FAISS 索引构建器。

    基于文档向量构建 IVF_FLAT 索引。
    """

    def __init__(self, dimension: int = 768, nlist: int = 100):
        ...

    def build(self, vectors: np.ndarray, doc_ids: list[str]) -> None:
        """从文档向量构建索引。

        Args:
            vectors: shape (n_docs, dimension) float32。
            doc_ids: 对应文档 ID 列表。
        """
        ...

    def save(self, path: str) -> None:
        """将索引序列化到磁盘。"""
        ...

    def load(self, path: str) -> None:
        """从磁盘加载索引。"""
        ...

    @property
    def index(self) -> faiss.Index:
        """原始 FAISS 索引（供 retriever 使用）。"""
        ...

# retriever.py
class Retriever:
    """统一检索接口（双链路共享）。"""

    def __init__(self, indexer: FAISSIndexer):
        ...

    def search(self, query_vector: np.ndarray, k: int = 10) -> tuple[list[str], list[float]]:
        """检索 top-k 结果。

        Args:
            query_vector: 768-dim 查询向量。
            k: 返回数量。

        Returns:
            (doc_ids, distances): 文档 ID 列表和对应距离。
            无结果时返回 ([], [])。
        """
        ...
```

#### 实现要点

| 要点 | 说明 |
|------|------|
| 索引类型 | IVF_FLAT（IndexIVFFlat），兼顾速度与精度 |
| nlist | 默认 100（NFCorpus 3.6K docs），数据集大时需调大 |
| 训练 | `build()` 内部自动 `index.train()` + `index.add()` |
| 多线程 | `faiss.omp_set_num_threads(4)` |
| 容错 | 空索引时 `search()` 返回 `([], [])`，不崩溃 |

#### 单元测试（`tests/test_baseline.py`）

| 测试 | 断言 |
|------|------|
| `test_index_build_and_search` | 建索引后再搜自己，top-1 doc_id 匹配 |
| `test_search_empty_index` | 未建索引时 search 返回 `([], [])` |
| `test_index_save_load_roundtrip` | save → load → search，结果一致 |
| `test_search_k_larger_than_n_docs` | k > n_docs 时不崩溃，返回全部 |

---

### 2.4 任务 1.4：数据集加载

**产出物：** `src/evaluation/dataset.py`

#### 接口定义

```python
@dataclass
class DatasetTriple:
    """数据集三元组。"""
    queries: dict[str, str]       # {query_id: query_text}
    corpus: dict[str, str]        # {doc_id: doc_text}
    qrels: dict[str, dict[str, int]]  # {query_id: {doc_id: relevance_score}}


def load_dataset(name: str, cache_dir: str | None = None) -> DatasetTriple:
    """加载 BEIR 格式数据集。

    Args:
        name: 数据集名称，支持 'nfcorpus', 'msmarco', 'nq', 'fiqa'。
        cache_dir: HuggingFace 数据集缓存路径，None 使用默认。

    Returns:
        DatasetTriple (queries, corpus, qrels)。

    Raises:
        ValueError: 不支持的数据集名称。
        RuntimeError: 下载失败（含网络错误）。
    """
    ...
```

#### 实现要点

| 要点 | 说明 |
|------|------|
| 加载方式 | 全部通过 HuggingFace `datasets.load_dataset()` |
| 数据集映射 | `nfcorpus` → `BeIR/nfcorpus`，`msmarco` → `BeIR/msmarco`，`nq` → `BeIR/nq`，`fiqa` → `BeIR/fiqa` |
| 格式转换 | BEIR 的 HuggingFace 格式 → `DatasetTriple` 统一结构 |
| 缓存重定向 | 接受 `cache_dir` 参数，Colab 时可指向 Google Drive |
| 错误处理 | 网络错误最多重试 3 次，每次间隔 5s |
| NFCorpus 调试 | 额外提供 `load_small_debug()` → 从 NFCorpus 取 50 条 query + 100 条 doc |

#### 测试方法

手动验证（不需要单元测试，因依赖网络）：

```bash
python -c "
from src.evaluation.dataset import load_dataset
data = load_dataset('nfcorpus')
print(f'queries: {len(data.queries)}, corpus: {len(data.corpus)}, qrels: {sum(len(v) for v in data.qrels.values())}')
# 期望: queries: 323, corpus: ~3600, qrels: ~3000
"
```

---

### 2.5 任务 1.5：核心指标计算

**产出物：** `src/evaluation/metrics.py`

#### 接口定义

```python
@dataclass
class MetricsResult:
    """单条查询的指标结果。"""
    query_id: str
    recall: dict[int, float]      # {k: recall@k}
    precision: dict[int, float]   # {k: precision@k}
    mrr: float
    ndcg: dict[int, float]        # {k: ndcg@k}
    hit_rate: dict[int, float]    # {k: hit_rate@k}


def compute_metrics(
    qrels: dict[str, int],
    results: list[str],
    k_values: list[int] = [5, 10, 20],
) -> MetricsResult:
    """计算单条查询的全部检索指标。

    Args:
        qrels: {doc_id: relevance_score}，该查询的真实相关文档。
        results: 检索返回的 doc_id 列表（按距离升序，最相关在前）。
        k_values: 计算哪些 k 位置的指标。

    Returns:
        MetricsResult 包含所有指标。

    Notes:
        - results 为空列表时，所有指标为 0.0。
        - qrels 中 relevance_score >= 1 视为相关。
    """
    ...


def compute_metrics_batch(
    all_qrels: dict[str, dict[str, int]],
    all_results: dict[str, list[str]],
    k_values: list[int] = [5, 10, 20],
) -> dict[str, MetricsResult]:
    """批量计算，返回 {query_id: MetricsResult}。"""
    ...
```

#### 指标计算公式

| 指标 | 公式 | 说明 |
|------|------|------|
| Recall@k | `#相关文档在 top-k 中 / #总相关文档` | 无相关文档时 = 0 |
| Precision@k | `#相关文档在 top-k 中 / k` | |
| MRR | `1 / 首个相关文档的排名` | 无相关文档时 = 0 |
| NDCG@k | `DCG@k / IDCG@k` | 使用 `scipy.stats.dcg_score` |
| HitRate@k | `1 if 任意相关文档在 top-k 中 else 0` | |

#### 单元测试（`tests/test_baseline.py`）

| 测试 | 断言 |
|------|------|
| `test_recall_perfect` | results 包含所有相关文档时 recall@k = 1.0 |
| `test_recall_zero` | results 为空时 recall = 0.0 |
| `test_mrr` | 首个相关文档在 rank=3 时 MRR = 1/3 |
| `test_ndcg_vs_known` | 手工计算验证 NDCG@10 |
| `test_hit_rate` | 有命中时 hit_rate = 1，无命中时 = 0 |

---

### 2.6 任务 1.6：端到端基线评测 CLI

**产出物：** `src/baseline/benchmark.py`

#### 接口定义

```python
def run_benchmark(
    dataset: str = "nfcorpus",
    encoder_name: str = "BAAI/bge-base-en-v1.5",
    index_nlist: int = 100,
    k_values: list[int] = [5, 10, 20],
    seed: int = 42,
    output_dir: str = "experiments/outputs",
) -> pd.DataFrame:
    """运行完整基线评测流程。

    流程:
        1. 固定随机种子
        2. 加载数据集
        3. 用 BaselineEncoder 编码所有文档 → 构建 FAISS 索引
        4. 用 BaselineEncoder 编码所有查询
        5. 逐条检索 + 计算指标
        6. 汇总结果并保存为 CSV

    Returns:
        包含每参数组（此阶段仅有一组）聚合指标的 DataFrame。
    """
    ...
```

#### CLI 入口（`python -m src.baseline.benchmark`）

```python
# src/baseline/benchmark.py (底部)
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Baseline 检索评测")
    parser.add_argument("--dataset", default="nfcorpus", choices=["nfcorpus", "msmarco", "nq", "fiqa"])
    parser.add_argument("--encoder", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="experiments/outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    df = run_benchmark(
        dataset=args.dataset,
        encoder_name=args.encoder,
        k_values=args.k,
        seed=args.seed,
        output_dir=args.output,
    )
    print(df.to_markdown())
```

#### 日志输出示例

```
$ python -m src.baseline.benchmark --dataset nfcorpus

[INFO] 2026-06-16 09:30:00 baseline.benchmark: 固定随机种子 seed=42
[INFO] 2026-06-16 09:30:01 evaluation.dataset: 加载数据集 nfcorpus
[INFO] 2026-06-16 09:30:01 evaluation.dataset:   queries=323, corpus=3630, qrels=3230
[INFO] 2026-06-16 09:30:05 baseline.encoder: 加载 BAAI/bge-base-en-v1.5 (device=cuda)
[INFO] 2026-06-16 09:30:08 vector_store.indexer: 编码 3630 篇文档
[INFO] 2026-06-16 09:30:15 vector_store.indexer: 构建 IVF_FLAT(nlist=100) 索引
[INFO] 2026-06-16 09:30:16 baseline.benchmark: 检索 323 条查询...
[INFO] 2026-06-16 09:30:18 baseline.benchmark: 评测完成
[INFO] 2026-06-16 09:30:18 baseline.benchmark: 结果已保存到 experiments/outputs/nfcorpus/baseline.csv

| dataset   |   recall@5 |   recall@10 |   recall@20 |   mrr |   ndcg@10 |
|-----------|------------|-------------|-------------|-------|-----------|
| nfcorpus  |      0.xxx |       0.xxx |       0.xxx | 0.xxx |     0.xxx |
```

---

## 三、基础设施文件清单

### 3.1 `src/config.py`

```python
"""全局配置。"""

# 路径
MODELS_DIR = "models"
DATA_DIR = "data"
EXPERIMENTS_DIR = "experiments"

# 向量维度
VECTOR_DIM = 768

# 默认检索参数
DEFAULT_K_VALUES = [5, 10, 20]
DEFAULT_INDEX_NLIST = 100
DEFAULT_ENCODER = "BAAI/bge-base-en-v1.5"

# 评测
DEFAULT_SEED = 42
DEFAULT_REPEATS = 3

# Colab 模式（Phase 4 启用）
COLAB_MODE = False
```

### 3.2 `src/utils/logger.py`（CODING_STYLE.md §4）

```python
import logging

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
        ))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger
```

### 3.3 `src/utils/seed.py`（CODING_STYLE.md §6）

```python
import random
import numpy as np
import torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

### 3.4 `src/utils/device.py`（CODING_STYLE.md §7 + 总体规划 §1.5）

```python
import torch

def get_device() -> str:
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[device] GPU: {gpu_name}, VRAM: {vram:.1f} GB")
    else:
        device = "cpu"
        print("[device] No GPU — CPU mode (debug only)")
    return device
```

### 3.5 `tests/conftest.py`

```python
import pytest
import numpy as np
from src.utils.seed import set_seed

@pytest.fixture(autouse=True)
def _fix_seed():
    """每个测试用例自动固定随机种子。"""
    set_seed(42)

@pytest.fixture
def sample_texts() -> list[str]:
    return ["What is the capital of France?", "Python programming language", "Climate change effects"]

@pytest.fixture
def dummy_vectors() -> np.ndarray:
    """10 个随机 768-dim 归一化向量。"""
    rng = np.random.RandomState(42)
    vecs = rng.randn(10, 768).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs
```

---

## 四、验收检查清单

### 4.1 逐任务验收

| 任务 | 验收命令 / 操作 | 预期结果 |
|------|----------------|---------|
| 1.1 环境 | `python -c "import torch, faiss, transformers, datasets"` | 无 ImportError |
| 1.1 环境 | `pip list \| grep -E "torch\|faiss\|transformers\|datasets"` | 版本符合 `requirements.txt` |
| 1.2 编码器 | `pytest tests/test_baseline.py::test_encode_shape -v` | PASSED |
| 1.2 编码器 | `pytest tests/test_baseline.py::test_encode_l2_norm -v` | PASSED |
| 1.3 索引+检索 | `pytest tests/test_baseline.py::test_index_build_and_search -v` | PASSED |
| 1.4 数据集 | `python -c "from src.evaluation.dataset import load_dataset; d=load_dataset('nfcorpus'); print(len(d.queries))"` | 323 |
| 1.5 指标 | `pytest tests/test_baseline.py::test_recall_perfect -v` | PASSED |
| 1.6 端到端 | `python -m src.baseline.benchmark --dataset nfcorpus` | 控制台打印指标表，`experiments/outputs/nfcorpus/baseline.csv` 存在 |

### 4.2 代码规范检查

```bash
# 格式
black . --line-length 100 --check

# import 顺序
isort . --check-only

# 类型注解
mypy src/ --strict

# 测试覆盖率
pytest --cov=src --cov-report=term-missing
```

### 4.3 完成标志

```
阶段 1 完成标志：
✅ 全部单元测试通过（pytest -v）
✅ 单命令 `python -m src.baseline.benchmark --dataset nfcorpus` 输出可复现指标
✅ 代码通过 black + isort + mypy 检查
✅ 测试覆盖率 ≥ 70%
```

---

## 五、本阶段风险与预案

| 风险 | 概率 | 影响 | 预案 |
|------|------|------|------|
| HuggingFace 下载缓慢或超时 | 中 | 高 | 设置 `HF_HUB_ENABLE_HF_TRANSFER=1`；使用 `--resume-download` |
| FAISS GPU 版本安装失败 | 低 | 中 | 先用 `faiss-cpu` 开发，评测时切换到 Colab GPU |
| BGE 模型 768-dim 与预期不符 | 低 | 高 | 加载后检查 `model.get_sentence_embedding_dimension()` |
| PyTorch CUDA 版本与本地驱动不兼容 | 低 | 中 | 使用 `pip install torch==2.x --index-url https://download.pytorch.org/whl/cu121` 匹配驱动 |
| NFCorpus 太小，无法暴露 batch 处理 bug | 中 | 中 | 从 MS-MARCO 随机取 1000 条构造 `tests/fixtures/small_corpus.json` 补充测试 |

---

## 六、文件创建顺序（建议开发顺序）

| 顺序 | 文件 | 依赖 |
|------|------|------|
| 1 | `pyproject.toml`, `requirements.txt`, `.gitignore` | — |
| 2 | `src/__init__.py`, `src/config.py` | — |
| 3 | `src/utils/logger.py`, `src/utils/seed.py`, `src/utils/device.py` | config |
| 4 | `src/baseline/encoder.py` | utils |
| 5 | `src/vector_store/indexer.py`, `src/vector_store/retriever.py` | encoder |
| 6 | `src/evaluation/dataset.py` | — |
| 7 | `src/evaluation/metrics.py` | — |
| 8 | `src/baseline/benchmark.py` | 所有上述模块 |
| 9 | `tests/conftest.py`, `tests/test_baseline.py` | 所有上述模块 |
| 10 | 端到端验收 | benchmark |
