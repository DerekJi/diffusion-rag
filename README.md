# Diffusion RAG

> 基于 ELF（Embedded Language Flows）扩散模型的语义检索增强实验原型  
> Phase 1：传统检索基线已就绪

## 快速开始

### 环境要求

- Python ≥ 3.10
- CUDA-capable GPU（可选，CPU 模式仅用于调试）
- 磁盘 ≥ 10GB（数据集 + 模型权重缓存）

### 安装

```bash
git clone <repo-url>
cd diffusion-rag
pip install -r requirements.txt
```

### 运行基线评测

```bash
# NFCorpus（小型，本机可跑）
python -m src.baseline.benchmark --dataset nfcorpus

# 指定数据集和 k 值
python -m src.baseline.benchmark --dataset nfcorpus --k 5 10 20

# 查看帮助
python -m src.baseline.benchmark --help
```

结果输出到 `experiments/outputs/<dataset>/baseline.csv`。

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

## 项目结构

```
diffusion-rag/
├── src/
│   ├── config.py                  # 全局配置
│   ├── baseline/
│   │   ├── encoder.py             # BGE-Mini 基线编码器
│   │   └── benchmark.py           # 基线评测 CLI 入口
│   ├── vector_store/
│   │   ├── indexer.py             # FAISS 索引构建
│   │   └── retriever.py           # 检索接口
│   ├── evaluation/
│   │   ├── dataset.py             # BEIR 数据集加载
│   │   └── metrics.py             # Recall/MRR/NDCG/HitRate
│   └── utils/
│       ├── logger.py              # 日志工厂
│       ├── seed.py                # 随机种子管理
│       └── device.py              # 设备自动检测
├── tests/
│   ├── conftest.py
│   ├── test_baseline.py           # 编码器 + FAISS + 指标测试
│   └── test_utils.py              # 配置 + 设备测试
├── data/                          # 数据集缓存（gitignored）
├── models/                        # 预训练权重（gitignored）
├── experiments/outputs/           # 实验结果（gitignored）
├── docs/                          # 文档
├── requirements.txt
└── pyproject.toml
```

## 开发阶段

| 阶段 | 状态 | 说明 |
|------|------|------|
| Phase 1 | ✅ 完成 | 传统检索基线（BGE + FAISS + BEIR 评测） |
| Phase 2 | 🔜 待开始 | ELF 检索接口封装 |
| Phase 3 | 🔜 待开始 | 增强链路集成 + 本地验证 |
| Phase 4 | 🔜 待开始 | Colab 迁移 |
| Phase 5 | 🔜 待开始 | 全量评测执行 |
| Phase 6 | 🔜 待开始 | 结论分析与文档输出 |

## 文档

- [整体规划](docs/diffusion-rag-plan.md)
- [Phase 1 详细计划](docs/phase1-detailed-plan.md)
- [代码规范](docs/CODING_STYLE.md)
