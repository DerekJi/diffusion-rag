# Diffusion RAG

> 基于 ELF（Embedded Language Flows）扩散模型的语义检索增强实验原型

---

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

# 快速测试（~1分钟，10条 query）
python -m src.baseline.benchmark --dataset nfcorpus --sample 10

# 或使用 Makefile
make benchmark       # 全量 NFCorpus
make benchmark-quick # 10条 query 快速验证
```

结果输出到 `experiments/outputs/<dataset>/baseline.csv`。

### 运行测试

```bash
# 全部测试（不含需要下载模型的 slow 测试）
pytest tests/ -v -m "not slow" --cov=src

# 含真实模型测试（需下载 BGE 权重）
pytest tests/ -v --cov=src

# 或使用 Makefile
make test        # 全部测试
make test-fast   # 快速测试（跳过模型下载）
make test-cov    # 测试 + 覆盖率报告
```

### 代码检查

```bash
make check      # black + isort + mypy 一键检查
make format     # 自动格式化
```

---

## GitHub Actions 自动化开发

本项目配置了完整的 CI/CD 工作流，**在 GitHub 云端完成开发、审查、修复**，绕过本地网络限制。

| 工作流 | 触发条件 | 功能 |
|--------|---------|------|
| `reasonix_develop.yml` | Issue 打上 `ai-dev` 标签 | 自动开发代码/文档 → 提 PR |
| `reasonix-agent.yml` | PR 创建/更新 | 自动审查 + 修复代码 → Push |
| `reasonix_pr_feedback.yml` | PR 评论区写 `/fix ...` | 根据反馈修改代码 → Push |

**详细文档**：[GitHub Actions 自动化开发工作流](docs/github-actions-dev.md)

### 快速使用

```bash
# 1. 创建 Issue（详见 docs/github-actions-dev.md）
gh issue create \
  --title "你的任务" \
  --body "type: code\n\n任务描述..."

# 2. 打标签触发
gh issue edit <number> --add-label ai-dev

# 3. 等待 AI 完成 → 审查 PR
# 4. 在 PR 评论 /fix ... 提出修改意见
```

---

## 项目结构

```
diffusion-rag/
├── .github/workflows/
│   ├── reasonix_develop.yml       # Issue 驱动自动开发
│   ├── reasonix-agent.yml         # PR 审查 + 自动修复
│   └── reasonix_pr_feedback.yml   # /fix 在线反馈闭环
├── .reasonix/commands/            # AI 开发规范
│   ├── develop.md                 # 开发流程
│   ├── code-review.md             # 审查标准
│   ├── fix-bug.md                 # 修复流程
│   └── feature.md                 # 功能实现协调
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
│   ├── conftest.py                # pytest fixtures
│   ├── test_baseline.py           # 编码器 + FAISS + 指标测试
│   └── test_utils.py              # 配置 + 设备测试
├── docs/
│   ├── diffusion-rag-plan.md      # 整体规划与架构
│   ├── phase1-detailed-plan.md    # Phase 1 详细计划
│   ├── CODING_STYLE.md            # 代码规范
│   └── github-actions-dev.md      # CI/CD 工作流文档
├── data/                          # 数据集缓存（gitignored）
├── models/                        # 预训练权重（gitignored）
├── experiments/outputs/           # 实验结果（gitignored）
├── .temp/                         # 临时文件（gitignored）
├── Makefile                       # 常用命令入口
├── requirements.txt
├── pyproject.toml
└── README.md
```

## 开发阶段

| 阶段 | 状态 | 说明 |
|------|------|------|
| Phase 1 | ✅ 完成 | 传统检索基线（BGE + FAISS + BEIR 评测） |
| Phase 2 | 🚧 进行中 | ELF 检索接口封装（3 个 Issue 已创建） |
| Phase 3 | 🔜 待开始 | 增强链路集成 + 本地验证 |
| Phase 4 | 🔜 待开始 | Colab 迁移 |
| Phase 5 | 🔜 待开始 | 全量评测执行 |
| Phase 6 | 🔜 待开始 | 结论分析与文档输出 |

> 整个项目的 Issue 已全部创建（共 10 个），打上 `ai-dev` 标签即可触发自动开发。
> 部分 Issue 标注了 `needs-manual` 标签，表示需要人工在 Colab 上执行。

## 文档

- [GitHub Actions 自动化开发工作流](docs/github-actions-dev.md)
- [整体规划](docs/diffusion-rag-plan.md)
- [Phase 1 详细计划](docs/phase1-detailed-plan.md)
- [代码规范](docs/CODING_STYLE.md)
