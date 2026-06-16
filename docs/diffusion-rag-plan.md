# Diffusion RAG — 整体规划、系统架构与开发计划

> 基于 ELF（Embedded Language Flows）扩散模型的语义检索增强方案  
> 开发语言：Python  
> 目标：验证扩散模型能否在不改变向量数据库架构的前提下，有效提升语义检索精度与鲁棒性

---

## 目录

1. [整体规划](#一整体规划)
2. [系统架构](#二系统架构)
3. [开发计划](#三开发计划)
4. [验收标准](#四验收标准)
5. [对比数据设计](#五对比数据设计)
6. [风险评估](#六风险评估)

---

## 一、整体规划

### 1.1 项目定位

本项目不是生产级 RAG 系统，而是 **"研究验证性原型"** — 核心目标是回答以下三个问题：

| # | 问题 | 检验方式 |
|---|------|---------|
| Q1 | 扩散模型增强查询向量能否比传统编码器 Recall@k 更高？ | 双链路（传统 vs ELF）在同一向量库、同一查询集上的对比 |
| Q2 | 扩散增强对拼写错误 / 缩写 / 口语化改写是否更鲁棒？ | 对标准查询施加人工噪声后，对比指标衰减幅度 |
| Q3 | CFG 引导能否提供可控的"精准 ↔ 扩展"调节能力？ | 固定查询，变化 CFG scale 观察召回范围与排序变化 |

### 1.2 范围定义

| 范围内 | 范围外 |
|--------|--------|
| 基于 ELF-B（105M）的查询端扩散增强 | 文档端重编码 / 文档端扩散增强 |
| FAISS 向量库，IVF_FLAT 索引 | Milvus / Qdrant / Weaviate 等分布式部署 |
| BEIR 公开评测集（三级策略: NFCorpus / MS-MARCO / NQ） | 超大（>100K query）评测集 |
| Recall / Precision / MRR / NDCG / 鲁棒性指标 | 端到端 RAG 生成质量（LLM 回答评估） |
| 延迟 / QPS 性能开销记录 | 生产级性能优化（量化、ONNX、TensorRT） |

### 1.3 总体时间线

```
里程碑                              预估工时    执行环境    产出物
──────────────────────────────────────────────────────────────────────────
M1: 环境与基准链路就绪               2-3天      本机        Baseline 检索链路 + 指标可跑通
M2: ELF 检索接口封装完成             1-2天      本机        encode/denoise/cfg_guide 三个接口 + 单元测试
M3: 增强链路集成 + 本地验证          1-2天      本机        双链路可切换 + NFCorpus 上走通
M4: Colab 迁移准备                   1天        本机/Colab  Colab Notebook + Drive 结构 + 小数据验证
M5: 全量评测采集                     1-2天      Colab      完整对比数据表 + 可视化图表
M6: 结论分析与文档输出               1天        本机        结论报告 + 可复现配置归档
```

### 1.4 资源需求

| 资源 | 规格 | 用途 | 备注 |
|------|------|------|------|
| 本地 GPU | NVIDIA RTX 1000 Ada (6GB) | 代码开发 + 单条调试 + 小规模验证 | 当前本机可用 GPU |
| 云端 GPU（可选） | Colab T4 (16GB) / A100 (40GB) | 大规模全量参数组扫描 + BEIR 评测 | 通过同一代码库 + Colab Notebook 一键切换 |
| Python | ≥3.10 | 主开发语言 | 本地 + Colab 版本需一致 |
| 磁盘 | ≥50GB | 数据集 + 模型权重 + 向量索引 | Colab 需挂载 Google Drive 做持久存储 |
| 依赖 | PyTorch ≥2.0, FAISS, transformers, datasets, ELF 官方库 | 见 `requirements.txt` | 同一份依赖双平台可用 |

#### 本地 GPU 实测可行性评估

| 场景 | 可行性 | 策略 |
|------|--------|------|
| 单条 query 编码 + 去噪推理（batch=1） | ✅ 完全可跑 | ELF-B (105M) FP16 仅 ~210MB 显存 |
| 文档库批量编码入库（batch=8~16） | ⚠️ 可跑但较慢 | 小数据集（<10K doc）本地完成，大规模集留给 Colab |
| 全参组合扫描（27组 × 3数据集） | ❌ 本地不现实 | 全部交给 Colab |
| 代码调试 / 单元测试 / 单点验证 | ✅ 完全适合 | 本地开发主力场景 |

### 1.5 Colab 迁移策略

#### 为什么需要 Colab

| 任务 | 本机 (RTX 1000 Ada 6GB) | Colab (T4 16GB) |
|------|------------------------|-----------------|
| NFCorpus (~3.6K docs) 调试 | ✅ 秒级 | ✅ 更快 |
| MS-MARCO (~8.8M docs) 编码+索引 | ❌ 显存不足 | ✅ 可完成 |
| 全参数组扫描 (12+1)×3 数据集 | ❌ 数小时+显存溢出 | ✅ 1-2小时 |
| 鲁棒性测试 (3 噪声类型) | ❌ 耗时过长 | ✅ 可行 |

**策略**：本地开发调链路 → Colab 跑全量实验

#### Colab 准备清单

迁移不是"复制代码到 Drive"这么简单，需要做以下准备工作：

| # | 准备项 | 说明 | 工作量 |
|---|--------|------|--------|
| 1 | 代码托管到 GitHub（私有仓库即可） | Colab 通过 `!git clone` 拉取，避免手动上传 | 一次性 |
| 2 | 编写 `scripts/colab_setup.ipynb` | 一个 Notebook 涵盖全部流程 | ~30min |
| 3 | 设置 Google Drive 持久化存储 | 数据集缓存、模型权重、实验结果三区分离，防 Colab 重启丢失 | ~15min |
| 4 | HuggingFace 数据集缓存重定向 | 设 `HF_HOME=/content/drive/MyDrive/.cache/huggingface`，避免每会话重新下载 | 1 行代码 |
| 5 | ELF 权重预下载到 Drive | 提前下载到 Drive 的 `cache/huggingface/hub/` 下，避免每会话重复下载 | ~10min |
| 6 | 验证 Colab 端 CUDA / PyTorch 版本 | Colab 预装 PyTorch 但版本可能滞后，确认与 ELF 兼容 | ~5min |
| 7 | 配置断点续跑 | Notebook Cell 检测已有输出文件，跳过已完成参数组 | 代码中实现 |
| 8 | 测试退出重进流程 | Colab 空闲超时断连后，重启 Notebook 能从断点继续 | 一次验证 |

**关键目录结构（Google Drive）**：

```
MyDrive/
└── diffusion-rag/
    ├── cache/                    # HF_DATASETS_CACHE + HF_HOME
    │   ├── huggingface/
    │   │   ├── datasets/         # BEIR / MS-MARCO 等数据集缓存
    │   │   └── hub/              # ELF / BGE 模型权重
    │   └── faiss_index/          # 预构建的 FAISS 索引（避免每次重建）
    ├── code/                     # git clone 到这里（可选，或直接 clone 到 Colab 临时盘）
    └── results/                  # 实验结果输出
        ├── msmarco/
        ├── nq/
        ├── fiqa/
        └── figures/
```

#### 代码层只需两个机制

#### 机制 A：设备自动检测（`src/utils/device.py`）

```python
import torch

def get_device():
    """自动选择最优设备——本地/Colab/无GPU均兼容"""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"[device] GPU: {gpu_name}, VRAM: {vram:.1f} GB")
    else:
        device = torch.device("cpu")
        print("[device] No GPU — CPU mode (debug only)")
    return device
```

所有模型初始化和张量操作统一通过 `get_device()` 获取设备，本地与 Colab 无需改一行代码。

#### 机制 B：Colab 一键启动入口（`scripts/colab_setup.ipynb`）

```python
# Cell 1: 克隆仓库 + 安装依赖
!git clone <repo-url>
%cd diffusion-rag
!pip install -r requirements.txt

# Cell 2: 下载 ELF 权重 + 评测数据
!python scripts/download_assets.py --all

# Cell 3: 运行全量对比实验（结果写入 Google Drive 持久化）
!python -m evaluation.runner \
    --datasets beir-msmarco beir-nq cmed-qa \
    --param_sweep \
    --output /content/drive/MyDrive/diffusion-rag-results/
```

#### 具体预留改动清单

| 文件 | 改动 | 工作量 |
|------|------|--------|
| `src/utils/device.py` | 新增 ~15 行，自动设备检测 | <10min |
| `src/utils/download.py` | 新增 ~30 行，支持从 HuggingFace / Google Drive / 本地缓存三路加载 | 30min |
| `src/config.py` | 增加 `colab_mode: bool` 标志位 | 1min |
| `scripts/colab_setup.ipynb` | 新增一个 Notebook，就是完整的 Colab 入口 | 15min |
| `requirements.txt` | 保持一致 — Colab 预装 PyTorch + CUDA，本地需手动装 | — |

---

## 二、系统架构

### 2.1 整体架构概览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        评测框架层                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────────┐  │
│  │ BEIR     │  │ MS-MARCO │  │ NFCorpus │  │ 带噪查询生成器     │  │
│  │ 数据集   │  │ 数据集   │  │ 调试集   │  │ (typo/缩写/改写)   │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └─────────┬──────────┘  │
│       └──────────────┴────────────┴───────────────────┘             │
│                               │ query texts                         │
│                               ▼                                     │
│                    ┌──────────────────┐                             │
│                    │  评测 Orchestrator│                             │
│                    │  (加载数据集 →   │                             │
│                    │   双链路对比 →   │                             │
│                    │   收集指标)      │                             │
│                    └────────┬─────────┘                             │
├─────────────────────────────┼───────────────────────────────────────┤
│                 路由选择     │                                        │
│                 ▼            ▼                                       │
│       ┌────────────┐  ┌──────────────┐                              │
│       │ 方案A:     │  │ 方案B:       │                              │
│       │ 传统基准   │  │ ELF 扩散增强  │                              │
│       └──────┬─────┘  └──────┬───────┘                              │
│              │               │                                       │
│              ▼               ▼                                       │
│       ┌────────────┐  ┌──────────────┐                              │
│       │ BGE-Mini   │  │ ELF Encode   │                              │
│       │ 编码器     │  │ → Denoise    │                              │
│       │            │  │ → CFG Guide  │                              │
│       └──────┬─────┘  └──────┬───────┘                              │
│              │               │                                       │
│              └───────┬───────┘                                       │
│                      ▼  768-dim 向量                                │
│             ┌────────────────────┐                                  │
│             │  FAISS 向量库       │                                  │
│             │  (IVF_FLAT)        │                                  │
│             │  共享索引, 共享文档  │                                  │
│             └────────┬───────────┘                                  │
│                      ▼  top-k results                                │
│             ┌────────────────────┐                                  │
│             │  指标计算器        │                                  │
│             │  (Recall/Precision │                                  │
│             │   MRR/NDCG/Hit)   │                                  │
│             └────────┬───────────┘                                  │
│                      ▼                                               │
│             ┌────────────────────┐                                  │
│             │  对比报告生成器    │                                  │
│             │  (CSV / 图表 /     │                                  │
│             │   结论摘要)        │                                  │
│             └────────────────────┘                                  │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 方案 A：传统基准链路

```
查询文本 ──→ BGE-Mini 编码器 ──→ 768-dim 向量 ──→ FAISS 检索 ──→ top-k 结果
```

- **编码器**: `BAAI/bge-small-zh-v1.5` 或 `BAAI/bge-base-en-v1.5`（取决于数据集语种）
- **向量维度**: 768（与 ELF 输出对齐）
- **不应用任何特殊处理**，保证是干净的基准线

### 2.3 方案 B：ELF 扩散增强链路

```
查询文本 ──→ ELF 编码器 ──→ 加噪 ──→ [去噪迭代 1~4步] ──→ [CFG 引导] ──→ 增强向量 ──→ FAISS 检索 ──→ top-k 结果
                                 ↑________________________|
                                  循环 denoise() 用 ODE 推进
```

**核心流程详解：**

```
Step 1: encode(text)
        └── 调用 model.encode(text) 得到原始向量 z_0 (768-dim, L2 归一化)

Step 2: add_noise(z_0, t=0.4)
        └── z_t = z_0 + σ(t)·ε, ε~N(0,1)
        └── t 控制噪声强度 (0=无噪声, 1=完全噪声)

Step 3: denoise(z_t, steps=2)
        └── 循环 steps 次, 每次调用 model.forward 预测速度场 v
        └── ODE 推进: z ← z + v·Δt
        └── 输出修复后的向量 z'

Step 4: cfg_guide(z', cond=z_0, scale=2.0)  [可选]
        └── v_cfg = scale·v_cond + (1-scale)·v_uncond
        └── scale>1: 更忠实原文（精准模式）
        └── scale<1: 更扩散探索（扩展召回模式）

Step 5: L2 normalize(z') → 送入 FAISS 检索
```

### 2.4 模块设计

```
diffusion-rag/
├── src/
│   ├── __init__.py
│   ├── config.py              # 全局配置 (模型路径 / 数据集 / 参数网格)
│   ├── baseline/
│   │   ├── __init__.py
│   │   └── encoder.py         # BGE-Mini 编码器封装
│   ├── elf/
│   │   ├── __init__.py
│   │   ├── encoder.py         # ELF encode() 封装
│   │   ├── diffusion.py       # add_noise / denoise / cfg_guide
│   │   └── pipeline.py        # 增强检索完整链路 (encode → denoise → cfg → norm)
│   ├── vector_store/
│   │   ├── __init__.py
│   │   ├── indexer.py         # FAISS 索引构建
│   │   └── retriever.py       # 检索接口 (共享, 双链路通用)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── metrics.py         # Recall / Precision / MRR / NDCG / HitRate
│   │   ├── dataset.py         # 数据集加载 + 带噪查询生成
│   │   ├── orchestrator.py    # 评测编排 (双链路循环)
│   │   └── reporter.py        # 报表生成 (CSV / Vega-Lite 图表)
│   └── utils/
│       ├── __init__.py
│       └── logger.py          # 实验日志 + 参数归档
├── experiments/
│   ├── configs/               # 实验配置 YAML 文件
│   ├── outputs/               # 实验结果 (CSV / JSON / PNG)
│   └── notebooks/             # Jupyter notebooks 用于深度分析
├── data/                      # 数据集缓存 (HuggingFace 自动下载)
├── models/                    # 预训练权重 (gitignored, 自动下载)
├── tests/                     # 单元测试
│   ├── test_elf.py
│   ├── test_retriever.py
│   └── test_metrics.py
├── docs/                      # 文档
├── requirements.txt
└── README.md
```

### 2.5 ELF 接口详细设计

三个核心接口的输入输出与实现策略：

#### `encode(text: str) -> np.ndarray`

| 项目 | 描述 |
|------|------|
| 输入 | 单条文本字符串 |
| 输出 | 768-dim float32 向量（L2 归一化） |
| 实现 | 调用 ELF 官方 `model.encode([text])[0]`，保证与 BGE 输出同维度同格式 |
| 单元测试 | 输入定长文本，检查输出 shape == (768,), dtype == float32, L2 norm ≈ 1.0 |

#### `add_noise(z: np.ndarray, t: float) -> np.ndarray`

| 项目 | 描述 |
|------|------|
| 输入 | 归一化向量 z, 噪声水平 t ∈ [0,1] |
| 输出 | 加噪后向量 z_t |
| 噪声策略 | 方差保持型 VP-SDE 噪声（与 ELF 训练一致）: z_t = √α̅(t)·z₀ + √(1-α̅(t))·ε |
| 单元测试 | t=0 输出 ≈ 输入; t=1 输出 ≈ 纯噪声; 多次采样 t=0.5 方差可控 |

#### `denoise(z: np.ndarray, t_start: float=0.4, steps: int=2) -> np.ndarray`

| 项目 | 描述 |
|------|------|
| 输入 | 加噪向量, 起始噪声水平, ODE 步数 |
| 输出 | 去噪修复后向量 |
| 实现 | 使用 ELF model.forward() 预测速度场 v；Euler ODE 推进 dt = t_start/steps |
| 关键 | 仅用 NFE 1~4 步 (非完整生成, 不需收敛到 t=0) |
| 单元测试 | 输入加噪向量, steps=0 输出应与输入相同; steps>0 输出与 clean 向量余弦相似度应高于输入 |

#### `cfg_guide(z: np.ndarray, cond: np.ndarray, scale: float=2.0) -> np.ndarray`

| 项目 | 描述 |
|------|------|
| 输入 | 当前向量 z, 条件向量 (原始编码), CFG scale |
| 输出 | CFG 引导后的向量 |
| 实现 | 两次 forward: v_cond = model(z, cond); v_uncond = model(z, null_cond); v = scale·v_cond + (1-scale)·v_uncond |
| 单元测试 | scale=1.0 输出 = 无条件输出; scale=0.0 输出完全无引导; scale>1 输出与 cond 相似度更高 |

---

## 三、开发计划

### 3.1 阶段划分与任务分解

#### 阶段 1：环境与基准链路搭建（本机，预估 2-3 天）

| 任务 ID | 任务 | 执行环境 | 产出物 | 验收标准 |
|---------|------|--------|---------|--------|
| 1.1 | 克隆 ELF 官方仓库，安装依赖，下载 ELF-B 权重 | 本机 | `models/elf-b/` 包含 `.pt` 权重 | `model.forward()` 可正常调用 |
| 1.2 | 搭建 BGE-Mini 基线编码器 | 本机 | `src/baseline/encoder.py` | 输入文本 → 输出 768-dim L2 归一化向量 |
| 1.3 | 搭建 FAISS 索引构建 + 检索接口 | 本机 | `src/vector_store/` | 构建 IVF_FLAT 索引，search() 返回 (ids, distances) |
| 1.4 | 下载并结构化 BEIR 数据集 | 本机 | `src/evaluation/dataset.py` | `load_dataset("nq")` / `load_dataset("nfcorpus")` 返回 (queries, corpus, qrels) 三元组 |
| 1.5 | 实现核心指标计算 | 本机 | `src/evaluation/metrics.py` | 给定 (qrels, results) 输出 Recall/MRR/NDCG/HitRate |
| 1.6 | 端到端走通基线评测 | 本机 | 控制台输出指标 | 运行 `python -m src.baseline benchmark` 产出 Recall@5/10/20 |

**阶段 1 完成标志**：Baseline 链路可在单条命令下输出可复现的评测指标。

#### 阶段 2：ELF 检索接口封装（本机，预估 1-2 天）

> ⚠️ 代码规范详见 `docs/CODING_STYLE.md` — 所有实现必须遵循。

| 任务 ID | 任务 | 执行环境 | 产出物 | 验收标准 |
|---------|------|--------|---------|--------|
| 2.1 | 实现 `ELFEncoder.encode()` | 本机 | `src/elf/encoder.py` | 输出 shape (768,), dtype float32, L2 norm ≈ 1.0 |
| 2.2 | 实现 `add_noise()` / `denoise()` | 本机 | `src/elf/diffusion.py` | 单元测试覆盖噪声注入与 ODE 去噪 |
| 2.3 | 实现 `cfg_guide()` | 本机 | `src/elf/diffusion.py` | 单元测试覆盖 scale 边界行为 |
| 2.4 | 实现 `ELFPipeline` 完整链路 | 本机 | `src/elf/pipeline.py` | 封装 encode → add_noise → denoise → cfg_guide → norm 完整流程 |
| 2.5 | 编写单元测试 | 本机 | `tests/test_elf.py` | ≥ 8 个测试用例，覆盖各接口输入输出边界 |

**阶段 2 完成标志**：`ELFPipeline(query).enhanced_vector()` 可稳定输出向量，全部单元测试通过。

#### 阶段 3：增强链路集成 + 本地验证（本机，预估 1-2 天）

| 任务 ID | 任务 | 执行环境 | 产出物 | 验收标准 |
|---------|------|--------|---------|--------|
| 3.1 | 将 ELF Pipeline 接入检索框架 | 本机 | 双链路共享 indexer/retriever | 仅替换 `encode_query()` 即可切换方案 |
| 3.2 | 配置参数网格 | 本机 | `experiments/configs/param_grid.yaml` | 12 组参数组合（见 3.3 节） |
| 3.3 | 实现评测编排器 | 本机 | `src/evaluation/orchestrator.py` | 遍历参数组合，自动跑双链路并记录指标 |
| 3.4 | 实现报表生成器 | 本机 | `src/evaluation/reporter.py` | 输出 CSV + Vega-Lite JSON 图表 |
| 3.5 | 本地验证跑通（NFCorpus） | 本机 | 对比 CSV 表 | **NFCorpus** 上所有 12 组 + baseline 均可产出非空指标，耗时 < 5分钟 |

**阶段 3 完成标志**：`python -m src.main --config experiments/configs/param_grid.yaml --dataset nfcorpus` 可在本机 GPU 上完成全部 13 组实验并输出指标。

> NFCorpus 仅 ~3,600 文档，本机 6GB 显存秒级跑完。此阶段只验证链路正确性，不验证结论。全量跑测在 Phase 5（Colab）进行。

#### 阶段 4：Colab 迁移与适配（本机 + Colab 混合，预估 1 天）

**目的**：将本地开发的代码迁移到 Colab 环境，确保全量评测可稳定执行。

| 任务 ID | 任务 | 执行环境 | 产出物 | 验收标准 |
|---------|------|---------|--------|---------|
| 4.1 | 编写 `colab_setup.ipynb`，覆盖克隆/安装/下载/运行全流程 | 本机 | `scripts/colab_setup.ipynb` | Notebook 可在 Colab 上无报错逐 Cell 执行 |
| 4.2 | 设置 Google Drive 持久化目录结构 | Colab | Drive 上 `diffusion-rag/` 目录 | 数据集缓存、模型权重、实验结果三区分离 |
| 4.3 | 在 Colab 上用 NFCorpus 做端到端验证 | Colab | 控制台输出指标 | Colab 上 NFCorpus 结果与本机一致（固定种子复现） |
| 4.4 | 测试 MS-MARCO 索引构建可行性 | Colab | 索引构建日志 + 显存监控 | 8.8M 文档编码 + FAISS 索引可在 Colab T4 内存预算内完成 |
| 4.5 | 设置断点续跑机制 | Colab | Notebook 增加 checkpoint 逻辑 | 会话中断后重启，可从已完成的参数组继续，不重复计算 |
| 4.6 | 将结果写入 Drive 并确认可从本机访问 | 双端 | Drive 上实验 `.csv` 文件 | `experiments/outputs/` 同步到 Google Drive，本机可 mount 或下载查看 |

**阶段 4 完成标志**：Colab 上可一键运行 NFCorpus 全流程并得到与本机一致的结果；MS-MARCO 索引构建成功。

#### 阶段 5：全量评测执行（Colab，预估 1-2 天）

> ⚠️ **全部在 Colab 上执行**。本机 6GB 显存不足以完成 MS-MARCO（8.8M 文档）的全参数组扫描。

| 任务 ID | 任务 | 执行环境 | 产出物 | 验收标准 |
|---------|------|---------|--------|---------|
| 5.1 | MS-MARCO 全量跑测（12组 + baseline） | Colab | `experiments/outputs/msmarco/` | 全部 13 组参数均产出非空指标列 |
| 5.2 | NQ 全量跑测 | Colab | `experiments/outputs/nq/` | 同上 |
| 5.3 | FiQA 领域迁移验证 | Colab | `experiments/outputs/fiqa.csv` | 同上 |
| 5.4 | 鲁棒性专项评测（3 噪声类型 × 3 数据集） | Colab | `experiments/outputs/robustness_*.csv` | 每个查询 3 个变体，对比指标衰减 |
| 5.5 | 性能开销记录 | Colab | `experiments/outputs/latency.csv` | 编码耗时、检索耗时、显存占用（T4 GPU 数据） |
| 5.6 | 生成对比图表 | Colab | `experiments/outputs/figures/*.png` | Recall@10 柱状图、鲁棒性衰减折线图、CFG 调控曲线 |
| 5.7 | 结论分析初稿 | 本机 | `docs/conclusion_draft.md` | 回答 Q1/Q2/Q3，引用 Colab 产出的图表路径 |

**阶段 5 完成标志**：Drive 上产出所有数据集的 `.csv` 文件和 `.png` 图表；`conclusion_draft.md` 已填充真实数据（暂时留空的指标位标记为 TBD）。

#### 阶段 6：结论分析与文档输出（本机，预估 1 天）

| 任务 ID | 任务 | 产出物 | 验收标准 |
|---------|------|--------|---------|
| 6.1 | 整理完整对比数据表 | `docs/conclusion.md` | Recall@10 总表、鲁棒性对比表、CFG 曲线、性能开销表均填入真实数据 |
| 6.2 | 撰写分析结论 | `docs/conclusion.md` | 回答 Q1/Q2/Q3，标注适用边界与例外场景 |
| 6.3 | 归档可复现配置 | `experiments/configs/full_benchmark.yaml` | 锁定所有参数版本、随机种子、评测命令 |
| 6.4 | 最终检视 | 全仓库 | README 包含复现步骤；`--help` 文档完备 |

**阶段 6 完成标志**：`docs/conclusion.md` 输出最终结论，仓库可复现全流程。

### 3.2 参数网格设计

ELF 增强链路的全部参数组合（12 组）：

| ID | 去噪步数 (steps) | 加噪起始 t | CFG scale | 说明 |
|----|------------------|-----------|-----------|------|
| ELF-01 | 1 | 0.3 | 1.0 (无 CFG) | 最轻量增强 |
| ELF-02 | 1 | 0.5 | 1.0 | 中等噪声 |
| ELF-03 | 1 | 0.3 | 2.0 | 轻量 + 精准 |
| ELF-04 | 1 | 0.5 | 2.0 | 中等 + 精准 |
| ELF-05 | 2 | 0.3 | 1.0 | 更充分去噪 |
| ELF-06 | 2 | 0.5 | 1.0 | 修正中等噪声 |
| ELF-07 | 2 | 0.3 | 2.0 | 推荐候选 |
| ELF-08 | 2 | 0.5 | 3.0 | 强精准 |
| ELF-09 | 4 | 0.4 | 1.0 | 充分去噪 |
| ELF-10 | 4 | 0.4 | 2.0 | 充分去噪 + 精准 |
| ELF-11 | 4 | 0.6 | 1.0 | 高噪声修复 |
| ELF-12 | 4 | 0.6 | 3.0 | 高噪声 + 严格引导 |

> **对照组**: Baseline (BGE-Mini, 无扩散增强)

### 3.3 数据集规划

**设计原则**：全部使用公开数据集，不自建数据。

- **可复现性** — 公开数据集他人可完全复现对比结果
- **有参考基线** — BEIR Leaderboard 上有大量公开结果可直接对照
- **零标注成本** — query-doc 相关性对已预标注
- **加载方式** — 通过 HuggingFace `datasets` 库一行下载，无需手动处理

```python
from datasets import load_dataset

# 开发调试
data = load_dataset("BeIR/nfcorpus", split="train")   # 仅 3K docs

# 全量评测
data = load_dataset("BeIR/msmarco", split="train")     # MS-MARCO
data = load_dataset("BeIR/nq", split="train")           # Natural Questions
```

#### 三级数据集策略

| 层级 | 数据集 | 规模 (docs) | queries | 用途 | 语种 |
|------|--------|-----------|---------|------|------|
| **L1 — 调试集** | BEIR NFCorpus | ~3,600 | ~323 test | **开发阶段**：链路调试、单元测试、ELF 接口验证，秒级跑通 | 英文 |
| **L2 — 主评测集** | MS-MARCO Passage | ~8.8M | ~6,980 dev | **全量对比**：标准检索评测，与公开 benchmark 直接对标 | 英文 |
| **L3 — 鲁棒性+多样性** | NQ (Natural Questions) | ~2.6M | ~3,452 test | **补充评测**：真实用户问题（含口语化表达），验证泛化能力 | 英文 |
| **L3 可选扩展** | BEIR FiQA | ~57K | ~648 test | 金融领域专项，检验领域迁移能力 | 英文 |

**开发流程**：
> L1 (NFCorpus) 调通链路 → L2 (MS-MARCO) 产出主对比数据 → L3 (NQ / FiQA) 验证泛化性

**不采用自建数据的原因**：① 自建集的标注质量不可靠；② 他人无法复现，削弱结论说服力；③ 公开集已有足够的领域多样性（通用 / 知识库 / 金融）。

### 3.4 鲁棒性测试设计

对每个标准查询，自动生成 3 个变体：

| 变体类型 | 操作 | 示例 |
|----------|------|------|
| T1: 错别字 | 每词 20% 概率随机替换一个字符（键盘邻近键） | "retrieval" → "retrieval" |
| T2: 缩写 | 常用缩写替换（API, DB, ML, NLP 等） | "natural language processing" → "NLP" |
| T3: 口语化改写 | 句式简化/口语化（规则+模板） | "what is the capital of France" → "France capital where?" |

**鲁棒性指标**：记录每个变体相对于原查询的指标衰减值，β = (Metric_clean - Metric_noisy) / Metric_clean × 100%。β 越小表示鲁棒性越好。

---

## 四、验收标准

### 4.1 功能验收

| 标准 | 检验方式 |
|------|---------|
| 双链路可切换 | 同一数据集、同一评测脚本，仅 `--mode baseline/elf` 切换 |
| ELF 接口输出与 BGE 同维度同格式 | `assert elf_vec.shape == bge_vec.shape == (768,)` |
| 全部 12 组参数可独立运行 | 每组产出一行对比指标 |
| 评测结果完全可复现 | 设置 `random_seed=42`，两次运行指标差异 <0.1% |

### 4.2 实验验收

| # | 验收项 | 最低标准 | 目标标准 |
|---|--------|---------|---------|
| A1 | Baseline Recall@10 | 在所有数据集上产出 | 与公开 benchmark 差异 <2% |
| A2 | ELF Recall@10 提升 | ≥ 1 组参数在 ≥ 1 个数据集上优于 baseline | ≥ 3 组参数在 ≥ 2 个数据集上 Recall@10 +3%↑ |
| A3 | 鲁棒性衰减 | ELF 在 ≥ 1 个噪声类型上衰减 ≤ baseline | ELF 在所有噪声类型上衰减均值低 20% |
| A4 | CFG 可调性 | scale=1 vs scale=3 的 Recall 趋势方向一致 | 存在单调区间: scale ↑ → precision ↑, recall ↓ |
| A5 | 性能开销（Colab T4 GPU 测量） | 单查询总耗时 ≤ 100ms | 单查询总耗时 ≤ 50ms |
| A6 | NFCorpus 本机 vs Colab 一致性 | 固定种子后两平台指标差异 < 0.1% | 差异 < 0.05% |
| A7 | Colab 断点续跑 | 手动中断后重启，已完成的参数组不重复计算 | 输出时间戳证明续跑只跑了剩余组 |

### 4.3 代码验收

| 标准 | 检验方式 | 对应 CODING_STYLE.md 章节 |
|------|---------|--------------------------|
| 单元测试覆盖率 > 70% | `pytest --cov=src --cov-report=term-missing` | §5 测试规范 |
| 代码通过 black + isort 格式化 | `black . --line-length 100 && isort .` | §1.1 格式化 |
| 类型注解完整（mypy --strict） | `mypy src/ --strict` (排除第三方) | §1.2 类型注解 |
| Google 风格 docstring | 抽查 3 个公共函数 | §2 文档字符串 |
| 无裸露 `print()` | `grep -r "print(" src/` 仅允许 utils/logger.py | §4 日志规范 |
| 所有实验配置 YAML 归档 | `experiments/configs/` 目录 Git 跟踪 | §9 配置管理 |
| 单命令复现全量实验 | `python -m src.main --config experiments/configs/full_benchmark.yaml` | — |

---

## 五、对比数据设计

### 5.1 数据收集策略

所有对比数据按以下结构统一收集：

```
experiments/outputs/
├── {dataset_name}/
│   ├── {timestamp}/
│   │   ├── params.yaml         # 本次实验参数快照
│   │   ├── baseline.csv        # Baseline 结果 (每条 query 一行)
│   │   │   └─ 列: query_id, recall@5, recall@10, recall@20, mrr, ndcg@10
│   │   ├── elf_results.csv     # ELF 结果
│   │   │   └─ 列: query_id, config_id, steps, t, cfg_scale,
│   │   │       recall@5, recall@10, recall@20, mrr, ndcg@10, latency_ms
│   │   ├── summary.csv         # 聚合汇总 (每组参数一行)
│   │   │   └─ 列: config_id, steps, t, cfg_scale,
│   │   │       avg_recall@10, avg_mrr, avg_ndcg@10, avg_latency_ms,
│   │   │       vs_baseline_recall@10_delta(%)
│   │   ├── robustness.csv      # 鲁棒性数据
│   │   │   └─ 列: query_id, config_id, noise_type,
│   │   │       recall@10_clean, recall@10_noisy, decay(%)
│   │   └── figures/            # 自动生成的图表
│   │       ├── recall_comparison.png
│   │       ├── robustness_decay.png
│   │       └── cfg_scale_curve.png
│   └── ...
```

### 5.2 对比报告模板

最终结论报告 `docs/conclusion.md` 应包含：

#### 5.2.1 总表：所有方案在所有数据集上的 Recall@10

| 方案 | NQ | MS-MARCO | FiQA | 均值 |
|------|----|----------|------|------|
| Baseline (BGE-Mini) | 0.xxx | 0.xxx | 0.xxx | 0.xxx |
| ELF-01 | 0.xxx | 0.xxx | 0.xxx | 0.xxx |
| ELF-02 | ... | ... | ... | ... |
| ... | ... | ... | ... | ... |
| **ELF-07 (最佳)** | 0.xxx | 0.xxx | 0.xxx | 0.xxx |
| **最佳提升 (%)** | +x.x% | +x.x% | +x.x% | +x.x% |

#### 5.2.2 鲁棒性对比表

| 方案 | 清洁查询 Recall@10 | 错字衰减(%) | 缩写衰减(%) | 口语衰减(%) | 平均衰减(%) |
|------|-------------------|------------|------------|------------|------------|
| Baseline | 0.xxx | -x.x% | -x.x% | -x.x% | -x.x% |
| ELF-最佳 | 0.xxx | -x.x% | -x.x% | -x.x% | -x.x% |

#### 5.2.3 CFG 调控曲线

绘制 `CFG scale (0.5 ~ 4.0)` vs `Recall@10 / Precision@10` 曲线，标注最佳操作点。

#### 5.2.4 性能开销

| 方案 | 编码耗时 (ms) | 去噪耗时 (ms) | 检索耗时 (ms) | 总耗时 (ms) | 显存 (MB) |
|------|--------------|--------------|--------------|------------|----------|
| Baseline | x.x | 0 | 1.x | x.x | xxx |
| ELF-01 | x.x | 1.x | 1.x | x.x | xxx |
| ELF-09 (4步) | x.x | 8.x | 1.x | x.x | xxx |

### 5.3 预期结论模板

```
## 结论

### 1. 扩散增强能否提升精度？
- ✅ / ❌ / 部分成立
- 最佳参数配置: steps={x}, t={x}, cfg_scale={x}
- 在 {dataset} 上 Recall@10 提升 {x.x}%
- 适用场景: {描述}

### 2. 鲁棒性是否更优？
- ✅ / ❌ / 部分成立
- 对错别字衰减降低 {x}%
- 对缩写/口语衰减降低 {x}%
- 适用场景: {描述}

### 3. CFG 是否具备可控性？
- ✅ / ❌ / 部分成立
- scale ↑ → recall {上升/下降}, precision {上升/下降}
- 最佳操作点: scale = {x.x}

### 4. 性能边界
- 单查询总耗时: {x}ms (T4), 相比 baseline 增加 {x}ms
- 显存开销: {x}MB
- 建议的使用策略: {描述}
```

---

## 六、风险评估

| 风险 | 概率 | 影响 | 缓解方案 |
|------|------|------|---------|
| ELF-B 模型权重下载失败 | 中 | 高 | 本地缓存权重文件；实现自动重试（最多3次）+ 超时60s；备选使用 mini 版本 |
| ELF 官方代码与 PyTorch 版本不兼容 | 低 | 中 | Docker 镜像锁定版本；提前在 CI 环境验证 |
| 去噪步数 > 2 时耗时不可接受 | 中 | 中 | 提前测量；4 步以上标记为"探索性实验"不作为强制验收项 |
| ELF 增强在部分数据集上无正向收益 | 中 | 低 | 记录为"非适用场景"，分析原因（数据集特性 vs 模型能力） |
| GPU 显存不足跑批量推理 | 低 | 中 | 实现自动 batch size 调优；fallback 到单条推理 |
| Colab 会话超时中断（空闲90分钟/总12小时） | 高 | 中 | 实现 checkpoint 断点续跑；每完成一个数据集保存结果到 Drive |
| Colab 预装 PyTorch 版本与 ELF 不兼容 | 中 | 高 | `colab_setup.ipynb` 首 Cell 显式 `!pip install torch==2.x` 锁定版本 |
| 评测结果浮动（随机种子敏感） | 中 | 中 | 固定 seed + 3 次重复取均值；记录标准差 |

---

## 附录

### A. 依赖清单

```txt
# requirements.txt
torch>=2.0.0
torchvision
faiss-cpu>=1.7.4     # CPU 版本用于开发调试
faiss-gpu>=1.7.4     # GPU 版本用于评测
transformers>=4.30.0
datasets>=2.14.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.2.0
pandas>=2.0.0
pyyaml>=6.0
matplotlib>=3.7.0
vega-altair>=5.1.0    # 交互式图表 (可选)
pytest>=7.4.0
pytest-cov>=4.1.0
mypy>=1.5.0
tqdm>=4.65.0
# ELF 官方依赖（见其仓库 requirements.txt）
```

### B. 快速启动命令

```bash
# ── 本机开发调试（Phase 1-3） ──

# 1. 安装
pip install -r requirements.txt

# 2. 下载数据集 & 模型权重
python -m src.utils.download

# 3. 基线评测（NFCorpus 小数据集调试链路）
python -m src.baseline.benchmark --dataset nfcorpus

# 4. ELF 增强评测（NFCorpus 验证）
python -m src.main --config experiments/configs/param_grid.yaml --dataset nfcorpus

# 5. 查看本机结果
ls experiments/outputs/

# ── Colab 全量评测（Phase 4-5） ──
# 打开 scripts/colab_setup.ipynb → 运行时 → 连接 Google Drive → 逐 Cell 执行
```

### C. 参考资源

- ELF 官方仓库: https://github.com/kaist-lklab/ELF
- BEIR 基准: https://github.com/beir-cellar/beir
- FAISS 文档: https://github.com/facebookresearch/faiss
- BGE 模型: https://huggingface.co/BAAI/bge-base-en-v1.5
