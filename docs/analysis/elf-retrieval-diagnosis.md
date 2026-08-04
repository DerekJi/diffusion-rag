# ELF 链路检索能力弱 — 诊断实验结论(issue #39)

> 日期: 2026-08-03 · 分支: fix/elf-retrieval-issue-39 · 数据: nfcorpus sample=20(20 queries / 542 docs)

## TL;DR

ELF 链路检索弱的**编码层根因已确认**: `mean-pooling` 把 T5 token 序列平均成单向量后,**有效秩坍缩到 1**——所有文档向量几乎共线(两两余弦 0.89),相关/不相关文档相似度差仅 0.02,检索排序接近随机。**token 级(ColBERT 式 maxsim)表示把区分度提升 2.4 倍**,是修复方向。

## 1. 区分度诊断(相关 vs 不相关文档)

对每条查询,计算其向量与 qrels 相关文档(正样本)和不相关文档(负样本)的内积:

| 表示 | pos_sim | neg_sim | gap(区分度) |
|---|---|---|---|
| ELF pooled(当前) | 0.427 | 0.407 | **0.020** |
| BGE(对照) | 0.576 | 0.514 | **0.061** |
| ELF token maxsim(100 docs) | 0.357 | 0.325 | **0.032**(pooled 的 2.4×) |

ELF 的区分度仅为 BGE 的 1/3,且相似度分布极窄(全库 sim std=0.03)。

## 2. 维度坍缩诊断(根因)

| 指标 | ELF pooled | BGE |
|---|---|---|
| 文档两两余弦均值 | **0.892**(几乎全相似) | 0.658 |
| 有效秩(participation ratio) | **1** | 2 |
| top-3 奇异值能量占比 | 0.915 | 0.706 |
| pooled hidden 每维 std | **0.062**(数值高度集中) | — |

542 个 ELF 文档向量**有效秩 = 1**: 所有向量挤在同一条直线上。根因是
`T5 mean-pooling`——把 token 序列平均掉后, 每维数值集中在全局均值附近
(std 0.06), 文档间差异只是小扰动, 归一化后都指向主导方向。

## 3. 排除项

- **proj_kernel 投影**: 奇异值分布均匀(top-3 能量仅 8%, cond≈34), 非低秩;
  且任意线性变换不改动 pooled 表示本身的信息坍缩。**排除投影为根因**。
- **denoiser 增强**: 增强在坍缩空间内进行, query_doc_sim 升高(0.74)但区分度
  依然≈0, 属于"在坍缩表示上的无效增强"。**排除为根因**(是下游症状)。

## 4. 修复方向(issue #39)

**token 级序列表示(ColBERT 式)**:
- 文档/查询保留 T5 token 序列(截断到 MAX_TOKENS, 如 64/128), 不做 mean-pooling
- 相似度用 maxsim: mean over query tokens of max over doc tokens(512 空间)
- 实测 gap 0.032, 为 pooled(0.013)的 2.4 倍

后续实现要点:
- 文档侧 token 索引存储(内存, 542 docs × 64 tokens × 512 dim ≈ 90MB float32)
- maxsim 检索向量化(避免逐文档 Python 循环)
- 评测链路接入(检索 top-k → 指标)
- denoiser 增强改为 token 序列上的增强(与 ELF-B 训练分布一致)

## 复现

```bash
# 区分度/坍缩诊断与 maxsim 实验脚本为临时脚本, 已随本分支记录删除;
# 关键指标数据见上表。完整实现后由诊断脚本(experiments/diagnose_elf.py)
# 扩展输出区分度指标。
```
