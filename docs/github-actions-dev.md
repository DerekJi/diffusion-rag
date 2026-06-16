# GitHub Actions 自动化开发工作流

> 本项目的开发、审查、修复、评测全流程由 GitHub Actions + Reasonix + DeepSeek 自动化完成。  
> **解决的问题**：本地连接 VPN 时无法使用 DeepSeek，所有 AI 操作迁移到云端 CI。

---

## 目录

1. [目标与优势](#1-目标与优势)
2. [操作链总览](#2-操作链总览)
3. [三工作流详解](#3-三工作流详解)
4. [Issue Body 配置参考](#4-issue-body-配置参考)
5. [工作流架构图](#5-工作流架构图)
6. [前置条件](#6-前置条件)

---

## 1. 目标与优势

### 目标

- 在 GitHub Actions 的 Ubuntu 虚拟机中运行 Reasonix + DeepSeek，绕过本地网络限制
- 实现 **Issue 驱动开发**：提需求 → AI 写代码 → 提 PR → Review → /fix 修改 → 合并
- 无需本地安装 CUDA / PyTorch / 模型权重，CI 自动处理一切

### 优势

| 优势 | 说明 |
|------|------|
| **绕过网络限制** | 在 GitHub 云端运行，不影响本地 VPN |
| **全自动闭环** | 开发 → 审查 → 修复 → 验证 自动化 |
| **节省本地资源** | 6GB 显存的本地 GPU 不需要跑模型推理 |
| **可复现** | CI 环境一致，每次从零安装依赖 |
| **省钱** | 默认 `deepseek-v4-flash`，仅复杂任务升级到 `pro` |

---

## 2. 操作链总览

```
                    ┌─────────────────────────┐
                    │  提 Issue → 打 ai-dev   │
                    │                         │
                    │  reasonix_develop.yml    │
                    │  AI 自动开发 + 测试      │
                    │  → 提 PR                │
                    └──────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │  reasonix-agent.yml      │
                    │  AI 审查 PR + 自动修复   │
                    │  → 评论审查报告          │
                    │  → 修复代码 → Push       │
                    └──────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │  你审查 PR，发现问题     │
                    │                         │
                    │  在评论区写：           │
                    │  /fix 这里需要加异常捕获 │
                    └──────────┬──────────────┘
                               │
                               ▼
                    ┌─────────────────────────┐
                    │  reasonix_pr_feedback.yml│
                    │  AI 读取意见 → 改代码    │
                    │  → pytest 验证          │
                    │  → git push 更新 PR     │
                    └──────────┬──────────────┘
                               │
                    循环直到你满意
```

### 操作步骤

#### 第 1 步：创建 Issue

```bash
# 或用 GitHub UI 直接创建
gh issue create \
  --title "Phase 2.1: 实现 ELF 编码器" \
  --body "
type: code
model: deepseek-v4-flash
level: medium

根据 docs/diffusion-rag-plan.md 实现 ELFEncoder...
"
```

#### 第 2 步：打标签触发开发

在 Issue 页面添加 `ai-dev` 标签（或移除后再添加以重新触发）。

#### 第 3 步：等待自动完成

CI 自动完成：开发 → 测试 → 提 PR → 审查 → 修复。

#### 第 4 步：人工审查 + /fix

在 PR 评论区写 `/fix ...` 提出修改意见，AI 自动修改并 Push。

---

## 3. 三工作流详解

### 3.1 `reasonix_develop.yml` — Issue 驱动开发

**触发**：Issue 被打上 `ai-dev` 标签时

**流程**：

| 步骤 | 说明 |
|------|------|
| 1. Checkout | 检出代码（含完整 git 历史） |
| 2. Python + Node | 安装 Python 3.10 + Node.js 20 |
| 3. HF Model Cache | 缓存 HuggingFace 模型权重 |
| 4. pip install | 安装 `requirements.txt` 依赖 |
| 5. Extract Config | 从 Issue body 提取 `model:`、`level:`、`type:` |
| 6. Formulate Prompt | 组装任务 Prompt，注入 `.reasonix/commands/` 规范 |
| 7. `reasonix run` | AI 根据 Issue 开发代码 |
| 8. Verify | `black` + `isort` + `mypy` + `pytest`（code 模式）/ 仅 `black` + `isort`（docs 模式） |
| 9. Auto-fix | 验证失败时自动修复 |
| 10. Git Config | 设置机器人身份 |
| 11. Create PR | 使用 `peter-evans/create-pull-request` 创建 PR |
| 12. Reply | 在 Issue 回复完成信息 |

### 3.2 `reasonix-agent.yml` — PR 审查 + 自动修复

**触发**：PR 被创建或更新时

**流程**：

| 步骤 | 说明 |
|------|------|
| 1-4. 准备环境 | 同 develop 流程 |
| 5. Generate Diff | 生成 `origin/main...HEAD` 的 diff |
| 6. Code Review | AI 按 `code-review.md` 审查 diff，输出到 `.temp/review-report.md` |
| 7. Post Comment | 审查报告评论到 PR |
| 8. Auto-fix | AI 按 `fix-bug.md` 修复审查发现的问题 |
| 9. Verify | `black` + `isort` + `pytest` |
| 10. Round 2 Fix | 验证失败时第二轮修复 |
| 11. Commit & Push | 推送修复到 PR 分支 |
| 12. Post Summary | 评论修复结果 |

### 3.3 `reasonix_pr_feedback.yml` — 在线 /fix 反馈

**触发**：PR 评论区出现包含 `/fix` 的评论

**流程**：

| 步骤 | 说明 |
|------|------|
| 1. Get PR Branch | 获取当前 PR 的分支名 |
| 2. Checkout | 检出 PR 分支 |
| 3-4. 准备环境 | Python + Node + 依赖 |
| 5. Extract Config | 从父 Issue 读取 `model:`、`level:` |
| 6. Formulate Prompt | 提取 `/fix` 后的意见 + 注入 `fix-bug.md` |
| 7. `reasonix run` | AI 修复代码 |
| 8. Verify | `black` + `isort` + `pytest` |
| 9. Auto-fix Round 2 | 修复失败时第二次 |
| 10. Commit & Push | 推送修复到 PR |
| 11. Reply | 回复用户完成信息 |

---

## 4. Issue Body 配置参考

在 Issue 正文顶部用 YAML 风格字段指定配置：

```yaml
---
type: code          # code（默认）| docs
model: deepseek-v4-flash  # flash（默认）| pro
level: medium        # low | medium（默认）| high
---
```

### 字段详解

| 字段 | 默认值 | 可选值 | 说明 |
|------|--------|--------|------|
| `type` | `code` | `code` / `docs` | 任务类型。`code` 触发完整测试流程；`docs` 仅检查格式 |
| `model` | `deepseek-v4-flash` | `flash` / `pro` | AI 模型。flash 省钱，pro 更强 |
| `level` | `medium` | `low` / `medium` / `high` | 推理深度。对应 `reasonix run --effort` |

**注意**：`ai-dev` 标签需要手动添加，Issue 创建时不自动打上，以实现完整控制。

### Issue 示例

```markdown
type: code
model: deepseek-v4-pro
level: high

根据 docs/diffusion-rag-plan.md 中 Phase 2.1 的描述，实现 src/elf/encoder.py。
验收标准：
- ELFEncoder.encode() 输出 shape (768,), dtype float32, L2 norm ≈ 1.0
- 单元测试覆盖正常路径 + 边界条件
- pytest tests/ -q -m "not slow" 全部通过
```

```markdown
type: docs
level: low

更新 README.md，增加 Phase 4 Colab 迁移的使用说明。
```

---

## 5. 工作流架构图

```
┌─────────────────────────────────────────────────────────────────────┐
│                        GitHub Repository                            │
│                                                                     │
│  ┌──────────────┐   ┌──────────────────┐   ┌──────────────────────┐ │
│  │ issue → ai-dev│   │   PR created     │   │   PR /fix comment   │ │
│  └──────┬───────┘   └────────┬─────────┘   └──────────┬───────────┘ │
│         │                    │                        │             │
│         ▼                    ▼                        ▼             │
│  ┌──────────────┐   ┌──────────────────┐   ┌──────────────────────┐ │
│  │ develop.yml  │   │   agent.yml      │   │  pr_feedback.yml     │ │
│  │              │   │                  │   │                      │ │
│  │ Extract      │   │ Generate Diff    │   │ Get PR Branch        │ │
│  │ Config       │   │                  │   │                      │ │
│  │              │   │ Review (code-    │   │ Extract Config       │ │
│  │ Formulate    │   │ review.md)       │   │                      │ │
│  │ Prompt       │   │                  │   │ Fix (fix-bug.md)     │ │
│  │              │   │ Fix (fix-bug.md) │   │                      │ │
│  │ run → verify │   │                  │   │ verify → push        │ │
│  │ → PR         │   │ verify → push    │   │ → reply              │ │
│  └──────┬───────┘   └────────┬─────────┘   └──────────┬───────────┘ │
│         │                    │                        │             │
│         ▼                    ▼                        ▼             │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  .reasonix/commands/                                        │   │
│  │  ├── develop.md      — 开发规范                             │   │
│  │  ├── code-review.md   — 审查标准                            │   │
│  │  ├── fix-bug.md       — 修复流程                            │   │
│  │  └── feature.md       — 功能实现流程                         │   │
│  └──────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 6. 前置条件

### GitHub Secrets

在仓库 Settings → Secrets → Actions 中配置：

| Secret | 说明 |
|--------|------|
| `DEEPSEEK_API_KEY` | **必需**。DeepSeek API 密钥 |
| `GITHUB_TOKEN` | 自动可用，无需配置 |
| `HF_TOKEN` | 可选。HuggingFace token，提升下载限速 |

### GitHub Labels

需要手动创建一个名为 `ai-dev` 的 label，用于触发 `reasonix_develop.yml`。

### 首次使用

```bash
# 1. 创建 ai-dev label
gh label create ai-dev --description "Trigger AI development workflow" --color "2DA44E"

# 2. 提 Issue 测试
gh issue create \
  --title "Hello AI" \
  --body "创建 scripts/hello.py，打印 Hello AI" \
  --label ai-dev
```

---

> 文档维护：Reasonix AI Developer Workflows  
> 更新日期：2026-06-16
