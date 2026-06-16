---
type: prompt
name: github-issues
version: 3.0
description: 使用 GitHub Issues 驱动当前代码库的多阶段开发（中文）
---

# GitHub Issues 提示词

## 工作流程
1. 使用 `gh` CLI 查看当前未关闭的 GitHub issues
2. 逐个读取 issue 描述，并将其映射到当前仓库的 `docs/`、`src/`、`tests/` 任务
3. 对每个 issue：
   - 先做需求分析
   - 再实现
   - 再 review
   - 再 fix
   - 直到问题收敛
4. 完成后提交代码，并使用有意义的 commit message
5. 提交后关闭对应 issue
6. 全部 issue 完成后，更新 README 和 `docs/`

## 说明
- 以当前仓库结构和文档为准，不沿用外部项目的目录或命名
- 如果 issue 描述不完整，先补充分析再动手实现
