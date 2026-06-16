---
name: github-issues
description: 执行Github Issues多阶段开发的 feature 核心调度智能体
---
0. 使用gh cli（用户名为DerekJi）查看当前所有未关闭的github issues

1. 逐个查看未关闭的github issues，根据描述进行响应开发
   * development：每个任务启用一个subagent实现，完成后要确保编译、单元测试能通过。开发要求详见 .reasonix/commands/development..md
   * review：实现完成后，启动一个subagent进行代码review，要求详见 .reasonix/commands/code-review..md
   * fix：review完成后，启动一个subagent修复发现的各种问题，并确保编译、单元测试能通过。修复要求详见 .reasonix/commands/fix..md
   * 迭代：重复上面的 review -> fix 过程，直到review不出什么问题为止。
      * 全部迭代完成后，提交任务代码，给予有意义的commit message
   * commit后，使用gh cli关闭相关GitHub issue(s)。

2. 全部任务完成后
   * review：启动一个subagent进行review第5阶段的全部代码，要求详见 .reasonix/commands/code-review.md
   * fix：review完成后，启动一个subagent修复发现的各种问题，并确保编译、单元测试。修复要求详见 .reasonix/commands/fix..md
   * 迭代：重复上面的 review -> fix 过程，直到review不出什么问题为止。a
      * 全部迭代完成后，提交任务代码，给予有意义的commit message
   * commit后，使用gh cli关闭相关GitHub issue(s)。

3. 更新README及docs目录中的文档