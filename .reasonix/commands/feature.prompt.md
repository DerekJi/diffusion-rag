
1. 根据已计划好的任务文档，逐项实现各项任务
   * development：每个任务启用一个subagent实现，完成后要确保编译、单元测试能通过。开发要求详见 .reasonix/commands/development.prompt.md
   * review：实现完成后，启动一个subagent进行代码review，要求详见 .reasonix/commands/code-review.prompt.md
   * fix：review完成后，启动一个subagent修复发现的各种问题，并确保编译、单元测试能通过。修复要求详见 .reasonix/commands/fix.prompt.md
   * 迭代：重复上面的 review -> fix 过程，直到review不出什么问题为止。

2. 全部任务完成后
   * review：启动一个subagent进行review第5阶段的全部代码，要求详见 .reasonix/commands/code-review.prompt.md
   * fix：review完成后，启动一个subagent修复发现的各种问题，并确保编译、单元测试。修复要求详见 .reasonix/commands/fix.prompt.md
   * 迭代：重复上面的 review -> fix 过程，直到review不出什么问题为止。

3. 更新README及docs目录中的文档