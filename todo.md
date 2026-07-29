# Phase 2.3 实现任务

## 任务列表
- [x] 需求分析完成
- [x] 创建 `src/elf/diffusion.py` — add_noise / denoise / cfg_guide 实现
- [x] 更新 `src/elf/__init__.py` — 导出三个函数
- [x] 创建 `tests/test_elf_diffusion.py` — 41 个单元测试，覆盖扩散正反向 + CFG scale 边界
- [x] 创建 `src/elf/pipeline.py` — ELFPipeline 整合 encode + add_noise + denoise + cfg_guide + L2 normalize
- [x] 更新 `src/elf/__init__.py` — 导出 ELFPipeline
- [x] 创建 `tests/test_elf_pipeline.py` — 29 个单元测试（mock 模式，无需真实模型）
- [x] 全部 104 测试通过 ✅
- [x] `make check` 通过 ✅
