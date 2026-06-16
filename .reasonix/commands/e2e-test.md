---
type: prompt
name: e2e-test
version: 1.0
description: VedaAgent E2E 测试
variables:
---

# E2E Test Prompt

## Role
You are a debugging specialist for VedaAgent, a Python agent framework with a TypeScript React frontend.

## Requirements
根据项目设计目标，以及docs/plan/中的文档，设计一些实际的测试案例（如果必要，还可以写成测试脚本），对已实现的代码进行测试。在测试过程中发现问题、解决问题，不断迭代，直至完全满足要求

我本地有Ollama，而且已经安装了一些模型，可以考虑使用。本机内存是64G，但显存只有6G。

$ ollama list
NAME                       ID              SIZE      MODIFIED     
qwen3.5:9b                 6488c96fa5fa    6.6 GB    24 hours ago    
qwen3.6:35b-a3b            07d35212591f    23 GB     38 hours ago    
llama3.2:latest            a80c4f17acd5    2.0 GB    6 weeks ago     
gemma4:e2b                 7fbdbf8f5e45    7.2 GB    7 weeks ago     
Moondream:latest           55fc3abd3867    1.7 GB    8 weeks ago     
nomic-embed-text:latest    0a109f422b47    274 MB    2 months ago    
bge-m3:latest              790764642607    1.2 GB    2 months ago    
qwen3-coder:30b            06c1097efce0    18 GB     3 months ago    
qwen3-vl:8b                901cae732162    6.1 GB    3 months ago    
deepseek-r1:32b            edba8017331d    19 GB     3 months ago    
mychen76/fin-r1:Q5         0769fc930775    5.4 GB    3 months ago    