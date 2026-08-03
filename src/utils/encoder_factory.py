"""编码器工厂：按 method 返回对应的文档编码器。

将 "按 method 选择编码器" 逻辑从 baseline/benchmark.py 中抽离，
避免 baseline → elf 的反向依赖。baseline 模块仅依赖本工厂，不再直接
导入 ELFPipeline。
"""

from __future__ import annotations

from src.baseline.encoder import BaselineEncoder
from src.config import DEFAULT_ENCODER, METHOD_ELF
from src.elf.pipeline import ELFPipeline


def create_encoder(
    method: str,
    encoder_name: str | None = None,
) -> tuple[BaselineEncoder | ELFPipeline, ELFPipeline | None]:
    """按 method 创建文档侧编码器。

    - 'baseline': BaselineEncoder(BGE)，elf_pipeline 为 None。
    - 'elf':      ELFPipeline(T5 原生编码)，同时返回 pipeline 自身。

    Args:
        method: 检索链路 ('baseline' / 'elf')。
        encoder_name: BGE 编码器名称，仅 method='baseline' 生效；
                      None 时使用 DEFAULT_ENCODER。

    Returns:
        (encoder, elf_pipeline) 元组。method='elf' 时二者指向同一实例。
    """
    if method == METHOD_ELF:
        elf_pipeline = ELFPipeline()
        return elf_pipeline, elf_pipeline
    encoder = BaselineEncoder(model_name=encoder_name or DEFAULT_ENCODER)
    return encoder, None
