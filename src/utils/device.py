"""设备自动检测模块。"""

import torch

from src.utils.logger import get_logger

logger = get_logger(__name__)


def get_device() -> str:
    """自动选择最优设备——本地/Colab/无GPU均兼容。

    Returns:
        "cuda" 或 "cpu"。
    """
    if torch.cuda.is_available():
        device = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info("GPU: %s, VRAM: %.1f GB", gpu_name, vram)
    else:
        device = "cpu"
        logger.warning("No GPU — CPU mode (debug only)")
    return device
