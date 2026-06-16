"""随机种子全局固定模块。"""

import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """固定所有随机源，保证结果可复现。

    Args:
        seed: 随机种子值。
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
