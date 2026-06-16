"""BGE-Mini 基线编码器。

封装 BAAI/bge-base-en-v1.5，输出 768-dim L2 归一化向量。
与 ELF 编码器输出维度、格式完全一致。
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from src.utils.device import get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaselineEncoder:
    """BGE-Mini 基线编码器。

    Attributes:
        model_name: HuggingFace 模型标识符。
        device: 计算设备字符串。
        _model: 底层 SentenceTransformer 实例。
    """

    def __init__(self, model_name: str = "BAAI/bge-base-en-v1.5", device: str = "auto") -> None:
        """初始化基线编码器。

        Args:
            model_name: HuggingFace 模型名称。
            device: 设备字符串，"auto" 表示自动检测。

        Raises:
            RuntimeError: 模型加载失败时抛出。
        """
        self.model_name = model_name
        self.device = get_device() if device == "auto" else device

        logger.info("加载 %s (device=%s)", model_name, self.device)
        try:
            self._model = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            logger.error("模型加载失败: %s", e)
            raise RuntimeError(f"无法加载模型 {model_name}: {e}") from e

    def encode(self, text: str) -> np.ndarray:
        """将单条文本编码为 768-dim 向量。

        Args:
            text: 输入文本。

        Returns:
            L2 归一化的 float32 向量，shape (768,)。

        Raises:
            ValueError: 当 text 为空字符串时。
        """
        if not text or not text.strip():
            raise ValueError("输入文本不能为空字符串")

        vec = self._model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )
        # 确保是 float32 且 1-dim
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        return vec

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        """批量编码。

        Args:
            texts: 文本列表。
            batch_size: 批次大小。

        Returns:
            shape (len(texts), 768) float32 数组。

        Raises:
            ValueError: 当 texts 为空列表时。
        """
        if not texts:
            raise ValueError("文本列表不能为空")

        vecs = self._model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )
        return np.asarray(vecs, dtype=np.float32)
