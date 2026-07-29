"""ELF 扩散模型编码器。

封装文本编码模型，输出 768-dim L2 归一化向量，
与 BGE 基线编码器 (BaselineEncoder) 输出维度、格式完全一致。

该编码器是 ELF 扩散增强检索链路的第一步：
  encode(text) → [add_noise → denoise → cfg_guide] → L2 normalize → FAISS search
  注：扩散相关步骤 (Phase 2.2+) 位于 src/elf/diffusion.py。
"""

import numpy as np
from numpy.typing import NDArray

from src.utils.device import get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)

try:
    from sentence_transformers import SentenceTransformer

    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:  # pragma: no cover
    SentenceTransformer = None  # type: ignore[assignment]
    _HAS_SENTENCE_TRANSFORMERS = False


class ELFEncoder:
    """ELF 扩散模型编码器。

    封装文本编码模型，输出 768-dim L2 归一化向量。
    当前底层使用 sentence-transformers 模型（与 BaselineEncoder 一致），
    后续可替换为 ELF 原生编码器。

    Attributes:
        model_name: HuggingFace 模型标识符。
        device: 计算设备字符串。
        _model: 底层 sentence-transformers 模型实例。
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "auto",
    ) -> None:
        """初始化 ELF 编码器。

        Args:
            model_name: HuggingFace 模型名称。默认使用与基线一致的 BGE 模型。
            device: 设备字符串，"auto" 表示自动检测。

        Raises:
            RuntimeError: 模型加载失败或 sentence-transformers 未安装时抛出。
        """
        self.model_name = model_name
        self.device = get_device() if device == "auto" else device

        if not _HAS_SENTENCE_TRANSFORMERS:
            raise RuntimeError(  # pragma: no cover
                "sentence-transformers 未安装。请运行: pip install sentence-transformers"
            )

        logger.info("加载 ELF 编码器 %s (device=%s)", model_name, self.device)
        try:
            self._model = SentenceTransformer(model_name, device=self.device)
        except Exception as e:
            logger.error("模型加载失败: %s", e)
            raise RuntimeError(f"无法加载模型 {model_name}: {e}") from e

    def encode(self, text: str) -> NDArray[np.float32]:
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
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        return vec

    def encode_batch(
        self,
        texts: list[str],
        batch_size: int = 32,
    ) -> NDArray[np.float32]:
        """批量编码文本列表。

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
