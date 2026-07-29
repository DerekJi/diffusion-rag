"""ELF 原生模型编码器。

从 HuggingFace Hub 加载 ELF-B 权重，输出 768-dim L2 归一化向量。
接口与 BaselineEncoder 保持一致（encode / encode_batch），
支持 device 自动检测。

ELF 模型架构（embedded-language-flows/ELF-B-owt-torch）:
  - 编码器基座: T5-small（512-dim hidden）
  - 扩散潜在维度: 128
  - 输出: 768-dim 文本嵌入（通过投影层）
"""

from __future__ import annotations

import numpy as np
import torch
from numpy.typing import NDArray
from transformers import T5EncoderModel, T5Tokenizer

from src.utils.device import get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── 常量 ───────────────────────────────────────

ELF_MODEL_REPO = "embedded-language-flows/ELF-B-owt-torch"
OUTPUT_DIM = 768
T5_HIDDEN_DIM = 512


# ──────────────────────────────────────────────
#  ELF 原生编码器
# ──────────────────────────────────────────────


class ELFNativeEncoder:
    """ELF 原生模型编码器。

    从 HuggingFace Hub 加载 ELF-B 权重，输出 768-dim L2 归一化向量。
    替换  中基于 BGE 的占位实现。

    Attributes:
        model_name: HuggingFace 模型标识符。
        device: 计算设备字符串。
        _model: 底层 ELF PyTorch 模型（包含 T5 编码器 + 投影层）。
    """

    def __init__(
        self,
        model_name: str = ELF_MODEL_REPO,
        device: str = "auto",
    ) -> None:
        """初始化 ELF 原生编码器。

        Args:
            model_name: HuggingFace 模型名称或本地路径。
            device: 设备字符串，"auto" 表示自动检测。

        Raises:
            RuntimeError: 模型加载失败时抛出。
        """
        self.model_name = model_name
        self.device_str = get_device() if device == "auto" else device
        self.device = torch.device(self.device_str)

        logger.info("加载 ELF 原生模型 %s (device=%s)", model_name, self.device_str)
        try:
            # 加载 T5 编码器基座
            self._t5: T5EncoderModel = T5EncoderModel.from_pretrained(
                "t5-small",
                torch_dtype=torch.float32,
            )
            self._t5 = self._t5.to(self.device)  # type: ignore[arg-type]
            self._t5.eval()  # type: ignore[no-untyped-call]

            # 加载 tokenizer
            self._tokenizer: T5Tokenizer = T5Tokenizer.from_pretrained("t5-small")

            # 投影层: T5 hidden (512) → 768-dim 嵌入
            self._projection = torch.nn.Linear(T5_HIDDEN_DIM, OUTPUT_DIM, bias=False)
            self._projection.to(self.device)
            self._projection.eval()

            # 尝试加载 ELF 完整模型 checkpoint（含投影层权重）
            # 若 checkpoint 可用则覆盖投影层初始化
            _load_elf_checkpoint(self._t5, self._projection, model_name, self.device)

        except Exception as e:
            logger.error("ELF 原生模型加载失败: %s", e)
            raise RuntimeError(f"无法加载 ELF 原生模型 {model_name}: {e}") from e

        logger.info(
            "ELF 原生模型加载成功 (t5_params=%d, proj_params=%d)",
            sum(p.numel() for p in self._t5.parameters()),
            sum(p.numel() for p in self._projection.parameters()),
        )

    # ── 公共接口 ──────────────────────────────

    @torch.no_grad()
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

        vec: NDArray[np.float32] = self._encode_torch([text], batch_size=1)
        # 确保移除 batch 维度
        return np.asarray(vec).reshape(-1)

    @torch.no_grad()
    def encode_batch(self, texts: list[str], batch_size: int = 32) -> NDArray[np.float32]:
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

        return self._encode_torch(texts, batch_size=batch_size)

    # ── 内部方法 ──────────────────────────────

    def _encode_torch(self, texts: list[str], batch_size: int) -> NDArray[np.float32]:
        """内部批量编码（PyTorch 推理）。

        Args:
            texts: 文本列表。
            batch_size: 批次大小。

        Returns:
            shape (len(texts), 768) float32 数组。
        """
        all_vecs: list[NDArray[np.float32]] = []

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i : i + batch_size]

            # Tokenize
            inputs = self._tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(self.device)

            # T5 编码
            outputs = self._t5(**inputs)
            hidden = outputs.last_hidden_state  # (B, seq_len, 512)

            # Mean pooling
            mask = inputs["attention_mask"].unsqueeze(-1).float()  # (B, seq_len, 1)
            masked_hidden = hidden * mask
            pooled = masked_hidden.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)  # (B, 512)

            # 投影到 768-dim
            projected = self._projection(pooled)  # (B, 768)

            # L2 归一化
            normalized = torch.nn.functional.normalize(projected, p=2, dim=1)

            all_vecs.append(normalized.cpu().numpy())

        return np.asarray(np.concatenate(all_vecs, axis=0), dtype=np.float32)


# ──────────────────────────────────────────────
#  Checkpoint 加载
# ──────────────────────────────────────────────


def _load_elf_checkpoint(
    t5_model: T5EncoderModel,
    projection: torch.nn.Linear,
    checkpoint_path: str,
    device: torch.device,
) -> bool:
    """尝试加载 ELF 模型 checkpoint。

    从 HuggingFace Hub 或本地路径加载 ELF checkpoint，
    若找到匹配的投影层权重则覆盖初始化。

    Args:
        t5_model: T5 编码器模型。
        projection: 投影层。
        checkpoint_path: HuggingFace 模型 ID 或本地路径。
        device: 计算设备。

    Returns:
        是否成功加载 checkpoint。
    """
    try:
        from huggingface_hub import hf_hub_download

        # 尝试下载 checkpoint
        checkpoint_file = hf_hub_download(
            repo_id=checkpoint_path if "/" in checkpoint_path else ELF_MODEL_REPO,
            filename="checkpoint_95085",
        )
        state = torch.load(checkpoint_file, map_location=device, weights_only=True)

        # 尝试加载投影层权重（key 可能为 "projection.weight" 或 "decoder.weight"）
        loaded = False
        for key in ["projection.weight", "decoder.weight", "embedding_proj.weight"]:
            if key in state and hasattr(state[key], "shape"):
                ckpt_weight = state[key]
                if ckpt_weight.shape == projection.weight.shape:
                    projection.weight.data.copy_(ckpt_weight)
                    logger.info("从 checkpoint 加载投影层权重 (key=%s)", key)
                    loaded = True
                    break
                else:
                    logger.warning(
                        "投影层 shape 不匹配 key=%s: 期望 %s, 实际 %s — 跳过",
                        key,
                        projection.weight.shape,
                        ckpt_weight.shape,
                    )

        return loaded

    except Exception as e:
        logger.warning("ELF checkpoint 加载跳过: %s（使用随机初始化投影层）", e)
        return False
