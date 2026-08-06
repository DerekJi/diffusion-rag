"""ColBERT 式多 token 检索器(issue #39)。

诊断确认 mean-pooled 单向量表示有效秩坍缩到 1(所有文档共线), 检索区分度
仅为 BGE 的 1/3。多 token 序列表示(保留 T5 token 级向量)的 maxsim 相似度
区分度提升 2.4 倍, 本模块实现该检索链路:

- TokenIndex: 文档 token 向量索引(内存, 可扩展磁盘序列化)
- ColBERTRetriever: maxsim 检索
  score(q, d) = mean over query tokens of max over doc tokens of cos(q_i, d_j)

用法::

    index = TokenIndex.build(encoder, doc_ids, doc_texts, max_tokens=64)
    retriever = ColBERTRetriever(index)
    doc_ids_found, scores = retriever.search(q_tokens, q_mask, k=10)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from src.utils.logger import get_logger

if TYPE_CHECKING:
    from src.elf.native_encoder import ELFNativeEncoder

logger = get_logger(__name__)


def _l2_normalize(x: NDArray[np.float32], axis: int) -> NDArray[np.float32]:
    """沿指定轴 L2 归一化, 零向量保持为零。"""
    norms = np.linalg.norm(x, axis=axis, keepdims=True)
    safe = np.where(norms > 1e-8, norms, 1.0)
    return np.asarray(x / safe, dtype=np.float32)


class TokenIndex:
    """文档 token 向量索引。

    Attributes:
        doc_ids: 文档 ID 列表(与 tokens 第 0 维对齐)。
        tokens: shape (n_docs, max_tokens, dim) float32, L2 归一化。
        mask: shape (n_docs, max_tokens) float32, 1=有效 token。
    """

    def __init__(
        self,
        doc_ids: list[str],
        tokens: NDArray[np.float32],
        mask: NDArray[np.float32],
    ) -> None:
        if len(doc_ids) != tokens.shape[0] or tokens.shape[0] != mask.shape[0]:
            raise ValueError("doc_ids / tokens / mask 第 0 维必须一致")
        if tokens.shape[1] != mask.shape[1]:
            raise ValueError("tokens 与 mask 的序列长度不一致")
        self.doc_ids = list(doc_ids)
        self.tokens = _l2_normalize(tokens, axis=2)
        self.mask = mask
        logger.info(
            "TokenIndex 构建完成: %d 篇文档, 每篇最多 %d tokens, 维度 %d",
            len(doc_ids),
            tokens.shape[1],
            tokens.shape[2],
        )

    @classmethod
    def build(
        cls,
        encoder: ELFNativeEncoder,
        doc_ids: list[str],
        doc_texts: list[str],
        max_tokens: int = 64,
    ) -> "TokenIndex":
        """从文本构建 token 索引。

        Args:
            encoder: ELF 原生编码器（T5 token 级向量, 提供 encode_tokens）。
            doc_ids: 文档 ID 列表。
            doc_texts: 文档文本列表。
            max_tokens: 截断的 token 数。

        Returns:
            构建完成的 TokenIndex。
        """
        tokens, mask = encoder.encode_tokens(doc_texts, max_tokens=max_tokens)
        return cls(doc_ids, tokens, mask)

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)


class ColBERTRetriever:
    """maxsim 检索器(ColBERT 风格)。

    score(q, d) = mean_{q_i in q} max_{d_j in d} cos(q_i, d_j)
    """

    def __init__(self, index: TokenIndex) -> None:
        self._index = index

    @property
    def index(self) -> TokenIndex:
        return self._index

    def search(
        self,
        query_tokens: NDArray[np.float32],
        query_mask: NDArray[np.float32],
        k: int = 10,
    ) -> tuple[list[str], list[float]]:
        """检索 top-k 文档。

        Args:
            query_tokens: shape (Lq, dim) float32。
            query_mask: shape (Lq,) float32(1=有效 token)。
            k: 返回数量。

        Returns:
            (doc_ids, scores): 按相似度降序。
        """
        if k <= 0:
            return ([], [])
        qt = _l2_normalize(np.asarray(query_tokens, dtype=np.float32), axis=1)
        qm = np.asarray(query_mask, dtype=np.float32)
        if np.all(qm <= 0):
            return ([], [])  # 查询无有效 token
        doc_tokens = self._index.tokens
        doc_mask = self._index.mask

        # 逐 token 相似度 (Lq, N, L)
        sims = np.einsum("qd,nld->qnl", qt, doc_tokens)
        sims = np.where(doc_mask[None] > 0, sims, -np.inf)  # 排除 padding

        # 对每个查询 token 取文档侧最大值 (Lq, N)
        doc_scores = sims.max(axis=2)
        doc_scores = np.where(qm[:, None] > 0, doc_scores, np.float32(np.nan))  # 排除 padding
        with np.errstate(all="ignore"):
            scores = np.nanmean(doc_scores, axis=0)  # (N,) 查询 token 均值

        order = np.argsort(scores)[::-1][:k]
        return [self._index.doc_ids[i] for i in order], [float(s) for s in scores[order]]
