"""ColBERT 式多 token 检索器单元测试(issue #39)。

使用合成向量验证 maxsim 语义, 无需模型加载。
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from src.elf.token_retriever import ColBERTRetriever, TokenIndex


def _unit(dim: int, idx: int) -> NDArray[np.float32]:
    """构造第 idx 维为 1 的单位向量。"""
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v


def _make_index(n_docs: int = 3, dim: int = 8, max_tokens: int = 4) -> TokenIndex:
    """3 篇文档, 每篇 token 指向不同方向, 便于断言排序。"""
    tokens = np.zeros((n_docs, max_tokens, dim), dtype=np.float32)
    mask = np.ones((n_docs, max_tokens), dtype=np.float32)
    # doc i 的第 0 个 token 指向第 i 维, 其余 token 指向无关方向
    for i in range(n_docs):
        tokens[i, 0] = _unit(dim, i)
        for j in range(1, max_tokens):
            tokens[i, j] = _unit(dim, dim - 1)
    return TokenIndex([f"d{i}" for i in range(n_docs)], tokens, mask)


class TestTokenIndex:
    def test_normalizes_tokens(self) -> None:
        """构建时 token 向量 L2 归一化。"""
        idx = _make_index()
        norms = np.linalg.norm(idx.tokens, axis=2)
        assert np.allclose(norms, 1.0)

    def test_shape_and_ids(self) -> None:
        """shape 与 doc_ids 对齐。"""
        idx = _make_index(3, 8, 4)
        assert idx.tokens.shape == (3, 4, 8)
        assert idx.doc_ids == ["d0", "d1", "d2"]
        assert idx.n_docs == 3

    def test_mismatched_lengths_raise(self) -> None:
        """doc_ids 与 tokens 数量不一致时抛 ValueError。"""
        with pytest.raises(ValueError, match="一致"):
            TokenIndex(
                ["d0"], np.zeros((2, 4, 8), dtype=np.float32), np.ones((2, 4), dtype=np.float32)
            )

    def test_zero_tokens_stay_zero(self) -> None:
        """零 token 向量归一化后保持为零(不产生 NaN)。"""
        tokens = np.zeros((1, 3, 8), dtype=np.float32)
        idx = TokenIndex(["d0"], tokens, np.ones((1, 3), dtype=np.float32))
        assert np.isfinite(idx.tokens).all()


class TestColBERTRetriever:
    def test_ranks_most_similar_doc_first(self) -> None:
        """查询 token 与 d1 最相似时, top1 应为 d1。"""
        idx = _make_index()
        ret = ColBERTRetriever(idx)
        q_tokens = np.stack([_unit(8, 1), _unit(8, 1)])  # 两个都指向 d1
        q_mask = np.ones(2, dtype=np.float32)
        found, scores = ret.search(q_tokens, q_mask, k=2)
        assert found[0] == "d1"
        assert len(found) == 2
        assert scores[0] >= scores[1]

    def test_respects_query_padding(self) -> None:
        """padding 查询 token(mask=0)不参与 mean。"""
        idx = _make_index()
        ret = ColBERTRetriever(idx)
        q_tokens = np.stack([_unit(8, 0), np.zeros(8, dtype=np.float32)])
        q_mask = np.array([1.0, 0.0])
        found, _ = ret.search(q_tokens, q_mask, k=1)
        assert found[0] == "d0"

    def test_respects_doc_padding(self) -> None:
        """文档 padding token(mask=0)不参与 max。"""
        idx = _make_index()
        # 把 d0 的 token 1..3 设为极强相似但 mask=0 → 应被排除
        idx.tokens[0, 1:] = np.stack([_unit(8, 1)] * 3)
        idx.mask[0, 1:] = 0.0
        ret = ColBERTRetriever(idx)
        q_tokens = np.stack([_unit(8, 1), _unit(8, 1)])
        q_mask = np.ones(2, dtype=np.float32)
        found, _ = ret.search(q_tokens, q_mask, k=1)
        assert found[0] == "d1"  # d0 的强相似 token 被 mask 排除

    def test_k_limits_results(self) -> None:
        """k 限制返回数量。"""
        idx = _make_index(3, 8, 4)
        ret = ColBERTRetriever(idx)
        q_tokens = np.stack([_unit(8, 0)])
        q_mask = np.ones(1, dtype=np.float32)
        found, _ = ret.search(q_tokens, q_mask, k=1)
        assert len(found) == 1

    def test_empty_query_returns_empty(self) -> None:
        """全部 padding 的查询返回空结果。"""
        idx = _make_index()
        ret = ColBERTRetriever(idx)
        q_tokens = np.zeros((2, 8), dtype=np.float32)
        q_mask = np.zeros(2, dtype=np.float32)
        found, _ = ret.search(q_tokens, q_mask, k=3)
        assert found == []
