"""Baseline 编码器 + FAISS 索引/检索 + 指标 单元测试。"""

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.baseline.encoder import BaselineEncoder
from src.evaluation.metrics import compute_metrics, compute_metrics_batch
from src.vector_store.indexer import FAISSIndexer
from src.vector_store.retriever import Retriever

# ── Encoder tests (mock 模式，不需下载模型) ─────────────────────


class TestBaselineEncoderMock:
    """使用 mock 的编码器测试，无需网络/模型缓存。

    通过 mock SentenceTransformer 类的构造函数，使其返回一个
    MagicMock 实例，encode 方法返回固定 768-dim 归一化向量。
    """

    @pytest.fixture(autouse=True)
    def _mock_model(self) -> None:
        """mock SentenceTransformer 类，返回 mock 实例。"""
        rng = np.random.RandomState(42)
        fixed_vec = rng.randn(768).astype(np.float32)
        fixed_vec /= np.linalg.norm(fixed_vec)

        mock_instance = MagicMock()
        mock_instance.encode.return_value = fixed_vec

        patcher = patch(
            "src.baseline.encoder.SentenceTransformer",
            return_value=mock_instance,
        )
        patcher.start()
        yield
        patcher.stop()

    def test_encode_shape(self) -> None:
        """输出 shape == (768,)。"""
        encoder = BaselineEncoder(device="cpu")
        vec = encoder.encode("Hello world")
        assert vec.shape == (768,)

    def test_encode_dtype(self) -> None:
        """输出 dtype == float32。"""
        encoder = BaselineEncoder(device="cpu")
        vec = encoder.encode("Hello world")
        assert vec.dtype == np.float32

    def test_encode_l2_norm(self) -> None:
        """L2 范数约为 1.0。"""
        encoder = BaselineEncoder(device="cpu")
        vec = encoder.encode("Hello world")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_encode_empty_string(self) -> None:
        """空字符串应 raise ValueError。"""
        encoder = BaselineEncoder(device="cpu")
        with pytest.raises(ValueError):
            encoder.encode("")

    def test_encode_batch_shape(self) -> None:
        """batch 输出 shape == (N, 768)。"""
        mock_batch = np.random.RandomState(42).randn(3, 768).astype(np.float32)
        with patch(
            "src.baseline.encoder.SentenceTransformer",
            return_value=MagicMock(encode=MagicMock(return_value=mock_batch)),
        ):
            encoder = BaselineEncoder(device="cpu")
            vecs = encoder.encode_batch(["a", "bb", "ccc"])
        assert vecs.shape == (3, 768)
        assert vecs.dtype == np.float32


# ── Encoder tests (真实模型，需下载) ────────────────────────────


@pytest.mark.slow
class TestBaselineEncoderReal:
    """需要真实 BGE 模型下载的集成测试。
    通过 pytest -m slow 执行，日常跳过。
    """

    @pytest.fixture(scope="class")
    def encoder(self) -> BaselineEncoder:
        return BaselineEncoder(device="cpu")

    def test_encode_deterministic(self, encoder: BaselineEncoder) -> None:
        """同一文本两次编码结果一致。"""
        v1 = encoder.encode("Test sentence")
        v2 = encoder.encode("Test sentence")
        assert np.allclose(v1, v2, atol=1e-6)

    def test_encode_batch_shape(self, encoder: BaselineEncoder) -> None:
        """batch 输出 shape == (N, 768)。"""
        texts = ["a", "bb", "ccc"]
        vecs = encoder.encode_batch(texts)
        assert vecs.shape == (3, 768)
        assert vecs.dtype == np.float32


# ── FAISS Indexer + Retriever tests ────────────────────────────


class TestFAISSIndexer:
    """FAISSIndexer 单元测试。"""

    @pytest.fixture
    def vecs(self) -> np.ndarray:
        """10 个 768-dim 归一化向量。"""
        rng = np.random.RandomState(42)
        v = rng.randn(10, 768).astype(np.float32)
        v /= np.linalg.norm(v, axis=1, keepdims=True)
        return v

    @pytest.fixture
    def doc_ids(self) -> list[str]:
        return [f"doc_{i}" for i in range(10)]

    def test_index_build_and_search(self, vecs: np.ndarray, doc_ids: list[str]) -> None:
        """建索引后再搜自己，top-1 doc_id 匹配。"""
        indexer = FAISSIndexer(dimension=768, nlist=5)
        indexer.build(vecs, doc_ids)
        retriever = Retriever(indexer)

        for i in range(10):
            result_ids, _ = retriever.search(vecs[i], k=1)
            assert len(result_ids) == 1
            assert result_ids[0] == doc_ids[i]

    def test_search_empty_index(self) -> None:
        """未建索引时 search 返回 ([], [])。"""
        indexer = FAISSIndexer(dimension=768, nlist=5)
        retriever = Retriever(indexer)
        ids, dists = retriever.search(np.random.randn(768).astype(np.float32), k=5)
        assert ids == []
        assert dists == []

    def test_index_save_load_roundtrip(self, vecs: np.ndarray, doc_ids: list[str]) -> None:
        """save → load → search，结果一致。"""
        indexer = FAISSIndexer(dimension=768, nlist=5)
        indexer.build(vecs, doc_ids)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "test.index")
            indexer.save(path)

            indexer2 = FAISSIndexer(dimension=768, nlist=5)
            indexer2.load(path)
            retriever2 = Retriever(indexer2)

            ids, _ = retriever2.search(vecs[0], k=3)
            assert len(ids) == 3
            assert ids[0] == doc_ids[0]

    def test_search_k_larger_than_n_docs(self, vecs: np.ndarray, doc_ids: list[str]) -> None:
        """k > n_docs 时不崩溃，返回全部。"""
        indexer = FAISSIndexer(dimension=768, nlist=5)
        indexer.build(vecs, doc_ids)
        retriever = Retriever(indexer)
        ids, _ = retriever.search(vecs[0], k=100)
        assert len(ids) == len(doc_ids)


# ── Metrics tests ──────────────────────────────────────────────


class TestMetrics:
    """指标计算单元测试。"""

    def test_recall_perfect(self) -> None:
        """results 包含所有相关文档时 recall@k = 1.0。"""
        qrels = {"d1": 1, "d2": 1, "d3": 1}
        results = ["d1", "d2", "d3", "d4", "d5"]
        m = compute_metrics(qrels, results, k_values=[3, 5])
        assert m.recall[3] == 1.0
        assert m.recall[5] == 1.0

    def test_recall_zero(self) -> None:
        """results 为空时 recall = 0.0。"""
        m = compute_metrics({"d1": 1}, [], k_values=[5])
        assert m.recall[5] == 0.0

    def test_mrr(self) -> None:
        """首个相关文档在 rank=3 时 MRR = 1/3。"""
        qrels = {"d3": 1}
        results = ["d1", "d2", "d3", "d4"]
        m = compute_metrics(qrels, results, k_values=[5])
        assert abs(m.mrr - 1.0 / 3) < 1e-9

    def test_ndcg_vs_known(self) -> None:
        """手工计算验证 NDCG。"""
        qrels = {"d1": 3, "d2": 2, "d3": 0}
        results = ["d1", "d3", "d2"]
        m = compute_metrics(qrels, results, k_values=[3])
        # rels from results: [3, 0, 2]
        # DCG:    3 + 0/log2(2) + 2/log2(3) = 3 + 0 + 1.262 = 4.262
        # IDCG:   sorted rels [3, 2, 0]: 3 + 2/log2(2) + 0/log2(3) = 3 + 2 + 0 = 5.0
        # NDCG ≈ 4.262 / 5.0 ≈ 0.852
        assert 0.84 < m.ndcg[3] < 0.86

    def test_hit_rate(self) -> None:
        """有命中时 hit_rate = 1，无命中时 = 0。"""
        qrels = {"d1": 1}
        m_hit = compute_metrics(qrels, ["d3", "d1"], k_values=[2])
        assert m_hit.hit_rate[2] == 1.0

        m_miss = compute_metrics(qrels, ["d3", "d4"], k_values=[2])
        assert m_miss.hit_rate[2] == 0.0

    def test_compute_metrics_batch(self) -> None:
        """批量计算指标返回正确字典。"""
        all_qrels = {
            "q1": {"d1": 1},
            "q2": {"d2": 1},
        }
        all_results = {
            "q1": ["d1", "d3"],
            "q2": ["d3", "d4"],
        }
        batch = compute_metrics_batch(all_qrels, all_results, k_values=[1])
        assert batch["q1"].hit_rate[1] == 1.0
        assert batch["q2"].hit_rate[1] == 0.0
