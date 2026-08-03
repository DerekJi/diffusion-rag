"""run_benchmark 双链路单元测试。

全部依赖均 mock（数据集 / 文档编码器 / ELFPipeline），
无需网络下载与 GPU，参考 TestBaselineEncoderMock 的 mock 模式。
"""

from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from numpy.typing import NDArray

import src.baseline.benchmark as benchmark
from src.baseline.benchmark import (
    BenchmarkContext,
    build_benchmark_context,
    run_benchmark,
)
from src.config import METHOD_BASELINE, METHOD_ELF
from src.evaluation.dataset import DatasetTriple


def _make_fake_data(n_queries: int = 4, n_docs: int = 8) -> DatasetTriple:
    """构造小型合成数据集（qrels 引用的 doc 均在 corpus 内）。"""
    queries = {f"q{i}": f"query text {i}" for i in range(n_queries)}
    corpus = {f"d{i}": f"doc text {i}" for i in range(n_docs)}
    qrels = {f"q{i}": {f"d{i}": 1, f"d{(i + 1) % n_docs}": 1} for i in range(n_queries)}
    return DatasetTriple(queries=queries, corpus=corpus, qrels=qrels)


class TestRunBenchmark:
    """run_benchmark 双链路切换测试。"""

    @pytest.fixture(autouse=True)
    def _mock_dependencies(self, tmp_path: Path) -> Iterator[None]:
        """mock 数据集加载、文档编码器与 ELFPipeline。"""
        rng = np.random.RandomState(42)
        self._fixed_vec = rng.randn(768).astype(np.float32)
        self._fixed_vec /= np.linalg.norm(self._fixed_vec)
        fixed_vec = self._fixed_vec

        def _encode(text: str | list[str], **kwargs: object) -> NDArray[np.float32]:
            if isinstance(text, str):
                return fixed_vec
            return np.tile(fixed_vec, (len(text), 1))

        doc_encoder = MagicMock()
        doc_encoder.encode.side_effect = _encode
        self._doc_encoder = doc_encoder

        self._elf_pipeline_mock = MagicMock()
        self._elf_pipeline_mock.enhance.return_value = fixed_vec
        # ELF 模式文档编码走 pipeline.encoder.encode_batch, 返回 (n, 768) 数组
        elf_inner_encoder = MagicMock()
        elf_inner_encoder.encode_batch.side_effect = lambda texts, batch_size=32: np.tile(
            fixed_vec, (len(texts), 1)
        )
        self._elf_pipeline_mock.encoder = elf_inner_encoder

        patchers = [
            patch("src.baseline.benchmark.load_dataset", return_value=_make_fake_data()),
            patch("src.baseline.encoder.SentenceTransformer", return_value=doc_encoder),
            patch("src.baseline.benchmark.ELFPipeline", return_value=self._elf_pipeline_mock),
            patch(
                "src.utils.encoder_factory.ELFPipeline",
                return_value=self._elf_pipeline_mock,
            ),
        ]
        for p in patchers:
            p.start()
        self._output_dir = tmp_path
        yield
        for p in patchers:
            p.stop()

    # ── baseline 链路 ─────────────────────────

    def test_default_method_is_baseline(self) -> None:
        """默认 method='baseline'，输出 baseline.csv。"""
        df = run_benchmark(output_dir=str(self._output_dir))
        assert df["method"].iloc[0] == METHOD_BASELINE
        assert (self._output_dir / "nfcorpus" / "baseline.csv").exists()

    def test_baseline_outputs_all_metrics(self) -> None:
        """baseline 输出全部指标列（recall/precision/ndcg/hit_rate/mrr）。"""
        df = run_benchmark(output_dir=str(self._output_dir))
        assert len(df) == 1
        for k in [5, 10, 20]:
            assert f"recall@{k}" in df.columns
            assert f"precision@{k}" in df.columns
            assert f"ndcg@{k}" in df.columns
            assert f"hit_rate@{k}" in df.columns
            assert 0.0 <= df[f"recall@{k}"].iloc[0] <= 1.0
        assert "mrr" in df.columns

    def test_baseline_does_not_touch_elf_pipeline(self) -> None:
        """baseline 链路不创建 ELFPipeline（不加载 ELF 模型）。"""
        run_benchmark(output_dir=str(self._output_dir))
        assert self._elf_pipeline_mock.enhance.call_count == 0

    # ── elf 链路 ──────────────────────────────

    def test_elf_method_uses_pipeline(self) -> None:
        """method='elf' 时每条 query 调用一次 enhance()，输出 elf.csv。"""
        df = run_benchmark(method=METHOD_ELF, output_dir=str(self._output_dir))
        assert df["method"].iloc[0] == METHOD_ELF
        assert df["steps"].iloc[0] == 2
        assert df["noise_t"].iloc[0] == 0.4
        assert df["cfg_scale"].iloc[0] == 2.0
        assert self._elf_pipeline_mock.enhance.call_count == 4
        assert (self._output_dir / "nfcorpus" / "elf.csv").exists()

    def test_elf_method_passes_params(self) -> None:
        """elf 增强参数（steps/noise_t/cfg_scale/rng）正确透传。"""
        run_benchmark(
            method=METHOD_ELF,
            elf_steps=4,
            elf_noise_t=0.5,
            elf_cfg_scale=3.0,
            output_dir=str(self._output_dir),
        )
        call = self._elf_pipeline_mock.enhance.call_args
        assert call.kwargs["steps"] == 4
        assert call.kwargs["noise_t"] == 0.5
        assert call.kwargs["cfg_scale"] == 3.0
        # 可复现性：传入顺序消费的 Generator
        assert isinstance(call.kwargs["rng"], np.random.Generator)

    def test_elf_reproducible_with_same_seed(self) -> None:
        """相同 seed 两次运行，enhance 收到的 rng 状态一致 → 结果可复现。"""
        run_benchmark(method=METHOD_ELF, seed=42, output_dir=str(self._output_dir))
        first_rng_state = self._elf_pipeline_mock.enhance.call_args.kwargs[
            "rng"
        ].bit_generator.state
        self._elf_pipeline_mock.reset_mock()
        run_benchmark(method=METHOD_ELF, seed=42, output_dir=str(self._output_dir))
        second_rng_state = self._elf_pipeline_mock.enhance.call_args.kwargs[
            "rng"
        ].bit_generator.state
        assert first_rng_state == second_rng_state

    # ── 共享上下文 (shared) ─────────────────────

    def test_shared_context_skips_preparation_and_reuses_pipeline(self, tmp_path: Path) -> None:
        """传入 shared 时跳过数据准备;ELF pipeline 惰性创建且跨组复用。"""
        retriever = MagicMock()
        retriever.search.side_effect = lambda qvec, k: (
            [f"d{i}" for i in range(min(k, 8))],
            [1.0] * min(k, 8),
        )
        ctx = BenchmarkContext(
            dataset="nfcorpus",
            method=METHOD_ELF,
            data=_make_fake_data(),
            encoder=self._doc_encoder,
            retriever=retriever,
        )
        with (
            patch("src.baseline.benchmark.build_benchmark_context") as mock_build,
            patch(
                "src.baseline.benchmark.ELFPipeline",
                return_value=self._elf_pipeline_mock,
            ) as factory,
        ):
            run_benchmark(method=METHOD_ELF, output_dir=str(self._output_dir), shared=ctx)
            run_benchmark(method=METHOD_ELF, output_dir=str(self._output_dir), shared=ctx)

        assert mock_build.call_count == 0  # 跳过数据加载/文档编码/建索引
        assert factory.call_count == 1  # ELF 模型只加载一次
        assert self._elf_pipeline_mock.enhance.call_count == 8  # 2 次运行 × 4 条 query
        assert ctx.elf_pipeline is self._elf_pipeline_mock  # 缓存在上下文中

    def test_shared_context_dataset_mismatch_raises(self, tmp_path: Path) -> None:
        """shared 上下文数据集与参数不一致时抛出 ValueError（防止写错输出目录）。"""
        ctx = BenchmarkContext(
            dataset="msmarco",
            method=METHOD_BASELINE,
            data=_make_fake_data(),
            encoder=self._doc_encoder,
            retriever=MagicMock(),
        )
        with pytest.raises(ValueError, match="不一致"):
            run_benchmark(dataset="nfcorpus", output_dir=str(self._output_dir), shared=ctx)

    # ── 文档编码按 method 切换 (issue #33) ─────

    def test_build_context_baseline_docs_use_bge(self) -> None:
        """method='baseline' 时文档用 BGE 编码, 不创建 ELFPipeline。"""
        with patch("src.utils.encoder_factory.ELFPipeline") as mock_elf_factory:
            ctx = build_benchmark_context(
                dataset="nfcorpus", method=METHOD_BASELINE, sample=2, index_nlist=2
            )
        assert ctx.method == METHOD_BASELINE
        assert ctx.elf_pipeline is None
        # ELFPipeline 从未被构造（create_encoder 在 method='baseline' 时不调用）
        assert mock_elf_factory.call_count == 0

    def test_build_context_elf_docs_use_elf_pipeline(self, tmp_path: Path) -> None:
        """method='elf' 时文档用 ELFPipeline 编码, 查询与文档同处 ELF 空间。"""
        with patch(
            "src.utils.encoder_factory.ELFPipeline",
            return_value=self._elf_pipeline_mock,
        ) as factory:
            ctx = build_benchmark_context(
                dataset="nfcorpus", method=METHOD_ELF, sample=2, index_nlist=2
            )
        # 文档编码走 pipeline.encoder.encode_batch(ELF 空间), 而非 BGE
        assert self._elf_pipeline_mock.encoder.encode_batch.call_count == 1
        assert ctx.encoder is self._elf_pipeline_mock
        assert ctx.elf_pipeline is self._elf_pipeline_mock
        assert ctx.method == METHOD_ELF

    # ── 采样 / 参数校验 ───────────────────────

    def test_sample_limits_queries(self) -> None:
        """sample=N 时只对前 N 条 query 编码检索。"""
        run_benchmark(method=METHOD_ELF, sample=2, output_dir=str(self._output_dir))
        assert self._elf_pipeline_mock.enhance.call_count == 2

    def test_invalid_method_raises(self) -> None:
        """非法 method 抛出 ValueError。"""
        with pytest.raises(ValueError, match="不支持的链路"):
            run_benchmark(method="invalid", output_dir=str(self._output_dir))

    def test_returns_dataframe(self) -> None:
        """返回单行聚合指标 DataFrame。"""
        df = run_benchmark(output_dir=str(self._output_dir))
        assert isinstance(df, pd.DataFrame)
        assert df["dataset"].iloc[0] == "nfcorpus"


def test_main_returns_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI 入口 `_main` 返回 int 退出码（修复 func-returns-value 类型错误）。"""
    monkeypatch.setattr("sys.argv", ["benchmark"])
    monkeypatch.setattr(benchmark, "set_seed", lambda seed: None)
    monkeypatch.setattr(
        benchmark,
        "run_benchmark",
        lambda **kwargs: pd.DataFrame({"dataset": ["nfcorpus"], "recall@10": [0.5]}),
    )
    exit_code = benchmark._main()
    assert exit_code == 0
    assert "recall@10" in capsys.readouterr().out


# ── _analyze_group 诊断指标参数组合测试 ─────

_MOCK_QUERY_IDS = ["q0", "q1", "q2"]


def _make_mock_ctx() -> MagicMock:
    """构造带 retriever 的 BenchmarkContext mock。"""
    ctx = MagicMock()
    ctx.retriever.search.side_effect = lambda qvec, k: (
        [f"d{i}" for i in range(min(k, 5))],
        [1.0 - i * 0.1 for i in range(min(k, 5))],
    )
    ctx.data.queries = {qid: f"query {qid}" for qid in _MOCK_QUERY_IDS}
    return ctx


def _make_qvecs() -> dict[str, np.ndarray]:
    """构造正交查询向量。"""
    rng = np.random.RandomState(42)
    qvecs: dict[str, np.ndarray] = {}
    for qid in _MOCK_QUERY_IDS:
        v = rng.randn(768).astype(np.float32)
        qvecs[qid] = v / np.linalg.norm(v)
    return qvecs


def _make_doc_vectors(n: int = 8) -> np.ndarray:
    """构造文档向量 (n, 768)。"""
    rng = np.random.RandomState(99)
    vecs = rng.randn(n, 768).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


class TestAnalyzeGroup:
    """_analyze_group 参数组合测试。"""

    def test_only_qvecs(self) -> None:
        """仅传入 qvecs 时返回基础指标(query_doc_sim/query_doc_top10_sim/n_queries)。"""
        from experiments.diagnose_elf import _analyze_group

        ctx = _make_mock_ctx()
        doc_vectors = _make_doc_vectors(8)
        id_to_idx = {f"d{i}": i for i in range(8)}
        qvecs = _make_qvecs()

        result = _analyze_group(ctx, doc_vectors, id_to_idx, _MOCK_QUERY_IDS, qvecs)

        assert "query_doc_sim" in result
        assert "query_doc_top10_sim" in result
        assert result["n_queries"] == 3.0
        assert "shift_cos" not in result
        assert "overlap_with_baseline" not in result

    def test_with_base_qvecs(self) -> None:
        """传入 base_qvecs 时附加 shift_cos / shift_l2。"""
        from experiments.diagnose_elf import _analyze_group

        ctx = _make_mock_ctx()
        doc_vectors = _make_doc_vectors(8)
        id_to_idx = {f"d{i}": i for i in range(8)}
        qvecs = _make_qvecs()
        base_qvecs = _make_qvecs()  # 不同随机种子 → 不同向量

        result = _analyze_group(
            ctx, doc_vectors, id_to_idx, _MOCK_QUERY_IDS, qvecs, base_qvecs=base_qvecs
        )

        assert "shift_cos" in result
        assert "shift_l2" in result
        assert "overlap_with_baseline" not in result
        assert -1.0 - 1e-6 <= result["shift_cos"] <= 1.0 + 1e-6
        assert result["shift_l2"] >= 0.0

    def test_with_baseline_top10(self) -> None:
        """传入 baseline_top10 时附加 overlap_with_baseline。"""
        from experiments.diagnose_elf import _analyze_group

        ctx = _make_mock_ctx()
        doc_vectors = _make_doc_vectors(8)
        id_to_idx = {f"d{i}": i for i in range(8)}
        qvecs = _make_qvecs()
        baseline_top10 = {
            "q0": ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d0", "d1"],
            "q1": ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d0", "d1"],
            "q2": ["d0", "d1", "d2", "d3", "d4", "d5", "d6", "d7", "d0", "d1"],
        }

        result = _analyze_group(
            ctx,
            doc_vectors,
            id_to_idx,
            _MOCK_QUERY_IDS,
            qvecs,
            baseline_top10=baseline_top10,
        )

        assert "overlap_with_baseline" in result
        assert "shift_cos" not in result
        assert 0.0 <= result["overlap_with_baseline"] <= 1.0

    def test_all_params(self) -> None:
        """同时传入 base_qvecs + baseline_top10 时返回全部指标。"""
        from experiments.diagnose_elf import _analyze_group

        ctx = _make_mock_ctx()
        doc_vectors = _make_doc_vectors(8)
        id_to_idx = {f"d{i}": i for i in range(8)}
        qvecs = _make_qvecs()
        base_qvecs = _make_qvecs()
        baseline_top10 = {qid: [f"d{i}" for i in range(10)] for qid in _MOCK_QUERY_IDS}

        result = _analyze_group(
            ctx,
            doc_vectors,
            id_to_idx,
            _MOCK_QUERY_IDS,
            qvecs,
            base_qvecs=base_qvecs,
            baseline_top10=baseline_top10,
        )

        assert "query_doc_sim" in result
        assert "query_doc_top10_sim" in result
        assert "shift_cos" in result
        assert "shift_l2" in result
        assert "overlap_with_baseline" in result
        assert result["n_queries"] == 3.0


class TestEncodeQueriesElfRaw:
    """_encode_queries_elf_raw smoke test。"""

    def test_encodes_all_queries_without_enhance(self) -> None:
        """跳过增强, 仅走 encoder.encode() 路径。"""
        from experiments.diagnose_elf import _encode_queries_elf_raw

        ctx = MagicMock()
        ctx.elf_pipeline.encode.side_effect = lambda text: np.ones(768, dtype=np.float32)
        ctx.data.queries = {"q0": "hello", "q1": "world"}

        result = _encode_queries_elf_raw(ctx, ["q0", "q1"])

        assert len(result) == 2
        assert ctx.elf_pipeline.encode.call_count == 2
        # enhance 从未被调用（仅 encode）
        assert ctx.elf_pipeline.enhance.call_count == 0

    def test_asserts_when_elf_pipeline_is_none(self) -> None:
        """ctx.elf_pipeline 为 None 时 assert 触发。"""
        from experiments.diagnose_elf import _encode_queries_elf_raw

        ctx = MagicMock()
        ctx.elf_pipeline = None

        with pytest.raises(AssertionError, match="elf_pipeline"):
            _encode_queries_elf_raw(ctx, ["q0"])
