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
from src.baseline.benchmark import run_benchmark
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

        self._elf_pipeline_mock = MagicMock()
        self._elf_pipeline_mock.enhance.return_value = fixed_vec

        patchers = [
            patch("src.baseline.benchmark.load_dataset", return_value=_make_fake_data()),
            patch("src.baseline.encoder.SentenceTransformer", return_value=doc_encoder),
            patch("src.baseline.benchmark.ELFPipeline", return_value=self._elf_pipeline_mock),
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
