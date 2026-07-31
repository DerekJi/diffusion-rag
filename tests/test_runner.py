"""src.evaluation.runner CLI 单元测试。"""

from collections.abc import Iterator
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.baseline.benchmark import _build_parser
from src.config import METHOD_BASELINE, METHOD_ELF
from src.evaluation.runner import main


@pytest.fixture
def _mock_run_benchmark() -> Iterator[MagicMock]:
    """mock run_benchmark，返回一行聚合指标。"""
    df = pd.DataFrame(
        [{"dataset": "nfcorpus", "method": "baseline", "recall@10": 0.5, "mrr": 0.25}]
    )
    with patch("src.evaluation.runner.run_benchmark", return_value=df) as mock_rb:
        yield mock_rb


class TestRunnerCLI:
    """统一评测入口 CLI 测试。"""

    def test_main_baseline_passthrough(self, _mock_run_benchmark: MagicMock) -> None:
        """--method baseline 正确透传到 run_benchmark。"""
        code = main(["--method", "baseline", "--dataset", "nfcorpus", "--sample", "2"])
        assert code == 0
        kwargs = _mock_run_benchmark.call_args.kwargs
        assert kwargs["method"] == METHOD_BASELINE
        assert kwargs["dataset"] == "nfcorpus"
        assert kwargs["sample"] == 2

    def test_main_elf_params_passthrough(self, _mock_run_benchmark: MagicMock) -> None:
        """ELF 增强参数（steps/noise-t/cfg-scale）正确透传。"""
        code = main(["--method", "elf", "--steps", "4", "--noise-t", "0.5", "--cfg-scale", "3.0"])
        assert code == 0
        kwargs = _mock_run_benchmark.call_args.kwargs
        assert kwargs["method"] == METHOD_ELF
        assert kwargs["elf_steps"] == 4
        assert kwargs["elf_noise_t"] == 0.5
        assert kwargs["elf_cfg_scale"] == 3.0

    def test_default_method_is_baseline(self, _mock_run_benchmark: MagicMock) -> None:
        """不传 --method 时默认 baseline。"""
        main([])
        assert _mock_run_benchmark.call_args.kwargs["method"] == METHOD_BASELINE

    def test_invalid_method_exits(self) -> None:
        """非法 method 触发 argparse SystemExit。"""
        with pytest.raises(SystemExit):
            main(["--method", "invalid"])

    def test_build_parser_elf_args(self) -> None:
        """参数解析器包含 ELF 专属参数。"""
        args = _build_parser().parse_args(
            ["--method", "elf", "--steps", "4", "--noise-t", "0.5", "--cfg-scale", "3.0"]
        )
        assert args.method == METHOD_ELF
        assert args.steps == 4
        assert args.noise_t == 0.5
        assert args.cfg_scale == 3.0
