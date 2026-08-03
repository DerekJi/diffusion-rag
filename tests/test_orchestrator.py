"""评测编排器 + 报表生成器单元测试。

mock run_benchmark / load_param_grid / build_benchmark_context 等外部依赖，
无需网络与模型下载。
"""

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.config import METHOD_BASELINE, METHOD_ELF
from src.evaluation.orchestrator import ExperimentConfig, load_param_grid, main, run_grid
from src.evaluation.reporter import write_json_summary, write_summary_csv

_REPO_ROOT = Path(__file__).resolve().parents[1]
_GRID_YAML = _REPO_ROOT / "experiments" / "configs" / "param_grid.yaml"


@contextmanager
def _patched_benchmark(
    side_effect: list[pd.DataFrame] | None = None,
    return_value: pd.DataFrame | None = None,
) -> Iterator[tuple[MagicMock, MagicMock]]:
    """同时 mock build_benchmark_context 与 run_benchmark。

    run_grid 现在会先构建共享上下文（真实实现会加载数据集 + 编码文档），
    测试中必须一并 mock，避免触发模型加载。返回 (mock_ctx, mock_run_benchmark)。
    """
    mock_ctx = MagicMock()
    with (
        patch(
            "src.evaluation.orchestrator.build_benchmark_context",
            return_value=mock_ctx,
        ),
        patch(
            "src.evaluation.orchestrator.run_benchmark",
            side_effect=side_effect,
            return_value=return_value,
        ) as mock_rb,
    ):
        yield mock_ctx, mock_rb


def _fake_summary_df(method: str = METHOD_BASELINE, **extra: object) -> pd.DataFrame:
    """构造 run_benchmark 返回的单行聚合指标 DataFrame。"""
    row: dict[str, object] = {
        "dataset": "nfcorpus",
        "method": method,
        "recall@5": 0.4,
        "recall@10": 0.5,
        "recall@20": 0.6,
        "precision@5": 0.1,
        "precision@10": 0.1,
        "precision@20": 0.1,
        "ndcg@5": 0.3,
        "ndcg@10": 0.4,
        "ndcg@20": 0.45,
        "hit_rate@5": 0.6,
        "hit_rate@10": 0.7,
        "hit_rate@20": 0.75,
        "mrr": 0.25,
        "n_queries": 5,
    }
    row.update(extra)
    return pd.DataFrame([row])


def _make_config(tmp_path: Path, n_groups: int = 2) -> ExperimentConfig:
    """构造小型参数网格配置（n_groups 组 ELF 参数）。"""
    return ExperimentConfig(
        dataset="nfcorpus",
        encoder="BAAI/bge-base-en-v1.5",
        index_nlist=100,
        k_values=[5, 10, 20],
        seed=42,
        sample=5,
        output_dir=str(tmp_path),
        elf_param_list=[
            {"id": f"ELF-0{i + 1}", "steps": steps, "noise_t": 0.3, "cfg_scale": 1.0}
            for i, steps in enumerate([1, 2][:n_groups])
        ],
    )


class TestLoadParamGrid:
    """load_param_grid 配置解析测试。"""

    def test_loads_12_groups_from_repo_yaml(self) -> None:
        """仓库内的 param_grid.yaml 应解析出 12 组参数 + 默认 eval 配置。"""
        config = load_param_grid(_GRID_YAML)
        assert config.dataset == "nfcorpus"
        assert config.seed == 42
        assert config.k_values == [5, 10, 20]
        assert config.sample is None
        assert len(config.elf_param_list) == 12

    def test_first_mid_last_groups(self) -> None:
        """首/中/尾三组参数与计划文档 §3.2 表格一致。"""
        config = load_param_grid(_GRID_YAML)
        assert config.elf_param_list[0] == {
            "id": "ELF-01",
            "steps": 1,
            "noise_t": 0.3,
            "cfg_scale": 1.0,
        }
        assert config.elf_param_list[6] == {
            "id": "ELF-07",
            "steps": 2,
            "noise_t": 0.3,
            "cfg_scale": 2.0,
        }
        assert config.elf_param_list[11] == {
            "id": "ELF-12",
            "steps": 4,
            "noise_t": 0.6,
            "cfg_scale": 3.0,
        }

    def test_missing_file_raises(self) -> None:
        """配置文件不存在时抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError):
            load_param_grid("does_not_exist.yaml")

    def test_invalid_top_level_raises(self, tmp_path: Path) -> None:
        """顶层不是 mapping 时抛出 ValueError。"""
        bad = tmp_path / "bad.yaml"
        bad.write_text("- a\n- b\n", encoding="utf-8")
        with pytest.raises(ValueError, match="mapping"):
            load_param_grid(bad)

    def test_empty_param_list_raises(self, tmp_path: Path) -> None:
        """elf.param_list 为空时抛出 ValueError。"""
        bad = tmp_path / "empty.yaml"
        bad.write_text("dataset: nfcorpus\nelf:\n  param_list: []\n", encoding="utf-8")
        with pytest.raises(ValueError, match="非空列表"):
            load_param_grid(bad)

    def test_duplicate_ids_raise(self, tmp_path: Path) -> None:
        """param_list 存在重复 id 时抛出 ValueError。"""
        bad = tmp_path / "dup.yaml"
        bad.write_text(
            "elf:\n  param_list:\n"
            "    - {id: X, steps: 1, noise_t: 0.3, cfg_scale: 1.0}\n"
            "    - {id: X, steps: 2, noise_t: 0.4, cfg_scale: 2.0}\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="重复"):
            load_param_grid(bad)


class TestRunGrid:
    """run_grid 编排遍历测试。"""

    def test_runs_baseline_then_each_elf_group(self, tmp_path: Path) -> None:
        """自动遍历 1 baseline + 每组 ELF 参数，参数正确透传。"""
        config = _make_config(tmp_path, n_groups=2)
        with _patched_benchmark(
            side_effect=[
                _fake_summary_df(method=METHOD_BASELINE),
                _fake_summary_df(method=METHOD_ELF, steps=1, noise_t=0.3, cfg_scale=1.0),
                _fake_summary_df(method=METHOD_ELF, steps=2, noise_t=0.3, cfg_scale=1.0),
            ]
        ) as (mock_ctx, mock_rb):
            summary = run_grid(config)

        assert mock_rb.call_count == 3
        calls = mock_rb.call_args_list
        assert calls[0].kwargs["method"] == METHOD_BASELINE
        assert calls[0].kwargs["sample"] == 5
        assert calls[1].kwargs["method"] == METHOD_ELF
        assert calls[1].kwargs["elf_steps"] == 1
        assert calls[1].kwargs["elf_noise_t"] == 0.3
        assert calls[1].kwargs["elf_cfg_scale"] == 1.0
        assert calls[2].kwargs["elf_steps"] == 2

        assert len(summary) == 3
        assert summary["config_id"].tolist() == ["baseline", "ELF-01", "ELF-02"]
        assert summary["method"].tolist() == [
            METHOD_BASELINE,
            METHOD_ELF,
            METHOD_ELF,
        ]

    def test_builds_context_once_per_method_and_shares_it(self, tmp_path: Path) -> None:
        """baseline/elf 各构建一次上下文,并按链路分别透传(issue #33)。"""
        config = _make_config(tmp_path, n_groups=2)
        base_ctx = MagicMock()
        elf_ctx = MagicMock()
        with (
            patch(
                "src.evaluation.orchestrator.build_benchmark_context",
                side_effect=[base_ctx, elf_ctx],
            ) as mock_build,
            patch(
                "src.evaluation.orchestrator.run_benchmark",
                side_effect=[
                    _fake_summary_df(method=METHOD_BASELINE),
                    _fake_summary_df(method=METHOD_ELF, steps=1, noise_t=0.3, cfg_scale=1.0),
                    _fake_summary_df(method=METHOD_ELF, steps=2, noise_t=0.3, cfg_scale=1.0),
                ],
            ) as mock_rb,
        ):
            run_grid(config)

        # 文档编码 + 建索引每条链路各做一遍(baseline → BGE, elf → ELF)
        assert mock_build.call_count == 2
        assert mock_build.call_args_list[0].kwargs["method"] == METHOD_BASELINE
        assert mock_build.call_args_list[1].kwargs["method"] == METHOD_ELF
        calls = mock_rb.call_args_list
        assert calls[0].kwargs["shared"] is base_ctx  # baseline 组用 BGE 文档库
        assert all(call.kwargs["shared"] is elf_ctx for call in calls[1:])  # ELF 组用 ELF 文档库

    def test_sample_argument_overrides_config(self, tmp_path: Path) -> None:
        """显式传入的 sample 覆盖 config.sample。"""
        config = _make_config(tmp_path, n_groups=1)
        with _patched_benchmark(
            return_value=_fake_summary_df(method=METHOD_BASELINE),
        ) as (_, mock_rb):
            run_grid(config, sample=2)
        assert mock_rb.call_args_list[0].kwargs["sample"] == 2

    def test_summary_contains_baseline_deltas(self, tmp_path: Path) -> None:
        """差值百分比列按 baseline 正确计算。"""
        config = _make_config(tmp_path, n_groups=1)
        base = _fake_summary_df(method=METHOD_BASELINE)  # recall@10=0.5, mrr=0.25, ndcg@10=0.4
        elf = _fake_summary_df(method=METHOD_ELF, steps=1, noise_t=0.3, cfg_scale=1.0)
        elf.loc[0, "recall@10"] = 0.55
        elf.loc[0, "mrr"] = 0.30
        elf.loc[0, "ndcg@10"] = 0.44
        with _patched_benchmark(side_effect=[base, elf]):
            summary = run_grid(config)

        assert summary["vs_baseline_recall@10_delta(%)"].tolist() == pytest.approx([0.0, 10.0])
        assert summary["vs_baseline_mrr_delta(%)"].tolist() == pytest.approx([0.0, 20.0])
        assert summary["vs_baseline_ndcg@10_delta(%)"].tolist() == pytest.approx([0.0, 10.0])

    def test_wall_clock_and_query_count_recorded(self, tmp_path: Path) -> None:
        """n_queries 与 wall_clock_per_query_ms 两列存在且为有限值。"""
        config = _make_config(tmp_path, n_groups=1)
        with _patched_benchmark(
            side_effect=[
                _fake_summary_df(method=METHOD_BASELINE),
                _fake_summary_df(method=METHOD_ELF, steps=1, noise_t=0.3, cfg_scale=1.0),
            ],
        ):
            summary = run_grid(config)

        assert summary["n_queries"].tolist() == [5, 5]
        assert summary["wall_clock_per_query_ms"].isna().sum() == 0

    def test_missing_baseline_raises(self) -> None:
        """汇总缺少 baseline 行时抛出 ValueError。"""
        from src.evaluation.orchestrator import _add_baseline_deltas

        df = pd.DataFrame([{"method": METHOD_ELF, "recall@10": 0.5, "mrr": 0.2, "ndcg@10": 0.3}])
        with pytest.raises(ValueError, match="baseline"):
            _add_baseline_deltas(df)


class TestReporter:
    """reporter 报表输出测试。"""

    def _run_summary(self, tmp_path: Path) -> pd.DataFrame:
        """通过 run_grid 构造含 baseline + 1 组 ELF 的汇总表。"""
        config = _make_config(tmp_path, n_groups=1)
        with _patched_benchmark(
            side_effect=[
                _fake_summary_df(method=METHOD_BASELINE),
                _fake_summary_df(method=METHOD_ELF, steps=1, noise_t=0.3, cfg_scale=1.0),
            ],
        ):
            return run_grid(config)

    def test_write_summary_csv(self, tmp_path: Path) -> None:
        """CSV 对比表落盘，含全部指标与差值列。"""
        summary = self._run_summary(tmp_path)
        csv_path = write_summary_csv(summary, str(tmp_path), "nfcorpus")

        assert csv_path.exists()
        assert csv_path.name == "summary.csv"
        loaded = pd.read_csv(csv_path)
        assert list(loaded.columns) == list(summary.columns)
        assert "config_id" in loaded.columns
        assert "vs_baseline_recall@10_delta(%)" in loaded.columns
        assert loaded["config_id"].tolist() == ["baseline", "ELF-01"]

    def test_write_json_summary(self, tmp_path: Path) -> None:
        """JSON 摘要含 baseline、最佳 ELF 参数组与全部明细。"""
        summary = self._run_summary(tmp_path)
        json_path = write_json_summary(summary, str(tmp_path), "nfcorpus")

        assert json_path.exists()
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["dataset"] == "nfcorpus"
        assert data["n_groups"] == 2
        assert data["baseline"]["recall@10"] == pytest.approx(0.5)
        assert data["best_elf"]["config_id"] == "ELF-01"
        assert len(data["groups"]) == 2

    def test_best_elf_is_max_recall10(self, tmp_path: Path) -> None:
        """best_elf 选取 recall@10 最高的 ELF 组。"""
        config = _make_config(tmp_path, n_groups=2)
        low = _fake_summary_df(method=METHOD_ELF, steps=1, noise_t=0.3, cfg_scale=1.0)
        high = _fake_summary_df(method=METHOD_ELF, steps=2, noise_t=0.4, cfg_scale=2.0)
        high.loc[0, "recall@10"] = 0.7
        with _patched_benchmark(
            side_effect=[
                _fake_summary_df(method=METHOD_BASELINE),
                low,
                high,
            ],
        ):
            summary = run_grid(config)
        json_path = write_json_summary(summary, str(tmp_path), "nfcorpus")
        data = json.loads(json_path.read_text(encoding="utf-8"))
        assert data["best_elf"]["config_id"] == "ELF-02"
        assert data["best_elf"]["recall@10"] == pytest.approx(0.7)


class TestMainCLI:
    """编排器 CLI 入口测试。"""

    def test_main_passes_overrides_and_writes_outputs(self, tmp_path: Path) -> None:
        """CLI 参数覆盖配置并调用 reporter 落盘。"""
        config = _make_config(tmp_path, n_groups=1)
        summary = pd.DataFrame([{"config_id": "baseline", "method": "baseline", "recall@10": 0.5}])
        with (
            patch("src.evaluation.orchestrator.load_param_grid", return_value=config),
            patch("src.evaluation.orchestrator.run_grid", return_value=summary) as mock_run,
            patch("src.evaluation.orchestrator.write_summary_csv") as mock_csv,
            patch("src.evaluation.orchestrator.write_json_summary") as mock_json,
        ):
            code = main(
                [
                    "--config",
                    str(_GRID_YAML),
                    "--dataset",
                    "nfcorpus",
                    "--sample",
                    "5",
                    "--seed",
                    "7",
                ]
            )

        assert code == 0
        assert config.dataset == "nfcorpus"
        assert config.sample == 5
        assert config.seed == 7
        mock_run.assert_called_once_with(config)
        mock_csv.assert_called_once_with(summary, config.output_dir, "nfcorpus")
        mock_json.assert_called_once_with(summary, config.output_dir, "nfcorpus")

    def test_main_keeps_config_defaults(self, tmp_path: Path) -> None:
        """未传 --dataset/--sample 时保留配置中的默认值。"""
        config = _make_config(tmp_path, n_groups=1)
        with (
            patch("src.evaluation.orchestrator.load_param_grid", return_value=config),
            patch("src.evaluation.orchestrator.run_grid", return_value=pd.DataFrame()),
            patch("src.evaluation.orchestrator.write_summary_csv"),
            patch("src.evaluation.orchestrator.write_json_summary"),
        ):
            code = main(["--config", str(_GRID_YAML)])

        assert code == 0
        assert config.dataset == "nfcorpus"
        assert config.sample == 5
        assert config.seed == 42
