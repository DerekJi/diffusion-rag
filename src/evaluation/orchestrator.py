#!/usr/bin/env python3
"""评测编排器。

加载参数网格 YAML（12 组 ELF 参数 + Baseline 对照组），逐组调用
run_benchmark() 跑双链路评测并收集指标，最后交给 reporter 输出
CSV 对比表 + JSON 摘要。

用法::

    python -m src.evaluation.orchestrator --config experiments/configs/param_grid.yaml \\
        --dataset nfcorpus --sample 5
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import cast

import pandas as pd
import yaml

from src.baseline.benchmark import run_benchmark
from src.config import (
    DEFAULT_ELF_CFG_SCALE,
    DEFAULT_ELF_NOISE_T,
    DEFAULT_ELF_STEPS,
    DEFAULT_ENCODER,
    DEFAULT_GRID_CONFIG,
    DEFAULT_INDEX_NLIST,
    DEFAULT_K_VALUES,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_SEED,
    METHOD_BASELINE,
    METHOD_ELF,
    SUPPORTED_DATASETS,
)
from src.evaluation.reporter import write_json_summary, write_summary_csv
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


@dataclass
class ExperimentConfig:
    """参数网格配置。

    Attributes:
        dataset: 数据集名称。
        encoder: 文档侧编码器（双链路共享）。
        index_nlist: FAISS IVF 聚类中心数。
        k_values: 评估的 k 值列表。
        seed: 随机种子。
        sample: 仅取前 N 条 query 快速验证，None 为全量。
        output_dir: 输出根目录。
        elf_param_list: ELF 参数组列表，每组含 id/steps/noise_t/cfg_scale。
    """

    dataset: str = "nfcorpus"
    encoder: str = DEFAULT_ENCODER
    index_nlist: int = DEFAULT_INDEX_NLIST
    k_values: list[int] = field(default_factory=lambda: list(DEFAULT_K_VALUES))
    seed: int = DEFAULT_SEED
    sample: int | None = None
    output_dir: str = DEFAULT_OUTPUT_DIR
    elf_param_list: list[dict[str, object]] = field(default_factory=list)

    def __post_init__(self) -> None:
        """校验配置字段的基本合法性，fail-fast 避免深层传播。"""
        if not all(k > 0 for k in self.k_values):
            raise ValueError(f"k_values 必须全部为正整数，got {self.k_values}")
        if self.seed < 0:
            raise ValueError(f"seed 不能为负数，got {self.seed}")
        if self.index_nlist <= 0:
            raise ValueError(f"index_nlist 必须为正整数，got {self.index_nlist}")


def load_param_grid(path: str | Path) -> ExperimentConfig:
    """加载参数网格 YAML 配置。

    Args:
        path: YAML 配置文件路径。

    Returns:
        解析后的 ExperimentConfig。

    Raises:
        FileNotFoundError: 配置文件不存在。
        ValueError: 配置格式错误或 elf.param_list 为空 / 存在重复 id。
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with config_path.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"配置文件顶层必须是 mapping，got {type(data).__name__}")

    baseline_cfg = data.get("baseline") or {}
    elf_cfg = data.get("elf") or {}
    eval_cfg = data.get("eval") or {}

    encoder = str(baseline_cfg.get("encoder") or elf_cfg.get("encoder") or DEFAULT_ENCODER)

    param_list_raw = elf_cfg.get("param_list")
    if not isinstance(param_list_raw, list) or not param_list_raw:
        raise ValueError("elf.param_list 必须是非空列表")

    elf_param_list: list[dict[str, object]] = []
    seen_ids: set[str] = set()
    for item in param_list_raw:
        if not isinstance(item, dict):
            raise ValueError(f"param_list 元素必须是 mapping，got {type(item).__name__}")
        config_id = str(item.get("id", ""))
        if not config_id:
            raise ValueError("param_list 元素缺少 id 字段")
        if config_id in seen_ids:
            raise ValueError(f"param_list 存在重复的 id: {config_id}")
        seen_ids.add(config_id)
        elf_param_list.append(
            {
                "id": config_id,
                "steps": int(item.get("steps", DEFAULT_ELF_STEPS)),
                "noise_t": float(item.get("noise_t", DEFAULT_ELF_NOISE_T)),
                "cfg_scale": float(item.get("cfg_scale", DEFAULT_ELF_CFG_SCALE)),
            }
        )

    k_raw = eval_cfg.get("k_values") or DEFAULT_K_VALUES
    sample_raw = eval_cfg.get("sample")

    return ExperimentConfig(
        dataset=str(data.get("dataset", "nfcorpus")),
        encoder=encoder,
        index_nlist=int(eval_cfg.get("nlist", DEFAULT_INDEX_NLIST)),
        k_values=[int(k) for k in k_raw],
        seed=int(eval_cfg.get("seed", DEFAULT_SEED)),
        sample=int(sample_raw) if sample_raw is not None else None,
        output_dir=str(eval_cfg.get("output_dir", DEFAULT_OUTPUT_DIR)),
        elf_param_list=elf_param_list,
    )


def run_grid(config: ExperimentConfig, sample: int | None = None) -> pd.DataFrame:
    """遍历参数组，跑双链路评测并收集指标。

    先运行 1 次 Baseline 对照组，再依次运行每组 ELF 参数（调用
    run_benchmark），最后追加相对 baseline 的差值百分比列。

    Args:
        config: 参数网格配置。
        sample: 覆盖配置中的采样查询数，None 表示使用 config.sample。

    Returns:
        汇总 DataFrame，每组参数一行。

    Raises:
        ValueError: 汇总结果缺少 baseline 对照行。
    """
    set_seed(config.seed)
    effective_sample = config.sample if sample is None else sample

    rows: list[dict[str, object]] = []

    logger.info("[grid] 运行 Baseline 对照组")
    rows.append(_run_group(config, method=METHOD_BASELINE, sample=effective_sample))

    for params in config.elf_param_list:
        logger.info(
            "[grid] 运行 %s: steps=%s, noise_t=%s, cfg_scale=%s",
            params["id"],
            params["steps"],
            params["noise_t"],
            params["cfg_scale"],
        )
        rows.append(
            _run_group(config, method=METHOD_ELF, sample=effective_sample, elf_params=params)
        )

    summary = _add_baseline_deltas(pd.DataFrame(rows))
    logger.info(
        "[grid] 完成: %d 组 (1 baseline + %d ELF)", len(summary), len(config.elf_param_list)
    )
    return summary


def _run_group(
    config: ExperimentConfig,
    method: str,
    sample: int | None,
    elf_params: dict[str, object] | None = None,
) -> dict[str, object]:
    """运行单个参数组并返回指标行（含 config_id / avg_latency_ms / n_queries）。

    Args:
        config: 参数网格配置。
        method: 检索链路，'baseline' 或 'elf'。
        sample: 采样查询数，None 为全量。
        elf_params: ELF 参数组（id/steps/noise_t/cfg_scale），仅 method='elf' 需要。

    Returns:
        该组聚合指标字典，作为汇总表的一行。

    Notes:
        wall_clock_per_query_ms 为整组 run_benchmark 调用（数据集加载 +
        文档编码 + 建索引 + 检索）的墙钟耗时均摊到每条 query 的近似值，
        并非纯检索延迟。

    Raises:
        ValueError: method='elf' 但未提供 elf_params。
    """
    t0 = time.perf_counter()
    if method == METHOD_ELF:
        if elf_params is None:
            raise ValueError("method='elf' 时必须提供 elf_params")
        df = run_benchmark(
            dataset=config.dataset,
            method=method,
            encoder_name=config.encoder,
            index_nlist=config.index_nlist,
            k_values=config.k_values,
            seed=config.seed,
            output_dir=config.output_dir,
            sample=sample,
            elf_steps=cast(int, elf_params["steps"]),
            elf_noise_t=cast(float, elf_params["noise_t"]),
            elf_cfg_scale=cast(float, elf_params["cfg_scale"]),
        )
        config_id = str(elf_params["id"])
    else:
        df = run_benchmark(
            dataset=config.dataset,
            method=method,
            encoder_name=config.encoder,
            index_nlist=config.index_nlist,
            k_values=config.k_values,
            seed=config.seed,
            output_dir=config.output_dir,
            sample=sample,
        )
        config_id = "baseline"
    elapsed_s = time.perf_counter() - t0

    row = dict(df.iloc[0].to_dict())
    row["config_id"] = config_id
    row["method"] = method
    if method == METHOD_ELF:
        assert elf_params is not None
        row["steps"] = elf_params["steps"]
        row["noise_t"] = elf_params["noise_t"]
        row["cfg_scale"] = elf_params["cfg_scale"]

    n_queries = int(row.get("n_queries", 0))
    row["n_queries"] = n_queries
    row["wall_clock_per_query_ms"] = (
        round(elapsed_s * 1000.0 / n_queries, 2) if n_queries > 0 else 0.0
    )
    return row


def _add_baseline_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    """为汇总表追加相对 baseline 的差值百分比列。

    Args:
        summary: 含 baseline 与 ELF 行的汇总 DataFrame。

    Returns:
        追加 vs_baseline_* 差值列后的 DataFrame。

    Raises:
        ValueError: 汇总表缺少 baseline 对照行。
    """
    baseline = summary[summary["method"] == METHOD_BASELINE]
    if baseline.empty:
        raise ValueError("汇总表缺少 baseline 对照行，无法计算差值")
    base_recall10 = float(baseline["recall@10"].iloc[0])
    base_mrr = float(baseline["mrr"].iloc[0])
    base_ndcg10 = float(baseline["ndcg@10"].iloc[0])

    out = summary.copy()
    out["vs_baseline_recall@10_delta(%)"] = out["recall@10"].map(
        lambda v: _pct_delta(float(v), base_recall10)
    )
    out["vs_baseline_mrr_delta(%)"] = out["mrr"].map(lambda v: _pct_delta(float(v), base_mrr))
    out["vs_baseline_ndcg@10_delta(%)"] = out["ndcg@10"].map(
        lambda v: _pct_delta(float(v), base_ndcg10)
    )
    return out


def _pct_delta(value: float, base: float) -> float:
    """相对 baseline 的百分比差值；base 为 0 时返回 0.0。

    Args:
        value: 当前组指标值。
        base: baseline 指标值。

    Returns:
        (value - base) / base * 100；base 为 0 时返回 0.0。
    """
    if base == 0:
        return 0.0
    return (value - base) / base * 100.0


def _build_parser() -> argparse.ArgumentParser:
    """构建编排器命令行参数解析器。

    Returns:
        配置完成的 ArgumentParser。
    """
    parser = argparse.ArgumentParser(description="评测编排器: 遍历参数网格，跑双链路对比评测")
    parser.add_argument(
        "--config",
        default=DEFAULT_GRID_CONFIG,
        help=f"参数网格 YAML 配置路径（默认 {DEFAULT_GRID_CONFIG}）",
    )
    parser.add_argument(
        "--dataset",
        default=None,
        choices=list(SUPPORTED_DATASETS),
        help="覆盖配置中的数据集名称",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="覆盖配置中的采样 query 数（仅取前 N 条）",
    )
    parser.add_argument("--seed", type=int, default=None, help="覆盖配置中的随机种子")
    parser.add_argument("--output", default=None, help="覆盖配置中的输出目录")
    return parser


def main(argv: list[str] | None = None) -> int:
    """评测编排器 CLI 入口。

    Args:
        argv: 命令行参数列表，None 表示使用 sys.argv[1:]。

    Returns:
        进程退出码（0 表示成功）。
    """
    args = _build_parser().parse_args(argv)

    config = load_param_grid(args.config)
    if args.dataset is not None:
        config.dataset = args.dataset
    if args.sample is not None:
        config.sample = args.sample
    if args.seed is not None:
        config.seed = args.seed
    if args.output is not None:
        config.output_dir = args.output

    summary = run_grid(config)
    csv_path = write_summary_csv(summary, config.output_dir, config.dataset)
    json_path = write_json_summary(summary, config.output_dir, config.dataset)

    logger.info(
        "评测编排完成: dataset=%s → %s / %s",
        config.dataset,
        csv_path,
        json_path,
    )
    # print() is intentional: CLI stdout output for the result table
    print(summary.to_markdown(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
