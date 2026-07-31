#!/usr/bin/env python3
"""统一评测入口 CLI：Baseline / ELF 双链路一键切换。

基于 src.baseline.benchmark.run_benchmark()，通过 --method 选择检索链路，
文档侧编码与 FAISS 索引在两条链路上保持一致（共享 indexer/retriever）。

用法::

    # Baseline 链路（BGE 编码）
    python -m src.evaluation.runner --method baseline --dataset nfcorpus --sample 10

    # ELF 增强链路（扩散增强查询向量）
    python -m src.evaluation.runner --method elf --dataset nfcorpus --sample 10
    python -m src.evaluation.runner --method elf --dataset nfcorpus \\
        --steps 4 --noise-t 0.5 --cfg-scale 3.0
"""

import argparse

from src.baseline.benchmark import run_benchmark
from src.config import (
    DEFAULT_ELF_CFG_SCALE,
    DEFAULT_ELF_NOISE_T,
    DEFAULT_ELF_STEPS,
    DEFAULT_ENCODER,
    DEFAULT_INDEX_NLIST,
    DEFAULT_K_VALUES,
    DEFAULT_SEED,
    METHOD_BASELINE,
    SUPPORTED_METHODS,
)
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """构建统一评测入口的参数解析器。

    Returns:
        配置完成的 ArgumentParser。
    """
    parser = argparse.ArgumentParser(description="Baseline / ELF 双链路统一评测入口")
    parser.add_argument(
        "--method",
        choices=SUPPORTED_METHODS,
        default=METHOD_BASELINE,
        help="检索链路: baseline（BGE 编码）或 elf（ELF 扩散增强），一键切换",
    )
    parser.add_argument(
        "--dataset",
        default="nfcorpus",
        choices=["nfcorpus", "msmarco", "nq", "fiqa"],
        help="数据集名称",
    )
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, help="编码器模型名称（文档侧）")
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_K_VALUES, help="k 值列表")
    parser.add_argument("--nlist", type=int, default=DEFAULT_INDEX_NLIST, help="IVF 聚类中心数")
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="仅取前 N 条 query 快速测试（默认全量）",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    parser.add_argument("--output", default="experiments/outputs", help="输出目录")
    parser.add_argument("--steps", type=int, default=DEFAULT_ELF_STEPS, help="ELF 去噪步数")
    parser.add_argument("--noise-t", type=float, default=DEFAULT_ELF_NOISE_T, help="ELF 加噪强度 t")
    parser.add_argument(
        "--cfg-scale", type=float, default=DEFAULT_ELF_CFG_SCALE, help="ELF CFG 引导强度"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """统一评测入口。

    Args:
        argv: 命令行参数列表，None 表示使用 sys.argv[1:]。

    Returns:
        进程退出码（0 表示成功）。
    """
    args = build_parser().parse_args(argv)

    set_seed(args.seed)
    df = run_benchmark(
        dataset=args.dataset,
        method=args.method,
        encoder_name=args.encoder,
        index_nlist=args.nlist,
        k_values=list(args.k),
        seed=args.seed,
        output_dir=args.output,
        sample=args.sample,
        elf_steps=args.steps,
        elf_noise_t=args.noise_t,
        elf_cfg_scale=args.cfg_scale,
    )
    logger.info(
        "评测完成: dataset=%s, method=%s → 指标已保存到 %s/%s/%s.csv",
        args.dataset,
        args.method,
        args.output,
        args.dataset,
        args.method,
    )
    # print() is intentional: CLI stdout output for the result table
    print(df.to_markdown())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
