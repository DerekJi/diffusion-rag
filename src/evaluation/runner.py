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

import sys

from src.baseline.benchmark import _build_parser, run_benchmark
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger(__name__)


def main(argv: list[str] | None = None) -> int:
    """统一评测入口。

    Args:
        argv: 命令行参数列表，None 表示使用 sys.argv[1:]。

    Returns:
        进程退出码（0 表示成功）。
    """
    args = _build_parser().parse_args(argv)

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
    sys.exit(main())
