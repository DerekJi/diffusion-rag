#!/usr/bin/env python3
"""基线评测 CLI 入口。

运行完整检索评测流程：加载数据集 → 编码 → 建索引 → 检索 → 计算指标 → 输出报告。

Phase 3.1 起支持双链路一键切换（仅替换查询编码方式）:
  - method="baseline": 查询用 BaselineEncoder 编码（BGE，无扩散增强）
  - method="elf":      查询用 ELFPipeline.enhance() 扩散增强

文档侧编码（BGE）与 FAISS 索引在两条链路上保持一致（共享 indexer/retriever）。
"""

import argparse
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm import tqdm

from src.baseline.encoder import BaselineEncoder
from src.config import (
    DEFAULT_ELF_CFG_SCALE,
    DEFAULT_ELF_NOISE_T,
    DEFAULT_ELF_STEPS,
    DEFAULT_ENCODER,
    DEFAULT_INDEX_NLIST,
    DEFAULT_K_VALUES,
    DEFAULT_SEED,
    METHOD_BASELINE,
    METHOD_ELF,
    SUPPORTED_METHODS,
)
from src.elf.pipeline import ELFPipeline
from src.evaluation.dataset import DatasetTriple, load_dataset
from src.evaluation.metrics import compute_metrics_batch
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.vector_store.indexer import FAISSIndexer
from src.vector_store.retriever import Retriever

logger = get_logger(__name__)


def run_benchmark(
    dataset: str = "nfcorpus",
    method: str = METHOD_BASELINE,
    encoder_name: str = DEFAULT_ENCODER,
    index_nlist: int = DEFAULT_INDEX_NLIST,
    k_values: list[int] | None = None,
    seed: int = DEFAULT_SEED,
    output_dir: str = "experiments/outputs",
    sample: int | None = None,
    elf_steps: int = DEFAULT_ELF_STEPS,
    elf_noise_t: float = DEFAULT_ELF_NOISE_T,
    elf_cfg_scale: float = DEFAULT_ELF_CFG_SCALE,
) -> pd.DataFrame:
    """运行完整检索评测流程（Baseline / ELF 双链路）。

    流程:
        1. 固定随机种子
        2. 加载数据集
        3. 用 BaselineEncoder 编码所有文档 → 构建 FAISS 索引（双链路共享）
        4. 按 method 编码所有查询（仅此步切换链路）
        5. 逐条检索 + 计算指标
        6. 汇总结果并保存为 CSV（baseline.csv / elf.csv）

    Args:
        dataset: 数据集名称。
        method: 检索链路，'baseline'（BGE 编码）或 'elf'（ELF 扩散增强）。
        encoder_name: HuggingFace 编码器名称（文档侧，两条链路共用）。
        index_nlist: FAISS IVF 聚类中心数。
        k_values: 评估的 k 值列表。
        seed: 随机种子。
        output_dir: 输出目录。
        sample: 仅取前 N 条有 qrels 的 query 快速验证，None 为全量。
        elf_steps: ELF 去噪步数（仅 method='elf' 生效）。
        elf_noise_t: ELF 加噪强度 t ∈ [0, 1]（仅 method='elf' 生效）。
        elf_cfg_scale: ELF CFG 引导强度（仅 method='elf' 生效）。

    Returns:
        包含聚合指标的 DataFrame（一行，列含 dataset/method 及各 k 指标）。

    Raises:
        ValueError: method 不在 SUPPORTED_METHODS 中。
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES
    if method not in SUPPORTED_METHODS:
        supported = ", ".join(SUPPORTED_METHODS)
        raise ValueError(f"不支持的链路 method='{method}'，支持: {supported}")

    set_seed(seed)

    # 1. 加载数据集
    data = load_dataset(dataset)

    # 采样模式：取前 sample 条有 qrels 的 query 及其相关文档
    # 使用新变量 sample_data，不修改原始 data
    sample_data = data
    if sample is not None and sample < len(data.queries):
        qids_with_qrels = sorted(q for q in data.queries if q in data.qrels)
        sampled_qids = qids_with_qrels[:sample]
        referenced: set[str] = set()
        for qid in sampled_qids:
            referenced.update(data.qrels[qid].keys())
        sample_data = DatasetTriple(
            queries={qid: data.queries[qid] for qid in sampled_qids},
            corpus={did: data.corpus[did] for did in referenced if did in data.corpus},
            qrels={qid: data.qrels[qid] for qid in sampled_qids},
        )
        data = sample_data
        logger.info("采样模式: %d queries, %d docs", len(data.queries), len(data.corpus))

    logger.info("数据集 %s: %d queries, %d docs", dataset, len(data.queries), len(data.corpus))

    # 2. 编码文档 + 建索引
    encoder = BaselineEncoder(model_name=encoder_name)
    doc_ids = sorted(data.corpus.keys())
    doc_texts = [data.corpus[did] for did in doc_ids]

    logger.info("编码 %d 篇文档...", len(doc_texts))
    doc_vectors = encoder.encode_batch(doc_texts)

    indexer = FAISSIndexer(dimension=768, nlist=index_nlist)
    indexer.build(doc_vectors, doc_ids)
    retriever = Retriever(indexer)

    # 3. 编码查询 + 检索（仅此处按 method 切换链路）
    query_ids = sorted(data.queries.keys())

    query_encoder: Callable[[str], NDArray[np.float32]]
    if method == METHOD_ELF:
        elf_pipeline = ELFPipeline()
        rng = np.random.default_rng(seed)
        logger.info(
            "ELF 增强链路已加载 (steps=%d, noise_t=%.2f, cfg_scale=%.1f)",
            elf_steps,
            elf_noise_t,
            elf_cfg_scale,
        )

        def _elf_query_encode(text: str) -> NDArray[np.float32]:
            """ELF 链路查询编码：编码 → 加噪 → 去噪 → CFG 引导。"""
            return elf_pipeline.enhance(
                text,
                steps=elf_steps,
                noise_t=elf_noise_t,
                cfg_scale=elf_cfg_scale,
                rng=rng,
            )

        query_encoder = _elf_query_encode
    else:
        query_encoder = encoder.encode

    logger.info("检索 %d 条查询 (method=%s)...", len(query_ids), method)
    all_results: dict[str, list[str]] = {}
    for qid in tqdm(query_ids, desc="检索中"):
        qvec = query_encoder(data.queries[qid])
        doc_ids_found, _ = retriever.search(qvec, k=max(k_values))
        all_results[qid] = doc_ids_found

    # 4. 计算指标
    logger.info("计算评测指标...")
    metrics = compute_metrics_batch(data.qrels, all_results, k_values)

    # 5. 聚合全部查询指标
    agg: dict[str, object] = {"dataset": dataset, "method": method}
    if method == METHOD_ELF:
        agg["steps"] = elf_steps
        agg["noise_t"] = elf_noise_t
        agg["cfg_scale"] = elf_cfg_scale
    for k in k_values:
        agg[f"recall@{k}"] = float(np.mean([m.recall[k] for m in metrics.values()]))
        agg[f"precision@{k}"] = float(np.mean([m.precision[k] for m in metrics.values()]))
        agg[f"ndcg@{k}"] = float(np.mean([m.ndcg[k] for m in metrics.values()]))
        agg[f"hit_rate@{k}"] = float(np.mean([m.hit_rate[k] for m in metrics.values()]))
    agg["mrr"] = float(np.mean([m.mrr for m in metrics.values()]))

    df = pd.DataFrame([agg])

    # 6. 保存（按链路区分文件名: baseline.csv / elf.csv）
    out_path = Path(output_dir) / dataset
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / f"{method}.csv"
    df.to_csv(csv_path, index=False)
    logger.info("结果已保存到 %s", csv_path)

    return df


def _main() -> None:
    parser = argparse.ArgumentParser(description="Baseline / ELF 双链路检索评测")
    parser.add_argument(
        "--dataset",
        default="nfcorpus",
        choices=["nfcorpus", "msmarco", "nq", "fiqa"],
        help="数据集名称",
    )
    parser.add_argument(
        "--method",
        choices=SUPPORTED_METHODS,
        default=METHOD_BASELINE,
        help="检索链路: baseline（BGE 编码）或 elf（ELF 扩散增强）",
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
    args = parser.parse_args()

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
    # print() is intentional: CLI stdout output for the result table
    print(df.to_markdown())


if __name__ == "__main__":
    _main()
