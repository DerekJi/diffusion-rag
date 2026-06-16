#!/usr/bin/env python3
"""基线评测 CLI 入口。

运行完整检索评测流程：加载数据集 → 编码 → 建索引 → 检索 → 计算指标 → 输出报告。
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.baseline.encoder import BaselineEncoder
from src.config import DEFAULT_ENCODER, DEFAULT_INDEX_NLIST, DEFAULT_K_VALUES, DEFAULT_SEED
from src.evaluation.dataset import load_dataset
from src.evaluation.metrics import compute_metrics_batch
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.vector_store.indexer import FAISSIndexer
from src.vector_store.retriever import Retriever

logger = get_logger(__name__)


def run_benchmark(
    dataset: str = "nfcorpus",
    encoder_name: str = DEFAULT_ENCODER,
    index_nlist: int = DEFAULT_INDEX_NLIST,
    k_values: list[int] | None = None,
    seed: int = DEFAULT_SEED,
    output_dir: str = "experiments/outputs",
) -> pd.DataFrame:
    """运行完整基线评测流程。

    流程:
        1. 固定随机种子
        2. 加载数据集
        3. 用 BaselineEncoder 编码所有文档 → 构建 FAISS 索引
        4. 用 BaselineEncoder 编码所有查询
        5. 逐条检索 + 计算指标
        6. 汇总结果并保存为 CSV

    Args:
        dataset: 数据集名称。
        encoder_name: HuggingFace 编码器名称。
        index_nlist: FAISS IVF 聚类中心数。
        k_values: 评估的 k 值列表。
        seed: 随机种子。
        output_dir: 输出目录。

    Returns:
        包含聚合指标的 DataFrame。
    """
    if k_values is None:
        k_values = DEFAULT_K_VALUES

    set_seed(seed)

    # 1. 加载数据集
    data = load_dataset(dataset)
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

    # 3. 编码查询 + 检索
    query_ids = sorted(data.queries.keys())
    logger.info("检索 %d 条查询...", len(query_ids))

    all_results: dict[str, list[str]] = {}
    for qid in tqdm(query_ids, desc="检索中"):
        qvec = encoder.encode(data.queries[qid])
        doc_ids_found, _ = retriever.search(qvec, k=max(k_values))
        all_results[qid] = doc_ids_found

    # 4. 计算指标
    logger.info("计算评测指标...")
    metrics = compute_metrics_batch(data.qrels, all_results, k_values)

    # 5. 聚合全部查询指标
    agg: dict[str, object] = {"dataset": dataset}
    for k in k_values:
        agg[f"recall@{k}"] = float(np.mean([m.recall[k] for m in metrics.values()]))
        agg[f"precision@{k}"] = float(np.mean([m.precision[k] for m in metrics.values()]))
        agg[f"ndcg@{k}"] = float(np.mean([m.ndcg[k] for m in metrics.values()]))
        agg[f"hit_rate@{k}"] = float(np.mean([m.hit_rate[k] for m in metrics.values()]))
    agg["mrr"] = float(np.mean([m.mrr for m in metrics.values()]))

    df = pd.DataFrame([agg])

    # 6. 保存
    out_path = Path(output_dir) / dataset
    out_path.mkdir(parents=True, exist_ok=True)
    csv_path = out_path / "baseline.csv"
    df.to_csv(csv_path, index=False)
    logger.info("结果已保存到 %s", csv_path)

    return df


def _main() -> None:
    parser = argparse.ArgumentParser(description="Baseline 检索评测")
    parser.add_argument(
        "--dataset",
        default="nfcorpus",
        choices=["nfcorpus", "msmarco", "nq", "fiqa"],
        help="数据集名称",
    )
    parser.add_argument("--encoder", default=DEFAULT_ENCODER, help="编码器模型名称")
    parser.add_argument("--k", type=int, nargs="+", default=DEFAULT_K_VALUES, help="k 值列表")
    parser.add_argument("--nlist", type=int, default=DEFAULT_INDEX_NLIST, help="IVF 聚类中心数")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="随机种子")
    parser.add_argument("--output", default="experiments/outputs", help="输出目录")
    args = parser.parse_args()

    set_seed(args.seed)
    df = run_benchmark(
        dataset=args.dataset,
        encoder_name=args.encoder,
        index_nlist=args.nlist,
        k_values=list(args.k),
        seed=args.seed,
        output_dir=args.output,
    )
    # print() is intentional: CLI stdout output for the result table
    print(df.to_markdown())


if __name__ == "__main__":
    _main()
