"""数据集采样工具。

从原始数据集中按 qrels 取前 sample 条 query 及其引用的文档，
用于快速验证。build_benchmark_context 与 diagnose_elf 共用。
"""

from __future__ import annotations

from src.evaluation.dataset import DatasetTriple
from src.utils.logger import get_logger

logger = get_logger(__name__)


def sample_dataset(data: DatasetTriple, sample: int) -> DatasetTriple:
    """取前 sample 条有 qrels 的 query 及其引用的文档。

    Args:
        data: 原始数据集。
        sample: 目标 query 数。

    Returns:
        采样后的 DatasetTriple（若 sample >= 总 query 数则返回原数据）。
    """
    if sample >= len(data.queries):
        return data
    qids_with_qrels = sorted(q for q in data.queries if q in data.qrels)
    sampled_qids = qids_with_qrels[:sample]
    referenced: set[str] = set()
    for qid in sampled_qids:
        referenced.update(data.qrels[qid].keys())
    sampled = DatasetTriple(
        queries={qid: data.queries[qid] for qid in sampled_qids},
        corpus={did: data.corpus[did] for did in referenced if did in data.corpus},
        qrels={qid: data.qrels[qid] for qid in sampled_qids},
    )
    logger.info("采样模式: %d queries, %d docs", len(sampled.queries), len(sampled.corpus))
    return sampled
