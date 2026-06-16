"""核心指标计算模块。

实现 Recall@k, Precision@k, MRR, NDCG@k, HitRate@k。
"""

import math
from dataclasses import dataclass, field


@dataclass
class MetricsResult:
    """单条查询的指标结果。

    Attributes:
        query_id: 查询 ID。
        recall: {k: recall@k}。
        precision: {k: precision@k}。
        mrr: 平均倒数排名。
        ndcg: {k: ndcg@k}。
        hit_rate: {k: hit_rate@k}。
    """

    query_id: str
    recall: dict[int, float] = field(default_factory=dict)
    precision: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    ndcg: dict[int, float] = field(default_factory=dict)
    hit_rate: dict[int, float] = field(default_factory=dict)


def compute_metrics(
    qrels: dict[str, int],
    results: list[str],
    k_values: list[int] | None = None,
    query_id: str = "",
) -> MetricsResult:
    """计算单条查询的全部检索指标。

    Args:
        qrels: {doc_id: relevance_score}，该查询的真实相关文档。
        results: 检索返回的 doc_id 列表（按距离升序，最相关在前）。
        k_values: 计算哪些 k 位置的指标。默认 [5, 10, 20]。

    Returns:
        MetricsResult 包含所有指标。

    Notes:
        - results 为空列表时，所有指标为 0.0。
        - qrels 中 relevance_score >= 1 视为相关。
    """
    if k_values is None:
        k_values = [5, 10, 20]

    # 相关文档集合
    relevant_docs = {doc_id for doc_id, score in qrels.items() if int(score) >= 1}
    total_relevant = len(relevant_docs)

    result = MetricsResult(query_id=query_id)

    # 空结果处理
    if not results:
        for k in k_values:
            result.recall[k] = 0.0
            result.precision[k] = 0.0
            result.ndcg[k] = 0.0
            result.hit_rate[k] = 0.0
        result.mrr = 0.0
        return result

    # MRR
    first_rank: int | None = None
    for rank, doc_id in enumerate(results, start=1):
        if doc_id in relevant_docs:
            first_rank = rank
            break
    result.mrr = 1.0 / first_rank if first_rank is not None else 0.0

    # 每个 k 的指标
    for k in k_values:
        top_k = results[:k]
        relevant_in_top_k = sum(1 for d in top_k if d in relevant_docs)

        # Recall@k
        result.recall[k] = relevant_in_top_k / total_relevant if total_relevant > 0 else 0.0

        # Precision@k
        result.precision[k] = relevant_in_top_k / k

        # HitRate@k
        result.hit_rate[k] = 1.0 if relevant_in_top_k > 0 else 0.0

        # NDCG@k
        true_relevances = [int(qrels.get(d, 0)) for d in top_k]
        ideal_order = sorted(qrels.values(), reverse=True)[:k]
        dcg_val = _dcg(true_relevances)
        idcg_val = _dcg(ideal_order)
        result.ndcg[k] = dcg_val / idcg_val if idcg_val > 0 else 0.0

    return result


def compute_metrics_batch(
    all_qrels: dict[str, dict[str, int]],
    all_results: dict[str, list[str]],
    k_values: list[int] | None = None,
) -> dict[str, MetricsResult]:
    """批量计算指标。

    Args:
        all_qrels: {query_id: {doc_id: relevance_score}}。
        all_results: {query_id: [doc_id, ...]}。
        k_values: 计算哪些 k 位置。

    Returns:
        {query_id: MetricsResult}。
    """
    if k_values is None:
        k_values = [5, 10, 20]

    results: dict[str, MetricsResult] = {}
    for qid in all_results:
        qrels = all_qrels.get(qid, {})
        results[qid] = compute_metrics(qrels, all_results[qid], k_values, query_id=qid)
    return results


def _dcg(relevances: list[int]) -> float:
    """计算 DCG（Discounted Cumulative Gain）。

    使用 TREC 标准公式:
        DCG = rel_1 + Σ_{i=2..n} rel_i / log₂(i)

    该公式将排名 1 的文档不做折扣，从排名 2 开始按 log₂(rank) 折损。
    与另一种常见定义 Σ rel_i / log₂(i+1) 相比，区别仅在于排名偏移。
    TREC 官方评测和 BEIR 基准均使用本实现。

    Args:
        relevances: 相关性分数列表，从最高排名开始。

    Returns:
        DCG 值。
    """
    if not relevances:
        return 0.0

    dcg = float(relevances[0])
    for i, rel in enumerate(relevances[1:], start=2):
        dcg += rel / math.log2(i)
    return dcg
