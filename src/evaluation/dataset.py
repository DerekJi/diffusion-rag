"""数据集加载模块。

支持 BEIR 格式数据集的加载与格式转换。
"""

import time
from dataclasses import dataclass, field

from datasets import load_dataset as hf_load_dataset

from src.utils.logger import get_logger

logger = get_logger(__name__)

# HuggingFace 数据集映射
_DATASET_MAP: dict[str, str] = {
    "nfcorpus": "BeIR/nfcorpus",
    "msmarco": "BeIR/msmarco",
    "nq": "BeIR/nq",
    "fiqa": "BeIR/fiqa",
}

# 最大重试次数
_MAX_RETRIES = 3
_RETRY_DELAY = 5.0  # 秒


@dataclass
class DatasetTriple:
    """数据集三元组。

    Attributes:
        queries: {query_id: query_text}。
        corpus: {doc_id: doc_text}。
        qrels: {query_id: {doc_id: relevance_score}}。
    """

    queries: dict[str, str] = field(default_factory=dict)
    corpus: dict[str, str] = field(default_factory=dict)
    qrels: dict[str, dict[str, int]] = field(default_factory=dict)


def load_dataset(name: str, cache_dir: str | None = None) -> DatasetTriple:
    """加载 BEIR 格式数据集。

    Args:
        name: 数据集名称，支持 'nfcorpus', 'msmarco', 'nq', 'fiqa'。
        cache_dir: HuggingFace 数据集缓存路径，None 使用默认。

    Returns:
        DatasetTriple (queries, corpus, qrels)。

    Raises:
        ValueError: 不支持的数据集名称。
        RuntimeError: 下载失败（含网络错误）。
    """
    if name not in _DATASET_MAP:
        supported = ", ".join(_DATASET_MAP.keys())
        raise ValueError(f"不支持的数据集 '{name}'，支持: {supported}")

    hf_name = _DATASET_MAP[name]
    logger.info("加载数据集 %s (HuggingFace: %s)", name, hf_name)

    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            ds = hf_load_dataset(hf_name, cache_dir=cache_dir, trust_remote_code=True)
            break
        except Exception as e:
            last_error = e
            logger.warning("数据集下载失败 (尝试 %d/%d): %s", attempt, _MAX_RETRIES, e)
            if attempt < _MAX_RETRIES:
                time.sleep(_RETRY_DELAY)
    else:
        raise RuntimeError(f"数据集 {hf_name} 下载失败，已重试 {_MAX_RETRIES} 次: {last_error}")

    # 转换为统一格式
    triple = DatasetTriple()

    # 查询
    for split in (
        ds.get("queries", {}).values()
        if hasattr(ds.get("queries", {}), "values")
        else [ds["queries"]] if "queries" in ds else []
    ):
        for row in split:
            triple.queries[row["_id"]] = row["text"]

    # 语料
    for split in (
        ds.get("corpus", {}).values()
        if hasattr(ds.get("corpus", {}), "values")
        else [ds["corpus"]] if "corpus" in ds else []
    ):
        for row in split:
            triple.corpus[row["_id"]] = row["text"]

    # 相关性标注
    for split in (
        ds.get("qrels", {}).values()
        if hasattr(ds.get("qrels", {}), "values")
        else [ds["qrels"]] if "qrels" in ds else []
    ):
        for row in split:
            qid = row["query-id"]
            did = row["corpus-id"]
            score = row["score"]
            if qid not in triple.qrels:
                triple.qrels[qid] = {}
            triple.qrels[qid][did] = int(score)

    logger.info(
        "  queries=%d, corpus=%d, qrels=%d",
        len(triple.queries),
        len(triple.corpus),
        sum(len(v) for v in triple.qrels.values()),
    )
    return triple


# Phase 2/3: consumed by fast debug test loops
def load_small_debug() -> DatasetTriple:
    """从 NFCorpus 取子集用于快速调试。

    Returns:
        DatasetTriple 包含 50 条 query + 100 条 doc。
    """
    full = load_dataset("nfcorpus")

    query_ids = sorted(full.queries.keys())[:50]
    debug = DatasetTriple(
        queries={qid: full.queries[qid] for qid in query_ids},
        corpus={},
        qrels={},
    )

    # 收集这些小查询相关的 doc
    referenced_docs: set[str] = set()
    for qid in query_ids:
        if qid in full.qrels:
            debug.qrels[qid] = full.qrels[qid]
            referenced_docs.update(full.qrels[qid].keys())

    # 最多取 100 个 doc，优先取引用到的
    selected_docs = list(referenced_docs)[:100]
    if len(selected_docs) < 100:
        remaining = [d for d in full.corpus if d not in selected_docs]
        selected_docs += remaining[: 100 - len(selected_docs)]

    debug.corpus = {did: full.corpus[did] for did in selected_docs}
    return debug
