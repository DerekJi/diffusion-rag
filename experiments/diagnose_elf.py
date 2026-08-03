#!/usr/bin/env python3
"""ELF 诊断脚本:定位 ELF 增强系统性落后于 baseline 的根因。

在评测(--sample N)跑完后运行,复用与评测完全相同的参数、数据集、
随机种子与 ELF 增强流程(逐条 query、每组从相同 seed 新建 rng),
计算四类诊断指标:

1. query_doc_sim        — 查询向量与全部文档向量的平均内积(向量空间对齐程度)
2. query_doc_top10_sim  — 查询向量与其检索到的 top10 文档的平均内积(检索质量)
3. overlap_with_baseline— top10 检索结果与 baseline 结果的重叠率
4. shift_cos / shift_l2 — ELF 增强前后查询向量的扰动幅度

输出:
    experiments/outputs/<dataset>/diagnosis/diagnosis.json  (结构化数据)
    experiments/outputs/<dataset>/diagnosis/diagnosis.md    (可读报告)

文档向量按 (dataset, sample, nlist, encoder) 缓存到
experiments/outputs/<dataset>/cache/,首次编码一次,之后秒级复用。

用法::

    python -m experiments.diagnose_elf --dataset nfcorpus --sample 20
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import faiss
import numpy as np

from src.baseline.benchmark import BenchmarkContext, build_benchmark_context
from src.baseline.encoder import BaselineEncoder
from src.config import METHOD_BASELINE, METHOD_ELF, SUPPORTED_DATASETS
from src.elf.pipeline import ELFPipeline
from src.evaluation.dataset import DatasetTriple, load_dataset
from src.evaluation.orchestrator import ExperimentConfig, load_param_grid
from src.utils.logger import get_logger
from src.utils.seed import set_seed
from src.vector_store.indexer import FAISSIndexer
from src.vector_store.retriever import Retriever

logger = get_logger(__name__)

_TOP_K = 10  # 检索重叠率与 top-k 相似度使用的 k


def _extract_doc_vectors(ctx: BenchmarkContext) -> tuple[np.ndarray, list[str]]:
    """从 FAISS 索引中还原文档向量与 doc_ids 顺序。

    文档向量在构建索引时按 add 顺序对齐 doc_ids,FAISS 的
    reconstruct(i) 按添加顺序取回,因此顺序一致。
    IVF 索引默认不初始化 DirectMap(直接映射),reconstruct 会抛
    "direct map not initialized",需要先启用 Hashtable 映射。

    Args:
        ctx: 共享评测上下文。

    Returns:
        (doc_vectors, doc_ids): shape (n_docs, 768) float32 与对应 id 列表。
    """
    # 诊断工具专用:Retriever 未公开 indexer,此处访问内部属性取回索引
    index = ctx.retriever._indexer.index  # type: ignore[attr-defined]
    n = index.ntotal
    if n == 0:
        raise RuntimeError("FAISS 索引为空,无法提取文档向量")
    # IVF 系列索引需要显式启用 DirectMap 才能 reconstruct
    if isinstance(index, faiss.IndexIVF):
        index.set_direct_map_type(faiss.DirectMap.Hashtable)
    vectors = np.vstack([index.reconstruct(i) for i in range(n)]).astype(np.float32)
    ids = list(ctx.retriever._indexer.doc_ids)  # type: ignore[attr-defined]
    if len(ids) != n:
        raise RuntimeError(f"doc_ids 数量 {len(ids)} 与索引向量数 {n} 不一致")
    return vectors, ids


def _cache_key(
    dataset: str, method: str, sample: int | None, nlist: int, encoder: str
) -> dict[str, object]:
    """文档向量缓存的参数指纹。"""
    return {
        "dataset": dataset,
        "method": method,
        "sample": sample,
        "nlist": nlist,
        "encoder": encoder,
    }


def _cache_paths(cache_dir: Path) -> tuple[Path, Path, Path]:
    """缓存文件路径:meta.json / doc_vectors.npy / doc_ids.json。"""
    return (
        cache_dir / "meta.json",
        cache_dir / "doc_vectors.npy",
        cache_dir / "doc_ids.json",
    )


def _load_cached_vectors(
    cache_dir: Path, params: dict[str, object]
) -> tuple[np.ndarray, list[str]] | None:
    """尝试从磁盘加载缓存的文档向量;参数不匹配时返回 None。"""
    meta_path, vec_path, ids_path = _cache_paths(cache_dir)
    if not (meta_path.exists() and vec_path.exists() and ids_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if meta != params:
        logger.info("缓存参数不匹配,忽略缓存")
        return None
    vectors = np.load(vec_path)
    ids = json.loads(ids_path.read_text(encoding="utf-8"))
    logger.info("命中文档向量缓存: %d 篇文档 (%s)", len(ids), vec_path)
    return vectors, ids


def _save_cached_vectors(
    cache_dir: Path,
    params: dict[str, object],
    vectors: np.ndarray,
    ids: list[str],
) -> None:
    """将文档向量与参数指纹写入磁盘缓存。"""
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path, vec_path, ids_path = _cache_paths(cache_dir)
    meta_path.write_text(
        json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    np.save(vec_path, vectors)
    ids_path.write_text(json.dumps(ids, ensure_ascii=False), encoding="utf-8")
    logger.info("文档向量已缓存: %d 篇文档 (%s)", len(ids), vec_path)


def _prepare_docs(
    dataset: str,
    method: str,
    sample: int | None,
    nlist: int,
    encoder: str,
    seed: int,
) -> tuple[BenchmarkContext, np.ndarray, list[str]]:
    """构建指定链路的评测上下文,返回 (ctx, doc_vectors, doc_ids)。

    优先复用磁盘缓存跳过文档编码;未命中时编码一次并落盘。
    缓存按 (dataset, method, sample, nlist, encoder) 区分。
    """
    cache_dir = Path("experiments/outputs") / dataset / "cache"
    params = _cache_key(dataset, method, sample, nlist, encoder)
    cached = _load_cached_vectors(cache_dir, params)
    if cached is not None:
        vectors, ids = cached
        ctx = _rebuild_context_from_vectors(
            dataset, method, vectors, ids, nlist, encoder, seed, sample
        )
        return ctx, vectors, ids

    ctx = build_benchmark_context(
        dataset=dataset,
        method=method,
        encoder_name=encoder,
        index_nlist=nlist,
        seed=seed,
        sample=sample,
    )
    vectors, ids = _extract_doc_vectors(ctx)
    _save_cached_vectors(cache_dir, params, vectors, ids)
    return ctx, vectors, ids


def _rebuild_context_from_vectors(
    dataset: str,
    method: str,
    vectors: np.ndarray,
    ids: list[str],
    nlist: int,
    encoder: str,
    seed: int,
    sample: int | None,
) -> BenchmarkContext:
    """用缓存的文档向量重建索引与上下文(跳过耗时编码)。

    按 method 重建对应的编码器:baseline → BGE;elf → ELFPipeline。
    """
    set_seed(seed)
    data = load_dataset(dataset)

    # 与 build_benchmark_context 相同的采样逻辑
    if sample is not None and sample < len(data.queries):
        qids_with_qrels = sorted(q for q in data.queries if q in data.qrels)
        sampled_qids = qids_with_qrels[:sample]
        referenced: set[str] = set()
        for qid in sampled_qids:
            referenced.update(data.qrels[qid].keys())
        data = DatasetTriple(
            queries={qid: data.queries[qid] for qid in sampled_qids},
            corpus={did: data.corpus[did] for did in referenced if did in data.corpus},
            qrels={qid: data.qrels[qid] for qid in sampled_qids},
        )
        logger.info("采样模式: %d queries, %d docs", len(data.queries), len(data.corpus))

    if method == METHOD_ELF:
        elf_pipeline = ELFPipeline()
        encoder_obj: BaselineEncoder | ELFPipeline = elf_pipeline
    else:
        encoder_obj = BaselineEncoder(model_name=encoder)
        elf_pipeline = None
    indexer = FAISSIndexer(dimension=vectors.shape[1], nlist=nlist)
    indexer.build(vectors, ids)
    retriever = Retriever(indexer)
    return BenchmarkContext(
        dataset=dataset,
        method=method,
        data=data,
        encoder=encoder_obj,
        retriever=retriever,
        elf_pipeline=elf_pipeline,
    )


def _encode_queries_baseline(
    ctx: BenchmarkContext, query_ids: list[str]
) -> dict[str, np.ndarray]:
    """baseline 查询向量:逐条 BGE 编码(与评测一致)。"""
    return {qid: ctx.encoder.encode(ctx.data.queries[qid]) for qid in query_ids}


def _encode_queries_elf(
    ctx: BenchmarkContext,
    query_ids: list[str],
    steps: int,
    noise_t: float,
    cfg_scale: float,
    seed: int,
) -> dict[str, np.ndarray]:
    """ELF 增强查询向量:逐条 enhance,每组从相同 seed 新建 rng(与评测一致)。"""
    if ctx.elf_pipeline is None:
        # 与 run_benchmark 相同的惰性创建:首个 ELF 组加载后复用
        ctx.elf_pipeline = ELFPipeline()
    pipeline = ctx.elf_pipeline
    rng = np.random.default_rng(seed)
    return {
        qid: pipeline.enhance(
            ctx.data.queries[qid],
            steps=steps,
            noise_t=noise_t,
            cfg_scale=cfg_scale,
            rng=rng,
        )
        for qid in query_ids
    }


def _encode_queries_elf_raw(
    ctx: BenchmarkContext, query_ids: list[str]
) -> dict[str, np.ndarray]:
    """ELF 编码器原始输出(不增强): 仅 T5 编码 + 投影, 跳过加噪/去噪。

    用于区分"编码空间错位"与"扩散破坏":若原始输出与文档向量已接近
    正交, 则根因在编码器空间, 与扩散参数无关。
    """
    if ctx.elf_pipeline is None:
        ctx.elf_pipeline = ELFPipeline()
    return {
        qid: ctx.elf_pipeline.encode(ctx.data.queries[qid]) for qid in query_ids
    }


def _topk_from_retriever(
    ctx: BenchmarkContext, qvec: np.ndarray, k: int = _TOP_K
) -> list[str]:
    """检索 top-k 文档 id(与评测相同的 retriever.search 路径)。"""
    doc_ids_found, _ = ctx.retriever.search(qvec, k=k)
    return doc_ids_found


def _overlap(a: list[str], b: list[str]) -> float:
    """两个结果列表的重叠率:交集大小 / 较短列表长度。"""
    if not a or not b:
        return 0.0
    return len(set(a) & set(b)) / min(len(a), len(b))


def _topk_sim(
    doc_vectors: np.ndarray, id_to_idx: dict[str, int], qvec: np.ndarray, doc_ids: list[str]
) -> float:
    """检索到的 top-k 文档与查询向量的平均内积。"""
    if not doc_ids:
        return 0.0
    sims = [float(doc_vectors[id_to_idx[did]] @ qvec) for did in doc_ids if did in id_to_idx]
    return float(np.mean(sims)) if sims else 0.0


def _analyze_group(
    ctx: BenchmarkContext,
    doc_vectors: np.ndarray,
    id_to_idx: dict[str, int],
    query_ids: list[str],
    qvecs: dict[str, np.ndarray],
    base_qvecs: dict[str, np.ndarray] | None = None,
    baseline_top10: dict[str, list[str]] | None = None,
) -> dict[str, float]:
    """计算一组查询向量的诊断指标。

    Args:
        ctx: 该链路自己的评测上下文（文档向量与该链路同空间）。
        doc_vectors: 该链路的文档向量（与 ctx 的索引一致）。
        id_to_idx: doc_id → 文档向量行号映射。
        query_ids: 查询 ID 列表。
        qvecs: 本组查询向量。
        base_qvecs: baseline(BGE) 查询向量,提供时计算 shift 扰动指标。
        baseline_top10: baseline 的 top10 检索结果 {qid: doc_ids},
                        提供时计算与 baseline 的重叠率。
    """
    avg_sims: list[float] = []
    top10_sims: list[float] = []
    overlaps: list[float] = []
    shifts_cos: list[float] = []
    shifts_l2: list[float] = []

    for qid in query_ids:
        qvec = qvecs[qid]
        all_sims = doc_vectors @ qvec  # 全部文档内积
        avg_sims.append(float(np.mean(all_sims)))
        top10_ids = _topk_from_retriever(ctx, qvec)
        top10_sims.append(_topk_sim(doc_vectors, id_to_idx, qvec, top10_ids))

        if base_qvecs is not None:
            base_vec = base_qvecs[qid]
            cos = float(np.dot(qvec, base_vec) / (np.linalg.norm(qvec) * np.linalg.norm(base_vec)))
            shifts_cos.append(cos)
            shifts_l2.append(float(np.linalg.norm(qvec - base_vec)))
        if baseline_top10 is not None:
            overlaps.append(_overlap(baseline_top10[qid], top10_ids))

    result: dict[str, float] = {
        "query_doc_sim": float(np.mean(avg_sims)),
        "query_doc_top10_sim": float(np.mean(top10_sims)),
    }
    if base_qvecs is not None:
        result["shift_cos"] = float(np.mean(shifts_cos))
        result["shift_l2"] = float(np.mean(shifts_l2))
    if baseline_top10 is not None:
        result["overlap_with_baseline"] = float(np.mean(overlaps))
    result["n_queries"] = float(len(query_ids))
    return result


def _build_report(
    dataset: str, sample: int | None, baseline_stats: dict[str, float],
    elf_stats: dict[str, dict[str, float]], elf_params: list[dict[str, object]],
) -> str:
    """生成 Markdown 诊断报告。"""
    lines = [
        f"# ELF 诊断报告 — {dataset}" + (f" (sample={sample})" if sample else ""),
        "",
        "> 由 experiments/diagnose_elf.py 生成,指标为全部采样 query 的均值。",
        "",
        "## 1. 向量对齐与检索质量",
        "",
        "| 组 | 参数 | query_doc_sim | query_doc_top10_sim | overlap_with_baseline |",
        "|---|---:|---:|---:|---:|",
    ]
    lines.append(
        f"| baseline | — | {baseline_stats['query_doc_sim']:.4f} | "
        f"{baseline_stats['query_doc_top10_sim']:.4f} | 1.0000 |"
    )
    if "elf-raw" in elf_stats:
        s = elf_stats["elf-raw"]
        lines.append(
            f"| elf-raw | 不增强(仅编码) | {s['query_doc_sim']:.4f} | "
            f"{s['query_doc_top10_sim']:.4f} | {s['overlap_with_baseline']:.4f} |"
        )
    for p in elf_params:
        cid = str(p["id"])
        s = elf_stats[cid]
        param_str = f"{p['steps']}/{p['noise_t']}/{p['cfg_scale']}"
        lines.append(
            f"| {cid} | s={param_str} | {s['query_doc_sim']:.4f} | "
            f"{s['query_doc_top10_sim']:.4f} | {s['overlap_with_baseline']:.4f} |"
        )

    lines += [
        "",
        "## 2. ELF 增强扰动幅度(相对 baseline 查询向量)",
        "",
        "| 组 | 参数 | shift_cos(越高越接近原向量) | shift_l2 |",
        "|---|---:|---:|---:|",
    ]
    if "elf-raw" in elf_stats:
        s = elf_stats["elf-raw"]
        lines.append(
            f"| elf-raw | 不增强(仅编码) | {s['shift_cos']:.4f} | {s['shift_l2']:.4f} |"
        )
    for p in elf_params:
        cid = str(p["id"])
        s = elf_stats[cid]
        param_str = f"{p['steps']}/{p['noise_t']}/{p['cfg_scale']}"
        lines.append(
            f"| {cid} | s={param_str} | {s['shift_cos']:.4f} | {s['shift_l2']:.4f} |"
        )

    lines += [
        "",
        "## 3. 速读指引",
        "",
        "- **elf-raw 的 query_doc_sim 已接近 0**: 根因是编码器向量空间错位"
        "(ELF/T5 投影空间 vs BGE 文档空间), 与扩散参数无关。",
        "- **query_doc_sim 显著低于 baseline**:增强把查询向量推离了文档分布,"
        "两条链路的向量空间不对齐。",
        "- **overlap_with_baseline 高但指标差**:问题不在检索,而在相关文档排序/向量本身。",
        "- **overlap_with_baseline 低**:ELF 检索到的是完全不同的文档,需检查增强方向。",
        "- **shift_cos 接近 1.0 但指标仍差**:扰动不是主因,怀疑文档侧向量空间问题。",
    ]
    return "\n".join(lines)


def _main() -> int:
    parser = argparse.ArgumentParser(description="ELF 诊断:定位 ELF 落后根因")
    parser.add_argument(
        "--config", default="experiments/configs/param_grid.yaml", help="参数网格 YAML 路径"
    )
    parser.add_argument(
        "--dataset",
        default=None,
        choices=list(SUPPORTED_DATASETS),
        help="数据集名称(默认取配置)",
    )
    parser.add_argument("--sample", type=int, default=None, help="采样 query 数(建议与评测一致)")
    parser.add_argument("--seed", type=int, default=None, help="随机种子(默认取配置)")
    parser.add_argument(
        "--output", default=None, help="诊断输出目录(默认 experiments/outputs/<dataset>/diagnosis)"
    )
    args = parser.parse_args()

    config: ExperimentConfig = load_param_grid(args.config)
    dataset = args.dataset or config.dataset
    sample = args.sample if args.sample is not None else config.sample
    seed = args.seed if args.seed is not None else config.seed

    if sample is None:
        raise SystemExit(
            "未指定 --sample 且配置中 sample 为 null(全量 3237 条 query 诊断过慢);"
            "请传 --sample N(建议与评测一致)"
        )

    out_dir = Path(args.output) if args.output else (
        Path("experiments/outputs") / dataset / "diagnosis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("诊断开始: dataset=%s, sample=%d, seed=%d", dataset, sample, seed)
    t0 = time.perf_counter()

    # 双链路各自独立构建(issue #33): baseline → BGE 文档库;
    # elf → ELF 文档库, 查询与文档同空间, 反映修复后的真实对齐度
    base_ctx, base_vectors, base_ids = _prepare_docs(
        dataset=dataset,
        method=METHOD_BASELINE,
        sample=sample,
        nlist=config.index_nlist,
        encoder=config.encoder,
        seed=seed,
    )
    elf_ctx, elf_vectors, elf_ids = _prepare_docs(
        dataset=dataset,
        method=METHOD_ELF,
        sample=sample,
        nlist=config.index_nlist,
        encoder=config.encoder,
        seed=seed,
    )
    query_ids = sorted(base_ctx.data.queries.keys())
    logger.info("查询数: %d, 文档数: baseline=%d / elf=%d",
                len(query_ids), len(base_ids), len(elf_ids))

    # baseline 查询向量与其 top10 检索结果(作为 ELF 组的对照)
    base_qvecs = _encode_queries_baseline(base_ctx, query_ids)
    base_id_to_idx = {did: i for i, did in enumerate(base_ids)}
    baseline_top10 = {
        qid: _topk_from_retriever(base_ctx, base_qvecs[qid]) for qid in query_ids
    }
    baseline_stats = _analyze_group(
        base_ctx, base_vectors, base_id_to_idx, query_ids, base_qvecs
    )
    logger.info("baseline 诊断完成: query_doc_sim=%.4f", baseline_stats["query_doc_sim"])

    # ELF 链路(文档为 ELF 编码, 与查询同空间)
    elf_id_to_idx = {did: i for i, did in enumerate(elf_ids)}
    elf_stats: dict[str, dict[str, float]] = {}
    # ELF 编码器原始输出对照(不增强): 区分"编码空间错位"与"扩散破坏"
    logger.info("分析 elf-raw: ELF 编码器原始输出(不加噪/不去噪)")
    raw_qvecs = _encode_queries_elf_raw(elf_ctx, query_ids)
    elf_stats["elf-raw"] = _analyze_group(
        elf_ctx, elf_vectors, elf_id_to_idx, query_ids,
        raw_qvecs, base_qvecs, baseline_top10,
    )
    for params in config.elf_param_list:
        cid = str(params["id"])
        steps = int(params["steps"])
        noise_t = float(params["noise_t"])
        cfg_scale = float(params["cfg_scale"])
        logger.info("分析 %s: steps=%d, noise_t=%.2f, cfg_scale=%.1f", cid, steps, noise_t, cfg_scale)
        elf_qvecs = _encode_queries_elf(elf_ctx, query_ids, steps, noise_t, cfg_scale, seed)
        elf_stats[cid] = _analyze_group(
            elf_ctx, elf_vectors, elf_id_to_idx, query_ids,
            elf_qvecs, base_qvecs, baseline_top10,
        )

    report_md = _build_report(
        dataset, sample, baseline_stats, elf_stats, config.elf_param_list
    )
    summary = {
        "dataset": dataset,
        "sample": sample,
        "seed": seed,
        "n_queries": len(query_ids),
        "n_docs": {"baseline": len(base_ids), "elf": len(elf_ids)},
        "baseline": baseline_stats,
        "elf_groups": elf_stats,
        "elf_params": config.elf_param_list,
    }

    md_path = out_dir / "diagnosis.md"
    json_path = out_dir / "diagnosis.json"
    md_path.write_text(report_md, encoding="utf-8")
    json_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    logger.info("诊断完成 (%.1fs): %s / %s", time.perf_counter() - t0, json_path, md_path)
    print(report_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
