#!/usr/bin/env python3
"""基线评测 CLI 入口。

运行完整检索评测流程：加载数据集 → 编码 → 建索引 → 检索 → 计算指标 → 输出报告。

Phase 3.1 起支持双链路一键切换（仅替换查询编码方式）:
  - method="baseline": 查询用 BaselineEncoder 编码（BGE，无扩散增强）
  - method="elf":      查询用 ELFPipeline.enhance() 扩散增强

文档侧编码（BGE）与 FAISS 索引在两条链路上保持一致（共享 indexer/retriever）。
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

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
    SUPPORTED_DATASETS,
    SUPPORTED_METHODS,
)
from src.elf.native_encoder import ELFNativeEncoder
from src.elf.pipeline import ELFPipeline
from src.evaluation.dataset import DatasetTriple, load_dataset
from src.evaluation.metrics import compute_metrics_batch
from src.utils.encoder_factory import create_encoder
from src.utils.logger import get_logger
from src.utils.sample import sample_dataset
from src.utils.seed import set_seed
from src.vector_store.indexer import FAISSIndexer
from src.vector_store.retriever import Retriever

if TYPE_CHECKING:
    from src.elf.token_retriever import ColBERTRetriever

logger = get_logger(__name__)


def _validate_token_retrieval_pipeline(elf_pipeline: ELFPipeline | None) -> ELFNativeEncoder:
    """校验 token 检索模式的前置条件并返回 ELFNativeEncoder。

    Args:
        elf_pipeline: ELFPipeline 实例（调用方已保证 method='elf' 时非 None）。

    Returns:
        pipeline 中的 ELFNativeEncoder 实例。

    Raises:
        RuntimeError: elf_pipeline 为 None 或其 encoder 不是 ELFNativeEncoder。
    """
    if elf_pipeline is None:
        raise RuntimeError("token 检索模式需要 ELFPipeline")
    if not isinstance(elf_pipeline.encoder, ELFNativeEncoder):
        raise RuntimeError("token 检索模式需要 ELFNativeEncoder (提供 encode_tokens)")
    return elf_pipeline.encoder


@dataclass
class BenchmarkContext:
    """一次构建、多组共享的评测上下文。

    包含采样后的数据集、文档编码器与检索器，供 run_grid 的
    baseline + 多组 ELF 复用，避免每组重复加载数据集、编码文档、
    建索引与加载 ELF 模型。

    Attributes:
        dataset: 上下文对应的数据集名称（用于 shared 模式下的一致性校验）。
        method: 上下文对应的检索链路（'baseline' / 'elf'）。
        data: 采样后的数据集三元组。
        encoder: 文档侧编码器（baseline → BGE；elf → ELFPipeline），
                 baseline 查询编码也复用该实例。
        retriever: 基于文档向量构建的 FAISS 检索器（token 模式下为 None）。
        elf_pipeline: ELFPipeline（仅 method='elf' 时非 None），
                      文档编码与查询增强共用同一实例。
        token_retriever: ColBERT 式多 token 检索器（method='elf' 且
                         use_token_retrieval 时非 None, issue #39）。
    """

    dataset: str
    method: str
    data: DatasetTriple
    encoder: BaselineEncoder | ELFPipeline
    retriever: Retriever | None = None
    elf_pipeline: ELFPipeline | None = None
    token_retriever: ColBERTRetriever | None = None


def build_benchmark_context(
    dataset: str = "nfcorpus",
    method: str = METHOD_BASELINE,
    encoder_name: str | None = None,
    index_nlist: int = DEFAULT_INDEX_NLIST,
    seed: int = DEFAULT_SEED,
    sample: int | None = None,
    use_token_retrieval: bool = False,
    max_tokens: int = 64,
) -> BenchmarkContext:
    """加载数据集、采样并编码文档建索引，返回可复用的评测上下文。

    等价于 run_benchmark() 的"数据准备"阶段（加载 → 采样 → 编码
    全部文档 → 构建 FAISS 索引），抽出来以便参数网格的 13 组评测
    只执行一次，其余组通过 run_benchmark(shared=ctx) 直接复用。

    文档侧编码按 method 切换（issue #33）：
    - 'baseline': 用 BGE 编码器，查询/文档同处 BGE 空间；
    - 'elf':      用 ELFPipeline（T5 原生编码），查询/文档同处 ELF 空间，
                  ELF 链路自建索引，避免与 BGE 文档空间错位。

    Args:
        dataset: 数据集名称。
        method: 检索链路，'baseline'（BGE 文档编码）或 'elf'（ELF 文档编码）。
        encoder_name: 文档侧编码器名称（仅 method='baseline' 生效；
                      None 时使用 DEFAULT_ENCODER）。
        index_nlist: FAISS IVF 聚类中心数。
        seed: 随机种子。
        sample: 仅取前 N 条有 qrels 的 query，None 为全量。
        use_token_retrieval: 启用 ColBERT 式多 token 检索（仅 method='elf' 生效）。
        max_tokens: token 序列截断长度（默认 64，仅 use_token_retrieval 时生效）。

    Returns:
        构建完成的 BenchmarkContext。

    Raises:
        ValueError: method 不在 SUPPORTED_METHODS 中。
    """
    if method not in SUPPORTED_METHODS:
        supported = ", ".join(SUPPORTED_METHODS)
        raise ValueError(f"不支持的链路 method='{method}'，支持: {supported}")

    set_seed(seed)

    data = load_dataset(dataset)

    # 采样模式：取前 sample 条有 qrels 的 query 及其相关文档
    if sample is not None and sample < len(data.queries):
        data = sample_dataset(data, sample)

    logger.info("数据集 %s: %d queries, %d docs", dataset, len(data.queries), len(data.corpus))

    # 编码文档 + 建索引（按 method 选择文档编码器, 查询与文档同空间）
    doc_ids = sorted(data.corpus.keys())
    doc_texts = [data.corpus[did] for did in doc_ids]
    logger.info("编码 %d 篇文档 (method=%s)...", len(doc_texts), method)

    encoder, elf_pipeline = create_encoder(method, encoder_name)

    token_retriever = None
    if method == METHOD_ELF and use_token_retrieval:
        # ColBERT 式多 token 检索(issue #39): 文档保留 T5 token 序列,
        # 不做 mean-pooling(诊断确认 pooled 表示有效秩坍缩到 1)
        token_encoder = _validate_token_retrieval_pipeline(elf_pipeline)
        from src.elf.token_retriever import ColBERTRetriever, TokenIndex

        index = TokenIndex.build(token_encoder, doc_ids, doc_texts, max_tokens=max_tokens)
        token_retriever = ColBERTRetriever(index)
        retriever = None
    else:
        # FAISS 路径：编码文档 → pooled 向量 → 建索引
        if elf_pipeline is not None:
            # method='elf'：encoder 与 elf_pipeline 是同一 ELFPipeline 实例,
            # 文档编码走 pipeline.encoder.encode_batch（ELF 空间）
            doc_vectors = elf_pipeline.encoder.encode_batch(doc_texts)
        else:
            # method='baseline'：encoder 为 BaselineEncoder（BGE 空间）
            assert isinstance(encoder, BaselineEncoder)
            doc_vectors = encoder.encode_batch(doc_texts)
        indexer = FAISSIndexer(dimension=768, nlist=index_nlist)
        indexer.build(doc_vectors, doc_ids)
        retriever = Retriever(indexer)
    return BenchmarkContext(
        dataset=dataset,
        method=method,
        data=data,
        encoder=encoder,
        retriever=retriever,
        elf_pipeline=elf_pipeline,
        token_retriever=token_retriever,
    )


def run_benchmark(
    dataset: str = "nfcorpus",
    method: str = METHOD_BASELINE,
    encoder_name: str | None = None,
    index_nlist: int = DEFAULT_INDEX_NLIST,
    k_values: list[int] | None = None,
    seed: int = DEFAULT_SEED,
    output_dir: str = "experiments/outputs",
    sample: int | None = None,
    elf_steps: int = DEFAULT_ELF_STEPS,
    elf_noise_t: float = DEFAULT_ELF_NOISE_T,
    elf_cfg_scale: float = DEFAULT_ELF_CFG_SCALE,
    shared: BenchmarkContext | None = None,
    use_token_retrieval: bool = False,
    max_tokens: int = 64,
) -> pd.DataFrame:
    """运行完整检索评测流程（Baseline / ELF 双链路）。

    流程:
        1. 准备数据与索引：加载数据集 → 采样 → 编码全部文档 → 建索引
           （传入 shared 时跳过，直接复用上下文中的数据与检索器）
        2. 按 method 编码所有查询（仅此步切换链路）
        3. 逐条检索 + 计算指标
        4. 汇总结果并保存为 CSV（baseline.csv / elf.csv）

    Args:
        dataset: 数据集名称。
        method: 检索链路，'baseline'（BGE 编码）或 'elf'（ELF 扩散增强）。
        encoder_name: HuggingFace 编码器名称（文档侧，仅 method='baseline' 生效；
                      None 时使用 DEFAULT_ENCODER）。
        index_nlist: FAISS IVF 聚类中心数。
        k_values: 评估的 k 值列表。
        seed: 随机种子。
        output_dir: 输出目录。
        sample: 仅取前 N 条有 qrels 的 query 快速验证，None 为全量。
        elf_steps: ELF 去噪步数（仅 method='elf' 生效）。
        elf_noise_t: ELF 加噪强度 t ∈ [0, 1]（仅 method='elf' 生效）。
        elf_cfg_scale: ELF CFG 引导强度（仅 method='elf' 生效）。
        shared: 预构建的共享上下文（数据集 + 编码器 + 检索器 + ELF pipeline）。
                传入时跳过数据加载 / 文档编码 / 建索引，供参数网格多组复用。
                注意：use_token_retrieval 仅在 shared=None 时生效（传入 shared 时
                假定上下文已配置好 token 检索器）。
        use_token_retrieval: 启用 ColBERT 式多 token 检索（仅 method='elf' 生效；
                            shared 非 None 时忽略，由上下文决定检索模式）。
        max_tokens: token 序列截断长度（默认 64）。shared 非 None 时仍用于查询侧
                    截断，应与 shared 上下文构建时保持一致。

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

    # 1. 数据准备（加载 + 采样 + 编码文档 + 建索引）
    #    传入 shared 时复用预构建上下文，跳过最耗时的重复计算
    if shared is None:
        ctx = build_benchmark_context(
            dataset=dataset,
            method=method,
            encoder_name=encoder_name,
            index_nlist=index_nlist,
            seed=seed,
            sample=sample,
            use_token_retrieval=use_token_retrieval,
            max_tokens=max_tokens,
        )
    else:
        ctx = shared
        if ctx.dataset != dataset:
            raise ValueError(
                f"shared 上下文的数据集 '{ctx.dataset}' 与参数 dataset='{dataset}' 不一致"
            )
        if ctx.method != method:
            raise ValueError(f"shared 上下文的链路 '{ctx.method}' 与参数 method='{method}' 不一致")
    data = ctx.data
    encoder = ctx.encoder
    retriever = ctx.retriever

    # 2. 编码查询 + 检索（仅此处按 method 切换链路）
    query_ids = sorted(data.queries.keys())

    query_encoder: Callable[[str], NDArray[np.float32]]
    if method == METHOD_ELF:
        if ctx.elf_pipeline is None:
            try:
                ctx.elf_pipeline = ELFPipeline()
            except Exception as e:
                logger.error("ELF 模型加载失败 (可能需要联网下载权重): %s", e)
                raise RuntimeError(f"ELF 模型加载失败: {e}") from e
        elf_pipeline = ctx.elf_pipeline

        if ctx.token_retriever is None:
            # FAISS 路径：ELF 增强编码
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
            # token 检索模式：查询编码由内联 encode_tokens 完成，
            # query_encoder 在此路径下不被调用
            query_encoder = lambda _: np.zeros(768, dtype=np.float32)  # unreachable
    else:
        query_encoder = encoder.encode

    logger.info("检索 %d 条查询 (method=%s)...", len(query_ids), method)
    # token 检索模式前置校验（仅一次，而非逐 query 重复断言）
    token_encoder: ELFNativeEncoder | None = None
    if ctx.token_retriever is not None:
        if shared is not None:
            # 外部传入上下文时重新校验
            token_encoder = _validate_token_retrieval_pipeline(ctx.elf_pipeline)
        else:
            # build_benchmark_context 中已完成校验，直接获取
            assert ctx.elf_pipeline is not None
            token_encoder = ctx.elf_pipeline.encoder
    all_results: dict[str, list[str]] = {}
    for qid in tqdm(query_ids, desc="检索中"):
        if ctx.token_retriever is not None:
            # 多 token 检索: 查询 token 编码 + maxsim
            if token_encoder is None:
                raise RuntimeError("token_encoder 未初始化（前置校验异常）")
            qt, qm = token_encoder.encode_tokens([data.queries[qid]], max_tokens=max_tokens)
            doc_ids_found, _ = ctx.token_retriever.search(qt[0], qm[0], k=max(k_values))
        else:
            if retriever is None:
                raise RuntimeError("FAISS 检索器未初始化")
            qvec = query_encoder(data.queries[qid])
            doc_ids_found, _ = retriever.search(qvec, k=max(k_values))
        all_results[qid] = doc_ids_found

    # 4. 计算指标
    logger.info("计算评测指标...")
    metrics = compute_metrics_batch(data.qrels, all_results, k_values)

    # 5. 聚合全部查询指标
    agg: dict[str, object] = {"dataset": dataset, "method": method}
    agg["n_queries"] = len(query_ids)
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


def _build_parser() -> argparse.ArgumentParser:
    """构建双链路评测参数解析器（benchmark / runner 共用）。

    Returns:
        配置完成的 ArgumentParser。
    """
    parser = argparse.ArgumentParser(description="Baseline / ELF 双链路检索评测")
    parser.add_argument(
        "--method",
        choices=SUPPORTED_METHODS,
        default=METHOD_BASELINE,
        help="检索链路: baseline（BGE 编码）或 elf（ELF 扩散增强）",
    )
    parser.add_argument(
        "--dataset",
        default="nfcorpus",
        choices=list(SUPPORTED_DATASETS),
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
    parser.add_argument(
        "--token-retrieval",
        action="store_true",
        help="使用 ColBERT 式多 token 检索(method=elf, issue #39)",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=64, help="token 检索的序列截断长度(默认 64)"
    )
    return parser


def _main() -> int:
    args = _build_parser().parse_args()

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
        use_token_retrieval=args.token_retrieval,
        max_tokens=args.max_tokens,
    )
    # print() is intentional: CLI stdout output for the result table
    print(df.to_markdown())
    return 0


if __name__ == "__main__":
    sys.exit(_main())
