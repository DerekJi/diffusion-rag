#!/usr/bin/env python3
"""资产下载脚本与缓存管理。

支持从 HuggingFace Hub 下载 ELF-B 模型权重和 BEIR 评测数据集，
实现三路加载策略（HuggingFace / 本地缓存），
并支持命令行调用。

Usage:
    python -m src.utils.download --all
    python -m src.utils.download --elf-weights
    python -m src.utils.download --dataset nfcorpus
    python -m src.utils.download --cache-dir /path/to/cache
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

from tqdm.auto import tqdm

from src.config import DATA_DIR, MODELS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# ── 常量 ───────────────────────────────────────

ELF_MODEL_REPO = "embedded-language-flows/ELF-B-owt-torch"

# 支持的数据集映射（与 src/evaluation/dataset.py 一致）
SUPPORTED_DATASETS: dict[str, str] = {
    "nfcorpus": "BeIR/nfcorpus",
    "msmarco": "BeIR/msmarco",
    "nq": "BeIR/nq",
    "fiqa": "BeIR/fiqa",
}

# 重试配置
_MAX_RETRIES = 3
_TIMEOUT_SECONDS = 60
_RETRY_DELAY = 5.0  # 秒


# ──────────────────────────────────────────────
#  下载函数
# ──────────────────────────────────────────────


def _resolve_cache_dir(cache_dir: str | None) -> tuple[Path, Path]:
    """根据用户参数解析模型和数据缓存目录。

    Args:
        cache_dir: 用户指定的缓存根目录，None 则使用默认值。

    Returns:
        (models_dir, data_dir): 模型和数据缓存目录的 Path 对象。
    """
    if cache_dir is not None:
        root = Path(cache_dir)
        return root / "models", root / "data"
    return Path(MODELS_DIR), Path(DATA_DIR)


def _check_elf_cached(models_dir: Path) -> bool:
    """检查 ELF-B 模型权重是否已缓存到本地。

    Args:
        models_dir: 模型缓存目录。

    Returns:
        是否已缓存。
    """
    # 检查 snapshot 目录或单个 checkpoint 文件
    elf_dir = models_dir / ELF_MODEL_REPO
    if elf_dir.is_dir() and any(elf_dir.iterdir()):
        return True
    checkpoint_file = models_dir / "checkpoint_95085"
    return checkpoint_file.is_file()


def download_elf_weights(cache_dir: str | None = None) -> Path:
    """从 HuggingFace Hub 下载 ELF-B 模型权重。

    检查本地缓存，已存在则跳过。
    若完整仓库下载失败，回退到仅下载 checkpoint 文件。

    Args:
        cache_dir: 缓存根目录，None 使用 ``models/``。

    Returns:
        模型文件所在的目录路径。

    Raises:
        RuntimeError: 所有下载尝试均失败时抛出。
    """
    models_dir, _ = _resolve_cache_dir(cache_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    if _check_elf_cached(models_dir):
        logger.info("ELF-B 权重已缓存，跳过下载")
        return models_dir / ELF_MODEL_REPO if (models_dir / ELF_MODEL_REPO).is_dir() else models_dir

    logger.info("下载 ELF-B 模型权重 %s", ELF_MODEL_REPO)

    # 尝试完整仓库下载
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            from huggingface_hub import snapshot_download

            local_dir = models_dir / ELF_MODEL_REPO
            with tqdm(desc="下载 ELF-B 权重", unit="B", unit_scale=True) as pbar:
                snapshot_download(
                    repo_id=ELF_MODEL_REPO,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    tqdm_class=type(pbar),
                )
            logger.info("ELF-B 权重下载完成: %s", local_dir)
            return local_dir
        except Exception as e:
            logger.warning(
                "ELF-B 完整仓库下载失败 (尝试 %d/%d): %s",
                attempt,
                _MAX_RETRIES,
                e,
            )
            if attempt < _MAX_RETRIES:
                _wait_with_timeout()

    # 回退：仅下载 checkpoint 文件
    logger.warning("完整仓库下载失败，回退到仅下载 checkpoint 文件")
    return _download_elf_checkpoint_fallback(models_dir)


def _download_elf_checkpoint_fallback(models_dir: Path) -> Path:
    """回退方案：仅下载 ELF checkpoint 文件。

    Args:
        models_dir: 模型缓存目录。

    Returns:
        模型文件所在目录路径。

    Raises:
        RuntimeError: 下载仍失败时抛出。
    """
    from huggingface_hub import hf_hub_download

    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            checkpoint_path = hf_hub_download(
                repo_id=ELF_MODEL_REPO,
                filename="checkpoint_95085",
                local_dir=str(models_dir),
                resume_download=True,
            )
            logger.info("ELF checkpoint 下载完成: %s", checkpoint_path)
            return Path(checkpoint_path).parent
        except Exception as e:
            logger.warning(
                "ELF checkpoint 下载失败 (尝试 %d/%d): %s",
                attempt,
                _MAX_RETRIES,
                e,
            )
            if attempt < _MAX_RETRIES:
                _wait_with_timeout()

    raise RuntimeError(
        f"ELF-B 模型下载失败，已重试 {_MAX_RETRIES} 次。"
        f"请检查网络连接或手动下载: https://huggingface.co/{ELF_MODEL_REPO}"
    )


def _wait_with_timeout() -> None:
    """等待重试间隔，同时计入总超时。"""
    time.sleep(_RETRY_DELAY)


def _check_dataset_cached(name: str, data_dir: Path) -> bool:
    """检查数据集是否已缓存。

    Args:
        name: 数据集名称（如 'nfcorpus'）。
        data_dir: 数据缓存目录。

    Returns:
        是否已缓存。
    """
    # HuggingFace datasets 缓存路径为 ~/.cache/huggingface/datasets
    # 我们不做深度扫描，依赖 datasets 库自身的缓存机制
    hf_name = SUPPORTED_DATASETS.get(name, name)
    # 简单检查 data_dir 下是否有标记文件
    marker = data_dir / f".{name}_downloaded"
    if marker.is_file():
        return True

    # 检查 HuggingFace 默认缓存
    hf_cache = Path(
        os.environ.get(
            "HF_DATASETS_CACHE", str(Path.home() / ".cache" / "huggingface" / "datasets")
        )
    )
    # 以 BeIR/nfcorpus 为例，缓存路径包含 repo 名称
    if (hf_cache / hf_name.replace("/", "_")).is_dir():
        return True
    # 也检查带下划线的变体
    if (hf_cache / hf_name.replace("/", "--")).is_dir():
        return True

    return False


def _mark_dataset_downloaded(name: str, data_dir: Path) -> None:
    """标记数据集已下载完成。"""
    data_dir.mkdir(parents=True, exist_ok=True)
    marker = data_dir / f".{name}_downloaded"
    marker.touch()


def download_dataset(name: str, cache_dir: str | None = None) -> None:
    """下载指定的 BEIR 评测数据集。

    Args:
        name: 数据集名称，支持 'nfcorpus', 'msmarco', 'nq', 'fiqa'。
        cache_dir: 缓存根目录，None 使用默认。

    Raises:
        ValueError: 不支持的数据集名称。
        RuntimeError: 下载失败（含网络错误）。
    """
    _, data_dir = _resolve_cache_dir(cache_dir)

    if name not in SUPPORTED_DATASETS:
        supported = ", ".join(SUPPORTED_DATASETS.keys())
        raise ValueError(f"不支持的数据集 '{name}'，支持: {supported}")

    hf_name = SUPPORTED_DATASETS[name]

    if _check_dataset_cached(name, data_dir):
        logger.info("数据集 %s 已缓存，跳过下载", name)
        return

    logger.info("下载数据集 %s (HuggingFace: %s)", name, hf_name)

    _download_hf_dataset_with_retry(hf_name, name, data_dir)

    _mark_dataset_downloaded(name, data_dir)
    logger.info("数据集 %s 下载完成", name)


def _download_hf_dataset_with_retry(
    hf_name: str,
    name: str,
    data_dir: Path,
) -> None:
    """带重试和进度条的 HuggingFace 数据集下载。

    Args:
        hf_name: HuggingFace 数据集标识符（如 'BeIR/nfcorpus'）。
        name: 数据集名称（用于日志）。
        data_dir: 数据缓存目录。

    Raises:
        RuntimeError: 所有重试均失败时抛出。
    """
    from datasets import load_dataset

    last_error: Exception | None = None
    for attempt in range(1, _MAX_RETRIES + 1):
        try:
            with tqdm(desc=f"下载 {name}", unit="B", unit_scale=True) as pbar:
                # 先下载并缓存数据集（不加载到内存）
                load_dataset(
                    hf_name,
                    cache_dir=str(data_dir),
                    download_mode="reuse_dataset_if_exists",
                )
                # 也下载 qrels
                load_dataset(
                    hf_name + "-qrels",
                    cache_dir=str(data_dir),
                    download_mode="reuse_dataset_if_exists",
                )
            # 快速验证：尝试加载一个子集
            ds = load_dataset(hf_name, "queries", cache_dir=str(data_dir), split="queries")
            pbar.update(0)  # 让进度条显示完成
            logger.info("数据集 %s 验证通过 (%d 条 queries)", name, len(ds))
            return
        except Exception as e:
            last_error = e
            logger.warning(
                "数据集 %s 下载失败 (尝试 %d/%d): %s",
                name,
                attempt,
                _MAX_RETRIES,
                e,
            )
            if attempt < _MAX_RETRIES:
                _wait_with_timeout()

    raise RuntimeError(
        f"数据集 {name} ({hf_name}) 下载失败，已重试 {_MAX_RETRIES} 次: {last_error}"
    )


def download_all(cache_dir: str | None = None) -> None:
    """下载全部资产（ELF-B 权重 + 所有 BEIR 数据集）。

    Args:
        cache_dir: 缓存根目录，None 使用默认。
    """
    logger.info("=== 开始下载全部资产 ===")

    logger.info("[1/2] 下载 ELF-B 模型权重...")
    download_elf_weights(cache_dir=cache_dir)

    logger.info("[2/2] 下载 BEIR 评测数据集...")
    for dataset_name in SUPPORTED_DATASETS:
        try:
            download_dataset(dataset_name, cache_dir=cache_dir)
        except (ValueError, RuntimeError) as e:
            logger.error("数据集 %s 下载失败: %s", dataset_name, e)

    logger.info("=== 全部资产下载完成 ===")


# ──────────────────────────────────────────────
#  CLI 入口
# ──────────────────────────────────────────────


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """解析命令行参数。

    Args:
        argv: 命令行参数列表，None 使用 sys.argv。

    Returns:
        解析后的命名空间。
    """
    parser = argparse.ArgumentParser(
        description="Diffusion RAG — 资产下载脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            "  %(prog)s --all                          下载全部资产\n"
            "  %(prog)s --elf-weights                  仅下载 ELF-B 模型权重\n"
            "  %(prog)s --dataset nfcorpus             仅下载 NFCorpus 数据集\n"
            "  %(prog)s --dataset nfcorpus --cache-dir /mnt/cache  指定缓存目录\n"
        ),
    )

    # 互斥操作
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--all",
        action="store_true",
        help="下载全部资产（ELF-B 权重 + 所有 BEIR 数据集）",
    )
    action.add_argument(
        "--elf-weights",
        action="store_true",
        help="仅下载 ELF-B 模型权重",
    )
    action.add_argument(
        "--dataset",
        type=str,
        choices=list(SUPPORTED_DATASETS.keys()),
        help="仅下载指定数据集",
    )

    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="缓存根目录（默认: models/ + data/）",
    )

    args = parser.parse_args(argv)

    # 未指定任何操作时，默认下载全部
    if not args.all and not args.elf_weights and not args.dataset:
        args.all = True

    return args


def main(argv: list[str] | None = None) -> None:
    """下载脚本入口。

    Args:
        argv: 命令行参数列表，None 使用 sys.argv。
    """
    args = _parse_args(argv)

    if args.all:
        download_all(cache_dir=args.cache_dir)
    elif args.elf_weights:
        download_elf_weights(cache_dir=args.cache_dir)
    elif args.dataset:
        download_dataset(args.dataset, cache_dir=args.cache_dir)


if __name__ == "__main__":
    main()
