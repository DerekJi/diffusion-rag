"""报表生成器。

输出 CSV 对比表 + JSON 摘要，供评测编排器 (orchestrator.py) 在全部
参数组跑完后调用。
"""

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import cast

import pandas as pd

from src.config import METHOD_BASELINE, METHOD_ELF
from src.utils.logger import get_logger

logger = get_logger(__name__)


def write_summary_csv(summary: pd.DataFrame, output_dir: str, dataset: str) -> Path:
    """输出 CSV 对比表（每组参数一行）。

    Args:
        summary: run_grid 返回的汇总 DataFrame（含 vs_baseline 差值列）。
        output_dir: 输出根目录，实际写入 output_dir/dataset/summary.csv。
        dataset: 数据集名称。

    Returns:
        写入的 CSV 文件路径。
    """
    out_dir = Path(output_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "summary.csv"
    summary.to_csv(csv_path, index=False)
    logger.info("对比表已保存: %s (%d 行)", csv_path, len(summary))
    return csv_path


def write_json_summary(summary: pd.DataFrame, output_dir: str, dataset: str) -> Path:
    """输出 JSON 摘要。

    包含 baseline 指标、最佳 ELF 参数组（按 recall@10 排序）及全部参数组明细。

    Args:
        summary: run_grid 返回的汇总 DataFrame。
        output_dir: 输出根目录，实际写入 output_dir/dataset/summary.json。
        dataset: 数据集名称。

    Returns:
        写入的 JSON 文件路径。
    """
    out_dir = Path(output_dir) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "summary.json"

    records = [_sanitize(row) for row in summary.to_dict(orient="records")]
    baseline_row = next((r for r in records if r.get("method") == METHOD_BASELINE), None)
    elf_rows = [r for r in records if r.get("method") == METHOD_ELF]
    best_elf: dict[str, object] | None = (
        max(elf_rows, key=lambda r: float(cast(float, r.get("recall@10", 0.0))))
        if elf_rows
        else None
    )

    payload: dict[str, object] = {
        "dataset": dataset,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "n_groups": len(records),
        "baseline": baseline_row,
        "best_elf": best_elf,
        "groups": records,
    }
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("JSON 摘要已保存: %s", json_path)
    return json_path


def _sanitize(row: dict[str, object]) -> dict[str, object]:
    """将 NaN/Inf 等非有限浮点值转为 None，保证 JSON 可解析。

    Args:
        row: 单行指标字典。

    Returns:
        清洗后的字典（非有限浮点值替换为 None）。
    """
    return {
        key: (None if isinstance(value, float) and not math.isfinite(value) else value)
        for key, value in row.items()
    }
