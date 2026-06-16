"""pytest fixtures。"""

import numpy as np
import pytest

from src.utils.seed import set_seed


@pytest.fixture(autouse=True)
def _fix_seed() -> None:
    """每个测试用例自动固定随机种子。"""
    set_seed(42)


# Phase 2 fixtures (consumed by test_elf.py)
@pytest.fixture
def sample_texts() -> list[str]:
    """三条英文样例文本。"""
    return [
        "What is the capital of France?",
        "Python programming language",
        "Climate change effects",
    ]


@pytest.fixture
def dummy_vectors() -> np.ndarray:
    """10 个随机 768-dim 归一化向量。"""
    rng = np.random.RandomState(42)
    vecs = rng.randn(10, 768).astype(np.float32)
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs
