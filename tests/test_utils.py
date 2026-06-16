"""配置与工具模块单元测试。"""

from src.config import (
    DEFAULT_ENCODER,
    DEFAULT_INDEX_NLIST,
    DEFAULT_K_VALUES,
    DEFAULT_SEED,
    VECTOR_DIM,
)
from src.utils.device import get_device


class TestConfig:
    """config 模块单元测试。"""

    def test_vector_dim(self) -> None:
        assert VECTOR_DIM == 768

    def test_default_k_values(self) -> None:
        assert DEFAULT_K_VALUES == [5, 10, 20]

    def test_default_seed(self) -> None:
        assert DEFAULT_SEED == 42

    def test_default_encoder(self) -> None:
        assert "bge" in DEFAULT_ENCODER.lower()

    def test_default_nlist(self) -> None:
        assert DEFAULT_INDEX_NLIST == 100


class TestDevice:
    """device 模块单元测试。"""

    def test_get_device_returns_string(self) -> None:
        dev = get_device()
        assert dev in ("cuda", "cpu")
