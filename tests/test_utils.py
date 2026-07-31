"""配置与工具模块单元测试。"""

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
    SUPPORTED_METHODS,
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

    def test_method_constants(self) -> None:
        assert METHOD_BASELINE == "baseline"
        assert METHOD_ELF == "elf"
        assert SUPPORTED_METHODS == (METHOD_BASELINE, METHOD_ELF)

    def test_elf_default_params(self) -> None:
        assert DEFAULT_ELF_STEPS == 2
        assert DEFAULT_ELF_NOISE_T == 0.4
        assert DEFAULT_ELF_CFG_SCALE == 2.0


class TestDevice:
    """device 模块单元测试。"""

    def test_get_device_returns_string(self) -> None:
        dev = get_device()
        assert dev in ("cuda", "cpu")
