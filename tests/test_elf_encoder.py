"""ELF 编码器单元测试。

使用 mock 模式，无须下载真实模型。
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.elf.encoder import ELFEncoder


class TestELFEncoderMock:
    """使用 mock 的 ELF 编码器测试，无需网络/模型缓存。

    通过 mock SentenceTransformer 类的构造函数，使其返回一个
    MagicMock 实例，encode 方法返回固定 768-dim 归一化向量。
    """

    @pytest.fixture(autouse=True)
    def _mock_model(self) -> None:
        """mock SentenceTransformer 类，返回 mock 实例。"""
        rng = np.random.RandomState(42)
        fixed_vec = rng.randn(768).astype(np.float32)
        fixed_vec /= np.linalg.norm(fixed_vec)

        mock_instance = MagicMock()
        mock_instance.encode.return_value = fixed_vec

        patcher = patch(
            "src.elf.encoder.SentenceTransformer",
            return_value=mock_instance,
        )
        patcher.start()
        yield
        patcher.stop()

    def test_encode_shape(self) -> None:
        """输出 shape == (768,)。"""
        encoder = ELFEncoder(device="cpu")
        vec = encoder.encode("Hello world")
        assert vec.shape == (768,)

    def test_encode_dtype(self) -> None:
        """输出 dtype == float32。"""
        encoder = ELFEncoder(device="cpu")
        vec = encoder.encode("Hello world")
        assert vec.dtype == np.float32

    def test_encode_l2_norm(self) -> None:
        """L2 范数约为 1.0。"""
        encoder = ELFEncoder(device="cpu")
        vec = encoder.encode("Hello world")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_encode_empty_string(self) -> None:
        """空字符串应 raise ValueError。"""
        encoder = ELFEncoder(device="cpu")
        with pytest.raises(ValueError):
            encoder.encode("")

    def test_encode_whitespace_only(self) -> None:
        """仅有空格的字符串也应 raise ValueError。"""
        encoder = ELFEncoder(device="cpu")
        with pytest.raises(ValueError):
            encoder.encode("   ")

    def test_encode_batch_shape(self) -> None:
        """batch 输出 shape == (N, 768)。"""
        mock_batch = np.random.RandomState(42).randn(3, 768).astype(np.float32)
        with patch(
            "src.elf.encoder.SentenceTransformer",
            return_value=MagicMock(encode=MagicMock(return_value=mock_batch)),
        ):
            encoder = ELFEncoder(device="cpu")
            vecs = encoder.encode_batch(["a", "bb", "ccc"])
        assert vecs.shape == (3, 768)
        assert vecs.dtype == np.float32

    def test_encode_batch_empty(self) -> None:
        """空列表应 raise ValueError。"""
        encoder = ELFEncoder(device="cpu")
        with pytest.raises(ValueError):
            encoder.encode_batch([])

    def test_encode_deterministic(self) -> None:
        """同一文本两次编码结果一致（mock 模式下）。"""
        encoder = ELFEncoder(device="cpu")
        v1 = encoder.encode("Test sentence")
        v2 = encoder.encode("Test sentence")
        assert np.allclose(v1, v2, atol=1e-6)

    def test_encode_single_character(self) -> None:
        """单字符输入应正常编码。"""
        encoder = ELFEncoder(device="cpu")
        vec = encoder.encode("a")
        assert vec.shape == (768,)
        assert vec.dtype == np.float32

    def test_model_name_default(self) -> None:
        """默认模型名为 BAAI/bge-base-en-v1.5。"""
        encoder = ELFEncoder(device="cpu")
        assert encoder.model_name == "BAAI/bge-base-en-v1.5"

    def test_model_loading_error(self) -> None:
        """模型加载失败应抛出 RuntimeError。"""
        # mock 构造函数抛出异常
        with patch(
            "src.elf.encoder.SentenceTransformer",
            side_effect=Exception("Download failed"),
        ):
            with pytest.raises(RuntimeError, match="无法加载模型"):
                ELFEncoder(device="cpu")
