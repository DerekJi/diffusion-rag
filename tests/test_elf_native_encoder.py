"""ELF 原生编码器单元测试。

使用 mock 模式，无须下载真实模型或 ELF 权重。
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.elf.native_encoder import ELFNativeEncoder


def _make_fixed_vector() -> np.ndarray:
    """生成固定 768-dim L2 归一化向量。"""
    rng = np.random.RandomState(42)
    vec = rng.randn(768).astype(np.float32)
    vec /= np.linalg.norm(vec)
    return vec


def _make_fixed_batch(n: int) -> np.ndarray:
    """生成固定批量向量。"""
    rng = np.random.RandomState(42)
    vecs = rng.randn(n, 768).astype(np.float32)
    for i in range(n):
        vecs[i] /= np.linalg.norm(vecs[i])
    return vecs


class TestELFNativeEncoderMock:
    """使用 mock 的 ELF 原生编码器测试，无需网络/模型缓存。

    通过 mock _encode_torch 和 from_pretrained，完全离线运行。
    """

    @pytest.fixture(autouse=True)
    def _mock_all(self) -> None:
        """mock from_pretrained 和 _encode_torch，完全离线。"""
        # mock 模型加载（避免下载 T5-small）
        mock_t5 = MagicMock()
        mock_tokenizer = MagicMock()
        mock_proj = MagicMock(spec=torch.nn.Linear)

        patchers = [
            patch("src.elf.native_encoder.T5EncoderModel.from_pretrained", return_value=mock_t5),
            patch(
                "src.elf.native_encoder.T5Tokenizer.from_pretrained", return_value=mock_tokenizer
            ),
            # 仅 patch native_encoder 模块内的 torch.nn.Linear 引用，
            # 不影响其他模块（function scope + autouse，单线程 pytest 安全）
            patch("src.elf.native_encoder.torch.nn.Linear", return_value=mock_proj),
            patch("src.elf.native_encoder._load_elf_checkpoint", return_value=False),
            # mock _encode_torch 返回固定向量
            patch.object(
                ELFNativeEncoder,
                "_encode_torch",
                return_value=_make_fixed_vector().reshape(1, 768),
            ),
        ]

        for p in patchers:
            p.start()
        yield
        for p in patchers:
            p.stop()

    def test_encode_shape(self) -> None:
        """输出 shape == (768,)。"""
        encoder = ELFNativeEncoder(device="cpu")
        vec = encoder.encode("Hello world")
        assert vec.shape == (768,)

    def test_encode_dtype(self) -> None:
        """输出 dtype == float32。"""
        encoder = ELFNativeEncoder(device="cpu")
        vec = encoder.encode("Hello world")
        assert vec.dtype == np.float32

    def test_encode_l2_norm(self) -> None:
        """L2 范数约为 1.0。"""
        encoder = ELFNativeEncoder(device="cpu")
        vec = encoder.encode("Hello world")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_encode_empty_string(self) -> None:
        """空字符串应 raise ValueError。"""
        encoder = ELFNativeEncoder(device="cpu")
        with pytest.raises(ValueError):
            encoder.encode("")

    def test_encode_whitespace_only(self) -> None:
        """仅有空格的字符串也应 raise ValueError。"""
        encoder = ELFNativeEncoder(device="cpu")
        with pytest.raises(ValueError):
            encoder.encode("   ")

    def test_encode_batch_shape(self) -> None:
        """batch 输出 shape == (N, 768)。"""
        fixed_batch = _make_fixed_batch(3)
        with patch.object(ELFNativeEncoder, "_encode_torch", return_value=fixed_batch):
            encoder = ELFNativeEncoder(device="cpu")
            vecs = encoder.encode_batch(["a", "bb", "ccc"])
        assert vecs.shape == (3, 768)
        assert vecs.dtype == np.float32

    def test_encode_batch_empty(self) -> None:
        """空列表应 raise ValueError。"""
        encoder = ELFNativeEncoder(device="cpu")
        with pytest.raises(ValueError):
            encoder.encode_batch([])

    def test_encode_deterministic(self) -> None:
        """同一文本两次编码结果一致（mock 模式下）。"""
        encoder = ELFNativeEncoder(device="cpu")
        v1 = encoder.encode("Test sentence")
        v2 = encoder.encode("Test sentence")
        assert np.allclose(v1, v2, atol=1e-6)

    def test_encode_single_character(self) -> None:
        """单字符输入应正常编码。"""
        encoder = ELFNativeEncoder(device="cpu")
        vec = encoder.encode("a")
        assert vec.shape == (768,)
        assert vec.dtype == np.float32

    def test_model_name_default(self) -> None:
        """默认模型名为 embedded-language-flows/ELF-B-owt-torch。"""
        encoder = ELFNativeEncoder(device="cpu")
        assert encoder.model_name == "embedded-language-flows/ELF-B-owt-torch"

    def test_model_loading_error(self) -> None:
        """模型加载失败应抛出 RuntimeError。"""
        with patch(
            "src.elf.native_encoder.T5EncoderModel.from_pretrained",
            side_effect=Exception("Download failed"),
        ):
            with pytest.raises(RuntimeError, match="无法加载 ELF 原生模型"):
                ELFNativeEncoder(device="cpu")
