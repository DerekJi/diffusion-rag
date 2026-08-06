"""ELF 原生编码器单元测试。

使用 mock 模式，无须下载真实模型或 ELF 权重。
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from src.elf.native_encoder import ELFNativeEncoder, _load_elf_checkpoint


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


class TestLoadElfCheckpoint:
    """_load_elf_checkpoint 的加载逻辑测试（离线，无网络）。"""

    @pytest.fixture()
    def projection(self) -> torch.nn.Linear:
        """固定随机初始化的投影层 (512 → 768, 无 bias)。"""
        torch.manual_seed(0)
        return torch.nn.Linear(512, 768, bias=False)

    def _write_train_ckpt(
        self, path, proj_kernel: torch.Tensor | None = None, with_ema: bool = True
    ) -> None:
        """写一个 ELF-B 训练检查点格式文件（params/ema_params1 嵌套）。"""
        params = {
            "blocks.0.attn.qkv.weight": torch.randn(768, 768),
            "text_proj.proj2.weight": torch.randn(768, 128),
        }
        state: dict = {"params": params, "opt_state": {}, "step": 100, "epoch": 5}
        if proj_kernel is not None:
            params["proj_kernel"] = proj_kernel
        if with_ema:
            ema = dict(params)
            if proj_kernel is not None:
                ema["proj_kernel"] = proj_kernel + 0.01  # EMA 与 params 不同，便于断言
            state["ema_params1"] = ema
        torch.save(state, path)

    def test_train_ckpt_uses_ema(self, projection, tmp_path) -> None:
        """训练检查点格式应解包并优先使用 ema_params1 的 proj_kernel。"""
        ckpt = tmp_path / "ckpt.pt"
        kernel = torch.randn(768, 512)
        self._write_train_ckpt(ckpt, proj_kernel=kernel, with_ema=True)

        loaded = _load_elf_checkpoint(projection, str(ckpt), torch.device("cpu"))

        assert loaded is True
        assert torch.equal(projection.weight.data, kernel + 0.01)

    def test_train_ckpt_fallback_params(self, projection, tmp_path) -> None:
        """无 ema_params1 时应回退 params 的 proj_kernel。"""
        ckpt = tmp_path / "ckpt.pt"
        kernel = torch.randn(768, 512)
        self._write_train_ckpt(ckpt, proj_kernel=kernel, with_ema=False)

        loaded = _load_elf_checkpoint(projection, str(ckpt), torch.device("cpu"))

        assert loaded is True
        assert torch.equal(projection.weight.data, kernel)

    def test_flat_ckpt_legacy_key(self, projection, tmp_path) -> None:
        """平铺格式的旧 key projection.weight 仍应兼容。"""
        ckpt = tmp_path / "ckpt.pt"
        kernel = torch.randn(768, 512)
        torch.save({"projection.weight": kernel}, ckpt)

        loaded = _load_elf_checkpoint(projection, str(ckpt), torch.device("cpu"))

        assert loaded is True
        assert torch.equal(projection.weight.data, kernel)

    def test_shape_mismatch_skipped(self, projection, tmp_path) -> None:
        """shape 不匹配的 key 应跳过并返回 False。"""
        ckpt = tmp_path / "ckpt.pt"
        before = projection.weight.data.clone()
        torch.save({"proj_kernel": torch.randn(128, 512)}, ckpt)

        loaded = _load_elf_checkpoint(projection, str(ckpt), torch.device("cpu"))

        assert loaded is False
        assert torch.equal(projection.weight.data, before)

    def test_local_dir_resolves_filename(self, projection, tmp_path) -> None:
        """传入本地目录时应拼接 checkpoint_95085。"""
        kernel = torch.randn(768, 512)
        repo_dir = tmp_path / "ELF-B"
        repo_dir.mkdir()
        torch.save({"params": {"proj_kernel": kernel}}, repo_dir / "checkpoint_95085")

        loaded = _load_elf_checkpoint(projection, str(repo_dir), torch.device("cpu"))

        assert loaded is True
        assert torch.equal(projection.weight.data, kernel)

    def test_hf_id_prefers_local_models(self, projection, tmp_path, monkeypatch) -> None:
        """HF 仓库 ID 且本地 models/ 已存在时应使用本地文件，不调用 hf_hub_download。"""
        kernel = torch.randn(768, 512)
        local_repo = Path("models") / "test-org" / "test-repo"
        local_repo.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"params": {"proj_kernel": kernel}, "ema_params1": {"proj_kernel": kernel}},
            local_repo / "checkpoint_95085",
        )
        monkeypatch.setattr(
            "huggingface_hub.hf_hub_download",
            lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应调用 hf_hub_download")),
        )
        try:
            loaded = _load_elf_checkpoint(projection, "test-org/test-repo", torch.device("cpu"))
            assert loaded is True
            assert torch.equal(projection.weight.data, kernel)
        finally:
            import shutil

            shutil.rmtree(Path("models") / "test-org")

    def test_missing_file_returns_false(self, projection, tmp_path) -> None:
        """文件不存在时应返回 False 而不是抛异常。"""
        loaded = _load_elf_checkpoint(projection, str(tmp_path / "nope.pt"), torch.device("cpu"))
        assert loaded is False


class TestEncodeTokens:
    """encode_tokens 方法单元测试（token 级编码，无 mean-pooling）。"""

    @pytest.fixture()
    def encoder(self) -> ELFNativeEncoder:
        """创建 mock encoder 并配置 _t5 / _tokenizer 返回固定值。"""
        mock_t5 = MagicMock()
        mock_tokenizer = MagicMock()
        mock_proj = MagicMock(spec=torch.nn.Linear)

        patchers = [
            patch("src.elf.native_encoder.T5EncoderModel.from_pretrained", return_value=mock_t5),
            patch(
                "src.elf.native_encoder.T5Tokenizer.from_pretrained", return_value=mock_tokenizer
            ),
            patch("src.elf.native_encoder.torch.nn.Linear", return_value=mock_proj),
            patch("src.elf.native_encoder._load_elf_checkpoint", return_value=False),
        ]
        for p in patchers:
            p.start()

        encoder = ELFNativeEncoder(device="cpu")

        # 配置 _tokenizer: 返回包含 attention_mask 的类 BatchEncoding 对象
        class _MockBatchEncoding(dict):
            def to(self, device):
                return self

        def _make_tokenizer_output(texts, **kwargs):
            max_len = kwargs.get("max_length", 64)
            B = len(texts)
            # 模拟每条文本不定长，padding 到 max_len
            mask = torch.zeros(B, max_len, dtype=torch.long)
            for b in range(B):
                n_tokens = min(len(texts[b].split()) + 2, max_len)
                mask[b, :n_tokens] = 1
            return _MockBatchEncoding({"attention_mask": mask})

        encoder._tokenizer.side_effect = _make_tokenizer_output  # type: ignore[attr-defined]

        # 配置 _t5: 返回 last_hidden_state (B, L, 512)
        def _make_t5_output(**inputs):
            mask = inputs.get("attention_mask", None)
            if mask is not None:
                B, L = mask.shape[:2]
            else:
                B, L = 1, 64
            hidden = torch.randn(B, L, 512, dtype=torch.float32)
            mock_result = MagicMock()
            mock_result.last_hidden_state = hidden
            return mock_result

        encoder._t5.side_effect = _make_t5_output  # type: ignore[attr-defined]

        yield encoder

        for p in patchers:
            p.stop()

    def test_single_text_shape(self, encoder: ELFNativeEncoder) -> None:
        """单文本编码 shape 应为 (1, L, 512) + mask (1, L)。"""
        tokens, mask = encoder.encode_tokens(["Hello world"], max_tokens=64)
        assert tokens.shape[0] == 1
        assert mask.shape[0] == 1
        assert tokens.shape[1] == mask.shape[1]
        assert tokens.shape[2] == 512
        assert tokens.dtype == np.float32
        assert mask.dtype == np.float32

    def test_batch_shape_alignment(self, encoder: ELFNativeEncoder) -> None:
        """批量文本各维度对齐。"""
        texts = ["a short text", "another somewhat longer sentence", "third"]
        tokens, mask = encoder.encode_tokens(texts, max_tokens=64)
        assert tokens.shape[0] == 3
        assert mask.shape[0] == 3
        assert tokens.shape[1] == mask.shape[1]
        assert tokens.shape[2] == 512
        assert tokens.dtype == np.float32

    def test_padding_mask_values(self, encoder: ELFNativeEncoder) -> None:
        """mask 值应为 0 或 1（有效 token=1, padding=0）。"""
        _, mask = encoder.encode_tokens(["test text"], max_tokens=64)
        assert mask.dtype == np.float32
        assert np.all((mask == 0.0) | (mask == 1.0))
        # 至少有一个有效 token（"test text" 不会全是 padding）
        assert mask.sum() > 0

    def test_empty_list_early_return(self, encoder: ELFNativeEncoder) -> None:
        """空列表应返回空 shape (0, 0, 512) + (0, 0)。"""
        tokens, mask = encoder.encode_tokens([], max_tokens=64)
        assert tokens.shape == (0, 0, 512)
        assert mask.shape == (0, 0)
        assert tokens.dtype == np.float32
        assert mask.dtype == np.float32
