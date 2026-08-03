"""ELF 原生 DiT denoiser 模型单元测试。

使用 mock checkpoint, 无须下载真实 ELF-B 权重。
"""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from src.elf.native_model import (
    DENOISER_NOISE_SCALE,
    LATENT_STD,
    T_EPS,
    BottleneckTextProj,
    ELFDenoiser,
    FinalLayer,
    RMSNorm,
    SwiGLUFFN,
    TimestepEmbedder,
)

# ── helpers ────────────────────────────────────


def _make_mini_checkpoint(tmp_path: Path) -> str:
    """构造一个最小 checkpoint, 包含 ELFDenoiser 所需全部权重 key。

    权重随机初始化, 仅用于 shape 校验与加载流程测试。
    """
    from src.elf.native_model import (
        _BOTTLENECK_DIM,
        _DEPTH,
        _HEAD_DIM,
        _HIDDEN_SIZE,
        _NUM_HEADS,
        _NUM_MODE_TOKENS,
        _NUM_SELF_COND_CFG_TOKENS,
        _NUM_TIME_TOKENS,
        _TEXT_ENCODER_DIM,
        _VOCAB_SIZE,
    )

    params: dict[str, torch.Tensor] = {}

    # self_cond_proj
    params["self_cond_proj.weight"] = torch.randn(_TEXT_ENCODER_DIM, _TEXT_ENCODER_DIM * 2)
    params["self_cond_proj.bias"] = torch.randn(_TEXT_ENCODER_DIM)

    # text_proj (BottleneckTextProj)
    params["text_proj.proj1.weight"] = torch.randn(_BOTTLENECK_DIM, _TEXT_ENCODER_DIM)
    params["text_proj.proj2.weight"] = torch.randn(_HIDDEN_SIZE, _BOTTLENECK_DIM)
    params["text_proj.proj2.bias"] = torch.randn(_HIDDEN_SIZE)

    # t_embedder
    params["t_embedder.mlp.0.weight"] = torch.randn(_HIDDEN_SIZE, 256)
    params["t_embedder.mlp.0.bias"] = torch.randn(_HIDDEN_SIZE)
    params["t_embedder.mlp.2.weight"] = torch.randn(_HIDDEN_SIZE, _HIDDEN_SIZE)
    params["t_embedder.mlp.2.bias"] = torch.randn(_HIDDEN_SIZE)

    # t_emb_tokens
    params["t_emb_tokens"] = torch.randn(1, _NUM_TIME_TOKENS, _HIDDEN_SIZE)

    # self_cond_cfg_embedder
    params["self_cond_cfg_embedder.mlp.0.weight"] = torch.randn(_HIDDEN_SIZE, 256)
    params["self_cond_cfg_embedder.mlp.0.bias"] = torch.randn(_HIDDEN_SIZE)
    params["self_cond_cfg_embedder.mlp.2.weight"] = torch.randn(_HIDDEN_SIZE, _HIDDEN_SIZE)
    params["self_cond_cfg_embedder.mlp.2.bias"] = torch.randn(_HIDDEN_SIZE)

    # self_cond_cfg_tokens
    params["self_cond_cfg_tokens"] = torch.randn(1, _NUM_SELF_COND_CFG_TOKENS, _HIDDEN_SIZE)

    # mode_tokens
    params["mode_tokens"] = torch.randn(1, _NUM_MODE_TOKENS, _HIDDEN_SIZE)

    # blocks
    for i in range(_DEPTH):
        prefix = f"blocks.{i}"
        params[f"{prefix}.norm1.weight"] = torch.randn(_HIDDEN_SIZE)
        params[f"{prefix}.attn.qkv.weight"] = torch.randn(_HIDDEN_SIZE * 3, _HIDDEN_SIZE)
        params[f"{prefix}.attn.qkv.bias"] = torch.randn(_HIDDEN_SIZE * 3)
        params[f"{prefix}.attn.q_norm.weight"] = torch.randn(_HEAD_DIM)
        params[f"{prefix}.attn.k_norm.weight"] = torch.randn(_HEAD_DIM)
        params[f"{prefix}.attn.proj.weight"] = torch.randn(_HIDDEN_SIZE, _HIDDEN_SIZE)
        params[f"{prefix}.attn.proj.bias"] = torch.randn(_HIDDEN_SIZE)
        params[f"{prefix}.norm2.weight"] = torch.randn(_HIDDEN_SIZE)
        inner_dim = int(_HIDDEN_SIZE * 4 * 2 / 3)
        params[f"{prefix}.mlp.w12.weight"] = torch.randn(inner_dim * 2, _HIDDEN_SIZE)
        params[f"{prefix}.mlp.w12.bias"] = torch.randn(inner_dim * 2)
        params[f"{prefix}.mlp.w3.weight"] = torch.randn(_HIDDEN_SIZE, inner_dim)
        params[f"{prefix}.mlp.w3.bias"] = torch.randn(_HIDDEN_SIZE)

    # final_layer
    params["final_layer.norm_final.weight"] = torch.randn(_HIDDEN_SIZE)
    params["final_layer.linear.weight"] = torch.randn(_TEXT_ENCODER_DIM, _HIDDEN_SIZE)
    params["final_layer.linear.bias"] = torch.randn(_TEXT_ENCODER_DIM)

    # decoder 分支权重
    params["proj_kernel"] = torch.randn(_HIDDEN_SIZE, _TEXT_ENCODER_DIM)
    params["proj_bias"] = torch.randn(_TEXT_ENCODER_DIM)
    params["unembed_kernel"] = torch.randn(_TEXT_ENCODER_DIM, _VOCAB_SIZE)
    params["unembed_bias"] = torch.randn(_VOCAB_SIZE)

    state = {"ema_params1": params}
    ckpt_path = tmp_path / "mini_checkpoint.pt"
    torch.save(state, ckpt_path)
    return str(ckpt_path)


# ── ELFDenoiser 模型测试 ──────────────────────


class TestELFDenoiser:
    """ELFDenoiser 模型单元测试（mock checkpoint）。"""

    @pytest.fixture(autouse=True)
    def _patch_default_checkpoint(self, tmp_path: Path, monkeypatch) -> None:
        """将 _DEFAULT_CHECKPOINT 替换为临时 mini checkpoint。"""
        ckpt = _make_mini_checkpoint(tmp_path)
        monkeypatch.setattr(
            "src.elf.native_model._DEFAULT_CHECKPOINT",
            Path(ckpt),
        )

    def test_init_loads_weights(self) -> None:
        """初始化应成功加载权重, 不抛异常。"""
        model = ELFDenoiser(device="cpu")
        assert model.hidden_size == 768
        assert len(model.blocks) == 12

    def test_forward_single(self) -> None:
        """单条 latent (1, 1, 512) forward 返回正确 shape。"""
        model = ELFDenoiser(device="cpu")
        z_t = torch.randn(1, 1, 512)
        t = torch.tensor([0.4])
        out = model(z_t, t)
        assert out.shape == (1, 1, 512)

    def test_forward_batch(self) -> None:
        """批量 latent (3, 1, 512) forward 返回正确 shape。"""
        model = ELFDenoiser(device="cpu")
        z_t = torch.randn(3, 1, 512)
        t = torch.tensor([0.3, 0.5, 0.7])
        out = model(z_t, t)
        assert out.shape == (3, 1, 512)

    def test_forward_scalar_t(self) -> None:
        """标量 t 自动扩展到 batch 维度。"""
        model = ELFDenoiser(device="cpu")
        z_t = torch.randn(2, 1, 512)
        out = model(z_t, torch.tensor(0.4))
        assert out.shape == (2, 1, 512)

    def test_forward_with_self_cond(self) -> None:
        """self_cond_cfg_scale 传入时正常返回。"""
        model = ELFDenoiser(device="cpu")
        z_t = torch.randn(1, 1, 512)
        t = torch.tensor([0.4])
        out = model(z_t, t, self_cond_cfg_scale=torch.tensor([1.0]))
        assert out.shape == (1, 1, 512)

    def test_forward_self_cond_double_dim(self) -> None:
        """最后一维为 1024(拼接 self-cond)时, 经 self_cond_proj 压回 512。"""
        model = ELFDenoiser(device="cpu")
        z_t = torch.randn(1, 1, 1024)
        t = torch.tensor([0.4])
        out = model(z_t, t)
        assert out.shape == (1, 1, 512)

    def test_missing_checkpoint_raises(self) -> None:
        """checkpoint 文件不存在时抛出 FileNotFoundError。"""
        with pytest.raises(FileNotFoundError, match="ELF checkpoint 不存在"):
            ELFDenoiser(checkpoint_path="/nonexistent/path.pt", device="cpu")

    def test_unexpected_keys_warning(self, tmp_path, caplog) -> None:
        """extra key 在 checkpoint 中应触发 warning 而非崩溃。"""
        ckpt = _make_mini_checkpoint(tmp_path)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        state["ema_params1"]["extra_surprise_key"] = torch.zeros(1)
        extra_ckpt = tmp_path / "extra.pt"
        torch.save(state, extra_ckpt)

        import logging

        with caplog.at_level(logging.WARNING):
            model = ELFDenoiser(checkpoint_path=str(extra_ckpt), device="cpu")
        assert model is not None
        # 应该有 warning 日志提及非预期权重键
        assert any("非预期权重键" in r.message for r in caplog.records)

    def test_missing_keys_raises(self, tmp_path) -> None:
        """缺少必需 key 应抛出 RuntimeError。"""
        ckpt = _make_mini_checkpoint(tmp_path)
        state = torch.load(ckpt, map_location="cpu", weights_only=True)
        del state["ema_params1"]["blocks.0.attn.qkv.weight"]
        bad_ckpt = tmp_path / "bad.pt"
        torch.save(state, bad_ckpt)

        with pytest.raises(RuntimeError, match="missing keys"):
            ELFDenoiser(checkpoint_path=str(bad_ckpt), device="cpu")

    def test_device_property(self) -> None:
        """self.device 应与构造参数一致。"""
        model = ELFDenoiser(device="cpu")
        assert model.device == torch.device("cpu")


# ── 子模块单元测试 ────────────────────────────


class TestRMSNorm:
    """RMSNorm 单元测试。"""

    def test_shape_preserved(self) -> None:
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64)
        out = norm(x)
        assert out.shape == (2, 10, 64)

    def test_dtype_preserved(self) -> None:
        norm = RMSNorm(64)
        x = torch.randn(2, 10, 64, dtype=torch.float16)
        out = norm(x)
        assert out.dtype == torch.float16


class TestSwiGLUFFN:
    """SwiGLUFFN 单元测试。"""

    def test_shape_preserved(self) -> None:
        ffn = SwiGLUFFN(768, 3072)
        x = torch.randn(2, 10, 768)
        out = ffn(x)
        assert out.shape == (2, 10, 768)


class TestTimestepEmbedder:
    """TimestepEmbedder 单元测试。"""

    def test_output_shape(self) -> None:
        emb = TimestepEmbedder(768)
        t = torch.tensor([0.3, 0.7])
        out = emb(t)
        assert out.shape == (2, 768)


class TestBottleneckTextProj:
    """BottleneckTextProj 单元测试。"""

    def test_output_shape(self) -> None:
        proj = BottleneckTextProj(512, 768, 128)
        x = torch.randn(3, 512)
        out = proj(x)
        assert out.shape == (3, 768)


class TestFinalLayer:
    """FinalLayer 单元测试。"""

    def test_output_shape(self) -> None:
        layer = FinalLayer(768, 512)
        x = torch.randn(2, 5, 768)
        out = layer(x)
        assert out.shape == (2, 5, 512)


# ── 公开常量测试 ──────────────────────────────


class TestPublicConstants:
    """验证公开常量值正确。"""

    def test_latent_std(self) -> None:
        assert LATENT_STD == 0.2

    def test_denoiser_noise_scale(self) -> None:
        assert DENOISER_NOISE_SCALE == 2.0

    def test_t_eps(self) -> None:
        assert T_EPS == 0.05
