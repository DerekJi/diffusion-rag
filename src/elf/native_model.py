"""ELF 原生 DiT denoiser 模型(issue #37)。

加载 checkpoint_95085 中完整的 ELF-B 权重(此前仅加载 proj_kernel),
实现真实的速度场: 在 T5 hidden(512-dim)空间做 flow matching,
网络输出 x0 预测, 速度 v = (x0_pred - z) / max(1 - t, t_eps)。

架构(参考 embedded-language-flows/ELF-B-owt-torch 权重与
Xrenya/ELF port, 键名与官方 checkpoint 严格对应):
- 12 个 DiT block: pre-norm(RMSNorm) + fused-qkv attention(q/k RMSNorm
  + RoPE) + SwiGLU MLP
- 输入序列: [t_emb_tokens(4) + t_embedder(t); self_cond_cfg_tokens(4) +
  self_cond_cfg_embedder(scale); mode_tokens(4); latent tokens]
- text_proj: 512 → 128(bottleneck) → 768(DiT hidden)
- final_layer: RMSNorm + Linear(768 → 512, 零初始化) 输出 x0 预测

用法::

    model = ELFDenoiser()
    x0 = model(z_t, t)          # z_t: (B, L, 512), t: (B,)
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from src.utils.logger import get_logger

logger = get_logger(__name__)

# 官方 checkpoint 路径与模型配置(config.yml 对应项)
_DEFAULT_CHECKPOINT = (
    Path("models") / "embedded-language-flows" / "ELF-B-owt-torch" / "checkpoint_95085"
)
_HIDDEN_SIZE = 768  # DiT hidden (blocks 维度)
_DEPTH = 12  # ELF-B
_NUM_HEADS = 12
_HEAD_DIM = _HIDDEN_SIZE // _NUM_HEADS  # 64
_TEXT_ENCODER_DIM = 512  # T5 hidden
_BOTTLENECK_DIM = 128
_NUM_TIME_TOKENS = 4
_NUM_SELF_COND_CFG_TOKENS = 4
_NUM_MODE_TOKENS = 4
LATENT_STD = 0.2  # config latent_std(用于 latent 归一化)
DENOISER_NOISE_SCALE = 2.0  # config denoiser_noise_scale
T_EPS = 0.05  # config t_eps
_VOCAB_SIZE = 32100  # T5 词表(unembed 分支)


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x[..., 0], x[..., 1]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


class TextRotaryEmbeddingFast(nn.Module):
    """RoPE, 前 num_empty_token 个位置不旋转(前缀 token)。"""

    def __init__(
        self,
        dim: int,
        pt_seq_len: int = 1024,
        theta: float = 10000.0,
        num_empty_token: int = 0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.pt_seq_len = pt_seq_len
        self.theta = theta
        self.num_empty_token = num_empty_token

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        seq_len = t.shape[-2]
        main_len = max(seq_len - self.num_empty_token, 0)
        device = t.device
        calc_dtype = torch.float32

        freqs = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.dim, 2, device=device, dtype=calc_dtype)[: self.dim // 2]
                / self.dim
            )
        )
        pos = torch.arange(self.ft_seq_len, device=device, dtype=calc_dtype)
        pos = pos / self.ft_seq_len * self.pt_seq_len
        freqs_main = torch.einsum("n,f->nf", pos, freqs)
        freqs_main = freqs_main.repeat_interleave(2, dim=-1)[:main_len]

        parts_cos: list[torch.Tensor] = []
        parts_sin: list[torch.Tensor] = []
        if self.num_empty_token > 0:
            empty = min(self.num_empty_token, seq_len)
            parts_cos.append(torch.ones(empty, self.dim, device=device, dtype=calc_dtype))
            parts_sin.append(torch.zeros(empty, self.dim, device=device, dtype=calc_dtype))
        if main_len > 0:
            parts_cos.append(torch.cos(freqs_main))
            parts_sin.append(torch.sin(freqs_main))

        cos = torch.cat(parts_cos, dim=0).to(dtype=t.dtype)
        sin = torch.cat(parts_sin, dim=0).to(dtype=t.dtype)
        return t * cos + _rotate_half(t) * sin

    @property
    def ft_seq_len(self) -> int:
        return self.pt_seq_len


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        variance = hidden_states.float().pow(2).mean(dim=-1, keepdim=True)
        inv_std = torch.rsqrt(variance + self.eps).to(dtype=input_dtype)
        return self.weight.to(dtype=input_dtype) * (hidden_states * inv_std)


class Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 12, qkv_bias: bool = True) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        head_dim = dim // num_heads
        self.q_norm = RMSNorm(head_dim)
        self.k_norm = RMSNorm(head_dim)
        self.proj = nn.Linear(dim, dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_fn: TextRotaryEmbeddingFast | None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, channels = x.shape
        head_dim = self.dim // self.num_heads
        qkv = self.qkv(x)
        qkv = qkv.reshape(bsz, seq_len, 3, self.num_heads, head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = self.q_norm(q)
        k = self.k_norm(k)
        if rope_fn is not None:
            q = rope_fn(q)
            k = rope_fn(k)
        x = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask)
        x = x.transpose(1, 2).reshape(bsz, seq_len, channels)
        return self.proj(x)  # type: ignore[no-any-return]


class SwiGLUFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        inner_dim = int(hidden_dim * 2 / 3)
        self.w12 = nn.Linear(dim, inner_dim * 2, bias=True)
        self.w3 = nn.Linear(inner_dim, dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(x1) * x2)  # type: ignore[no-any-return]


class ELFBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.norm1 = RMSNorm(hidden_size, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True)
        self.norm2 = RMSNorm(hidden_size, eps=1e-6)
        self.mlp = SwiGLUFFN(hidden_size, mlp_hidden_dim)

    def forward(
        self,
        x: torch.Tensor,
        rope_fn: TextRotaryEmbeddingFast | None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), rope_fn, attention_mask=attention_mask)
        x = x + self.mlp(self.norm2(x))
        return x


class TimestepEmbedder(nn.Module):
    """时间步嵌入: sin 嵌入(256) → Linear → SiLU → Linear。"""

    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    @staticmethod
    def timestep_embedding(t: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(0, half, device=t.device, dtype=torch.float32)
            / half
        )
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(  # type: ignore[no-any-return]
            self.timestep_embedding(t, self.frequency_embedding_size)
        )


class BottleneckTextProj(nn.Module):
    """T5 hidden(512) → bottleneck(128) → DiT hidden(768)。"""

    def __init__(self, text_encoder_dim: int, hidden_size: int, bottleneck_dim: int) -> None:
        super().__init__()
        self.proj1 = nn.Linear(text_encoder_dim, bottleneck_dim, bias=False)
        self.proj2 = nn.Linear(bottleneck_dim, hidden_size, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj2(self.proj1(x))  # type: ignore[no-any-return]


class FinalLayer(nn.Module):
    """DiT 输出(768) → x0 预测(512), Linear 零初始化。"""

    def __init__(self, hidden_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(self.norm_final(x))  # type: ignore[no-any-return]


class ELFDenoiser(nn.Module):
    """ELF-B 原生 DiT denoiser(加载完整 checkpoint 权重)。

    输入: latent z(B, L, 512) + 时间步 t(B,);输出: x0 预测(B, L, 512)。
    """

    def __init__(
        self,
        checkpoint_path: str | Path | None = None,
        device: str = "cpu",
    ) -> None:
        super().__init__()
        if checkpoint_path is None:
            checkpoint_path = str(_DEFAULT_CHECKPOINT)
        self.text_encoder_dim = _TEXT_ENCODER_DIM
        self.max_length = 1024
        self.hidden_size = _HIDDEN_SIZE
        self.num_heads = _NUM_HEADS
        self.num_time_tokens = _NUM_TIME_TOKENS
        self.num_self_cond_cfg_tokens = _NUM_SELF_COND_CFG_TOKENS
        self.num_model_mode_tokens = _NUM_MODE_TOKENS

        self.self_cond_proj = nn.Linear(_TEXT_ENCODER_DIM * 2, _TEXT_ENCODER_DIM)
        self.text_proj = BottleneckTextProj(_TEXT_ENCODER_DIM, _HIDDEN_SIZE, _BOTTLENECK_DIM)
        self.t_embedder = TimestepEmbedder(_HIDDEN_SIZE)
        self.t_emb_tokens = nn.Parameter(torch.empty(1, _NUM_TIME_TOKENS, _HIDDEN_SIZE))
        self.self_cond_cfg_embedder = TimestepEmbedder(_HIDDEN_SIZE)
        self.self_cond_cfg_tokens = nn.Parameter(
            torch.empty(1, _NUM_SELF_COND_CFG_TOKENS, _HIDDEN_SIZE)
        )
        self.mode_tokens = nn.Parameter(torch.empty(1, _NUM_MODE_TOKENS, _HIDDEN_SIZE))

        prefix_total = _NUM_MODE_TOKENS + _NUM_TIME_TOKENS + _NUM_SELF_COND_CFG_TOKENS
        self.feat_rope = TextRotaryEmbeddingFast(
            dim=_HEAD_DIM,
            pt_seq_len=self.max_length,
            num_empty_token=prefix_total,
        )

        self.blocks = nn.ModuleList(
            [ELFBlock(_HIDDEN_SIZE, _NUM_HEADS, mlp_ratio=4.0) for _ in range(_DEPTH)]
        )
        self.final_layer = FinalLayer(_HIDDEN_SIZE, _TEXT_ENCODER_DIM)

        # decoder 分支权重(官方 checkpoint 含, 检索增强不使用, 仅完整承载)
        self.proj_kernel = nn.Parameter(torch.empty(_HIDDEN_SIZE, _TEXT_ENCODER_DIM))
        self.proj_bias = nn.Parameter(torch.empty(_TEXT_ENCODER_DIM))
        self.unembed_kernel = nn.Parameter(torch.empty(_TEXT_ENCODER_DIM, _VOCAB_SIZE))
        self.unembed_bias = nn.Parameter(torch.empty(_VOCAB_SIZE))

        self.device = torch.device(device)
        self._load_checkpoint(checkpoint_path)
        self.to(self.device)
        self.eval()
        logger.info(
            "ELFDenoiser 加载完成 (depth=%d, hidden=%d, params=%d, device=%s)",
            _DEPTH,
            _HIDDEN_SIZE,
            sum(p.numel() for p in self.parameters()),
            device,
        )

    # ── 权重加载 ──────────────────────────────

    def _load_checkpoint(self, checkpoint_path: str | Path) -> None:
        """从官方 checkpoint 的 ema_params1 加载权重。

        checkpoint 键名已是 PyTorch 风格, 仅 t_embedder / self_cond_cfg_embedder
        的 mlp_0 / mlp_2 需归一化为 nn.Sequential 的 mlp.0 / mlp.2。
        """
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"ELF checkpoint 不存在: {path}")
        state = torch.load(path, map_location="cpu", weights_only=True)
        weights = state.get("ema_params1") or state.get("params") or state

        normalized: dict[str, torch.Tensor] = {}
        for key, value in weights.items():
            if not isinstance(value, torch.Tensor):
                continue
            norm_key = key.replace("t_embedder.mlp_", "t_embedder.mlp.")
            norm_key = norm_key.replace(
                "self_cond_cfg_embedder.mlp_", "self_cond_cfg_embedder.mlp."
            )
            normalized[norm_key] = value

        missing, unexpected = self.load_state_dict(normalized, strict=False)
        if missing:
            raise RuntimeError(f"ELFDenoiser 权重不匹配: missing keys={missing[:10]}")
        if unexpected:
            logger.warning("ELFDenoiser checkpoint 含非预期权重键(将忽略): %s", unexpected[:10])
        logger.info("ELFDenoiser checkpoint 加载成功 (%s)", path)

    # ── 前向 ──────────────────────────────────

    def _make_prefix(self, emb: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.expand(emb.shape[0], -1, -1) + emb[:, None, :]

    def build_context(
        self, t: torch.Tensor, self_cond_cfg_scale: torch.Tensor | None = None
    ) -> torch.Tensor:
        """构造 DiT 输入前缀: 时间步嵌入 token + 可选 self-cond cfg token。

        Args:
            t: 时间步, (B,)。
            self_cond_cfg_scale: self-cond cfg 标量 (B,), 可选。

        Returns:
            前缀张量, shape (B, num_prefix_tokens, hidden_size)。
        """
        prefix_tokens = [self._make_prefix(self.t_embedder(t), self.t_emb_tokens)]
        if self_cond_cfg_scale is not None:
            prefix_tokens.append(
                self._make_prefix(
                    self.self_cond_cfg_embedder(self_cond_cfg_scale),
                    self.self_cond_cfg_tokens,
                )
            )
        return torch.cat(prefix_tokens, dim=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        self_cond_cfg_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """预测 x0。

        Args:
            x: latent, (B, L, 512);若最后一维为 1024(拼接 self-cond),
               先经 self_cond_proj 压回 512。
            t: 时间步, (B,) 或标量。
            self_cond_cfg_scale: self-cond cfg 标量 (B,), 可选。

        Returns:
            x0 预测, (B, L, 512)。
        """
        if t.ndim == 0:
            t = t.expand(x.shape[0])
        elif t.shape[0] != x.shape[0]:
            # 批量前向: 单值时间步展开到 batch 大小
            t = t.expand(x.shape[0])
        if self_cond_cfg_scale is not None:
            if self_cond_cfg_scale.ndim == 0:
                self_cond_cfg_scale = self_cond_cfg_scale.expand(x.shape[0])
            elif self_cond_cfg_scale.shape[0] != x.shape[0]:
                self_cond_cfg_scale = self_cond_cfg_scale.expand(x.shape[0])

        bsz = x.shape[0]
        if x.shape[-1] == 2 * self.text_encoder_dim:
            x = self.self_cond_proj(x.float())
        x = self.text_proj(x.float())
        context_prefix_tokens = self.build_context(t, self_cond_cfg_scale)

        # 序列: [context(t + self_cond); mode(4, gate=0 全零); latent]
        mode_tokens = self.mode_tokens.expand(bsz, -1, -1) * 0.0
        x = torch.cat([mode_tokens, x], dim=1)
        model_mode_offset = self.num_model_mode_tokens

        prefix_len = context_prefix_tokens.shape[1]
        x = torch.cat([context_prefix_tokens, x], dim=1)

        for block in self.blocks:
            x = block(x, rope_fn=self.feat_rope)

        x = x[:, prefix_len + model_mode_offset :]
        return self.final_layer(x.float())  # type: ignore[no-any-return]
