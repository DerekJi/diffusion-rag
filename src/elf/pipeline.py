"""ELF 扩散增强链路。

整合编码器 (ELFEncoder)、扩散正反向 (add_noise / denoise_with_cfg) 与
Velocity 级 CFG 去噪的完整链路。

典型用法::

    pipe = ELFPipeline()
    vec = pipe.enhance("query text", steps=2, noise_t=0.4, cfg_scale=2.0)
    # 或仅编码不增强:
    vec = pipe.encode("query text")
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from numpy.typing import NDArray

from src.elf.diffusion import DEFAULT_NOISE_T, add_noise, denoise_with_cfg, sigma
from src.elf.encoder import ELFEncoder
from src.elf.native_encoder import ELFNativeEncoder
from src.elf.native_model import (
    _DENOISER_NOISE_SCALE,
    _LATENT_STD,
    _T_EPS,
    ELFDenoiser,
)
from src.utils.device import get_device
from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
#  默认速度场（用于测试/回退）
# ──────────────────────────────────────────────


def _default_model_fn(z: NDArray[np.float32], t: float) -> NDArray[np.float32]:
    """简易去噪速度场: 朝原点收缩。

    当真实扩散模型不可用时回退使用。
    速度方向为 -z，大小与当前噪声水平成反比。

    Args:
        z: 当前向量。shape (d,) 或 (n, d)。
        t: 当前时间步 ∈ [0, 1]。

    Returns:
        速度向量，shape 与 z 相同。
    """
    noise_scale = sigma(t)
    if noise_scale > 1e-8:
        return -z / noise_scale
    return np.zeros_like(z)


# ──────────────────────────────────────────────
#  ELFPipeline
# ──────────────────────────────────────────────

_DEFAULT_STEPS: int = 2
_DEFAULT_CFG_SCALE: float = 2.0


class ELFPipeline:
    """ELF 扩散增强链路。

    整合编码器 (ELFNativeEncoder)、扩散正反向 (add_noise / denoise)
    与 CFG 引导 (cfg_guide) 的完整链路。

    默认使用 ELFNativeEncoder（原生 ELF 模型），可通过 use_native=False
    切换为 BGE 基线编码器。

    Attributes:
        encoder: 底层文本编码器 (ELFNativeEncoder | ELFEncoder)。
        device: 实际使用的设备字符串。
    """

    def __init__(
        self,
        encoder: ELFEncoder | ELFNativeEncoder | None = None,
        model_fn_cond: Callable[[NDArray[np.float32], float], NDArray[np.float32]] | None = None,
        model_fn_uncond: Callable[[NDArray[np.float32], float], NDArray[np.float32]] | None = None,
        device: str = "auto",
        use_native: bool = True,
    ) -> None:
        """初始化 ELF 增强链路。

        Args:
            encoder: 编码器实例。若为 None 则根据 use_native 自动创建。
            model_fn_cond: 条件去噪速度场函数。若为 None 使用默认简易模型。
            model_fn_uncond: 无条件去噪速度场函数。
                            若为 None 则与 model_fn_cond 相同。
            device: 设备字符串，"auto" 表示自动检测。
            use_native: 为 True 时默认使用 ELFNativeEncoder，
                       为 False 时使用 ELFEncoder（BGE 基线）。
        """
        self.device = get_device() if device == "auto" else device
        if encoder is not None:
            self.encoder = encoder
        elif use_native:
            self.encoder = ELFNativeEncoder(device=self.device)
        else:
            self.encoder = ELFEncoder(device=self.device)
        self._model_fn_cond = model_fn_cond
        self._model_fn_uncond = model_fn_uncond or model_fn_cond

        # 真实 ELF-B DiT denoiser(issue #37): use_native 且底层为
        # ELFNativeEncoder 时加载完整 checkpoint 权重, 替换占位速度场。
        self._denoiser: ELFDenoiser | None = None
        if use_native and isinstance(self.encoder, ELFNativeEncoder):
            try:
                self._denoiser = ELFDenoiser(device=self.device)
            except Exception as e:  # pragma: no cover - 权重缺失时的回退
                logger.warning("ELFDenoiser 加载失败, 回退占位速度场: %s", e)

        logger.info(
            "ELFPipeline 已初始化 (device=%s, denoiser=%s, cond_fn=%s, uncond_fn=%s)",
            self.device,
            "ELF-B" if self._denoiser is not None else "none(占位)",
            type(model_fn_cond).__name__ if model_fn_cond is not None else "default",
            type(model_fn_uncond).__name__ if model_fn_uncond is not None else "default",
        )

    # ── 公共方法 ──────────────────────────────

    def encode(self, text: str) -> NDArray[np.float32]:
        """仅编码，不增强。

        Args:
            text: 输入文本。

        Returns:
            L2 归一化的 float32 向量，shape (768,)。
        """
        return self.encoder.encode(text)

    def enhance(
        self,
        text: str,
        steps: int = _DEFAULT_STEPS,
        noise_t: float = DEFAULT_NOISE_T,
        cfg_scale: float = _DEFAULT_CFG_SCALE,
        rng: np.random.Generator | None = None,
        blend_alpha: float = 0.5,
    ) -> NDArray[np.float32]:
        """完整增强链路: encode → 加噪 → 去噪(真实 DiT denoiser)→ 与原始编码 blend → L2 normalize。

        当加载了 ELF-B denoiser(issue #37)时, 在 T5 hidden(512)空间做
        flow matching 增强: latent 归一化 → 插值加噪 → ODE 去噪(网络预测
        x0, v = (x0 - z)/max(1-t, t_eps))→ denormalize → 投影回 768。

        denoiser 会把查询向量推向文本流形高密度区、损失查询特异性
        (sample=20 检索指标为负), 因此与原始编码按 blend_alpha 混合:
        out = (1-α)·raw + α·denoised。实验显示 α≈0.5~0.75 检索最佳
        (recall@10 提升约 4 倍, 纯 denoiser α=1.0 最差)。

        回退路径(denoiser 不可用): 使用占位速度场 denoise_with_cfg。

        Args:
            text: 输入文本。
            steps: 去噪步数。
            noise_t: 加噪强度 ∈ [0, 1]。0=无噪声, 1=完全噪声。
            cfg_scale: CFG 引导强度。1.0 表示不使用 CFG。
            rng: 可选随机数生成器（固定种子用）。
            blend_alpha: denoiser 输出权重 ∈ [0, 1]。
                        0=纯原始编码, 1=纯 denoiser 输出。

        Returns:
            L2 归一化的 float32 增强向量，shape (768,)。
        """
        # 真实 denoiser 路径: 仅当未显式注入自定义速度场时使用(issue #37)
        if (
            self._denoiser is not None
            and self._model_fn_cond is None
            and isinstance(self.encoder, ELFNativeEncoder)
        ):
            return self._enhance_with_denoiser(
                text, steps, noise_t, cfg_scale, rng, blend_alpha
            )

        # ── 回退路径: 占位速度场(测试 / denoiser 不可用) ──
        # Step 1: 编码
        z_0 = self.encoder.encode(text)

        # Step 2: 加噪
        z_t = add_noise(z_0, t=noise_t, rng=rng)

        # Step 3: Velocity 级 CFG 去噪（每步混合速度场）
        cond_fn = self._model_fn_cond or _default_model_fn
        uncond_fn = self._model_fn_uncond or self._model_fn_cond or _default_model_fn
        z_out = denoise_with_cfg(
            z_t,
            cond_fn=cond_fn,
            uncond_fn=uncond_fn,
            steps=steps,
            cfg_scale=cfg_scale,
            t_start=noise_t,
        )

        # Step 4: L2 归一化
        norm = float(np.linalg.norm(z_out))
        if norm > 1e-8:
            z_out = z_out / norm

        logger.debug(
            "enhance(fallback): text=%r, steps=%d, noise_t=%.2f, cfg_scale=%.1f, norm=%.4f",
            text[:50],
            steps,
            noise_t,
            cfg_scale,
            norm,
        )
        return z_out

    # ── 真实 denoiser 增强(issue #37) ─────────

    def _enhance_with_denoiser(
        self,
        text: str,
        steps: int,
        noise_t: float,
        cfg_scale: float,
        rng: np.random.Generator | None,
        blend_alpha: float,
    ) -> NDArray[np.float32]:
        """在 T5 hidden 空间做 flow matching 增强, 与原始编码混合后投影回 768。"""
        assert self._denoiser is not None
        assert isinstance(self.encoder, ELFNativeEncoder)

        pooled = self.encoder.encode_pooled(text)  # (512,) raw hidden
        z0 = np.asarray(pooled, dtype=np.float32) / _LATENT_STD  # latent 归一化
        if rng is None:
            rng = np.random.default_rng()
        eps = rng.standard_normal(512, dtype=np.float32) * _DENOISER_NOISE_SCALE
        z_t = ((1.0 - noise_t) * z0 + noise_t * eps).astype(np.float32)

        # ODE 去噪: t 从 noise_t 线性降至 t_eps, 网络预测 x0, v = (x0 - z)/(1-t)
        t_vals = np.linspace(noise_t, _T_EPS, steps + 1)
        x_pred_prev: NDArray[np.float32] | None = None
        for i in range(steps):
            v, x = self._denoiser_step(z_t, float(t_vals[i]), x_pred_prev, cfg_scale)
            z_t = (z_t + (t_vals[i + 1] - t_vals[i]) * v).astype(np.float32)
            x_pred_prev = x
        z0_pred = x_pred_prev if x_pred_prev is not None else z_t

        pooled_out = np.asarray(z0_pred, dtype=np.float32) * _LATENT_STD  # denormalize
        denoised = self.encoder.embed_from_pooled(pooled_out)  # 512 → 768 L2

        # 与原始编码 blend: 纯 denoiser 输出会损失查询特异性(issue #37 调优)
        raw = self.encoder.encode(text)
        vec = raw * (1.0 - blend_alpha) + denoised * blend_alpha
        norm = float(np.linalg.norm(vec))
        if norm > 1e-8:
            vec = vec / norm

        logger.debug(
            "enhance(denoiser): text=%r, steps=%d, noise_t=%.2f, cfg_scale=%.1f, "
            "blend_alpha=%.2f",
            text[:50],
            steps,
            noise_t,
            cfg_scale,
            blend_alpha,
        )
        return vec

    def _denoiser_step(
        self,
        z: NDArray[np.float32],
        t: float,
        x_pred_prev: NDArray[np.float32] | None,
        cfg_scale: float,
    ) -> tuple[NDArray[np.float32], NDArray[np.float32]]:
        """单步去噪: 模型预测 x0, 由插值公式反推速度; cfg_scale>1 时 CFG 外推。"""
        assert self._denoiser is not None
        z_t = torch.from_numpy(np.asarray(z, dtype=np.float32)).view(1, 1, -1)
        t_t = torch.tensor([t])
        scale_t = torch.tensor([1.0])
        x_prev = (
            torch.zeros_like(z_t)
            if x_pred_prev is None
            else torch.from_numpy(np.asarray(x_pred_prev, dtype=np.float32)).view(1, 1, -1)
        )
        x_den = max(1.0 - t, _T_EPS)

        # 条件预测
        net = self._denoiser(
            torch.cat([z_t, x_prev], dim=-1), t_t, self_cond_cfg_scale=scale_t
        )
        x_cond = net[0, 0].detach().cpu().numpy().astype(np.float32)
        v_cond = ((x_cond - z) / x_den).astype(np.float32)

        if abs(cfg_scale - 1.0) < 1e-6:
            return v_cond, x_cond

        # 无条件预测: latent 清零(z=0, self-cond=0)
        z_u = torch.zeros_like(z_t)
        net_u = self._denoiser(
            torch.cat([z_u, z_u], dim=-1), t_t, self_cond_cfg_scale=scale_t
        )
        x_uncond = net_u[0, 0].detach().cpu().numpy().astype(np.float32)
        v_uncond = (x_uncond / x_den).astype(np.float32)  # z=0

        v = (v_uncond + cfg_scale * (v_cond - v_uncond)).astype(np.float32)
        x = (x_uncond + cfg_scale * (x_cond - x_uncond)).astype(np.float32)
        return v, x

    def enhance_batch(
        self,
        texts: list[str],
        steps: int = _DEFAULT_STEPS,
        noise_t: float = DEFAULT_NOISE_T,
        cfg_scale: float = _DEFAULT_CFG_SCALE,
        rng: np.random.Generator | None = None,
        encode_batch_size: int = 32,
    ) -> NDArray[np.float32]:
        """批量增强: 先批量编码，再在向量维度上批量执行扩散步骤。

        相比逐条串行调用 enhance()，本方法只调用一次批量编码
        （encode_batch），且加噪 / 去噪 / 归一化全部走 numpy 批量
        向量运算，大幅减少模型前向传播次数。

        rng 语义：批量路径用同一个 rng 按行（row-major）消费噪声流，
        因此结果等价于把同一个 rng 实例逐条传给 enhance() 后堆叠；
        若逐条调用时各自新建相同种子的 rng，则每条会得到相同噪声，
        与批量路径不一致。

        Args:
            texts: 文本列表。
            steps: 去噪步数。
            noise_t: 加噪强度。
            cfg_scale: CFG 引导强度。
            rng: 可选随机数生成器（固定种子用）。
            encode_batch_size: 编码器内部批次大小。

        Returns:
            shape (len(texts), 768) float32 数组。
        """
        if not texts:
            raise ValueError("文本列表不能为空")

        # Step 1: 批量编码 (N, 768)
        z_0 = self.encoder.encode_batch(texts, batch_size=encode_batch_size)

        # Step 2: 批量加噪 (N, 768)
        z_t = add_noise(z_0, t=noise_t, rng=rng)

        # Step 3: Velocity 级 CFG 批量去噪（每步同时处理全部 N 条向量）
        cond_fn = self._model_fn_cond or _default_model_fn
        uncond_fn = self._model_fn_uncond or self._model_fn_cond or _default_model_fn
        z_out = denoise_with_cfg(
            z_t,
            cond_fn=cond_fn,
            uncond_fn=uncond_fn,
            steps=steps,
            cfg_scale=cfg_scale,
            t_start=noise_t,
        )

        # Step 4: 逐行 L2 归一化
        norms = np.linalg.norm(z_out, axis=1, keepdims=True)
        safe_norms = np.where(norms > 1e-8, norms, 1.0)
        z_out = np.asarray(z_out / safe_norms, dtype=np.float32)

        logger.debug(
            "enhance_batch: n=%d, steps=%d, noise_t=%.2f, cfg_scale=%.1f",
            len(texts),
            steps,
            noise_t,
            cfg_scale,
        )
        return z_out
