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
from numpy.typing import NDArray

from src.elf.diffusion import DEFAULT_NOISE_T, add_noise, denoise_with_cfg, sigma
from src.elf.encoder import ELFEncoder
from src.elf.native_encoder import ELFNativeEncoder
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

        logger.info(
            "ELFPipeline 已初始化 (device=%s, cond_fn=%s, uncond_fn=%s)",
            self.device,
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
    ) -> NDArray[np.float32]:
        """完整增强链路: encode → add_noise → denoise_with_cfg → L2 normalize。

        Args:
            text: 输入文本。
            steps: 去噪步数。
            noise_t: 加噪强度 ∈ [0, 1]。0=无噪声, 1=完全噪声。
            cfg_scale: CFG 引导强度。1.0 表示不使用 CFG。
            rng: 可选随机数生成器（固定种子用）。

        Returns:
            L2 归一化的 float32 增强向量，shape (768,)。
        """
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
            "enhance: text=%r, steps=%d, noise_t=%.2f, cfg_scale=%.1f, norm=%.4f",
            text[:50],
            steps,
            noise_t,
            cfg_scale,
            norm,
        )
        return z_out

    def enhance_batch(
        self,
        texts: list[str],
        steps: int = _DEFAULT_STEPS,
        noise_t: float = DEFAULT_NOISE_T,
        cfg_scale: float = _DEFAULT_CFG_SCALE,
        rng: np.random.Generator | None = None,
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

        Returns:
            shape (len(texts), 768) float32 数组。
        """
        if not texts:
            raise ValueError("文本列表不能为空")

        # Step 1: 批量编码 (N, 768)
        z_0 = self.encoder.encode_batch(texts)

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
        norms = np.asarray(np.linalg.norm(z_out, axis=1, keepdims=True))
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
