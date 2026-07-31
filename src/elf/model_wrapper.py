"""ELF 模型速度场包装器。

将 ELF 原生模型的  包装为与 denoise() 兼容的速度场函数，
支持条件/无条件两种模式（用于 CFG 引导）。
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
#  ELF 模型速度场包装器
# ──────────────────────────────────────────────


class ELFModelWrapper:
    """ELF 模型速度场包装器。

    将 ELF 原生模型的 forward() 包装为与 denoise() 兼容的速度场函数，
    支持条件/无条件两种模式（用于 CFG 引导）。

    签名兼容::

        model_fn(z: NDArray[np.float32], t: float) -> NDArray[np.float32]

    Attributes:
        cond_vec: 条件文本的编码向量（条件模式使用）。
        use_conditioning: 是否使用条件编码。

    .. note::

        当前速度场实现为数学模拟（占位），后续 Phase 将接入
        ELF 原生模型的真实 forward() 推理。详见内部方法 docstring。

        FIXME(Phase 2.3+): 替换 _velocity_* 为真实 ELF 模型前向调用。
    """

    def __init__(
        self,
        cond_vec: NDArray[np.float32] | None = None,
        use_conditioning: bool = True,
    ) -> None:
        """初始化速度场包装器。

        Args:
            cond_vec: 条件文本的编码向量，shape (768,)。
                      若为 None 且 use_conditioning=True，则回退到无条件速度场。
            use_conditioning: 是否使用条件编码。False 时为无条件模式。
        """
        self.cond_vec = cond_vec
        self.use_conditioning = use_conditioning

        logger.debug(
            "ELFModelWrapper 已初始化 (use_conditioning=%s, cond_vec=%s)",
            use_conditioning,
            "provided" if cond_vec is not None else "None",
        )

    def __call__(self, z: NDArray[np.float32], t: float) -> NDArray[np.float32]:
        """预测速度场。

        根据当前状态 z 和时间 t，结合条件向量（若有）预测速度。

        Args:
            z: 当前（加噪）向量。shape (d,) 或 (n, d)。
            t: 当前时间步 ∈ [0, 1]。

        Returns:
            速度向量 v，shape 与 z 相同。
        """
        if z.dtype != np.float32:
            raise ValueError(f"z 必须是 float32，got {z.dtype}")

        is_1d = z.ndim == 1
        z_2d = np.atleast_2d(z)

        # 计算速度场
        if self.use_conditioning and self.cond_vec is not None:
            v = self._velocity_with_condition(z_2d, t)
        else:
            v = self._velocity_unconditioned(z_2d, t)

        return v.reshape(-1) if is_1d else v

    # ── 内部速度场计算 ────────────────────────

    def _velocity_with_condition(
        self,
        z: NDArray[np.float32],
        t: float,
    ) -> NDArray[np.float32]:
        """带条件的速度场预测。

        FIXME(Phase 2.3+): 当前为数学模拟占位，将当前状态推向条件向量方向。
        后续应替换为 ELF 原生模型的真实 forward() 推理。

        Args:
            z: 当前向量，shape (n, d)。
            t: 当前时间步。

        Returns:
            速度向量，shape (n, d)。
        """
        cond = np.asarray(self.cond_vec, dtype=np.float32)
        # 速度方向: 朝向条件向量，幅度与噪声水平成反比
        direction = cond - z  # (n, d)
        noise_scale = max(t, 1e-8)
        v = direction / noise_scale
        return v

    def _velocity_unconditioned(
        self,
        z: NDArray[np.float32],
        t: float,
    ) -> NDArray[np.float32]:
        """无条件速度场预测。

        FIXME(Phase 2.3+): 当前为数学模拟占位，向原点收缩。
        后续应替换为 ELF 原生模型的真实 forward() 推理。

        Args:
            z: 当前向量，shape (n, d)。
            t: 当前时间步。

        Returns:
            速度向量，shape (n, d)。
        """
        noise_scale = max(t, 1e-8)
        v = -z / noise_scale
        return v


# ──────────────────────────────────────────────
#  工厂函数
# ──────────────────────────────────────────────


def create_model_pair(
    cond_vec: NDArray[np.float32],
) -> tuple[
    Callable[[NDArray[np.float32], float], NDArray[np.float32]],
    Callable[[NDArray[np.float32], float], NDArray[np.float32]],
]:
    """创建条件/无条件速度场函数对。

    用于 Velocity 级 CFG 去噪：

        z_cfg = denoise_with_cfg(z_t, cond_fn, uncond_fn, steps=2, cfg_scale=2.0)

    Args:
        cond_vec: 条件文本的编码向量，shape (768,)。

    Returns:
        (cond_fn, uncond_fn): 条件/无条件速度场函数。
    """
    cond_fn = ELFModelWrapper(cond_vec=cond_vec, use_conditioning=True)
    uncond_fn = ELFModelWrapper(cond_vec=None, use_conditioning=False)
    return cond_fn, uncond_fn
