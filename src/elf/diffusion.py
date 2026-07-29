"""扩散正反向与 CFG 引导接口。

提供三个核心函数:
  add_noise  — 前向加噪
  denoise    — 反向去噪（ODE 推进）
  cfg_guide  — 无分类器引导

与 ELFEncoder 配合构成完整增强链路:
  encode(text) → add_noise → denoise → cfg_guide → L2 normalize → FAISS search
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ──────────────────────────────────────────────
#  噪声调度
# ──────────────────────────────────────────────


def sigma(t: float) -> float:
    """线性噪声调度 σ(t) = t。

    Args:
        t: 时间步 ∈ [0, 1]。

    Returns:
        对应时间步的噪声标准差。
    """
    return t


# ──────────────────────────────────────────────
#  前向加噪
# ──────────────────────────────────────────────


def add_noise(
    z_0: NDArray[np.float32],
    t: float = 0.4,
    rng: np.random.Generator | None = None,
) -> NDArray[np.float32]:
    """前向加噪: z_t = z_0 + σ(t) · ε, ε ~ N(0, I)

    Args:
        z_0: 干净向量。shape (d,) 或 (n, d)。
        t: 噪声强度 ∈ [0, 1]。0=无噪声, 1=完全噪声。
        rng: 可选随机数生成器（固定种子用）。

    Returns:
        加噪后的向量，shape 与 z_0 相同。

    Raises:
        ValueError: t 不在 [0, 1] 范围内。
        ValueError: z_0 不是 float32 数组。

    Examples:
        >>> z_0 = np.random.randn(768).astype(np.float32)
        >>> z_0 /= np.linalg.norm(z_0)
        >>> z_t = add_noise(z_0, t=0.4)
        >>> z_t.shape
        (768,)
    """
    if not 0.0 <= t <= 1.0:
        raise ValueError(f"t 必须在 [0, 1] 范围内，got {t}")

    if z_0.dtype != np.float32:
        raise ValueError(f"z_0 必须是 float32，got {z_0.dtype}")

    if rng is None:
        rng = np.random.default_rng()

    eps = rng.standard_normal(size=z_0.shape, dtype=np.float32)
    z_t = z_0 + sigma(t) * eps

    logger.debug("add_noise: t=%.2f, shape=%s", t, z_0.shape)
    return z_t.astype(np.float32)


# ──────────────────────────────────────────────
#  反向去噪（Euler ODE）
# ──────────────────────────────────────────────


def denoise(
    z_t: NDArray[np.float32],
    model_fn: Callable[[NDArray[np.float32], float], NDArray[np.float32]],
    steps: int = 2,
    t_start: float | None = None,
    t_end: float = 0.0,
) -> NDArray[np.float32]:
    """反向去噪: 通过 Euler ODE 推进迭代去噪。

    循环 steps 次，每次调用 model_fn(z, t) 预测速度场 v，
    然后按 ODE 推进: z ← z + v · Δt。

    Args:
        z_t: 加噪向量。shape (d,) 或 (n, d)。
        model_fn: 可调用对象，签名 model_fn(z, t) → v。
                  其中 z、v 均为 ndarray，t 为 float。
        steps: 去噪步数（ODE 步数）。
        t_start: 起始时间。默认与 z_t 对应噪声水平匹配。
                 若为 None，尝试从加噪幅度推断。
        t_end: 终止时间，通常为 0.0（干净数据）。

    Returns:
        去噪后的向量，shape 与 z_t 相同。

    Raises:
        ValueError: steps <= 0。
        ValueError: z_t 不是 float32。

    Examples:
        >>> z_t = np.random.randn(768).astype(np.float32)
        >>> model_fn = lambda z, t: -z / max(t, 1e-6)  # 简易去噪
        >>> z_0 = denoise(z_t, model_fn, steps=2, t_start=0.4)
        >>> z_0.shape
        (768,)
    """
    if steps <= 0:
        raise ValueError(f"steps 必须为正整数，got {steps}")
    if z_t.dtype != np.float32:
        raise ValueError(f"z_t 必须是 float32，got {z_t.dtype}")

    if t_start is None:
        t_start = 0.4  # 参数网格中最常用的默认值

    is_1d = z_t.ndim == 1
    z = np.atleast_2d(z_t).copy().astype(np.float32)

    dt = (t_end - t_start) / steps

    for i in range(steps):
        t_curr = t_start + i * dt
        v = model_fn(z, t_curr)
        z = z + v * dt

    logger.debug("denoise: steps=%d, t_start=%.2f, shape=%s", steps, t_start, z_t.shape)
    return z.reshape(-1) if is_1d else z


# ──────────────────────────────────────────────
#  CFG 引导
# ──────────────────────────────────────────────


def cfg_guide(
    z_cond: NDArray[np.float32],
    z_uncond: NDArray[np.float32],
    scale: float = 2.0,
) -> NDArray[np.float32]:
    """无分类器引导（Classifier-Free Guidance）。

    z_cfg = z_uncond + scale · (z_cond - z_uncond)

    等价形式: z_cfg = scale · z_cond + (1 - scale) · z_uncond

    scale=1.0 时返回 z_cond（无 CFG 效果）。
    scale>1.0 时更靠近条件预测（精准模式）。
    0<scale<1.0 时更靠近无条件预测（扩展召回模式）。

    Args:
        z_cond: 条件预测向量。
        z_uncond: 无条件预测向量。
        scale: CFG 引导强度。默认 2.0。

    Returns:
        引导后的向量，shape 与输入相同。

    Raises:
        ValueError: z_cond 和 z_uncond shape 不一致。

    Examples:
        >>> z_cond = np.array([0.5, 0.3], dtype=np.float32)
        >>> z_uncond = np.array([0.1, 0.2], dtype=np.float32)
        >>> z_cfg = cfg_guide(z_cond, z_uncond, scale=2.0)
        >>> z_cfg.shape
        (2,)
    """
    if z_cond.shape != z_uncond.shape:
        raise ValueError(
            f"z_cond 和 z_uncond shape 必须一致，" f"got {z_cond.shape} vs {z_uncond.shape}"
        )

    z_cfg = z_uncond + scale * (z_cond - z_uncond)

    logger.debug("cfg_guide: scale=%.1f, shape=%s", scale, z_cond.shape)
    return z_cfg.astype(np.float32)
