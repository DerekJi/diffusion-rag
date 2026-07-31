"""扩散接口单元测试。

使用纯 numpy 数值验证，无需真实模型或 GPU。
"""

import numpy as np
import pytest

from src.elf.diffusion import add_noise, cfg_guide, denoise, denoise_with_cfg, sigma

# ──────────────────────────────────────────────
#  sigma 调度
# ──────────────────────────────────────────────


class TestSigma:
    """噪声调度测试。"""

    def test_sigma_zero(self) -> None:
        """t=0 时 σ=0。"""
        assert sigma(0.0) == 0.0

    def test_sigma_one(self) -> None:
        """t=1 时 σ=1。"""
        assert sigma(1.0) == 1.0

    def test_sigma_half(self) -> None:
        """t=0.5 时 σ=0.5。"""
        assert sigma(0.5) == 0.5

    def test_sigma_linear(self) -> None:
        """σ(t) = t 是线性关系。"""
        ts = [0.0, 0.2, 0.5, 0.8, 1.0]
        for t in ts:
            assert sigma(t) == pytest.approx(t)


# ──────────────────────────────────────────────
#  add_noise
# ──────────────────────────────────────────────


class TestAddNoise:
    """前向加噪测试。"""

    def test_shape_1d(self) -> None:
        """输入 1d 向量，输出 shape 不变。"""
        z_0 = np.random.randn(768).astype(np.float32)
        z_0 /= np.linalg.norm(z_0)
        z_t = add_noise(z_0, t=0.4, rng=np.random.default_rng(42))
        assert z_t.shape == (768,)

    def test_shape_2d(self) -> None:
        """输入 2d batch，输出 shape 不变。"""
        z_0 = np.random.randn(5, 768).astype(np.float32)
        z_0 /= np.linalg.norm(z_0, axis=1, keepdims=True)
        z_t = add_noise(z_0, t=0.4, rng=np.random.default_rng(42))
        assert z_t.shape == (5, 768)

    def test_dtype(self) -> None:
        """输出 dtype 为 float32。"""
        z_0 = np.random.randn(768).astype(np.float32)
        z_0 /= np.linalg.norm(z_0)
        z_t = add_noise(z_0, t=0.4, rng=np.random.default_rng(42))
        assert z_t.dtype == np.float32

    def test_t_zero_no_noise(self) -> None:
        """t=0 时输出应与输入一致（无噪声）。"""
        z_0 = np.random.randn(768).astype(np.float32)
        z_0 /= np.linalg.norm(z_0)
        z_t = add_noise(z_0, t=0.0, rng=np.random.default_rng(42))
        assert np.allclose(z_t, z_0, atol=1e-6)

    def test_t_one_full_noise(self) -> None:
        """t=1 时输出与输入显著不同。"""
        z_0 = np.random.randn(768).astype(np.float32)
        z_0 /= np.linalg.norm(z_0)
        z_t = add_noise(z_0, t=1.0, rng=np.random.default_rng(42))
        # 加噪后不再是单位范数
        assert not np.allclose(z_t, z_0, atol=0.5)

    def test_noise_level_increasing(self) -> None:
        """t 越大，噪声水平越高（与原始向量的距离越大）。"""
        z_0 = np.random.randn(768).astype(np.float32)
        z_0 /= np.linalg.norm(z_0)
        rng = np.random.default_rng(42)
        # 使用不同 t 值但确保噪声种子一致
        z_t_low = add_noise(z_0.copy(), t=0.2, rng=np.random.default_rng(42))
        z_t_high = add_noise(z_0.copy(), t=0.8, rng=np.random.default_rng(42))
        dist_low = float(np.linalg.norm(z_t_low - z_0))
        dist_high = float(np.linalg.norm(z_t_high - z_0))
        assert dist_high > dist_low

    def test_deterministic_with_seed(self) -> None:
        """相同种子产生相同噪声。"""
        z_0 = np.random.randn(768).astype(np.float32)
        z_0 /= np.linalg.norm(z_0)
        z_t1 = add_noise(z_0.copy(), t=0.5, rng=np.random.default_rng(42))
        z_t2 = add_noise(z_0.copy(), t=0.5, rng=np.random.default_rng(42))
        assert np.allclose(z_t1, z_t2, atol=1e-6)

    def test_different_seed_different_noise(self) -> None:
        """不同种子产生不同噪声。"""
        z_0 = np.random.randn(768).astype(np.float32)
        z_0 /= np.linalg.norm(z_0)
        z_t1 = add_noise(z_0.copy(), t=0.5, rng=np.random.default_rng(42))
        z_t2 = add_noise(z_0.copy(), t=0.5, rng=np.random.default_rng(99))
        assert not np.allclose(z_t1, z_t2, atol=0.1)

    def test_default_rng(self) -> None:
        """默认 rng=None 时不应报错。"""
        z_0 = np.random.randn(768).astype(np.float32)
        z_0 /= np.linalg.norm(z_0)
        z_t = add_noise(z_0, t=0.5)  # rng=None
        assert z_t.shape == (768,)
        assert z_t.dtype == np.float32

    def test_invalid_t_below_zero(self) -> None:
        """t < 0 应抛出 ValueError。"""
        z_0 = np.random.randn(768).astype(np.float32)
        with pytest.raises(ValueError, match="必须在"):
            add_noise(z_0, t=-0.1)

    def test_invalid_t_above_one(self) -> None:
        """t > 1 应抛出 ValueError。"""
        z_0 = np.random.randn(768).astype(np.float32)
        with pytest.raises(ValueError, match="必须在"):
            add_noise(z_0, t=1.5)

    def test_invalid_dtype(self) -> None:
        """非 float32 输入应抛出 ValueError。"""
        z_0 = np.random.randn(768)
        with pytest.raises(ValueError, match="float32"):
            add_noise(z_0, t=0.5)

    def test_zero_vector(self) -> None:
        """零向量加噪后应非零。"""
        z_0 = np.zeros(768, dtype=np.float32)
        z_t = add_noise(z_0, t=0.5, rng=np.random.default_rng(42))
        assert not np.allclose(z_t, 0.0, atol=1e-6)


# ──────────────────────────────────────────────
#  denoise
# ──────────────────────────────────────────────


class TestDenoise:
    """反向去噪测试。"""

    @staticmethod
    def _identity_model(z: np.ndarray, _t: float) -> np.ndarray:
        """恒等模型: 返回输入本身。"""
        return z

    @staticmethod
    def _zero_model(z: np.ndarray, _t: float) -> np.ndarray:
        """零速度模型: 返回零向量。"""
        return np.zeros_like(z)

    def test_shape_1d(self) -> None:
        """输入 1d，输出 1d shape 不变。"""
        z_t = np.random.randn(768).astype(np.float32)
        z_0 = denoise(z_t, self._zero_model, steps=1, t_start=0.4)
        assert z_0.shape == (768,)

    def test_shape_2d(self) -> None:
        """输入 2d batch，输出 shape 不变。"""
        z_t = np.random.randn(5, 768).astype(np.float32)
        z_0 = denoise(z_t, self._zero_model, steps=1, t_start=0.4)
        assert z_0.shape == (5, 768)

    def test_dtype(self) -> None:
        """输出 dtype 为 float32。"""
        z_t = np.random.randn(768).astype(np.float32)
        z_0 = denoise(z_t, self._zero_model, steps=1, t_start=0.4)
        assert z_0.dtype == np.float32

    def test_zero_velocity_no_change(self) -> None:
        """零速度模型: 输出应与输入一致。"""
        z_t = np.random.randn(768).astype(np.float32)
        z_0 = denoise(z_t, self._zero_model, steps=2, t_start=0.4)
        assert np.allclose(z_0, z_t, atol=1e-6)

    def test_identity_model_euler_exact(self) -> None:
        """恒等速度模型，1 步时 z_1 = z_0 + z_0 * dt。

        dt = (0 - t_start) / 1 = -t_start
        z_1 = z_0 + z_0 * (-t_start) = z_0 * (1 - t_start)
        """
        z_t = np.array([1.0, 2.0], dtype=np.float32)
        t_start = 0.4
        z_0 = denoise(z_t, self._identity_model, steps=1, t_start=t_start)
        expected = z_t * (1 - t_start)
        assert np.allclose(z_0, expected, atol=1e-6)

    def test_multi_step_improvement(self) -> None:
        """多步去噪的 Euler 逼近应更精确（对非恒常速度场）。"""
        z_t = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        # 速度场随状态变化: v = -0.5 * z  (指数衰减 dz/dt = -0.5z)
        def stateful_model(z: np.ndarray, t: float) -> np.ndarray:
            return -0.5 * z

        # t_start=0.6, t_end=0.0
        # dt = -0.6 / steps
        # Euler 1步: z₁ = z_t + (-0.5*z_t)*(-0.6) = z_t * (1 + 0.3) = 1.3 * z_t
        # Euler 2步: z₁ = z_t * (1+0.15) = 1.15*z_t, z₂ = 1.15*z_t * (1+0.15) = 1.3225 * z_t
        # 真实解:   dz/dt = -0.5·z → z(t) = z₀·exp(-0.5·t)
        #         → z₀ = z_t·exp(0.5·0.6) = z_t·exp(0.3) ≈ 1.3499·z_t
        # Euler 随步数增加应更接近真实解（截断误差更小）
        z_1step = denoise(z_t.copy(), stateful_model, steps=1, t_start=0.6)
        z_2step = denoise(z_t.copy(), stateful_model, steps=2, t_start=0.6)

        # 计算与真实解的差距
        z_true = z_t * np.exp(0.3)  # exp(-0.5 * 0.6), 反向积分时放大
        err_1step = float(np.linalg.norm(z_1step - z_true))
        err_2step = float(np.linalg.norm(z_2step - z_true))
        assert err_2step < err_1step

    def test_invalid_steps_zero(self) -> None:
        """steps <= 0 应抛出 ValueError。"""
        z_t = np.random.randn(768).astype(np.float32)
        with pytest.raises(ValueError, match="正整数"):
            denoise(z_t, self._zero_model, steps=0)

    def test_invalid_steps_negative(self) -> None:
        """steps 负值应抛出 ValueError。"""
        z_t = np.random.randn(768).astype(np.float32)
        with pytest.raises(ValueError, match="正整数"):
            denoise(z_t, self._zero_model, steps=-1)

    def test_invalid_dtype(self) -> None:
        """非 float32 输入应抛出 ValueError。"""
        z_t = np.random.randn(768)
        with pytest.raises(ValueError, match="float32"):
            denoise(z_t, self._zero_model, steps=1)

    def test_t_end_nonzero(self) -> None:
        """t_end > 0 时应在中间时间停止。"""
        z_t = np.random.randn(768).astype(np.float32)
        z_0_full = denoise(z_t, self._identity_model, steps=1, t_start=0.4, t_end=0.0)
        z_0_half = denoise(z_t, self._identity_model, steps=1, t_start=0.4, t_end=0.2)
        # 使用恒等模型:
        # dt_full = -0.4, z_full = z_t + z_t * (-0.4) = z_t * 0.6
        # dt_half = -0.2, z_half = z_t + z_t * (-0.2) = z_t * 0.8
        # 所以 z_full 变化更大
        assert float(np.linalg.norm(z_0_full - z_t)) > float(np.linalg.norm(z_0_half - z_t))

    def test_t_start_default(self) -> None:
        """t_start=None 使用默认值 0.4。"""
        z_t = np.random.randn(768).astype(np.float32)
        z_explicit = denoise(z_t, self._zero_model, steps=1, t_start=0.4)
        z_default = denoise(z_t, self._zero_model, steps=1)
        assert np.allclose(z_explicit, z_default, atol=1e-6)


# ──────────────────────────────────────────────
#  cfg_guide
# ──────────────────────────────────────────────


class TestCFGGuide:
    """CFG 引导测试。"""

    def test_shape_1d(self) -> None:
        """输入 1d，输出 shape 一致。"""
        z_cond = np.array([0.5, 0.3], dtype=np.float32)
        z_uncond = np.array([0.1, 0.2], dtype=np.float32)
        z_cfg = cfg_guide(z_cond, z_uncond, scale=2.0)
        assert z_cfg.shape == (2,)

    def test_shape_2d(self) -> None:
        """输入 2d batch，输出 shape 一致。"""
        z_cond = np.random.randn(5, 768).astype(np.float32)
        z_uncond = np.random.randn(5, 768).astype(np.float32)
        z_cfg = cfg_guide(z_cond, z_uncond, scale=2.0)
        assert z_cfg.shape == (5, 768)

    def test_dtype(self) -> None:
        """输出 dtype 为 float32。"""
        z_cond = np.array([0.5, 0.3], dtype=np.float32)
        z_uncond = np.array([0.1, 0.2], dtype=np.float32)
        z_cfg = cfg_guide(z_cond, z_uncond, scale=2.0)
        assert z_cfg.dtype == np.float32

    def test_scale_one_returns_cond(self) -> None:
        """scale=1.0 应返回 z_cond（无 CFG 效果）。"""
        z_cond = np.array([0.5, 0.3], dtype=np.float32)
        z_uncond = np.array([0.1, 0.2], dtype=np.float32)
        z_cfg = cfg_guide(z_cond, z_uncond, scale=1.0)
        assert np.allclose(z_cfg, z_cond, atol=1e-6)

    def test_scale_zero_returns_uncond(self) -> None:
        """scale=0.0 应返回 z_uncond。"""
        z_cond = np.array([0.5, 0.3], dtype=np.float32)
        z_uncond = np.array([0.1, 0.2], dtype=np.float32)
        z_cfg = cfg_guide(z_cond, z_uncond, scale=0.0)
        assert np.allclose(z_cfg, z_uncond, atol=1e-6)

    def test_scale_two_formula(self) -> None:
        """scale=2.0 验证公式: z_uncond + 2 * (z_cond - z_uncond)。"""
        z_cond = np.array([0.5, 0.3], dtype=np.float32)
        z_uncond = np.array([0.1, 0.2], dtype=np.float32)
        z_cfg = cfg_guide(z_cond, z_uncond, scale=2.0)
        expected = z_uncond + 2.0 * (z_cond - z_uncond)
        assert np.allclose(z_cfg, expected, atol=1e-6)

    def test_scale_three_precision_mode(self) -> None:
        """scale=3.0 应更接近条件向量。"""
        z_cond = np.array([1.0, 0.0], dtype=np.float32)
        z_uncond = np.array([0.0, 1.0], dtype=np.float32)
        z_cfg = cfg_guide(z_cond, z_uncond, scale=3.0)
        dist_to_cond = float(np.linalg.norm(z_cfg - z_cond))
        dist_to_uncond = float(np.linalg.norm(z_cfg - z_uncond))
        # scale>1 时更靠近条件
        assert dist_to_cond < dist_to_uncond

    def test_scale_half_recall_mode(self) -> None:
        """scale=0.25 应更接近无条件向量。"""
        z_cond = np.array([1.0, 0.0], dtype=np.float32)
        z_uncond = np.array([0.0, 1.0], dtype=np.float32)
        z_cfg = cfg_guide(z_cond, z_uncond, scale=0.25)
        dist_to_cond = float(np.linalg.norm(z_cfg - z_cond))
        dist_to_uncond = float(np.linalg.norm(z_cfg - z_uncond))
        # scale<1 时更靠近无条件
        assert dist_to_uncond < dist_to_cond

    def test_negative_scale(self) -> None:
        """负 scale 应反推方向。"""
        z_cond = np.array([1.0, 0.0], dtype=np.float32)
        z_uncond = np.array([0.0, 1.0], dtype=np.float32)
        z_cfg = cfg_guide(z_cond, z_uncond, scale=-1.0)
        expected = z_uncond - 1.0 * (z_cond - z_uncond)
        assert np.allclose(z_cfg, expected, atol=1e-6)

    def test_shape_mismatch(self) -> None:
        """z_cond 和 z_uncond shape 不一致应抛出 ValueError。"""
        z_cond = np.array([0.5, 0.3], dtype=np.float32)
        z_uncond = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        with pytest.raises(ValueError, match="必须一致"):
            cfg_guide(z_cond, z_uncond, scale=2.0)

    def test_batch_shape_mismatch(self) -> None:
        """batch 维度不匹配应抛出 ValueError。"""
        z_cond = np.random.randn(3, 768).astype(np.float32)
        z_uncond = np.random.randn(5, 768).astype(np.float32)
        with pytest.raises(ValueError, match="必须一致"):
            cfg_guide(z_cond, z_uncond, scale=2.0)

    def test_identical_vectors(self) -> None:
        """条件与无条件相同时，任何 scale 输出不变。"""
        z = np.array([0.5, 0.3], dtype=np.float32)
        for scale in [0.0, 0.5, 1.0, 2.0, 3.0]:
            z_cfg = cfg_guide(z, z, scale=scale)
            assert np.allclose(z_cfg, z, atol=1e-6)

    def test_768_dim(self) -> None:
        """768-dim 向量 CFG 正确工作。"""
        z_cond = np.random.randn(768).astype(np.float32)
        z_uncond = np.random.randn(768).astype(np.float32)
        z_cfg = cfg_guide(z_cond, z_uncond, scale=2.0)
        assert z_cfg.shape == (768,)
        assert z_cfg.dtype == np.float32


# ──────────────────────────────────────────────
#  denoise_with_cfg (velocity 级 CFG)
# ──────────────────────────────────────────────


class TestDenoiseWithCFG:
    """Velocity 级 CFG 去噪测试。"""

    @staticmethod
    def _zero_model(z: np.ndarray, _t: float) -> np.ndarray:
        """零速度模型。"""
        return np.zeros_like(z)

    # ── 基本功能 ─────────────────────────────

    def test_shape_1d(self) -> None:
        """输入 1d，输出 1d shape 不变。"""
        z_t = np.random.randn(768).astype(np.float32)
        z_0 = denoise_with_cfg(
            z_t, self._zero_model, self._zero_model, steps=1, cfg_scale=2.0, t_start=0.4
        )
        assert z_0.shape == (768,)

    def test_shape_2d(self) -> None:
        """输入 2d batch，输出 shape 不变。"""
        z_t = np.random.randn(5, 768).astype(np.float32)
        z_0 = denoise_with_cfg(
            z_t, self._zero_model, self._zero_model, steps=1, cfg_scale=2.0, t_start=0.4
        )
        assert z_0.shape == (5, 768)

    def test_dtype(self) -> None:
        """输出 dtype 为 float32。"""
        z_t = np.random.randn(768).astype(np.float32)
        z_0 = denoise_with_cfg(
            z_t, self._zero_model, self._zero_model, steps=1, cfg_scale=2.0, t_start=0.4
        )
        assert z_0.dtype == np.float32

    # ── cfg_scale=1.0 行为 ──────────────────

    def test_cfg_scale_one_calls_cond_only(self) -> None:
        """cfg_scale=1.0 时只调用 cond_fn（不调 uncond_fn）。"""
        cond_calls: list[float] = []
        uncond_calls: list[float] = []

        def cond_fn(z: np.ndarray, t: float) -> np.ndarray:
            cond_calls.append(t)
            return np.zeros_like(z)

        def uncond_fn(z: np.ndarray, t: float) -> np.ndarray:
            uncond_calls.append(t)
            return np.zeros_like(z)

        z_t = np.random.randn(768).astype(np.float32)
        denoise_with_cfg(z_t, cond_fn, uncond_fn, steps=4, cfg_scale=1.0, t_start=0.4)
        assert len(cond_calls) == 4
        assert len(uncond_calls) == 0

    def test_cfg_scale_one_equals_denoise(self) -> None:
        """cfg_scale=1.0 时结果应与纯 denoise 相同。"""
        z_t = np.random.randn(768).astype(np.float32)

        def model_fn(z: np.ndarray, t: float) -> np.ndarray:
            return -z * 0.5

        z_plain = denoise(z_t, model_fn, steps=3, t_start=0.4)
        z_cfg = denoise_with_cfg(z_t, model_fn, model_fn, steps=3, cfg_scale=1.0, t_start=0.4)
        assert np.allclose(z_plain, z_cfg, atol=1e-6)

    # ── velocity 级混合验证 ────────────────

    def test_cfg_scale_two_mixes_velocities(self) -> None:
        """cfg_scale=2.0 时每步混合速度场，验证公式 v_uncond + 2·(v_cond - v_uncond)。"""
        z_t = np.array([1.0, 0.0], dtype=np.float32)

        cond_calls: list[np.ndarray] = []
        uncond_calls: list[np.ndarray] = []

        def cond_fn(z: np.ndarray, t: float) -> np.ndarray:
            v = np.array([1.0, 0.0], dtype=np.float32)
            cond_calls.append(v)
            return v

        def uncond_fn(z: np.ndarray, t: float) -> np.ndarray:
            v = np.array([0.0, 1.0], dtype=np.float32)
            uncond_calls.append(v)
            return v

        # 1 步: dt = (0 - 0.4) / 1 = -0.4
        # v_cond = [1, 0], v_uncond = [0, 1]
        # v_cfg = [0, 1] + 2 * ([1, 0] - [0, 1]) = [0, 1] + 2*[1, -1] = [2, -1]
        # z = [1, 0] + [2, -1] * (-0.4) = [1, 0] + [-0.8, 0.4] = [0.2, 0.4]
        z_out = denoise_with_cfg(z_t, cond_fn, uncond_fn, steps=1, cfg_scale=2.0, t_start=0.4)
        expected = np.array([0.2, 0.4], dtype=np.float32)
        assert np.allclose(z_out, expected, atol=1e-6)
        assert len(cond_calls) == 1
        assert len(uncond_calls) == 1

    def test_cfg_scale_zero_uses_uncond(self) -> None:
        """cfg_scale=0.0 时 v_cfg = v_uncond（仅无条件速度场）。"""
        z_t = np.array([1.0, 0.0], dtype=np.float32)

        def cond_fn(z: np.ndarray, t: float) -> np.ndarray:
            return np.array([10.0, 0.0], dtype=np.float32)

        def uncond_fn(z: np.ndarray, t: float) -> np.ndarray:
            return np.array([1.0, 0.0], dtype=np.float32)

        # v_cfg = v_uncond + 0 * (v_cond - v_uncond) = v_uncond
        # 应与纯 denoise(uncond_fn) 等价
        z_plain = denoise(z_t, uncond_fn, steps=1, t_start=0.4)
        z_cfg = denoise_with_cfg(z_t, cond_fn, uncond_fn, steps=1, cfg_scale=0.0, t_start=0.4)
        assert np.allclose(z_plain, z_cfg, atol=1e-6)

    # ── 调用次数验证 ───────────────────────

    def test_call_count_with_cfg(self) -> None:
        """cfg_scale≠1.0 时每步调用 cond_fn 和 uncond_fn 各一次。"""
        cond_calls: list[float] = []
        uncond_calls: list[float] = []

        def cond_fn(z: np.ndarray, t: float) -> np.ndarray:
            cond_calls.append(t)
            return np.zeros_like(z)

        def uncond_fn(z: np.ndarray, t: float) -> np.ndarray:
            uncond_calls.append(t)
            return np.zeros_like(z)

        z_t = np.random.randn(768).astype(np.float32)
        denoise_with_cfg(z_t, cond_fn, uncond_fn, steps=5, cfg_scale=2.5, t_start=0.5)
        assert len(cond_calls) == 5
        assert len(uncond_calls) == 5

    # ── 错误处理 ───────────────────────────

    def test_invalid_steps_zero(self) -> None:
        """steps <= 0 应抛出 ValueError。"""
        z_t = np.random.randn(768).astype(np.float32)
        with pytest.raises(ValueError, match="正整数"):
            denoise_with_cfg(z_t, self._zero_model, self._zero_model, steps=0, cfg_scale=2.0)

    def test_invalid_steps_negative(self) -> None:
        """steps 负值应抛出 ValueError。"""
        z_t = np.random.randn(768).astype(np.float32)
        with pytest.raises(ValueError, match="正整数"):
            denoise_with_cfg(z_t, self._zero_model, self._zero_model, steps=-1, cfg_scale=2.0)

    def test_invalid_dtype(self) -> None:
        """非 float32 输入应抛出 ValueError。"""
        z_t = np.random.randn(768)
        with pytest.raises(ValueError, match="float32"):
            denoise_with_cfg(z_t, self._zero_model, self._zero_model, steps=1, cfg_scale=2.0)

    # ── t_start 默认 ───────────────────────

    def test_t_start_default(self) -> None:
        """t_start=None 使用默认值 0.4。"""
        z_t = np.random.randn(768).astype(np.float32)
        z_explicit = denoise_with_cfg(
            z_t, self._zero_model, self._zero_model, steps=1, cfg_scale=2.0, t_start=0.4
        )
        z_default = denoise_with_cfg(
            z_t, self._zero_model, self._zero_model, steps=1, cfg_scale=2.0
        )
        assert np.allclose(z_explicit, z_default, atol=1e-6)

    # ── multi_step ─────────────────────────

    def test_multi_step_with_cfg(self) -> None:
        """多步 velocity 级 CFG 正常工作。"""
        z_t = np.array([1.0, 2.0, 3.0], dtype=np.float32)

        def cond_fn(z: np.ndarray, t: float) -> np.ndarray:
            return -z * 0.3

        def uncond_fn(z: np.ndarray, t: float) -> np.ndarray:
            return -z * 0.1

        z_out = denoise_with_cfg(z_t, cond_fn, uncond_fn, steps=3, cfg_scale=2.0, t_start=0.6)
        assert z_out.shape == (3,)
        assert z_out.dtype == np.float32
