"""ELF 模型速度场包装器单元测试。

验证速度场输出 shape/dtype 与输入一致。
"""

import numpy as np
import pytest

from src.elf.model_wrapper import ELFModelWrapper, create_model_pair


class TestELFModelWrapper:
    """ELF 模型速度场包装器测试。"""

    def test_shape_1d(self) -> None:
        """输入 1d，输出 shape 一致。"""
        z = np.random.randn(768).astype(np.float32)
        cond = np.random.randn(768).astype(np.float32)
        wrapper = ELFModelWrapper(cond_vec=cond, use_conditioning=True)
        v = wrapper(z, t=0.4)
        assert v.shape == (768,)

    def test_shape_2d(self) -> None:
        """输入 2d batch，输出 shape 一致。"""
        z = np.random.randn(5, 768).astype(np.float32)
        cond = np.random.randn(768).astype(np.float32)
        wrapper = ELFModelWrapper(cond_vec=cond, use_conditioning=True)
        v = wrapper(z, t=0.4)
        assert v.shape == (5, 768)

    def test_dtype(self) -> None:
        """输出 dtype 为 float32。"""
        z = np.random.randn(768).astype(np.float32)
        cond = np.random.randn(768).astype(np.float32)
        wrapper = ELFModelWrapper(cond_vec=cond, use_conditioning=True)
        v = wrapper(z, t=0.4)
        assert v.dtype == np.float32

    def test_unconditioned_shape(self) -> None:
        """无条件模式，输出 shape 一致。"""
        z = np.random.randn(768).astype(np.float32)
        wrapper = ELFModelWrapper(use_conditioning=False)
        v = wrapper(z, t=0.4)
        assert v.shape == (768,)

    def test_unconditioned_dtype(self) -> None:
        """无条件模式，输出 dtype 为 float32。"""
        z = np.random.randn(768).astype(np.float32)
        wrapper = ELFModelWrapper(use_conditioning=False)
        v = wrapper(z, t=0.4)
        assert v.dtype == np.float32

    def test_unconditioned_direction(self) -> None:
        """无条件模式，速度方向指向原点（与 z 反向）。"""
        z = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        wrapper = ELFModelWrapper(use_conditioning=False)
        v = wrapper(z, t=0.5)
        # v 应该与 z 反向（都朝原点收缩）
        assert np.dot(v, z) < 0

    def test_conditioned_direction(self) -> None:
        """条件模式，速度方向朝向条件向量。"""
        z = np.array([1.0, 0.0], dtype=np.float32)
        cond = np.array([2.0, 0.0], dtype=np.float32)
        wrapper = ELFModelWrapper(cond_vec=cond, use_conditioning=True)
        v = wrapper(z, t=0.5)
        # 速度方向应朝向条件向量（从 z 指向 cond）
        assert v[0] > 0  # x 方向朝向 cond 的 x

    def test_velocity_magnitude_increases_with_noise(self) -> None:
        """噪声水平 t 越大，速度幅度越大（对抗更大噪声）。"""
        z = np.array([1.0, 0.0], dtype=np.float32)
        cond = np.array([2.0, 0.0], dtype=np.float32)
        wrapper = ELFModelWrapper(cond_vec=cond, use_conditioning=True)
        v_low = wrapper(z, t=0.2)
        v_high = wrapper(z, t=0.8)
        # |direction| = 1.0, v = direction / t
        # t=0.2: |v| = 1.0/0.2 = 5.0
        # t=0.8: |v| = 1.0/0.8 = 1.25
        # So |v_low| > |v_high|
        assert float(np.linalg.norm(v_low)) > float(np.linalg.norm(v_high))

    def test_no_cond_vec_fallback(self) -> None:
        """条件模式但 cond_vec 为 None，使用零向量作为条件。"""
        z = np.array([1.0, 0.0], dtype=np.float32)
        wrapper = ELFModelWrapper(cond_vec=None, use_conditioning=True)
        v = wrapper(z, t=0.5)
        # cond is None → 默认零向量 → direction = 0 - z = -z → v = -z / t
        expected = -z / 0.5
        assert np.allclose(v, expected, atol=1e-6)

    def test_invalid_dtype(self) -> None:
        """非 float32 输入应抛出 ValueError。"""
        z = np.random.randn(768)  # float64
        cond = np.random.randn(768).astype(np.float32)
        wrapper = ELFModelWrapper(cond_vec=cond, use_conditioning=True)
        with pytest.raises(ValueError, match="float32"):
            wrapper(z, t=0.5)

    def test_short_input(self) -> None:
        """低维输入（如 128-dim）也能工作。"""
        z = np.random.randn(128).astype(np.float32)
        cond = np.random.randn(128).astype(np.float32)
        wrapper = ELFModelWrapper(cond_vec=cond, use_conditioning=True)
        v = wrapper(z, t=0.4)
        assert v.shape == (128,)
        assert v.dtype == np.float32

    def test_t_zero(self) -> None:
        """t=0 时速度应有限，不除以零。"""
        z = np.random.randn(768).astype(np.float32)
        cond = np.random.randn(768).astype(np.float32)
        wrapper = ELFModelWrapper(cond_vec=cond, use_conditioning=True)
        v = wrapper(z, t=0.0)
        assert v.shape == (768,)
        assert np.all(np.isfinite(v))


class TestCreateModelPair:
    """create_model_pair 工厂函数测试。"""

    def test_returns_callables(self) -> None:
        """返回两个可调用对象。"""
        cond = np.random.randn(768).astype(np.float32)
        cond_fn, uncond_fn = create_model_pair(cond)
        assert callable(cond_fn)
        assert callable(uncond_fn)

    def test_cond_fn_uses_conditioning(self) -> None:
        """条件函数使用给定的条件向量。"""
        cond = np.array([2.0, 0.0], dtype=np.float32)
        z = np.array([1.0, 0.0], dtype=np.float32)
        cond_fn, _ = create_model_pair(cond)
        v = cond_fn(z, t=0.5)
        # direction = 2-1 = 1.0, v = 1.0/0.5 = 2.0
        assert abs(v[0] - 2.0) < 1e-6

    def test_uncond_fn_no_conditioning(self) -> None:
        """无条件函数不使用条件向量。"""
        cond = np.array([2.0, 0.0], dtype=np.float32)
        z = np.array([1.0, 0.0], dtype=np.float32)
        _, uncond_fn = create_model_pair(cond)
        v = uncond_fn(z, t=0.5)
        # unconditioned: v = -z / t = -1/0.5 = -2.0
        assert abs(v[0] + 2.0) < 1e-6

    def test_output_shape_matches_input(self) -> None:
        """两个函数的输出 shape 与输入一致。"""
        cond = np.random.randn(768).astype(np.float32)
        z = np.random.randn(5, 768).astype(np.float32)
        cond_fn, uncond_fn = create_model_pair(cond)
        v_cond = cond_fn(z, t=0.4)
        v_uncond = uncond_fn(z, t=0.4)
        assert v_cond.shape == (5, 768)
        assert v_uncond.shape == (5, 768)
        assert v_cond.dtype == np.float32
        assert v_uncond.dtype == np.float32
