"""ELF 增强链路单元测试。

使用 mock 模式，无须下载真实模型。
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.elf.pipeline import ELFPipeline, _default_model_fn

# ──────────────────────────────────────────────
#  Fixtures
# ──────────────────────────────────────────────


class _FakeEncoder:
    """测试用假编码器：每段文本映射到固定的 768-dim 向量。

    实现 encode / encode_batch，并统计调用次数，
    用于验证批量路径只调用一次批量编码。
    """

    def __init__(self) -> None:
        rng = np.random.RandomState(42)
        self._vecs = rng.randn(64, 768).astype(np.float32)
        self._vecs /= np.linalg.norm(self._vecs, axis=1, keepdims=True)
        self._table: dict[str, np.ndarray] = {}
        self.encode_calls = 0
        self.encode_batch_calls = 0

    def _vec_for(self, text: str) -> np.ndarray:
        if text not in self._table:
            self._table[text] = self._vecs[len(self._table) % len(self._vecs)]
        return self._table[text]

    def encode(self, text: str) -> np.ndarray:
        self.encode_calls += 1
        return self._vec_for(text)

    def encode_batch(self, texts: list[str], batch_size: int = 32) -> np.ndarray:
        self.encode_batch_calls += 1
        return np.stack([self._vec_for(text) for text in texts])


@pytest.fixture(autouse=True)
def _mock_encoder() -> None:
    """mock ELFEncoder 的 SentenceTransformer，返回固定 768-dim 向量。

    所有测试自动生效。
    """
    rng = np.random.RandomState(42)
    fixed_vec = rng.randn(768).astype(np.float32)
    fixed_vec /= np.linalg.norm(fixed_vec)

    mock_instance = MagicMock()
    mock_instance.encode.return_value = fixed_vec

    patcher = patch(
        "src.elf.encoder.SentenceTransformer",
        return_value=mock_instance,
    )
    patcher.start()
    yield
    patcher.stop()


# ──────────────────────────────────────────────
#  测试 _default_model_fn
# ──────────────────────────────────────────────


class TestDefaultModelFn:
    """默认速度场函数测试。"""

    def test_shape_preserved(self) -> None:
        """输出 shape 与输入一致。"""
        z = np.random.randn(768).astype(np.float32)
        v = _default_model_fn(z, t=0.4)
        assert v.shape == (768,)

    def test_dtype(self) -> None:
        """输出 dtype 为 float32。"""
        z = np.random.randn(768).astype(np.float32)
        v = _default_model_fn(z, t=0.4)
        assert v.dtype == np.float32

    def test_direction_toward_origin(self) -> None:
        """速度方向指向原点（与 z 反向）。"""
        z = np.array([1.0, -2.0, 3.0], dtype=np.float32)
        v = _default_model_fn(z, t=0.5)
        # v 应该与 z 反向
        assert np.dot(v, z) < 0

    def test_velocity_equals_neg_z_at_t_one(self) -> None:
        """t=1 时 sigma(1)=1, 速度 v = -z（等于输入向量取反）。"""
        z = np.random.randn(768).astype(np.float32)
        v = _default_model_fn(z, t=1.0)
        # v = -z / sigma(1) = -z / 1.0 = -z
        assert np.allclose(v, -z, atol=1e-6)

    def test_magnitude_inversely_related(self) -> None:
        """速度大小与噪声水平成反比。"""
        z = np.array([5.0, 0.0], dtype=np.float32)
        v_small_t = _default_model_fn(z, t=0.2)  # v = -z/0.2
        v_large_t = _default_model_fn(z, t=0.8)  # v = -z/0.8
        assert float(np.linalg.norm(v_small_t)) > float(np.linalg.norm(v_large_t))

    def test_zero_input(self) -> None:
        """零向量输入速度也为零。"""
        z = np.zeros(768, dtype=np.float32)
        v = _default_model_fn(z, t=0.4)
        assert np.allclose(v, 0.0, atol=1e-6)


# ──────────────────────────────────────────────
#  ELFPipeline
# ──────────────────────────────────────────────


class TestELFPipeline:
    """增强链路集成测试。"""

    # ── encode ──────────────────────────────

    def test_encode_shape(self) -> None:
        """encode 输出 shape (768,)。"""
        pipe = ELFPipeline(device="cpu")
        vec = pipe.encode("Hello world")
        assert vec.shape == (768,)

    def test_encode_dtype(self) -> None:
        """encode 输出 float32。"""
        pipe = ELFPipeline(device="cpu")
        vec = pipe.encode("Hello world")
        assert vec.dtype == np.float32

    def test_encode_l2_norm(self) -> None:
        """encode 输出 L2 归一化。"""
        pipe = ELFPipeline(device="cpu")
        vec = pipe.encode("Hello world")
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    # ── enhance (无 CFG) ─────────────────────

    def test_enhance_shape(self) -> None:
        """enhance 输出 shape (768,)。"""
        pipe = ELFPipeline(device="cpu")
        vec = pipe.enhance("query", steps=2, noise_t=0.4, cfg_scale=1.0)
        assert vec.shape == (768,)

    def test_enhance_dtype(self) -> None:
        """enhance 输出 float32。"""
        pipe = ELFPipeline(device="cpu")
        vec = pipe.enhance("query", steps=2, noise_t=0.4, cfg_scale=1.0)
        assert vec.dtype == np.float32

    def test_enhance_l2_norm(self) -> None:
        """enhance 输出 L2 归一化。"""
        pipe = ELFPipeline(device="cpu")
        vec = pipe.enhance("query", steps=2, noise_t=0.4, cfg_scale=1.0)
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_enhance_t_zero_no_noise(self) -> None:
        """noise_t=0 时无噪声，增强后仍归一化。"""
        pipe = ELFPipeline(device="cpu")
        vec = pipe.enhance("query", steps=2, noise_t=0.0, cfg_scale=1.0)
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_enhance_cfg_scale_one(self) -> None:
        """cfg_scale=1.0 时跳过 CFG，与仅去噪等价。"""
        pipe = ELFPipeline(device="cpu")
        vec = pipe.enhance("query", steps=2, noise_t=0.4, cfg_scale=1.0)
        assert vec.shape == (768,)
        assert vec.dtype == np.float32

    # ── enhance (有 CFG) ─────────────────────

    def test_enhance_with_cfg_shape(self) -> None:
        """enhance + CFG 输出 shape (768,)。"""
        pipe = ELFPipeline(device="cpu")
        vec = pipe.enhance("query", steps=2, noise_t=0.4, cfg_scale=2.0)
        assert vec.shape == (768,)

    def test_enhance_with_cfg_dtype(self) -> None:
        """enhance + CFG 输出 float32。"""
        pipe = ELFPipeline(device="cpu")
        vec = pipe.enhance("query", steps=2, noise_t=0.4, cfg_scale=2.0)
        assert vec.dtype == np.float32

    def test_enhance_with_cfg_normalized(self) -> None:
        """enhance + CFG 输出 L2 归一化。"""
        pipe = ELFPipeline(device="cpu")
        vec = pipe.enhance("query", steps=2, noise_t=0.4, cfg_scale=2.0)
        norm = float(np.linalg.norm(vec))
        assert abs(norm - 1.0) < 1e-5

    def test_cfg_scale_effect_differs(self) -> None:
        """不同 cfg_scale 产生不同向量（去噪后输出不同方向）。"""

        def cond_fn(z: np.ndarray, t: float) -> np.ndarray:
            return -z * 0.1  # 朝原点收缩

        def uncond_fn(z: np.ndarray, t: float) -> np.ndarray:
            # 朝完全不同方向偏移（对第一维施加正速度）
            result = np.zeros_like(z)
            result[..., 0] = 1.0
            return result

        pipe = ELFPipeline(device="cpu", model_fn_cond=cond_fn, model_fn_uncond=uncond_fn)
        rng42 = np.random.default_rng(42)
        vec1 = pipe.enhance("query", steps=2, noise_t=0.4, cfg_scale=1.0, rng=rng42)
        rng42 = np.random.default_rng(42)
        vec2 = pipe.enhance("query", steps=2, noise_t=0.4, cfg_scale=3.0, rng=rng42)
        assert not np.allclose(vec1, vec2, atol=1e-4)

    # ── custom model_fn ─────────────────────

    def test_custom_model_fn(self) -> None:
        """自定义 model_fn 被正确使用。"""
        calls_cond: list[float] = []
        calls_uncond: list[float] = []

        def cond_fn(z: np.ndarray, t: float) -> np.ndarray:
            calls_cond.append(t)
            return np.zeros_like(z)

        def uncond_fn(z: np.ndarray, t: float) -> np.ndarray:
            calls_uncond.append(t)
            return np.zeros_like(z)

        pipe = ELFPipeline(
            device="cpu",
            model_fn_cond=cond_fn,
            model_fn_uncond=uncond_fn,
        )
        vec = pipe.enhance("query", steps=2, noise_t=0.4, cfg_scale=2.0)
        assert vec.shape == (768,)
        # 条件函数应被调用 steps 次，无条件函数也应被调用 steps 次
        assert len(calls_cond) == 2
        assert len(calls_uncond) == 2

    def test_custom_model_fn_no_cfg(self) -> None:
        """cfg_scale=1.0 时不调用无条件函数。"""
        cond_calls: list[float] = []
        uncond_calls: list[float] = []

        def cond_fn(z: np.ndarray, t: float) -> np.ndarray:
            cond_calls.append(t)
            return -z * 0.1

        def uncond_fn(z: np.ndarray, t: float) -> np.ndarray:
            uncond_calls.append(t)
            return np.zeros_like(z)

        pipe = ELFPipeline(device="cpu", model_fn_cond=cond_fn, model_fn_uncond=uncond_fn)
        pipe.enhance("query", steps=3, noise_t=0.4, cfg_scale=1.0)
        assert len(cond_calls) == 3
        # cfg_scale=1.0 时不走 CFG 分支，不调用 uncond_fn
        assert len(uncond_calls) == 0

    # ── 确定性 ─────────────────────────────

    def test_deterministic_with_seed(self) -> None:
        """相同种子 + 相同输入产生相同输出。"""
        pipe = ELFPipeline(device="cpu")
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        v1 = pipe.enhance("query", steps=2, noise_t=0.4, cfg_scale=2.0, rng=rng1)
        v2 = pipe.enhance("query", steps=2, noise_t=0.4, cfg_scale=2.0, rng=rng2)
        assert np.allclose(v1, v2, atol=1e-5)

    def test_different_seed_different(self) -> None:
        """不同种子产生不同输出。"""
        pipe = ELFPipeline(device="cpu")
        v1 = pipe.enhance(
            "query", steps=2, noise_t=0.5, cfg_scale=2.0, rng=np.random.default_rng(42)
        )
        v2 = pipe.enhance(
            "query", steps=2, noise_t=0.5, cfg_scale=2.0, rng=np.random.default_rng(99)
        )
        assert not np.allclose(v1, v2, atol=0.1)

    # ── 不同参数 ───────────────────────────

    def test_enhance_one_step(self) -> None:
        """steps=1 正常工作。"""
        pipe = ELFPipeline(device="cpu")
        vec = pipe.enhance("query", steps=1, noise_t=0.3, cfg_scale=1.0)
        assert vec.shape == (768,)

    def test_enhance_four_steps(self) -> None:
        """steps=4 正常工作。"""
        pipe = ELFPipeline(device="cpu")
        vec = pipe.enhance("query", steps=4, noise_t=0.5, cfg_scale=1.0)
        assert vec.shape == (768,)

    def test_enhance_high_noise(self) -> None:
        """高噪声 t=0.8 正常工作。"""
        pipe = ELFPipeline(device="cpu")
        vec = pipe.enhance("query", steps=2, noise_t=0.8, cfg_scale=1.0)
        assert vec.shape == (768,)

    # ── enhance_batch ──────────────────────

    def test_enhance_batch_shape(self) -> None:
        """enhance_batch 输出 shape (N, 768)。"""
        pipe = ELFPipeline(device="cpu")
        texts = ["query one", "query two", "query three"]
        vecs = pipe.enhance_batch(texts, steps=2, noise_t=0.4, cfg_scale=1.0)
        assert vecs.shape == (3, 768)

    def test_enhance_batch_dtype(self) -> None:
        """enhance_batch 输出 float32。"""
        pipe = ELFPipeline(device="cpu")
        vecs = pipe.enhance_batch(["a", "b"], steps=2, noise_t=0.4, cfg_scale=1.0)
        assert vecs.dtype == np.float32

    def test_enhance_batch_empty_raises(self) -> None:
        """空列表应抛出 ValueError。"""
        pipe = ELFPipeline(device="cpu")
        with pytest.raises(ValueError, match="不能为空"):
            pipe.enhance_batch([])

    def test_enhance_batch_encodes_once(self) -> None:
        """批量编码只调用 1 次（不是逐条调用 N 次）。"""
        encoder = _FakeEncoder()
        pipe = ELFPipeline(device="cpu", encoder=encoder)
        pipe.enhance_batch(["a", "b", "c"], steps=2, noise_t=0.4, cfg_scale=1.0)
        assert encoder.encode_batch_calls == 1
        assert encoder.encode_calls == 0

    def test_enhance_batch_matches_individual(self) -> None:
        """批量输出与逐条输出数值一致（相同 seed + 相同输入）。

        批量路径用同一个 rng 按行消费噪声流，等价于把同一个 rng
        逐条传给 enhance()。
        """
        pipe = ELFPipeline(device="cpu", encoder=_FakeEncoder())
        texts = ["query one", "query two", "query three"]
        batch_vecs = pipe.enhance_batch(
            texts, steps=2, noise_t=0.4, cfg_scale=2.0, rng=np.random.default_rng(42)
        )
        shared_rng = np.random.default_rng(42)
        per_item = np.stack(
            [
                pipe.enhance(text, steps=2, noise_t=0.4, cfg_scale=2.0, rng=shared_rng)
                for text in texts
            ]
        )
        assert batch_vecs.shape == (3, 768)
        assert np.allclose(batch_vecs, per_item, atol=1e-6)

    def test_enhance_batch_larger_batch(self) -> None:
        """批量大小验证: 更大批量仍返回 (N, 768) 且只编码一次。"""
        encoder = _FakeEncoder()
        pipe = ELFPipeline(device="cpu", encoder=encoder)
        texts = [f"query number {i}" for i in range(8)]
        vecs = pipe.enhance_batch(
            texts, steps=2, noise_t=0.4, cfg_scale=1.0, rng=np.random.default_rng(42)
        )
        assert vecs.shape == (8, 768)
        assert vecs.dtype == np.float32
        assert encoder.encode_batch_calls == 1

    def test_enhance_batch_model_fn_batched(self) -> None:
        """性能验证: 批量模式下速度场每步只调用一次，且收到完整 batch。"""
        cond_shapes: list[tuple[int, ...]] = []
        uncond_shapes: list[tuple[int, ...]] = []

        def cond_fn(z: np.ndarray, t: float) -> np.ndarray:
            cond_shapes.append(z.shape)
            return -z * 0.1

        def uncond_fn(z: np.ndarray, t: float) -> np.ndarray:
            uncond_shapes.append(z.shape)
            return -z * 0.1

        pipe = ELFPipeline(
            device="cpu",
            encoder=_FakeEncoder(),
            model_fn_cond=cond_fn,
            model_fn_uncond=uncond_fn,
        )
        texts = [f"query number {i}" for i in range(5)]
        pipe.enhance_batch(
            texts, steps=3, noise_t=0.4, cfg_scale=2.0, rng=np.random.default_rng(42)
        )
        # 每步调用一次（共 steps 次），而非逐条串行的 steps × N 次
        assert len(cond_shapes) == 3
        assert len(uncond_shapes) == 3
        assert all(shape == (5, 768) for shape in cond_shapes)
        assert all(shape == (5, 768) for shape in uncond_shapes)

    # ── 默认构造函数 ───────────────────────

    def test_default_construction(self) -> None:
        """无参数构造不报错。"""
        pipe = ELFPipeline(device="cpu")
        assert pipe.encoder is not None
