"""全局配置。"""

# 路径 (consumed in Phase 3+)
MODELS_DIR = "models"
DATA_DIR = "data"
EXPERIMENTS_DIR = "experiments"

# 向量维度 (consumed in Phase 2+)
VECTOR_DIM = 768

# 默认检索参数
DEFAULT_K_VALUES = [5, 10, 20]
DEFAULT_INDEX_NLIST = 100
DEFAULT_ENCODER = "BAAI/bge-base-en-v1.5"

# 评测
DEFAULT_SEED = 42
DEFAULT_REPEATS = 3  # Phase 5

# 双链路方法常量 (Phase 3.1)
METHOD_BASELINE = "baseline"
METHOD_ELF = "elf"
SUPPORTED_METHODS: tuple[str, ...] = (METHOD_BASELINE, METHOD_ELF)

# ELF 增强默认参数（与 src/elf/pipeline.py 默认值一致）
DEFAULT_ELF_STEPS = 2
DEFAULT_ELF_NOISE_T = 0.4
DEFAULT_ELF_CFG_SCALE = 2.0

# Colab 模式（Phase 4 启用）
COLAB_MODE = False

# BEIR 数据集映射（供 download.py 与 dataset.py 共用）
BEIR_DATASET_MAP: dict[str, str] = {
    "nfcorpus": "BeIR/nfcorpus",
    "msmarco": "BeIR/msmarco",
    "nq": "BeIR/nq",
    "fiqa": "BeIR/fiqa",
}
