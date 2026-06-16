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

# Colab 模式（Phase 4 启用）
COLAB_MODE = False
