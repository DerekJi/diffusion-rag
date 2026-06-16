"""统一检索接口（双链路共享）。"""

import numpy as np

from src.utils.logger import get_logger
from src.vector_store.indexer import FAISSIndexer

logger = get_logger(__name__)


class Retriever:
    """统一检索接口。

    Attributes:
        indexer: FAISSIndexer 实例。
    """

    def __init__(self, indexer: FAISSIndexer) -> None:
        """初始化检索器。

        Args:
            indexer: 已构建的 FAISSIndexer 实例。
        """
        self._indexer = indexer

    def search(self, query_vector: np.ndarray, k: int = 10) -> tuple[list[str], list[float]]:
        """检索 top-k 结果。

        Args:
            query_vector: 768-dim 查询向量。
            k: 返回数量。

        Returns:
            (doc_ids, distances): 文档 ID 列表和对应距离。
            无结果时返回 ([], [])。
        """
        try:
            index = self._indexer.index
        except RuntimeError:
            logger.warning("检索失败: 索引尚未构建")
            return ([], [])

        if index.ntotal == 0:
            logger.warning("检索失败: 索引为空")
            return ([], [])

        query = query_vector.astype(np.float32).reshape(1, -1)
        distances, indices = index.search(query, min(k, index.ntotal))

        valid_indices = indices[0][indices[0] >= 0]
        if len(valid_indices) == 0:
            return ([], [])

        doc_ids = [self._indexer.doc_ids[i] for i in valid_indices]
        dists = [float(d) for d in distances[0][: len(valid_indices)]]

        return (doc_ids, dists)
