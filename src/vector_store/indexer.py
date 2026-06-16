"""FAISS 索引构建器。

基于文档向量构建 IVF_FLAT 索引。
"""

import json
import os

import faiss
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class FAISSIndexer:
    """FAISS 索引构建器。

    Attributes:
        dimension: 向量维度。
        nlist: IVF 聚类中心数。
        index: 底层 FAISS 索引实例。
        _doc_ids: 文档 ID 列表。
    """

    def __init__(self, dimension: int = 768, nlist: int = 100) -> None:
        """初始化 FAISS 索引构建器。

        Args:
            dimension: 向量维度，默认 768。
            nlist: IVF 聚类中心数，默认 100。
        """
        self.dimension = dimension
        self.nlist = nlist
        self._doc_ids: list[str] = []
        self._index: faiss.Index | None = None

    @property
    def index(self) -> faiss.Index:
        """原始 FAISS 索引（供 retriever 使用）。"""
        if self._index is None:
            raise RuntimeError("索引尚未构建，请先调用 build()")
        return self._index

    @property
    def doc_ids(self) -> list[str]:
        """文档 ID 列表。"""
        return self._doc_ids

    def build(self, vectors: np.ndarray, doc_ids: list[str]) -> None:
        """从文档向量构建索引。

        Args:
            vectors: shape (n_docs, dimension) float32。
            doc_ids: 对应文档 ID 列表。

        Raises:
            ValueError: 向量维度与索引维度不匹配。
        """
        if vectors.shape[1] != self.dimension:
            raise ValueError(f"向量维度 {vectors.shape[1]} 与索引维度 {self.dimension} 不匹配")

        faiss.omp_set_num_threads(4)

        n_docs = vectors.shape[0]
        nlist = min(self.nlist, n_docs)

        # 当文档数不足以支撑 IVF 聚类时，使用 FlatIP 精确搜索
        if n_docs < 100 or nlist < 2:
            self._index = faiss.IndexFlatIP(self.dimension)
            self._index.add(vectors)
            self._doc_ids = list(doc_ids)
            logger.info("FAISS 索引构建完成 (IndexFlatIP): %d 个向量", len(doc_ids))
        else:
            quantizer = faiss.IndexFlatIP(self.dimension)
            self._index = faiss.IndexIVFFlat(
                quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT
            )
            self._index.train(vectors)
            self._index.add(vectors)
            self._doc_ids = list(doc_ids)
            logger.info("FAISS 索引构建完成 (IVF_FLAT): %d 个向量, nlist=%d", len(doc_ids), nlist)

    def save(self, path: str) -> None:
        """将索引及 doc_ids 序列化到磁盘。

        Args:
            path: 输出路径（索引本体）。doc_ids 保存在 path + ".ids.json"。

        Raises:
            RuntimeError: 索引尚未构建。
        """
        if self._index is None:
            raise RuntimeError("索引尚未构建，无法保存")
        faiss.write_index(self._index, path)
        ids_path = path + ".ids.json"
        with open(ids_path, "w", encoding="utf-8") as f:
            json.dump(self._doc_ids, f)
        logger.info("索引已保存到 %s (doc_ids: %s)", path, ids_path)

    def load(self, path: str) -> None:
        """从磁盘加载索引及 doc_ids。

        Args:
            path: 索引文件路径。doc_ids 从 path + ".ids.json" 读取。

        Raises:
            FileNotFoundError: 索引文件或 ids 文件不存在。
            RuntimeError: 加载失败。
        """
        try:
            self._index = faiss.read_index(path)
            self.dimension = self._index.d
        except Exception as e:
            logger.error("索引加载失败: %s", e)
            raise RuntimeError(f"无法加载索引 {path}: {e}") from e

        ids_path = path + ".ids.json"
        if not os.path.exists(ids_path):
            raise FileNotFoundError(f"doc_ids 文件不存在: {ids_path}")
        with open(ids_path, encoding="utf-8") as f:
            self._doc_ids = json.load(f)
        logger.info("索引已从 %s 加载 (%d 维, %d 个文档)", path, self.dimension, len(self._doc_ids))
