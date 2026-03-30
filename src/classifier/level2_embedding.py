"""
Level 2: Embedding语义匹配模块

使用 sentence-transformers 将文本转为向量，并计算相似度。
MVP阶段使用内存存储，生产环境可替换为Milvus。
"""

import asyncio
import numpy as np
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

# Default configuration constants
DEFAULT_SIMILARITY_THRESHOLD = 0.85
DEFAULT_TOP_K = 5
DEFAULT_MAX_VECTOR_STORE_SIZE = 100000
DEFAULT_EMBEDDING_DIMENSION = 384  # all-MiniLM-L6-v2 output dimension


@dataclass
class SimilarityResult:
    """相似度匹配结果"""

    query_text: str
    similar_text: str
    similarity_score: float
    route_decision: str
    complexity_score: Optional[float]
    confidence: Optional[float]


class EmbeddingService:
    """
    Embedding服务
    将文本转为向量表示
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        初始化Embedding服务

        Args:
            model_name: sentence-transformers模型名称
        """
        self.model_name = model_name
        self._model = None
        self._dimension = 384  # all-MiniLM-L6-v2的输出维度

    def _load_model(self):
        """懒加载模型"""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info(f"Loading embedding model: {self.model_name}")
                self._model = SentenceTransformer(self.model_name)
                logger.info("Embedding model loaded successfully")
            except ImportError:
                logger.error(
                    "sentence-transformers not installed. Run: pip install sentence-transformers"
                )
                raise

    def encode(self, text: str) -> np.ndarray:
        """
        将文本编码为向量

        Args:
            text: 输入文本

        Returns:
            numpy array: 384维向量
        """
        self._load_model()

        # 清理文本
        text = text.strip()
        if not text:
            # 空文本返回零向量
            return np.zeros(self._dimension)

        # 编码
        embedding = self._model.encode(text, convert_to_numpy=True)
        return embedding

    def encode_batch(self, texts: List[str]) -> np.ndarray:
        """
        批量编码文本

        Args:
            texts: 文本列表

        Returns:
            numpy array: (n, 384) 矩阵
        """
        self._load_model()

        # 清理文本
        texts = [t.strip() for t in texts]

        # 批量编码
        embeddings = self._model.encode(texts, convert_to_numpy=True)
        return embeddings

    def cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        计算余弦相似度

        Args:
            vec1: 向量1
            vec2: 向量2

        Returns:
            float: 相似度分数 (-1 到 1)
        """
        # 归一化
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)

        # 点积
        similarity = np.dot(vec1_norm, vec2_norm)

        return float(similarity)


class VectorStore:
    """
    向量存储
    MVP阶段使用内存存储，生产环境应使用Milvus
    """

    def __init__(self, max_size: int = DEFAULT_MAX_VECTOR_STORE_SIZE):
        """
        初始化向量存储

        Args:
            max_size: 最大存储记录数
        """
        self.max_size = max_size
        self._texts: List[str] = []
        self._embeddings: List[np.ndarray] = []
        self._metadata: List[Dict] = []
        self._lock = asyncio.Lock()

    async def add(self, text: str, embedding: np.ndarray, metadata: Dict):
        """
        添加记录到存储

        Args:
            text: 原始文本
            embedding: 向量表示
            metadata: 元数据（包含route_decision等）
        """
        async with self._lock:
            # 如果超过最大容量，移除最旧的记录（FIFO）
            if len(self._texts) >= self.max_size:
                self._texts.pop(0)
                self._embeddings.pop(0)
                self._metadata.pop(0)

            self._texts.append(text)
            self._embeddings.append(embedding)
            self._metadata.append(metadata)

            logger.debug(f"Added to vector store. Current size: {len(self._texts)}")

    def search(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float, Dict]]:
        """
        搜索相似记录

        Args:
            query_embedding: 查询向量
            top_k: 返回最相似的k条记录

        Returns:
            List of tuples: (text, similarity_score, metadata)
        """
        if len(self._embeddings) == 0:
            return []

        # 计算与所有记录的相似度
        similarities = []
        for i, emb in enumerate(self._embeddings):
            # 余弦相似度
            similarity = self._cosine_similarity(query_embedding, emb)
            similarities.append((i, similarity))

        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)

        # 返回top_k
        results = []
        for i, score in similarities[:top_k]:
            results.append((self._texts[i], score, self._metadata[i]))

        return results

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """计算余弦相似度"""
        vec1_norm = vec1 / (np.linalg.norm(vec1) + 1e-8)
        vec2_norm = vec2 / (np.linalg.norm(vec2) + 1e-8)
        return float(np.dot(vec1_norm, vec2_norm))

    def size(self) -> int:
        """返回存储的记录数"""
        return len(self._texts)


class Level2SimilarityMatcher:
    """
    Level 2: Embedding语义匹配器
    基于向量相似度匹配历史问题
    """

    def __init__(
        self,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
        top_k: int = DEFAULT_TOP_K,
    ):
        """
        初始化语义匹配器

        Args:
            similarity_threshold: 相似度阈值，超过此值认为匹配成功
            top_k: 查询时返回的最相似记录数
        """
        self.embedding_service = EmbeddingService()
        self.vector_store = VectorStore()
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

        logger.info(
            f"Level2SimilarityMatcher initialized with "
            f"threshold={similarity_threshold}, top_k={top_k}"
        )

    async def find_similar(self, query_text: str) -> Optional[SimilarityResult]:
        """
        查找相似的历史问题

        Args:
            query_text: 查询文本

        Returns:
            SimilarityResult: 如果找到相似问题且相似度超过阈值
            None: 未找到足够相似的问题
        """
        try:
            # 1. 编码查询文本
            query_embedding = self.embedding_service.encode(query_text)

            # 2. 搜索相似记录
            results = self.vector_store.search(query_embedding, top_k=self.top_k)

            if not results:
                logger.debug("No similar records found in vector store")
                return None

            # 3. 检查最相似记录的相似度
            best_match_text, best_score, best_metadata = results[0]

            logger.debug(
                f"Best similarity score: {best_score:.4f} "
                f"(threshold: {self.similarity_threshold})"
            )

            if best_score >= self.similarity_threshold:
                # 找到足够相似的问题
                logger.info(
                    f"Found similar question with score {best_score:.4f}. "
                    f"Reusing route decision: {best_metadata.get('route_decision')}"
                )

                return SimilarityResult(
                    query_text=query_text,
                    similar_text=best_match_text,
                    similarity_score=best_score,
                    route_decision=best_metadata.get("route_decision", "tier3"),
                    complexity_score=best_metadata.get("complexity_score"),
                    confidence=best_metadata.get("confidence"),
                )

            # 相似度不够，返回None
            return None

        except Exception as e:
            logger.error(f"Error in find_similar: {e}", exc_info=True)
            # 出错时返回None，让调用方继续到Level 3
            return None

    async def add_record(
        self,
        text: str,
        route_decision: str,
        complexity_score: Optional[float] = None,
        confidence: Optional[float] = None,
    ):
        """
        添加记录到向量存储

        Args:
            text: 问题文本
            route_decision: 路由决策结果
            complexity_score: 复杂度评分
            confidence: 置信度
        """
        try:
            # 编码文本
            embedding = self.embedding_service.encode(text)

            # 构造元数据
            metadata = {
                "route_decision": route_decision,
                "complexity_score": complexity_score,
                "confidence": confidence,
            }

            # 添加到存储
            await self.vector_store.add(text, embedding, metadata)

            logger.debug(
                f"Added record to vector store. Total: {self.vector_store.size()}"
            )

        except Exception as e:
            logger.error(f"Error adding record: {e}", exc_info=True)

    def get_stats(self) -> Dict:
        """获取统计信息"""
        return {
            "vector_store_size": self.vector_store.size(),
            "similarity_threshold": self.similarity_threshold,
            "top_k": self.top_k,
        }
