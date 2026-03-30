import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, patch
from src.classifier.level2_embedding import (
    EmbeddingService,
    VectorStore,
    Level2SimilarityMatcher,
    SimilarityResult,
    DEFAULT_SIMILARITY_THRESHOLD,
    DEFAULT_TOP_K,
)


class TestEmbeddingService:
    """测试Embedding服务"""

    def test_encode_empty_text(self):
        """测试空文本编码"""
        service = EmbeddingService()
        # 使用Mock避免加载真实模型
        with patch.object(service, "_model"):
            result = service.encode("")
            assert isinstance(result, np.ndarray)
            assert result.shape[0] == 384
            assert np.allclose(result, 0)

    def test_cosine_similarity_identical(self):
        """测试相同向量的相似度"""
        service = EmbeddingService()
        vec = np.array([1.0, 0.0, 0.0])
        similarity = service.cosine_similarity(vec, vec)
        assert similarity == pytest.approx(1.0, abs=1e-6)

    def test_cosine_similarity_orthogonal(self):
        """测试正交向量的相似度"""
        service = EmbeddingService()
        vec1 = np.array([1.0, 0.0, 0.0])
        vec2 = np.array([0.0, 1.0, 0.0])
        similarity = service.cosine_similarity(vec1, vec2)
        assert similarity == pytest.approx(0.0, abs=1e-6)


class TestVectorStore:
    """测试向量存储"""

    @pytest.mark.asyncio
    async def test_add_and_search(self):
        """测试添加和搜索"""
        store = VectorStore(max_size=100)

        # 添加测试数据
        embedding1 = np.array([1.0, 0.0, 0.0])
        await store.add("text1", embedding1, {"route": "tier1"})

        embedding2 = np.array([0.9, 0.1, 0.0])
        await store.add("text2", embedding2, {"route": "tier2"})

        # 搜索
        query = np.array([1.0, 0.0, 0.0])
        results = store.search(query, top_k=2)

        assert len(results) == 2
        assert results[0][0] == "text1"  # 最相似
        assert results[0][2]["route"] == "tier1"
        assert results[0][1] > 0.9  # 高相似度

    @pytest.mark.asyncio
    async def test_max_size_limit(self):
        """测试存储容量限制"""
        store = VectorStore(max_size=2)

        # 添加超过容量的数据
        for i in range(5):
            await store.add(
                f"text{i}", np.array([float(i), 0.0, 0.0]), {"route": f"tier{i}"}
            )

        # 只保留最新的2条
        assert store.size() == 2

    def test_empty_search(self):
        """测试空存储搜索"""
        store = VectorStore()
        query = np.array([1.0, 0.0, 0.0])
        results = store.search(query)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """测试并发访问安全性"""
        store = VectorStore(max_size=1000)

        async def add_records(start_idx):
            for i in range(10):
                await store.add(
                    f"text_{start_idx}_{i}", np.random.rand(384), {"route": "tier1"}
                )

        # 并发添加记录
        await asyncio.gather(*[add_records(i) for i in range(5)])

        # 验证所有记录都正确添加
        assert store.size() == 50


class TestLevel2SimilarityMatcher:
    """测试Level 2相似度匹配器"""

    @pytest.fixture
    def matcher(self):
        """创建测试用的matcher"""
        return Level2SimilarityMatcher(similarity_threshold=0.85, top_k=5)

    @pytest.fixture
    def mock_embedding_service(self):
        """Mock Embedding服务"""
        service = Mock()

        # 返回简单的one-hot向量
        def mock_encode(text):
            if text == "hello":
                return np.array([1.0, 0.0, 0.0, 0.0])
            elif text == "hi":
                return np.array([0.9, 0.1, 0.0, 0.0])  # 相似
            elif text == "goodbye":
                return np.array([0.0, 0.0, 1.0, 0.0])  # 不相似
            return np.random.rand(4)

        service.encode = mock_encode
        return service

    @pytest.mark.asyncio
    async def test_find_similar_match(self, matcher, mock_embedding_service):
        """测试找到相似匹配"""
        matcher.embedding_service = mock_embedding_service

        # 添加一条记录
        await matcher.add_record("hello", "tier1", 0.2, 0.9)

        # 搜索相似文本
        result = await matcher.find_similar("hi")

        assert result is not None
        assert result.similarity_score >= 0.85
        assert result.route_decision == "tier1"

    @pytest.mark.asyncio
    async def test_find_similar_no_match(self, matcher, mock_embedding_service):
        """测试未找到足够相似的匹配"""
        matcher.embedding_service = mock_embedding_service

        # 添加一条记录
        await matcher.add_record("hello", "tier1", 0.2, 0.9)

        # 搜索不相似的文本
        result = await matcher.find_similar("goodbye")

        # 相似度不够，应该返回None
        assert result is None

    @pytest.mark.asyncio
    async def test_find_similar_empty_store(self, matcher):
        """测试空存储"""
        result = await matcher.find_similar("hello")
        assert result is None

    @pytest.mark.asyncio
    async def test_add_record(self, matcher):
        """测试添加记录"""
        # Mock embedding service
        matcher.embedding_service.encode = Mock(
            return_value=np.array([1.0, 0.0, 0.0, 0.0])
        )

        await matcher.add_record(
            text="test question",
            route_decision="tier2",
            complexity_score=0.5,
            confidence=0.8,
        )

        assert matcher.vector_store.size() == 1

    def test_get_stats(self, matcher):
        """测试获取统计信息"""
        stats = matcher.get_stats()

        assert "vector_store_size" in stats
        assert "similarity_threshold" in stats
        assert "top_k" in stats
        assert stats["similarity_threshold"] == 0.85
        assert stats["top_k"] == 5

    @pytest.mark.asyncio
    async def test_error_handling_find_similar(self, matcher):
        """测试find_similar错误处理"""
        # Mock encode方法抛出异常
        matcher.embedding_service.encode = Mock(
            side_effect=Exception("Encoding failed")
        )

        # 应该返回None而不是抛出异常
        result = await matcher.find_similar("test")
        assert result is None

    @pytest.mark.asyncio
    async def test_error_handling_add_record(self, matcher):
        """测试add_record错误处理"""
        # Mock encode方法抛出异常
        matcher.embedding_service.encode = Mock(
            side_effect=Exception("Encoding failed")
        )

        # 不应该抛出异常
        await matcher.add_record("test", "tier1")

        # 存储应该保持为空
        assert matcher.vector_store.size() == 0


class TestEmbeddingServiceExtended:
    """扩展测试Embedding服务"""

    def test_encode_batch(self):
        """测试批量编码"""
        service = EmbeddingService()

        # Mock模型避免加载真实模型
        with patch.object(service, "_model") as mock_model:
            mock_model.encode.return_value = np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )

            texts = ["text1", "text2", "text3"]
            result = service.encode_batch(texts)

            assert result.shape == (3, 3)
            mock_model.encode.assert_called_once_with(texts, convert_to_numpy=True)

    def test_encode_batch_with_whitespace(self):
        """测试批量编码去除空白"""
        service = EmbeddingService()

        with patch.object(service, "_model") as mock_model:
            mock_model.encode.return_value = np.array([[1.0, 0.0]])

            # 带空格的文本
            texts = ["  hello  ", "  world  "]
            service.encode_batch(texts)

            # 验证去除空格后的文本被编码
            call_args = mock_model.encode.call_args[0][0]
            assert call_args == ["hello", "world"]


class TestDefaultConstants:
    """测试默认常量"""

    def test_default_similarity_threshold(self):
        """验证默认相似度阈值"""
        assert DEFAULT_SIMILARITY_THRESHOLD == 0.85

    def test_default_top_k(self):
        """验证默认top_k"""
        assert DEFAULT_TOP_K == 5

    def test_matcher_uses_default_constants(self):
        """验证matcher使用默认常量"""
        matcher = Level2SimilarityMatcher()
        assert matcher.similarity_threshold == DEFAULT_SIMILARITY_THRESHOLD
        assert matcher.top_k == DEFAULT_TOP_K

    def test_vector_store_uses_default_max_size(self):
        """验证VectorStore使用默认最大容量"""
        store = VectorStore()
        assert store.max_size == 100000
