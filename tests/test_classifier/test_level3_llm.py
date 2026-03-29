import pytest
from unittest.mock import Mock, AsyncMock
from src.classifier.level3_llm import Level3Classifier, ComplexityResult


@pytest.fixture
def mock_llm_client():
    client = Mock()
    client.classify = AsyncMock()
    return client


@pytest.fixture
def classifier(mock_llm_client):
    return Level3Classifier(llm_client=mock_llm_client)


class TestLevel3Classifier:
    @pytest.mark.asyncio
    async def test_classify_simple_question(self, classifier, mock_llm_client):
        # Mock LLM返回简单分类
        mock_llm_client.classify.return_value = {
            "complexity_score": 0.2,
            "confidence": 0.9,
            "reasoning": "简单问候",
        }

        result = await classifier.classify("你好，今天天气怎么样？")

        assert result.complexity_score == 0.2
        assert result.confidence == 0.9
        assert result.route_decision == "tier1"

    @pytest.mark.asyncio
    async def test_classify_complex_question(self, classifier, mock_llm_client):
        # Mock LLM返回复杂分类
        mock_llm_client.classify.return_value = {
            "complexity_score": 0.8,
            "confidence": 0.85,
            "reasoning": "需要深入分析",
        }

        result = await classifier.classify("请详细分析这个算法的时间复杂度")

        assert result.complexity_score == 0.8
        assert result.route_decision == "tier3"

    @pytest.mark.asyncio
    async def test_classify_medium_question(self, classifier, mock_llm_client):
        # Mock LLM返回中等分类
        mock_llm_client.classify.return_value = {
            "complexity_score": 0.5,
            "confidence": 0.8,
            "reasoning": "中等复杂度",
        }

        result = await classifier.classify("解释一下这个概念")

        assert result.complexity_score == 0.5
        assert result.route_decision == "tier2"

    @pytest.mark.asyncio
    async def test_low_confidence_defaults_to_tier3(self, classifier, mock_llm_client):
        # Mock LLM返回低置信度
        mock_llm_client.classify.return_value = {
            "complexity_score": 0.5,
            "confidence": 0.4,  # 低置信度
            "reasoning": "不确定",
        }

        result = await classifier.classify("某个模糊的问题")

        # 低置信度时应该走GLM5以确保质量
        assert result.route_decision == "tier3"
