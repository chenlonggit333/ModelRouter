import pytest
from unittest.mock import Mock, AsyncMock
from src.classifier.router import ClassificationRouter
from src.classifier.level1_rules import RouteDecision


@pytest.fixture
def mock_level1():
    classifier = Mock()
    classifier.classify = Mock()
    return classifier


@pytest.fixture
def mock_level3():
    classifier = Mock()
    classifier.classify = AsyncMock()
    return classifier


@pytest.fixture
def router(mock_level1, mock_level3):
    return ClassificationRouter(
        level1_classifier=mock_level1, level3_classifier=mock_level3
    )


class TestClassificationRouter:
    @pytest.mark.asyncio
    async def test_level1_simple_route(self, router, mock_level1):
        # Level 1直接判定为简单
        from src.classifier.level1_rules import ClassificationResult

        mock_level1.classify.return_value = ClassificationResult(
            decision=RouteDecision.TIER1_SIMPLE, reason="短文本"
        )

        result = await router.route("你好", token_count=10)

        assert result.decision == "tier1"
        assert result.path == ["level1_rules"]

    @pytest.mark.asyncio
    async def test_level1_complex_route(self, router, mock_level1):
        # Level 1直接判定为复杂
        from src.classifier.level1_rules import ClassificationResult

        mock_level1.classify.return_value = ClassificationResult(
            decision=RouteDecision.TIER3_COMPLEX, reason="包含代码"
        )

        result = await router.route("编写代码", token_count=50)

        assert result.decision == "tier3"
        assert result.path == ["level1_rules"]

    @pytest.mark.asyncio
    async def test_fallback_to_level3(self, router, mock_level1, mock_level3):
        # Level 1无法判断，走到Level 3
        from src.classifier.level1_rules import ClassificationResult
        from src.classifier.level3_llm import ComplexityResult

        mock_level1.classify.return_value = ClassificationResult(
            decision=RouteDecision.CONTINUE, reason="无法判断"
        )

        mock_level3.classify.return_value = ComplexityResult(
            complexity_score=0.4,
            confidence=0.8,
            reasoning="中等",
            route_decision="tier2",
        )

        result = await router.route("某个问题", token_count=500)

        assert result.decision == "tier2"
        assert "level1_rules" in result.path
        assert "level3_llm" in result.path
