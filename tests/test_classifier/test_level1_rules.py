import pytest
from src.classifier.level1_rules import Level1Classifier, RouteDecision


@pytest.fixture
def classifier():
    rules = {
        "simple_keywords": ["你好", "谢谢", "什么是"],
        "complex_keywords": ["代码", "分析", "推理"],
        "thresholds": {"tier1": 0.3, "tier2": 0.7},
        "token_count": {"simple_max": 100, "complex_min": 2000},
    }
    return Level1Classifier(rules)


class TestLevel1Classifier:
    def test_simple_greeting(self, classifier):
        result = classifier.classify("你好，介绍一下自己", token_count=10)
        assert result.decision == RouteDecision.TIER1_SIMPLE

    def test_complex_code_request(self, classifier):
        result = classifier.classify("请编写一段代码", token_count=50)
        assert result.decision == RouteDecision.TIER3_COMPLEX

    def test_long_complex_text(self, classifier):
        result = classifier.classify("请详细分析这个问题", token_count=2500)
        assert result.decision == RouteDecision.TIER3_COMPLEX

    def test_ambiguous_needs_more_info(self, classifier):
        result = classifier.classify("这个问题怎么解决", token_count=500)
        assert result.decision == RouteDecision.CONTINUE

    def test_simple_no_keywords_short(self, classifier):
        result = classifier.classify("hello world", token_count=50)
        assert result.decision == RouteDecision.TIER1_SIMPLE
