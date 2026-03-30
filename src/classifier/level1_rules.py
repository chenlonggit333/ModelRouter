from enum import Enum
from typing import Dict, Any, List


class RouteDecision(Enum):
    TIER1_SIMPLE = "tier1"  # 轻量模型
    TIER3_COMPLEX = "tier3"  # GLM5
    CONTINUE = "continue"  # 继续到Level 3分类


class ClassificationResult:
    def __init__(
        self, decision: RouteDecision, reason: str, path: str = "level1_rules"
    ):
        self.decision = decision
        self.reason = reason
        self.path = path

    def __repr__(self):
        return f"ClassificationResult(decision={self.decision}, reason={self.reason})"


class Level1Classifier:
    """Level 1: 极速规则筛选 (<1ms)"""

    def __init__(self, rules: Dict[str, Any]):
        self.simple_keywords = set(rules.get("simple_keywords", []))
        self.complex_keywords = set(rules.get("complex_keywords", []))
        self.thresholds = rules.get("thresholds", {})
        self.token_thresholds = rules.get("token_count", {})

    def classify(self, input_text: str, token_count: int) -> ClassificationResult:
        """
        基于规则进行快速分类

        Args:
            input_text: 用户输入文本
            token_count: 输入token数量

        Returns:
            ClassificationResult: 路由决策结果
        """
        text_lower = input_text.lower()

        # 规则1: 明显简单 - 短文本且无复杂关键词
        if token_count < self.token_thresholds.get("simple_max", 100):
            has_complex = any(kw in text_lower for kw in self.complex_keywords)
            if not has_complex:
                return ClassificationResult(
                    decision=RouteDecision.TIER1_SIMPLE,
                    reason=f"短文本(token={token_count})且无复杂关键词",
                )

        # 规则2: 明显复杂 - 长文本或明确复杂意图
        if token_count > self.token_thresholds.get("complex_min", 2000):
            return ClassificationResult(
                decision=RouteDecision.TIER3_COMPLEX,
                reason=f"长文本(token={token_count})",
            )

        has_complex = any(kw in text_lower for kw in self.complex_keywords)
        if has_complex:
            return ClassificationResult(
                decision=RouteDecision.TIER3_COMPLEX, reason="包含复杂关键词"
            )

        # 规则3: 无法判断，需要Level 3
        return ClassificationResult(
            decision=RouteDecision.CONTINUE,
            reason=f"规则无法判断(token={token_count})，需要进一步分类",
        )

    def has_simple_indicators(self, text: str) -> bool:
        """检查是否包含简单指示词"""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.simple_keywords)

    def has_complex_indicators(self, text: str) -> bool:
        """检查是否包含复杂指示词"""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.complex_keywords)
