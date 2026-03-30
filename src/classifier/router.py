from dataclasses import dataclass, field
from typing import List, Optional, Any
from src.classifier.level1_rules import RouteDecision
import time


@dataclass
class RouteResult:
    """最终路由结果"""

    decision: str  # tier1, tier2, tier3
    path: List[str]  # 路由决策路径
    complexity_score: Optional[float] = None
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    latency_ms: int = 0


class ClassificationRouter:
    """
    分类路由整合器
    整合Level 1和Level 3分类器，实现完整的路由决策流程
    """

    def __init__(self, level1_classifier, level3_classifier):
        self.level1 = level1_classifier
        self.level3 = level3_classifier

    async def route(self, input_text: str, token_count: int) -> RouteResult:
        """
        执行完整的路由决策流程

        流程：Level 1 → (需要时) Level 3
        """
        start_time = time.time()
        path = []

        # Step 1: Level 1规则筛选
        level1_result = self.level1.classify(input_text, token_count)
        path.append("level1_rules")

        if level1_result.decision == RouteDecision.TIER1_SIMPLE:
            latency_ms = int((time.time() - start_time) * 1000)
            return RouteResult(
                decision="tier1",
                path=path,
                reasoning=level1_result.reason,
                latency_ms=latency_ms,
            )

        if level1_result.decision == RouteDecision.TIER3_COMPLEX:
            latency_ms = int((time.time() - start_time) * 1000)
            return RouteResult(
                decision="tier3",
                path=path,
                reasoning=level1_result.reason,
                latency_ms=latency_ms,
            )

        # Step 2: Level 1无法判断，走到Level 3
        level3_result = await self.level3.classify(input_text)
        path.append("level3_llm")

        latency_ms = int((time.time() - start_time) * 1000)
        return RouteResult(
            decision=level3_result.route_decision,
            path=path,
            complexity_score=level3_result.complexity_score,
            confidence=level3_result.confidence,
            reasoning=level3_result.reasoning,
            latency_ms=latency_ms,
        )
