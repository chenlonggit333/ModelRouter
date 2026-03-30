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
    整合Level 1、Level 2和Level 3分类器，实现完整的三级路由决策流程
    """

    def __init__(self, level1_classifier, level3_classifier, level2_matcher=None):
        """
        初始化分类路由整合器

        Args:
            level1_classifier: Level 1规则分类器
            level3_classifier: Level 3 LLM分类器
            level2_matcher: Level 2 Embedding相似度匹配器（可选）
        """
        self.level1 = level1_classifier
        self.level3 = level3_classifier
        self.level2 = level2_matcher  # Level 2可选，不传入则跳过

    async def route(self, input_text: str, token_count: int) -> RouteResult:
        """
        执行完整的三级路由决策流程

        流程：Level 1 → Level 2 (可选) → Level 3
        """
        start_time = time.time()
        path = []

        # Step 1: Level 1规则筛选
        level1_result = self.level1.classify(input_text, token_count)
        path.append("level1_rules")

        if level1_result.decision == RouteDecision.TIER1_SIMPLE:
            latency_ms = int((time.time() - start_time) * 1000)
            result = RouteResult(
                decision="tier1",
                path=path,
                reasoning=level1_result.reason,
                latency_ms=latency_ms,
            )
            # 添加到Level 2存储（如果启用）
            await self._add_to_level2_storage(input_text, result)
            return result

        if level1_result.decision == RouteDecision.TIER3_COMPLEX:
            latency_ms = int((time.time() - start_time) * 1000)
            result = RouteResult(
                decision="tier3",
                path=path,
                reasoning=level1_result.reason,
                latency_ms=latency_ms,
            )
            # 添加到Level 2存储（如果启用）
            await self._add_to_level2_storage(input_text, result)
            return result

        # Step 2: Level 2 Embedding语义匹配（如果启用）
        if self.level2 is not None:
            level2_result = await self.level2.find_similar(input_text)
            if level2_result is not None:
                # 找到相似问题，复用历史决策
                path.append("level2_embedding")
                latency_ms = int((time.time() - start_time) * 1000)
                return RouteResult(
                    decision=level2_result.route_decision,
                    path=path,
                    complexity_score=level2_result.complexity_score,
                    confidence=level2_result.confidence,
                    reasoning=f"Similar to: {level2_result.similar_text[:50]}... (score: {level2_result.similarity_score:.2f})",
                    latency_ms=latency_ms,
                )

        # Step 3: Level 1无法判断且Level 2未匹配，走到Level 3
        level3_result = await self.level3.classify(input_text)
        path.append("level3_llm")

        latency_ms = int((time.time() - start_time) * 1000)
        result = RouteResult(
            decision=level3_result.route_decision,
            path=path,
            complexity_score=level3_result.complexity_score,
            confidence=level3_result.confidence,
            reasoning=level3_result.reasoning,
            latency_ms=latency_ms,
        )

        # 添加到Level 2存储（如果启用）
        await self._add_to_level2_storage(
            input_text, result, level3_result.complexity_score, level3_result.confidence
        )

        return result

    async def _add_to_level2_storage(
        self,
        text: str,
        result: RouteResult,
        complexity_score: Optional[float] = None,
        confidence: Optional[float] = None,
    ):
        """
        将结果添加到Level 2存储（如果启用）

        Args:
            text: 问题文本
            result: 路由结果
            complexity_score: 复杂度评分（Level 3有，Level 1没有）
            confidence: 置信度（Level 3有，Level 1没有）
        """
        if self.level2 is not None:
            try:
                await self.level2.add_record(
                    text=text,
                    route_decision=result.decision,
                    complexity_score=complexity_score,
                    confidence=confidence,
                )
            except Exception as e:
                # 添加失败不应影响主流程
                import logging

                logging.getLogger(__name__).warning(
                    f"Failed to add to Level 2 storage: {e}"
                )
