from dataclasses import dataclass
from typing import Dict, Any, Optional
import json


class MockLLMClient:
    """占位用的LLM客户端，用于测试和开发阶段"""

    async def classify(self, prompt: str):
        # 简化实现，实际应调用vLLM服务
        return {"complexity_score": 0.5, "confidence": 0.8, "reasoning": "mock"}


@dataclass
class ComplexityResult:
    """复杂度分类结果"""

    complexity_score: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0
    reasoning: str
    route_decision: str  # tier1, tier2, tier3


class Level3Classifier:
    """Level 3: 小模型智能分类 (50-100ms)"""

    def __init__(
        self,
        llm_client,
        tier1_threshold: float = 0.3,
        tier2_threshold: float = 0.7,
        min_confidence: float = 0.7,
    ):
        self.llm_client = llm_client
        self.tier1_threshold = tier1_threshold
        self.tier2_threshold = tier2_threshold
        self.min_confidence = min_confidence

    async def classify(self, input_text: str) -> ComplexityResult:
        """
        使用LLM进行复杂度分类

        Args:
            input_text: 用户输入文本

        Returns:
            ComplexityResult: 复杂度分析结果
        """
        # 构造分类prompt
        prompt = self._build_classification_prompt(input_text)

        # 调用LLM（实际实现中会调用vLLM服务）
        response = await self.llm_client.classify(prompt)

        # 解析结果
        complexity_score = response.get("complexity_score", 0.5)
        confidence = response.get("confidence", 0.5)
        reasoning = response.get("reasoning", "")

        # 根据分数和置信度做路由决策
        route_decision = self._decide_route(complexity_score, confidence)

        return ComplexityResult(
            complexity_score=complexity_score,
            confidence=confidence,
            reasoning=reasoning,
            route_decision=route_decision,
        )

    def _build_classification_prompt(self, question: str) -> str:
        """构建分类prompt"""
        return f"""你是一个问题复杂度评估专家。请评估以下问题的复杂度：

问题：{question}

请从以下维度评估（0-10分）：
1. 推理深度：是否需要多步逻辑推理
2. 知识广度：是否涉及跨领域知识
3. 创造性：是否需要生成创新内容
4. 精确性：是否要求严格的准确性和完整性

请以JSON格式输出：
{{
  "complexity_score": 0.0-1.0,  // 综合复杂度评分
  "confidence": 0.0-1.0,        // 置信度
  "reasoning": "string",        // 主要理由（50字以内）
  "dimensions": {{
    "reasoning_depth": 0-10,
    "knowledge_breadth": 0-10,
    "creativity": 0-10,
    "accuracy_requirement": 0-10
  }}
}}"""

    def _decide_route(self, complexity_score: float, confidence: float) -> str:
        """
        根据复杂度评分和置信度决定路由

        策略：
        - 低置信度(<0.7)时，保守策略走GLM5
        - 根据复杂度评分分配到不同层级
        """
        # 低置信度时保守处理
        if confidence < self.min_confidence:
            return "tier3"

        # 根据复杂度分配
        if complexity_score < self.tier1_threshold:
            return "tier1"
        elif complexity_score < self.tier2_threshold:
            return "tier2"
        else:
            return "tier3"
