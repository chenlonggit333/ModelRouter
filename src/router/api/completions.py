import os
from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from src.router.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Usage,
    Choice,
    RouterInfo,
)
from src.router.config import settings
from src.classifier.level1_rules import Level1Classifier
from src.classifier.level3_llm import Level3Classifier, MockLLMClient
from src.classifier.router import ClassificationRouter
from src.models.pool import ModelPool, ModelInstance
from src.models.load_balancer import LoadBalancer
from src.models.glm5_client import GLM5Client
from src.models.lightweight_client import LightweightModelClient

router = APIRouter()

# 全局组件（简化版，生产环境应使用依赖注入）
_classifier_router = None
_model_pool = None
_load_balancer = None
_glm5_client = None
_lightweight_client = None


def _get_components():
    """获取或初始化组件"""
    global \
        _classifier_router, \
        _model_pool, \
        _load_balancer, \
        _glm5_client, \
        _lightweight_client

    if _classifier_router is None:
        # 初始化分类器
        level1 = Level1Classifier(settings.rules.dict())
        # Level 3分类器需要LLM客户端（这里简化处理，实际应连接到vLLM服务）
        level3_llm_client = MockLLMClient()  # 占位
        level3 = Level3Classifier(level3_llm_client)
        _classifier_router = ClassificationRouter(level1, level3)

    if _model_pool is None:
        _model_pool = ModelPool()
        # 从配置加载实例
        _load_model_instances()

    if _load_balancer is None:
        _load_balancer = LoadBalancer(strategy="least_connection")

    if _glm5_client is None:
        # 从配置或环境变量加载GLM5地址
        glm5_url = getattr(settings, "glm5_base_url", None) or os.getenv(
            "GLM5_BASE_URL", "http://localhost:8000"
        )
        _glm5_client = GLM5Client(base_url=glm5_url, timeout=300)

    if _lightweight_client is None:
        # 从配置加载轻量模型地址列表
        lightweight_urls = getattr(
            settings, "lightweight_base_urls", None
        ) or os.getenv("LIGHTWEIGHT_BASE_URLS", "http://localhost:8001").split(",")
        lightweight_model = getattr(settings, "lightweight_model_name", "qwen2.5-7b")
        _lightweight_client = LightweightModelClient(
            base_urls=lightweight_urls, model_name=lightweight_model, timeout=60
        )

    return (
        _classifier_router,
        _model_pool,
        _load_balancer,
        _glm5_client,
        _lightweight_client,
    )


def _load_model_instances():
    """从配置加载模型实例"""
    # 实际应用中从配置文件或配置中心加载
    global _model_pool

    # GLM5实例
    _model_pool.register(
        ModelInstance(
            id="glm5-001",
            tier="tier3",
            host="glm5.internal",
            port=8000,
            max_concurrency=100,
        )
    )

    # 轻量模型实例
    _model_pool.register(
        ModelInstance(
            id="qwen-001", tier="tier1", host="localhost", port=8001, max_concurrency=50
        )
    )


def _count_tokens(text: str) -> int:
    """估算token数（简化版，实际应使用tiktoken）"""
    # 简单估算：中文1个字符≈1个token，英文4个字符≈1个token
    import re

    chinese_chars = len(re.findall(r"[\u4e00-\u9fff]", text))
    other_chars = len(text) - chinese_chars
    return chinese_chars + (other_chars // 4) + 1


@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    x_router_debug: Optional[str] = Header(None, alias="X-Router-Debug"),
):
    """
    OpenAI兼容的聊天完成接口

    - model="auto": 自动路由
    - model="glm5"/"light"/"medium": 强制指定模型
    """
    try:
        (
            classifier_router,
            model_pool,
            load_balancer,
            glm5_client,
            lightweight_client,
        ) = _get_components()

        # 计算token数
        input_text = request.messages[-1].content if request.messages else ""
        token_count = _count_tokens(input_text)

        # 路由决策
        if request.model == "auto":
            route_result = await classifier_router.route(input_text, token_count)
            target_tier = route_result.decision
        else:
            # 强制指定模型
            target_tier = request.model
            route_result = None

        # 调用对应模型
        if target_tier == "tier1" or target_tier == "light":
            model_response = await lightweight_client.chat_completion(
                messages=[m.model_dump() for m in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            model_name = "qwen2.5-7b"
        else:
            # tier2/tier3都先走GLM5（MVP简化）
            model_response = await glm5_client.chat_completion(
                messages=[m.model_dump() for m in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens,
            )
            model_name = "glm5"

        # 构造响应
        response = ChatCompletionResponse(
            model=model_name,
            choices=[Choice(**choice) for choice in model_response["choices"]],
            usage=Usage(**model_response["usage"]),
        )

        # 添加路由信息（如果请求了debug模式）
        if x_router_debug:
            response.router_info = RouterInfo(
                complexity_score=route_result.complexity_score
                if route_result
                else None,
                route_decision=target_tier,
                classification_time_ms=route_result.latency_ms if route_result else 0,
                routing_path=route_result.path if route_result else ["forced"],
            )

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
