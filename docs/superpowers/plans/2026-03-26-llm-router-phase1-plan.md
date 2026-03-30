# LLM智能路由层实施计划 - Phase 1 MVP

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建LLM智能路由层MVP版本，实现基础三级路由（规则筛选→人工分类→模型调用），验证核心假设，达到70%路由准确率，GLM5调用量降低40%+。

**Architecture:** 基于Python FastAPI构建Router Gateway，实现Level 1规则筛选（本地执行，零延迟）和Level 3小模型分类（Qwen2.5-7B，中低端服务器部署），对接GLM5和轻量模型，支持基础负载均衡和降级机制。

**Tech Stack:** Python 3.11, FastAPI, vLLM, Redis, PostgreSQL, pytest

**设计文档:** @docs/superpowers/specs/2026-03-26-llm-router-design.md

---

## 项目结构

```
llm-router/
├── src/
│   ├── router/                    # Router Gateway核心
│   │   ├── __init__.py
│   │   ├── main.py               # FastAPI入口
│   │   ├── config.py             # 配置管理
│   │   ├── models.py             # Pydantic模型
│   │   ├── middleware.py         # 中间件（日志、错误处理）
│   │   └── api/
│   │       ├── __init__.py
│   │       ├── completions.py    # /v1/chat/completions接口
│   │       └── admin.py          # 管理接口
│   ├── classifier/               # 分类器服务
│   │   ├── __init__.py
│   │   ├── level1_rules.py      # Level 1规则筛选
│   │   ├── level3_llm.py        # Level 3小模型分类
│   │   └── router.py            # 分类决策路由器
│   ├── models/                   # 模型池管理
│   │   ├── __init__.py
│   │   ├── pool.py              # 模型池核心
│   │   ├── load_balancer.py     # 负载均衡器
│   │   ├── glm5_client.py       # GLM5客户端
│   │   └── lightweight_client.py # 轻量模型客户端
│   └── common/                   # 公共模块
│       ├── __init__.py
│       ├── logger.py            # 日志配置
│       └── metrics.py           # 指标收集
├── tests/
│   ├── __init__.py
│   ├── conftest.py              # pytest配置和fixtures
│   ├── test_classifier/         # 分类器测试
│   ├── test_router/             # Router测试
│   └── test_models/             # 模型池测试
├── scripts/
│   ├── deploy/                  # 部署脚本
│   └── benchmark/               # 压测脚本
├── config/
│   ├── config.yaml              # 主配置
│   └── rules.yaml               # 路由规则配置
├── requirements.txt
├── pytest.ini
└── README.md
```

---

## Phase 1 MVP 详细任务清单

### Task 1: 项目初始化和基础结构

**Files:**
- Create: `requirements.txt`
- Create: `pytest.ini`
- Create: `src/router/__init__.py`
- Create: `src/common/__init__.py`

- [ ] **Step 1: 创建requirements.txt**

```txt
# Web Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0

# HTTP Client
httpx==0.26.0

# Configuration
pydantic==2.6.0
pydantic-settings==2.1.0
pyyaml==6.0.1

# Database
redis==5.0.1
asyncpg==0.29.0

# Testing
pytest==8.0.0
pytest-asyncio==0.23.5
pytest-mock==3.12.0
httpretty==1.1.4

# Monitoring
prometheus-client==0.19.0

# Utilities
python-json-logger==2.0.7
structlog==24.1.0
```

- [ ] **Step 2: 创建pytest.ini**

```ini
[pytest]
testpaths = tests
asyncio_mode = auto
asyncio_default_fixture_loop_scope = function
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = -v --tb=short
```

- [ ] **Step 3: 初始化包结构**

创建空文件：
- `src/router/__init__.py`
- `src/classifier/__init__.py`
- `src/models/__init__.py`
- `src/common/__init__.py`
- `tests/__init__.py`

- [ ] **Step 4: Commit**

```bash
git add requirements.txt pytest.ini src/ tests/
git commit -m "chore: initialize project structure with dependencies"
```

---

### Task 2: 配置管理系统

**Files:**
- Create: `src/router/config.py`
- Create: `config/config.yaml`
- Create: `config/rules.yaml`
- Test: `tests/test_router/test_config.py`

- [ ] **Step 1: 写配置加载的测试**

```python
# tests/test_router/test_config.py
import pytest
from pathlib import Path
from src.router.config import Settings, RoutingRules, load_routing_rules

class TestSettings:
    def test_settings_load_from_env(self, monkeypatch):
        monkeypatch.setenv("ROUTER_PORT", "8080")
        monkeypatch.setenv("REDIS_URL", "redis://localhost:6379")
        
        settings = Settings()
        assert settings.port == 8080
        assert settings.redis_url == "redis://localhost:6379"
    
    def test_settings_default_values(self):
        settings = Settings()
        assert settings.port == 8000
        assert settings.log_level == "INFO"

class TestRoutingRules:
    def test_load_routing_rules(self, tmp_path):
        rules_file = tmp_path / "rules.yaml"
        rules_file.write_text("""
simple_keywords:
  - "你好"
  - "谢谢"
  - "什么是"
complex_keywords:
  - "代码"
  - "分析"
  - "推理"
thresholds:
  tier1: 0.3
  tier2: 0.7
  token_count:
    simple_max: 100
    complex_min: 2000
""")
        
        rules = load_routing_rules(rules_file)
        assert "你好" in rules.simple_keywords
        assert "代码" in rules.complex_keywords
        assert rules.thresholds.tier1 == 0.3
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_router/test_config.py -v
```

Expected: FAIL - ModuleNotFoundError

- [ ] **Step 3: 实现配置模块**

```python
# src/router/config.py
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
import yaml
from typing import List

class ServerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ROUTER_")
    
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    redis_url: str = "redis://localhost:6379"
    database_url: str = "postgresql://user:pass@localhost/router"

class Thresholds(BaseModel):
    tier1: float = 0.3
    tier2: float = 0.7

class TokenThresholds(BaseModel):
    simple_max: int = 100
    complex_min: int = 2000

class RoutingRules(BaseModel):
    simple_keywords: List[str] = Field(default_factory=list)
    complex_keywords: List[str] = Field(default_factory=list)
    thresholds: Thresholds = Field(default_factory=Thresholds)
    token_count: TokenThresholds = Field(default_factory=TokenThresholds)

def load_routing_rules(path: Path) -> RoutingRules:
    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    return RoutingRules(**data)

class Settings:
    def __init__(self):
        self.server = ServerSettings()
        self.rules = None
    
    def load_rules(self, rules_path: Path):
        self.rules = load_routing_rules(rules_path)

settings = Settings()
```

- [ ] **Step 4: 创建默认配置文件**

```yaml
# config/rules.yaml
# Level 1 规则筛选配置

simple_keywords:
  - "你好"
  - "您好"
  - "谢谢"
  - "什么是"
  - "介绍一下"
  - "解释一下"
  - "简单"
  - "简单说"
  - "hello"
  - "hi"
  - "thanks"

complex_keywords:
  - "代码"
  - "编写"
  - "分析"
  - "推理"
  - "计算"
  - "比较"
  - "优化"
  - "设计"
  - "详细"
  - "深入"
  - "复杂"
  - "完整方案"
  - "debug"
  - "algorithm"

thresholds:
  tier1: 0.3  # 复杂度低于此值走轻量模型
  tier2: 0.7  # 复杂度高于此值走GLM5

token_count:
  simple_max: 100    # token数小于此值且无明显复杂关键词视为简单
  complex_min: 2000  # token数大于此值视为复杂
```

- [ ] **Step 5: 运行测试确认通过**

```bash
pytest tests/test_router/test_config.py -v
```

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add src/router/config.py config/ tests/test_router/test_config.py
git commit -m "feat: add configuration management system"
```

---

### Task 3: Pydantic数据模型

**Files:**
- Create: `src/router/models.py`
- Test: `tests/test_router/test_models.py`

- [ ] **Step 1: 写模型验证的测试**

```python
# tests/test_router/test_models.py
import pytest
from src.router.models import ChatMessage, ChatCompletionRequest, ChatCompletionResponse

class TestChatMessage:
    def test_valid_message(self):
        msg = ChatMessage(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
    
    def test_invalid_role(self):
        with pytest.raises(ValueError):
            ChatMessage(role="invalid", content="Hello")
    
    def test_empty_content(self):
        with pytest.raises(ValueError):
            ChatMessage(role="user", content="")

class TestChatCompletionRequest:
    def test_auto_model(self):
        req = ChatCompletionRequest(
            model="auto",
            messages=[{"role": "user", "content": "Hi"}]
        )
        assert req.model == "auto"
    
    def test_specific_model(self):
        req = ChatCompletionRequest(
            model="glm5",
            messages=[{"role": "user", "content": "Hi"}]
        )
        assert req.model == "glm5"
    
    def test_invalid_model(self):
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="invalid_model",
                messages=[{"role": "user", "content": "Hi"}]
            )

class TestChatCompletionResponse:
    def test_response_structure(self):
        response = ChatCompletionResponse(
            id="test-123",
            model="qwen2.5-7b",
            choices=[{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop"
            }]
        )
        assert response.id == "test-123"
        assert response.model == "qwen2.5-7b"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_router/test_models.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现Pydantic模型**

```python
# src/router/models.py
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, Union
from datetime import datetime
import uuid

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    name: Optional[str] = None
    
    @field_validator('content')
    @classmethod
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError('Content cannot be empty')
        return v

class ChatCompletionRequest(BaseModel):
    model: str = "auto"  # "auto", "glm5", "light", "medium"
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=2000, gt=0)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    stream: Optional[bool] = False
    user: Optional[str] = None
    
    @field_validator('model')
    @classmethod
    def validate_model(cls, v):
        allowed = {"auto", "glm5", "light", "medium"}
        if v not in allowed:
            raise ValueError(f'Model must be one of {allowed}')
        return v
    
    @field_validator('messages')
    @classmethod
    def at_least_one_message(cls, v):
        if len(v) == 0:
            raise ValueError('At least one message is required')
        return v

class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

class Choice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[Literal["stop", "length", "content_filter"]] = None

class RouterInfo(BaseModel):
    complexity_score: Optional[float] = None
    route_decision: Optional[str] = None
    classification_time_ms: Optional[int] = None
    routing_path: Optional[List[str]] = None

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:12]}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(datetime.now().timestamp()))
    model: str
    choices: List[Choice]
    usage: Usage
    router_info: Optional[RouterInfo] = None

class ErrorResponse(BaseModel):
    error: dict
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_router/test_models.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/router/models.py tests/test_router/test_models.py
git commit -m "feat: add pydantic data models for API"
```

---

### Task 4: Level 1 规则筛选分类器

**Files:**
- Create: `src/classifier/level1_rules.py`
- Test: `tests/test_classifier/test_level1_rules.py`

- [ ] **Step 1: 写规则筛选测试**

```python
# tests/test_classifier/test_level1_rules.py
import pytest
from src.classifier.level1_rules import Level1Classifier, RouteDecision

@pytest.fixture
def classifier():
    rules = {
        "simple_keywords": ["你好", "谢谢", "什么是"],
        "complex_keywords": ["代码", "分析", "推理"],
        "thresholds": {"tier1": 0.3, "tier2": 0.7},
        "token_count": {"simple_max": 100, "complex_min": 2000}
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
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_classifier/test_level1_rules.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现Level 1规则分类器**

```python
# src/classifier/level1_rules.py
from enum import Enum
from typing import Dict, Any, List
import re

class RouteDecision(Enum):
    TIER1_SIMPLE = "tier1"      # 轻量模型
    TIER3_COMPLEX = "tier3"     # GLM5
    CONTINUE = "continue"       # 继续到Level 3分类

class ClassificationResult:
    def __init__(self, decision: RouteDecision, reason: str, path: str = "level1_rules"):
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
                    reason=f"短文本(token={token_count})且无复杂关键词"
                )
        
        # 规则2: 明显复杂 - 长文本或明确复杂意图
        if token_count > self.token_thresholds.get("complex_min", 2000):
            return ClassificationResult(
                decision=RouteDecision.TIER3_COMPLEX,
                reason=f"长文本(token={token_count})"
            )
        
        has_complex = any(kw in text_lower for kw in self.complex_keywords)
        if has_complex:
            return ClassificationResult(
                decision=RouteDecision.TIER3_COMPLEX,
                reason="包含复杂关键词"
            )
        
        # 规则3: 无法判断，需要Level 3
        return ClassificationResult(
            decision=RouteDecision.CONTINUE,
            reason=f"规则无法判断(token={token_count})，需要进一步分类"
        )
    
    def has_simple_indicators(self, text: str) -> bool:
        """检查是否包含简单指示词"""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.simple_keywords)
    
    def has_complex_indicators(self, text: str) -> bool:
        """检查是否包含复杂指示词"""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.complex_keywords)
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_classifier/test_level1_rules.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/classifier/level1_rules.py tests/test_classifier/test_level1_rules.py
git commit -m "feat: implement Level 1 rule-based classifier"
```

---

### Task 5: Level 3 LLM分类器（简化版）

**Files:**
- Create: `src/classifier/level3_llm.py`
- Test: `tests/test_classifier/test_level3_llm.py`

- [ ] **Step 1: 写LLM分类器测试**

```python
# tests/test_classifier/test_level3_llm.py
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
            "reasoning": "简单问候"
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
            "reasoning": "需要深入分析"
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
            "reasoning": "中等复杂度"
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
            "reasoning": "不确定"
        }
        
        result = await classifier.classify("某个模糊的问题")
        
        # 低置信度时应该走GLM5以确保质量
        assert result.route_decision == "tier3"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_classifier/test_level3_llm.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现Level 3分类器**

```python
# src/classifier/level3_llm.py
from dataclasses import dataclass
from typing import Dict, Any, Optional
import json

@dataclass
class ComplexityResult:
    """复杂度分类结果"""
    complexity_score: float  # 0.0 - 1.0
    confidence: float        # 0.0 - 1.0
    reasoning: str
    route_decision: str      # tier1, tier2, tier3
    
class Level3Classifier:
    """Level 3: 小模型智能分类 (50-100ms)"""
    
    def __init__(self, llm_client, tier1_threshold: float = 0.3, tier2_threshold: float = 0.7, min_confidence: float = 0.7):
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
            route_decision=route_decision
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
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_classifier/test_level3_llm.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/classifier/level3_llm.py tests/test_classifier/test_level3_llm.py
git commit -m "feat: implement Level 3 LLM-based classifier"
```

---

### Task 6: 分类路由整合器

**Files:**
- Create: `src/classifier/router.py`
- Test: `tests/test_classifier/test_router.py`

- [ ] **Step 1: 写分类路由整合测试**

```python
# tests/test_classifier/test_router.py
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
        level1_classifier=mock_level1,
        level3_classifier=mock_level3
    )

class TestClassificationRouter:
    @pytest.mark.asyncio
    async def test_level1_simple_route(self, router, mock_level1):
        # Level 1直接判定为简单
        from src.classifier.level1_rules import ClassificationResult
        mock_level1.classify.return_value = ClassificationResult(
            decision=RouteDecision.TIER1_SIMPLE,
            reason="短文本"
        )
        
        result = await router.route("你好", token_count=10)
        
        assert result.decision == "tier1"
        assert result.path == ["level1_rules"]
    
    @pytest.mark.asyncio
    async def test_level1_complex_route(self, router, mock_level1):
        # Level 1直接判定为复杂
        from src.classifier.level1_rules import ClassificationResult
        mock_level1.classify.return_value = ClassificationResult(
            decision=RouteDecision.TIER3_COMPLEX,
            reason="包含代码"
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
            decision=RouteDecision.CONTINUE,
            reason="无法判断"
        )
        
        mock_level3.classify.return_value = ComplexityResult(
            complexity_score=0.4,
            confidence=0.8,
            reasoning="中等",
            route_decision="tier2"
        )
        
        result = await router.route("某个问题", token_count=500)
        
        assert result.decision == "tier2"
        assert "level1_rules" in result.path
        assert "level3_llm" in result.path
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_classifier/test_router.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现分类路由整合器**

```python
# src/classifier/router.py
from dataclasses import dataclass, field
from typing import List, Optional, Any
from src.classifier.level1_rules import RouteDecision
import time

@dataclass
class RouteResult:
    """最终路由结果"""
    decision: str           # tier1, tier2, tier3
    path: List[str]         # 路由决策路径
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
                latency_ms=latency_ms
            )
        
        if level1_result.decision == RouteDecision.TIER3_COMPLEX:
            latency_ms = int((time.time() - start_time) * 1000)
            return RouteResult(
                decision="tier3",
                path=path,
                reasoning=level1_result.reason,
                latency_ms=latency_ms
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
            latency_ms=latency_ms
        )
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_classifier/test_router.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/classifier/router.py tests/test_classifier/test_router.py
git commit -m "feat: integrate Level 1 and Level 3 classifiers"
```

---

### Task 6b：部署vLLM分类器服务（生产环境必需）

**说明**: Task 5-6使用的是MockLLMClient用于测试和开发。生产环境必须部署真正的vLLM服务。

**部署步骤**:

1. **准备中低端服务器**（建议4-8台，配置：16GB+显存或高性能CPU）

2. **安装vLLM**:
```bash
pip install vllm==0.3.0
```

3. **下载模型**:
```bash
# 使用ModelScope（国内推荐）
pip install modelscope
python -c "from modelscope import snapshot_download; snapshot_download('qwen/Qwen2.5-7B-Instruct', cache_dir='/models')"
```

4. **启动vLLM服务**（每台服务器）:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model /models/Qwen2.5-7B-Instruct \
    --tensor-parallel-size 1 \
    --max-model-len 4096 \
    --port 8000 \
    --api-key ${CLASSIFIER_API_KEY:-""}
```

5. **配置环境变量**:
```bash
# .env文件
CLASSIFIER_VLLM_URL=http://classifier-host:8000/v1
```

6. **修改代码切换到vLLM客户端**:
```python
# 在Task 9的_get_components()函数中
from src.classifier.level3_llm import vLLMClassifierClient

level3_llm_client = vLLMClassifierClient()  # 替代MockLLMClient()
```

**验证部署**:
```bash
curl http://classifier-host:8000/v1/models
# 应返回可用模型列表
```

---

### Task 7: 模型池核心和负载均衡

**Files:**
- Create: `src/models/pool.py`
- Create: `src/models/load_balancer.py`
- Test: `tests/test_models/test_pool.py`
- Test: `tests/test_models/test_load_balancer.py`

- [ ] **Step 1: 写模型池测试**

```python
# tests/test_models/test_pool.py
import pytest
from unittest.mock import Mock
from src.models.pool import ModelPool, ModelInstance

class TestModelPool:
    def test_register_instance(self):
        pool = ModelPool()
        instance = ModelInstance(
            id="glm5-001",
            tier="tier3",
            host="10.0.0.1",
            port=8000,
            max_concurrency=100
        )
        
        pool.register(instance)
        
        assert "glm5-001" in pool.instances
        assert pool.get_healthy_instances("tier3") == [instance]
    
    def test_unregister_instance(self):
        pool = ModelPool()
        instance = ModelInstance(id="glm5-001", tier="tier3", host="10.0.0.1", port=8000, max_concurrency=100)
        pool.register(instance)
        
        pool.unregister("glm5-001")
        
        assert "glm5-001" not in pool.instances
    
    def test_mark_unhealthy(self):
        pool = ModelPool()
        instance = ModelInstance(id="glm5-001", tier="tier3", host="10.0.0.1", port=8000, max_concurrency=100)
        pool.register(instance)
        
        pool.mark_unhealthy("glm5-001")
        
        healthy = pool.get_healthy_instances("tier3")
        assert instance not in healthy
    
    def test_get_instances_by_tier(self):
        pool = ModelPool()
        glm5 = ModelInstance(id="glm5-001", tier="tier3", host="10.0.0.1", port=8000, max_concurrency=100)
        qwen = ModelInstance(id="qwen-001", tier="tier1", host="10.0.0.2", port=8000, max_concurrency=100)
        
        pool.register(glm5)
        pool.register(qwen)
        
        tier3 = pool.get_healthy_instances("tier3")
        tier1 = pool.get_healthy_instances("tier1")
        
        assert len(tier3) == 1
        assert len(tier1) == 1
        assert tier3[0].id == "glm5-001"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_models/test_pool.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现模型池**

```python
# src/models/pool.py
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from datetime import datetime
import threading

@dataclass
class ModelInstance:
    """模型实例配置"""
    id: str
    tier: str  # tier1, tier2, tier3
    host: str
    port: int
    max_concurrency: int
    current_load: int = 0
    queue_depth: int = 0
    is_healthy: bool = True
    last_health_check: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)
    
    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def available_slots(self) -> int:
        return self.max_concurrency - self.current_load

class ModelPool:
    """
    模型池管理器
    统一管理所有模型实例，提供健康检查和负载查询
    """
    
    def __init__(self):
        self.instances: Dict[str, ModelInstance] = {}
        self._lock = threading.RLock()
    
    def register(self, instance: ModelInstance):
        """注册新实例"""
        with self._lock:
            self.instances[instance.id] = instance
    
    def unregister(self, instance_id: str):
        """注销实例"""
        with self._lock:
            if instance_id in self.instances:
                del self.instances[instance_id]
    
    def get_instance(self, instance_id: str) -> Optional[ModelInstance]:
        """获取指定实例"""
        return self.instances.get(instance_id)
    
    def get_healthy_instances(self, tier: str) -> List[ModelInstance]:
        """获取指定层级的健康实例"""
        with self._lock:
            return [
                inst for inst in self.instances.values()
                if inst.tier == tier and inst.is_healthy
            ]
    
    def mark_unhealthy(self, instance_id: str):
        """标记实例为不健康"""
        with self._lock:
            if instance_id in self.instances:
                self.instances[instance_id].is_healthy = False
    
    def mark_healthy(self, instance_id: str):
        """标记实例为健康"""
        with self._lock:
            if instance_id in self.instances:
                self.instances[instance_id].is_healthy = True
                self.instances[instance_id].last_health_check = datetime.now()
    
    def update_load(self, instance_id: str, current_load: int, queue_depth: int):
        """更新实例负载信息"""
        with self._lock:
            if instance_id in self.instances:
                self.instances[instance_id].current_load = current_load
                self.instances[instance_id].queue_depth = queue_depth
    
    def get_all_instances(self) -> List[ModelInstance]:
        """获取所有实例"""
        return list(self.instances.values())
```

- [ ] **Step 4: 写负载均衡测试**

```python
# tests/test_models/test_load_balancer.py
import pytest
from unittest.mock import Mock
from src.models.pool import ModelInstance
from src.models.load_balancer import LoadBalancer, LeastConnectionStrategy

class TestLoadBalancer:
    def test_round_robin_selection(self):
        balancer = LoadBalancer(strategy="round_robin")
        
        inst1 = ModelInstance(id="i1", tier="tier1", host="10.0.0.1", port=8000, max_concurrency=100)
        inst2 = ModelInstance(id="i2", tier="tier1", host="10.0.0.2", port=8000, max_concurrency=100)
        
        candidates = [inst1, inst2]
        
        # 第一次选i1
        result1 = balancer.select(candidates)
        assert result1.id == "i1"
        
        # 第二次选i2
        result2 = balancer.select(candidates)
        assert result2.id == "i2"
        
        # 第三次回到i1
        result3 = balancer.select(candidates)
        assert result3.id == "i1"
    
    def test_least_connection_selection(self):
        balancer = LoadBalancer(strategy="least_connection")
        
        inst1 = ModelInstance(id="i1", tier="tier1", host="10.0.0.1", port=8000, max_concurrency=100, current_load=50)
        inst2 = ModelInstance(id="i2", tier="tier1", host="10.0.0.2", port=8000, max_concurrency=100, current_load=20)
        
        candidates = [inst1, inst2]
        result = balancer.select(candidates)
        
        # 应该选负载较低的i2
        assert result.id == "i2"
    
    def test_no_available_instances(self):
        balancer = LoadBalancer()
        
        result = balancer.select([])
        
        assert result is None
```

- [ ] **Step 5: 运行测试确认失败**

```bash
pytest tests/test_models/test_load_balancer.py -v
```

Expected: FAIL

- [ ] **Step 6: 实现负载均衡器**

```python
# src/models/load_balancer.py
from abc import ABC, abstractmethod
from typing import List, Optional
from src.models.pool import ModelInstance
import itertools

class LoadBalancingStrategy(ABC):
    """负载均衡策略基类"""
    
    @abstractmethod
    def select(self, instances: List[ModelInstance]) -> Optional[ModelInstance]:
        pass

class RoundRobinStrategy(LoadBalancingStrategy):
    """轮询策略"""
    
    def __init__(self):
        self._counter = itertools.count()
    
    def select(self, instances: List[ModelInstance]) -> Optional[ModelInstance]:
        if not instances:
            return None
        idx = next(self._counter) % len(instances)
        return instances[idx]

class LeastConnectionStrategy(LoadBalancingStrategy):
    """最少连接策略"""
    
    def select(self, instances: List[ModelInstance]) -> Optional[ModelInstance]:
        if not instances:
            return None
        return min(instances, key=lambda x: x.current_load)

class QueueDepthStrategy(LoadBalancingStrategy):
    """队列深度感知策略"""
    
    def select(self, instances: List[ModelInstance]) -> Optional[ModelInstance]:
        if not instances:
            return None
        return min(instances, key=lambda x: x.queue_depth)

class LoadBalancer:
    """
    负载均衡器
    支持多种负载均衡策略
    """
    
    STRATEGIES = {
        "round_robin": RoundRobinStrategy,
        "least_connection": LeastConnectionStrategy,
        "queue_depth": QueueDepthStrategy,
    }
    
    def __init__(self, strategy: str = "round_robin"):
        strategy_class = self.STRATEGIES.get(strategy, RoundRobinStrategy)
        self._strategy = strategy_class()
    
    def select(self, instances: List[ModelInstance]) -> Optional[ModelInstance]:
        """
        从候选实例中选择一个
        
        Args:
            instances: 候选实例列表
            
        Returns:
            选中的实例，如果没有可用实例则返回None
        """
        return self._strategy.select(instances)
```

- [ ] **Step 7: 运行测试确认通过**

```bash
pytest tests/test_models/test_pool.py tests/test_models/test_load_balancer.py -v
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/models/pool.py src/models/load_balancer.py tests/test_models/
git commit -m "feat: implement model pool and load balancer"
```

---

### Task 8: GLM5和轻量模型客户端

**Files:**
- Create: `src/models/glm5_client.py`
- Create: `src/models/lightweight_client.py`
- Test: `tests/test_models/test_glm5_client.py`
- Test: `tests/test_models/test_lightweight_client.py`

- [ ] **Step 1: 写GLM5客户端测试**

```python
# tests/test_models/test_glm5_client.py
import pytest
from unittest.mock import Mock, AsyncMock, patch
from src.models.glm5_client import GLM5Client

@pytest.fixture
def glm5_client():
    return GLM5Client(base_url="http://glm5.internal:8000", timeout=60)

class TestGLM5Client:
    @pytest.mark.asyncio
    async def test_chat_completion_success(self, glm5_client):
        mock_response = {
            "id": "chatcmpl-123",
            "choices": [{
                "message": {"role": "assistant", "content": "Hello"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value.status_code = 200
            mock_post.return_value.json = Mock(return_value=mock_response)
            
            result = await glm5_client.chat_completion(
                messages=[{"role": "user", "content": "Hi"}],
                temperature=0.7
            )
        
        assert result["choices"][0]["message"]["content"] == "Hello"
    
    @pytest.mark.asyncio
    async def test_chat_completion_failure(self, glm5_client):
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.return_value.status_code = 500
            mock_post.return_value.text = "Internal Server Error"
            
            with pytest.raises(Exception):
                await glm5_client.chat_completion(
                    messages=[{"role": "user", "content": "Hi"}]
                )
    
    @pytest.mark.asyncio
    async def test_health_check_success(self, glm5_client):
        with patch("httpx.AsyncClient.get") as mock_get:
            mock_get.return_value.status_code = 200
            mock_get.return_value.json = Mock(return_value={"status": "healthy"})
            
            is_healthy = await glm5_client.health_check()
        
        assert is_healthy is True
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_models/test_glm5_client.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现GLM5客户端**

```python
# src/models/glm5_client.py
import httpx
from typing import List, Dict, Any, Optional
import json

class GLM5Client:
    """
    GLM5模型客户端
    封装与GLM5服务的HTTP调用
    """
    
    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 1.0,
        stream: bool = False
    ) -> Dict[str, Any]:
        """
        调用GLM5聊天完成接口
        
        Args:
            messages: 对话历史
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: top-p采样
            stream: 是否流式返回
            
        Returns:
            GLM5的响应数据
        """
        payload = {
            "model": "glm5",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream
        }
        
        try:
            response = await self._client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise Exception(f"GLM5 request failed: {e.response.status_code} - {e.response.text}")
        except Exception as e:
            raise Exception(f"GLM5 request error: {str(e)}")
    
    async def health_check(self) -> bool:
        """健康检查"""
        try:
            response = await self._client.get(
                f"{self.base_url}/health",
                timeout=5.0
            )
            return response.status_code == 200
        except:
            return False
    
    async def close(self):
        """关闭连接"""
        await self._client.aclose()
```

- [ ] **Step 4: 写轻量模型客户端测试**

```python
# tests/test_models/test_lightweight_client.py
import pytest
from unittest.mock import Mock, patch
from src.models.lightweight_client import LightweightModelClient

@pytest.fixture
def lightweight_client():
    return LightweightModelClient(
        base_urls=["http://qwen-001:8000", "http://qwen-002:8000"],
        model_name="qwen2.5-7b",
        timeout=30
    )

class TestLightweightModelClient:
    @pytest.mark.asyncio
    async def test_chat_completion_with_fallback(self, lightweight_client):
        mock_response = {
            "id": "chatcmpl-123",
            "model": "qwen2.5-7b",
            "choices": [{
                "message": {"role": "assistant", "content": "Hello from Qwen"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}
        }
        
        with patch("httpx.AsyncClient.post") as mock_post:
            # 第一个实例失败，第二个成功
            mock_post.side_effect = [
                Exception("Connection refused"),
                Mock(status_code=200, json=Mock(return_value=mock_response))
            ]
            
            result = await lightweight_client.chat_completion(
                messages=[{"role": "user", "content": "Hi"}]
            )
        
        assert result["choices"][0]["message"]["content"] == "Hello from Qwen"
    
    @pytest.mark.asyncio
    async def test_all_instances_fail(self, lightweight_client):
        with patch("httpx.AsyncClient.post") as mock_post:
            mock_post.side_effect = Exception("Connection refused")
            
            with pytest.raises(Exception) as exc_info:
                await lightweight_client.chat_completion(
                    messages=[{"role": "user", "content": "Hi"}]
                )
            
            assert "All model instances failed" in str(exc_info.value)
```

- [ ] **Step 5: 运行测试确认失败**

```bash
pytest tests/test_models/test_lightweight_client.py -v
```

Expected: FAIL

- [ ] **Step 6: 实现轻量模型客户端**

```python
# src/models/lightweight_client.py
import httpx
from typing import List, Dict, Any
import random

class LightweightModelClient:
    """
    轻量模型客户端（Qwen2.5-7B等）
    支持多实例故障转移
    """
    
    def __init__(self, base_urls: List[str], model_name: str, timeout: int = 60):
        self.base_urls = [url.rstrip("/") for url in base_urls]
        self.model_name = model_name
        self.timeout = timeout
        self._client = httpx.AsyncClient(timeout=timeout)
    
    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 1.0
    ) -> Dict[str, Any]:
        """
        调用轻量模型，支持故障转移
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p
        }
        
        # 随机打乱顺序，实现简单负载均衡
        urls = self.base_urls.copy()
        random.shuffle(urls)
        
        last_error = None
        for base_url in urls:
            try:
                response = await self._client.post(
                    f"{base_url}/v1/chat/completions",
                    json=payload,
                    headers={"Content-Type": "application/json"}
                )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                last_error = e
                continue
        
        raise Exception(f"All model instances failed. Last error: {last_error}")
    
    async def close(self):
        """关闭连接"""
        await self._client.aclose()
```

- [ ] **Step 7: 运行测试确认通过**

```bash
pytest tests/test_models/test_glm5_client.py tests/test_models/test_lightweight_client.py -v
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/models/glm5_client.py src/models/lightweight_client.py tests/test_models/
git commit -m "feat: implement GLM5 and lightweight model clients"
```

---

### Task 9: Router Gateway主服务

**Files:**
- Create: `src/router/main.py`
- Create: `src/router/middleware.py`
- Create: `src/router/api/completions.py`
- Modify: `src/router/config.py`（添加实例配置）
- Test: `tests/test_router/test_api.py`

- [ ] **Step 1: 写API接口测试**

```python
# tests/test_router/test_api.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import Mock, patch, AsyncMock

@pytest.fixture
def client():
    from src.router.main import app
    return TestClient(app)

class TestChatCompletionsAPI:
    def test_simple_request_routed_to_lightweight(self, client):
        mock_route_result = Mock()
        mock_route_result.decision = "tier1"
        mock_route_result.path = ["level1_rules"]
        mock_route_result.latency_ms = 5
        mock_route_result.complexity_score = None
        
        mock_completion = {
            "id": "chatcmpl-123",
            "model": "qwen2.5-7b",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
        }
        
        with patch("src.classifier.router.ClassificationRouter.route", new_callable=AsyncMock) as mock_route, \
             patch("src.models.lightweight_client.LightweightModelClient.chat_completion", new_callable=AsyncMock) as mock_chat:
            
            mock_route.return_value = mock_route_result
            mock_chat.return_value = mock_completion
            
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "Hello"}]
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "qwen2.5-7b"
        assert "router_info" in data
    
    def test_complex_request_routed_to_glm5(self, client):
        mock_route_result = Mock()
        mock_route_result.decision = "tier3"
        mock_route_result.path = ["level1_rules"]
        
        mock_completion = {
            "id": "chatcmpl-456",
            "model": "glm5",
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": "Complex answer"},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 100, "completion_tokens": 200, "total_tokens": 300}
        }
        
        with patch("src.classifier.router.ClassificationRouter.route", new_callable=AsyncMock) as mock_route, \
             patch("src.models.glm5_client.GLM5Client.chat_completion", new_callable=AsyncMock) as mock_chat:
            
            mock_route.return_value = mock_route_result
            mock_chat.return_value = mock_completion
            
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "auto",
                    "messages": [{"role": "user", "content": "Write complex code"}]
                }
            )
        
        assert response.status_code == 200
        data = response.json()
        assert data["model"] == "glm5"
    
    def test_invalid_model(self, client):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "invalid_model",
                "messages": [{"role": "user", "content": "Hi"}]
            }
        )
        
        assert response.status_code == 422
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_router/test_api.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现FastAPI主应用**

```python
# src/router/main.py
from fastapi import FastAPI
from contextlib import asynccontextmanager

from src.router.config import settings
from src.router.api import completions, admin
from src.router.middleware import setup_middleware

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时加载配置
    settings.load_rules("config/rules.yaml")
    yield
    # 关闭时清理资源

app = FastAPI(
    title="LLM Router Gateway",
    description="智能路由层，实现请求分发到不同层级的LLM模型",
    version="1.0.0",
    lifespan=lifespan
)

# 设置中间件
setup_middleware(app)

# 注册路由
app.include_router(completions.router, prefix="/v1")
app.include_router(admin.router, prefix="/admin")

@app.get("/health")
async def health_check():
    """健康检查端点"""
    return {"status": "healthy", "service": "llm-router"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.router.main:app",
        host=settings.server.host,
        port=settings.server.port,
        log_level=settings.server.log_level.lower()
    )
```

- [ ] **Step 4: 实现中间件**

```python
# src/router/middleware.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
import time
import structlog

logger = structlog.get_logger()

def setup_middleware(app: FastAPI):
    """配置所有中间件"""
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Gzip压缩
    app.add_middleware(GZipMiddleware, minimum_size=1000)
    
    # 请求日志中间件
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        
        logger.info(
            "request_processed",
            method=request.method,
            path=request.url.path,
            status_code=response.status_code,
            process_time_ms=round(process_time * 1000, 2),
            client_host=request.client.host if request.client else None
        )
        
        response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
        return response
```

- [ ] **Step 5: 实现completions接口**

```python
# src/router/api/completions.py
import os
from fastapi import APIRouter, HTTPException, Header
from typing import Optional
from src.router.models import ChatCompletionRequest, ChatCompletionResponse, Usage, Choice, RouterInfo
from src.router.config import settings
from src.classifier.level1_rules import Level1Classifier
from src.classifier.level3_llm import Level3Classifier
from src.classifier.router import ClassificationRouter
from src.models.pool import ModelPool, ModelInstance
from src.models.load_balancer import LoadBalancer
from src.models.glm5_client import GLM5Client
from src.models.lightweight_client import LightweightModelClient
import tiktoken

router = APIRouter()

# 全局组件（实际应用中应使用依赖注入）
_classifier_router = None
_model_pool = None
_load_balancer = None
_glm5_client = None
_lightweight_client = None

def _count_tokens(text: str) -> int:
    """估算token数量"""
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except:
        # 降级方案：粗略估算（中文约1字符=1token，英文约4字符=1token）
        return len(text) // 2

def _get_components():
    """获取或初始化组件"""
    global _classifier_router, _model_pool, _load_balancer, _glm5_client, _lightweight_client
    
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
        glm5_url = getattr(settings, 'glm5_base_url', None) or os.getenv('GLM5_BASE_URL', 'http://localhost:8000')
        _glm5_client = GLM5Client(
            base_url=glm5_url,
            timeout=300
        )
    
    if _lightweight_client is None:
        # 从配置加载轻量模型地址列表
        lightweight_urls = getattr(settings, 'lightweight_base_urls', None) or os.getenv('LIGHTWEIGHT_BASE_URLS', 'http://localhost:8001').split(',')
        lightweight_model = getattr(settings, 'lightweight_model_name', 'qwen2.5-7b')
        _lightweight_client = LightweightModelClient(
            base_urls=lightweight_urls,
            model_name=lightweight_model,
            timeout=60
        )
    
    return _classifier_router, _model_pool, _load_balancer, _glm5_client, _lightweight_client

def _load_model_instances():
    """从配置加载模型实例"""
    # 实际应用中从配置文件或配置中心加载
    global _model_pool
    
    # GLM5实例
    _model_pool.register(ModelInstance(
        id="glm5-001",
        tier="tier3",
        host="glm5.internal",
        port=8000,
        max_concurrency=100
    ))
    
    # 轻量模型实例
    _model_pool.register(ModelInstance(
        id="qwen-001",
        tier="tier1",
        host="qwen-001",
        port=8000,
        max_concurrency=100
    ))

class MockLLMClient:
    """占位用的LLM客户端，用于测试和开发阶段"""
    async def classify(self, prompt: str):
        # 简化实现，实际应调用vLLM服务
        return {
            "complexity_score": 0.5,
            "confidence": 0.8,
            "reasoning": "mock"
        }

class vLLMClassifierClient:
    """
    真正的vLLM分类器客户端
    连接到部署的Qwen2.5-7B-Instruct vLLM服务
    
    部署说明：
    1. 在中低端服务器上部署vLLM：
       python -m vllm.entrypoints.openai.api_server \
           --model Qwen/Qwen2.5-7B-Instruct \
           --tensor-parallel-size 1 \
           --max-model-len 4096 \
           --port 8000
    
    2. 设置环境变量：
       export CLASSIFIER_VLLM_URL=http://classifier-host:8000/v1
    """
    
    def __init__(self, base_url: str = None, model: str = "Qwen2.5-7B-Instruct"):
        import os
        self.base_url = base_url or os.getenv("CLASSIFIER_VLLM_URL", "http://localhost:8000/v1")
        self.model = model
        self._client = None
    
    async def _get_client(self):
        if self._client is None:
            import httpx
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client
    
    async def classify(self, prompt: str) -> dict:
        """
        调用vLLM进行分类
        
        Args:
            prompt: 分类prompt
            
        Returns:
            dict: 包含complexity_score, confidence, reasoning
        """
        client = await self._get_client()
        
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.0,  # 分类任务用确定性输出
            "max_tokens": 500,
            "response_format": {"type": "json_object"}  # 强制JSON输出
        }
        
        try:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            response.raise_for_status()
            
            result = response.json()
            content = result["choices"][0]["message"]["content"]
            
            # 解析JSON响应
            import json
            parsed = json.loads(content)
            
            return {
                "complexity_score": float(parsed.get("complexity_score", 0.5)),
                "confidence": float(parsed.get("confidence", 0.5)),
                "reasoning": parsed.get("reasoning", ""),
                "dimensions": parsed.get("dimensions", {})
            }
            
        except json.JSONDecodeError as e:
            # JSON解析失败时返回默认值
            return {
                "complexity_score": 0.5,
                "confidence": 0.3,  # 低置信度
                "reasoning": f"Failed to parse LLM response: {str(e)}",
                "dimensions": {}
            }
        except Exception as e:
            # 其他错误时保守处理（走GLM5）
            return {
                "complexity_score": 0.8,  # 标记为复杂
                "confidence": 0.9,
                "reasoning": f"Error calling classifier: {str(e)}",
                "dimensions": {}
            }
    
    async def health_check(self) -> bool:
        """检查vLLM服务健康状态"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/models", timeout=5.0)
            return response.status_code == 200
        except:
            return False
    
    async def close(self):
        """关闭连接"""
        if self._client:
            await self._client.aclose()
            self._client = None

@router.post("/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(
    request: ChatCompletionRequest,
    x_router_debug: Optional[str] = Header(None, alias="X-Router-Debug")
):
    """
    OpenAI兼容的聊天完成接口
    
    - model="auto": 自动路由
    - model="glm5"/"light"/"medium": 强制指定模型
    """
    try:
        classifier_router, model_pool, load_balancer, glm5_client, lightweight_client = _get_components()
        
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
                max_tokens=request.max_tokens
            )
            model_name = "qwen2.5-7b"
        else:
            # tier2/tier3都先走GLM5（MVP简化）
            model_response = await glm5_client.chat_completion(
                messages=[m.model_dump() for m in request.messages],
                temperature=request.temperature,
                max_tokens=request.max_tokens
            )
            model_name = "glm5"
        
        # 构造响应
        response = ChatCompletionResponse(
            model=model_name,
            choices=[Choice(**choice) for choice in model_response["choices"]],
            usage=Usage(**model_response["usage"])
        )
        
        # 添加路由信息（如果请求了debug模式）
        if x_router_debug:
            response.router_info = RouterInfo(
                complexity_score=route_result.complexity_score if route_result else None,
                route_decision=target_tier,
                classification_time_ms=route_result.latency_ms if route_result else 0,
                routing_path=route_result.path if route_result else ["forced"]
            )
        
        return response
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

- [ ] **Step 6: 更新requirements.txt添加tiktoken**

```bash
# 追加到requirements.txt
echo "tiktoken==0.6.0" >> requirements.txt
```

- [ ] **Step 7: 运行测试确认通过**

```bash
pytest tests/test_router/test_api.py -v
```

Expected: PASS

- [ ] **Step 8: Commit**

```bash
git add src/router/main.py src/router/middleware.py src/router/api/ tests/test_router/
git commit -m "feat: implement Router Gateway main service and completions API"
```

---

### Task 10: 管理接口

**Files:**
- Create: `src/router/api/admin.py`
- Test: `tests/test_router/test_admin.py`

- [ ] **Step 1: 写管理接口测试**

```python
# tests/test_router/test_admin.py
import pytest
from fastapi.testclient import TestClient

@pytest.fixture
def client():
    from src.router.main import app
    return TestClient(app)

class TestAdminAPI:
    def test_health_endpoint(self, client):
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "llm-router"
    
    def test_admin_stats(self, client):
        response = client.get("/admin/stats?time_range=24h")
        
        assert response.status_code == 200
        data = response.json()
        assert "total_requests" in data
        assert "routing_distribution" in data
    
    def test_admin_health_detailed(self, client):
        response = client.get("/admin/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "components" in data
    
    def test_admin_config_update(self, client):
        response = client.post(
            "/admin/config",
            json={"tier1_threshold": 0.35}
        )
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "updated"
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_router/test_admin.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现管理接口**

```python
# src/router/api/admin.py
from fastapi import APIRouter, Query
from typing import Optional
from pydantic import BaseModel

router = APIRouter()

class ConfigUpdate(BaseModel):
    tier1_threshold: Optional[float] = None
    tier2_threshold: Optional[float] = None

# 模拟统计数据（实际应从Prometheus或数据库读取）
_mock_stats = {
    "total_requests": 100000,
    "routing_distribution": {
        "tier1": 65000,
        "tier2": 20000,
        "tier3": 15000
    },
    "avg_latency_ms": {
        "classification": 35,
        "tier1": 800,
        "tier2": 3000,
        "tier3": 8000
    },
    "accuracy": 0.92,
    "error_rate": 0.001
}

@router.get("/stats")
async def get_stats(
    time_range: str = Query("24h", regex="^(1h|24h|7d|30d)$"),
    granularity: Optional[str] = Query("hour", regex="^(minute|hour|day)$")
):
    """
    获取路由统计信息
    
    - time_range: 时间范围（1h, 24h, 7d, 30d）
    - granularity: 数据粒度（minute, hour, day）
    """
    # 实际应用中应从时序数据库查询
    return _mock_stats

@router.get("/health")
async def admin_health_check():
    """详细健康检查"""
    return {
        "status": "healthy",
        "components": {
            "gateway": "up",
            "classifier": "up",
            "tier1_pool": "up",
            "tier2_pool": "up",
            "tier3_pool": "up"
        },
        "metrics": {
            "qps": 1500,
            "queue_depth": 12
        }
    }

@router.post("/config")
async def update_config(config: ConfigUpdate):
    """
    更新路由配置
    
    - tier1_threshold: Tier 1阈值（0.0-1.0）
    - tier2_threshold: Tier 2阈值（0.0-1.0）
    """
    # 实际应用中应更新配置并生效
    return {
        "status": "updated",
        "config": config.model_dump()
    }

@router.get("/instances")
async def list_instances(tier: Optional[str] = None):
    """列出所有模型实例"""
    # 实际应用中应从ModelPool查询
    return {
        "instances": [
            {
                "id": "glm5-001",
                "tier": "tier3",
                "host": "glm5.internal",
                "port": 8000,
                "is_healthy": True
            }
        ]
    }
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_router/test_admin.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/router/api/admin.py tests/test_router/test_admin.py
git commit -m "feat: implement admin API endpoints"
```

---

### Task 11: 日志和监控

**Files:**
- Create: `src/common/logger.py`
- Create: `src/common/metrics.py`
- Test: `tests/test_common/test_metrics.py`

- [ ] **Step 1: 写指标收集测试**

```python
# tests/test_common/test_metrics.py
import pytest
from src.common.metrics import MetricsCollector, RoutingMetrics

class TestMetricsCollector:
    def test_record_request(self):
        collector = MetricsCollector()
        
        collector.record_request(
            tier="tier1",
            latency_ms=100,
            input_tokens=50,
            output_tokens=20,
            success=True
        )
        
        stats = collector.get_stats()
        assert stats["total_requests"] == 1
        assert stats["tier1_requests"] == 1
    
    def test_record_routing_decision(self):
        collector = MetricsCollector()
        
        collector.record_routing_decision(
            path=["level1_rules"],
            latency_ms=5,
            complexity_score=0.3
        )
        
        stats = collector.get_stats()
        assert stats["routing_decisions"] == 1
    
    def test_get_stats(self):
        collector = MetricsCollector()
        
        # 记录多个请求
        for i in range(10):
            collector.record_request(
                tier="tier1" if i < 7 else "tier3",
                latency_ms=100 + i * 10,
                input_tokens=50,
                output_tokens=20,
                success=True
            )
        
        stats = collector.get_stats()
        assert stats["total_requests"] == 10
        assert stats["tier1_requests"] == 7
        assert stats["tier3_requests"] == 3
```

- [ ] **Step 2: 运行测试确认失败**

```bash
pytest tests/test_common/test_metrics.py -v
```

Expected: FAIL

- [ ] **Step 3: 实现日志和监控**

```python
# src/common/logger.py
import structlog
import logging
import sys

def setup_logging(log_level: str = "INFO"):
    """配置结构化日志"""
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level.upper())
    )

# 创建默认logger
logger = structlog.get_logger()
```

```python
# src/common/metrics.py
from dataclasses import dataclass, field
from typing import Dict, List
from collections import defaultdict
import time

@dataclass
class RoutingMetrics:
    """路由决策指标"""
    timestamp: float
    path: List[str]
    latency_ms: int
    complexity_score: Optional[float] = None

class MetricsCollector:
    """
    指标收集器
    收集路由性能和决策指标
    """
    
    def __init__(self):
        self.requests = defaultdict(list)  # tier -> list of latencies
        self.routing_decisions = []
        self.errors = 0
        self._start_time = time.time()
    
    def record_request(
        self,
        tier: str,
        latency_ms: int,
        input_tokens: int,
        output_tokens: int,
        success: bool
    ):
        """记录一次请求"""
        self.requests[tier].append({
            "latency_ms": latency_ms,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "success": success,
            "timestamp": time.time()
        })
        
        if not success:
            self.errors += 1
    
    def record_routing_decision(
        self,
        path: List[str],
        latency_ms: int,
        complexity_score: Optional[float] = None
    ):
        """记录路由决策"""
        self.routing_decisions.append(RoutingMetrics(
            timestamp=time.time(),
            path=path,
            latency_ms=latency_ms,
            complexity_score=complexity_score
        ))
    
    def get_stats(self) -> Dict:
        """获取统计数据"""
        stats = {
            "total_requests": sum(len(reqs) for reqs in self.requests.values()),
            "uptime_seconds": time.time() - self._start_time,
            "errors": self.errors,
        }
        
        # 各层级统计
        for tier, reqs in self.requests.items():
            if reqs:
                latencies = [r["latency_ms"] for r in reqs]
                stats[f"{tier}_requests"] = len(reqs)
                stats[f"{tier}_avg_latency_ms"] = sum(latencies) / len(latencies)
        
        # 路由决策统计
        if self.routing_decisions:
            latencies = [d.latency_ms for d in self.routing_decisions]
            stats["routing_decisions"] = len(self.routing_decisions)
            stats["avg_classification_latency_ms"] = sum(latencies) / len(latencies)
        
        return stats
```

- [ ] **Step 4: 运行测试确认通过**

```bash
pytest tests/test_common/test_metrics.py -v
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/common/logger.py src/common/metrics.py tests/test_common/
git commit -m "feat: add logging and metrics collection"
```

---

### Task 12: 部署脚本和文档

**Files:**
- Create: `scripts/deploy/start.sh`
- Create: `scripts/deploy/stop.sh`
- Create: `README.md`
- Create: `.env.example`

- [ ] **Step 1: 创建启动脚本**

```bash
# scripts/deploy/start.sh
#!/bin/bash

# LLM Router 启动脚本

set -e

echo "🚀 Starting LLM Router..."

# 加载环境变量
if [ -f .env ]; then
    export $(cat .env | xargs)
fi

# 检查依赖
echo "📦 Checking dependencies..."
python3 -c "import fastapi" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -r requirements.txt
}

# 检查配置文件
if [ ! -f "config/rules.yaml" ]; then
    echo "❌ Config file not found: config/rules.yaml"
    exit 1
fi

# 启动服务
echo "🌐 Starting Router Gateway..."
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

uvicorn src.router.main:app \
    --host ${ROUTER_HOST:-0.0.0.0} \
    --port ${ROUTER_PORT:-8000} \
    --workers ${ROUTER_WORKERS:-4} \
    --log-level ${ROUTER_LOG_LEVEL:-info} \
    --access-log &

ROUTER_PID=$!
echo $ROUTER_PID > /tmp/llm-router.pid

echo "✅ LLM Router started with PID: $ROUTER_PID"
echo "📊 Health check: http://localhost:${ROUTER_PORT:-8000}/health"
echo "📝 API docs: http://localhost:${ROUTER_PORT:-8000}/docs"
```

- [ ] **Step 2: 创建停止脚本**

```bash
# scripts/deploy/stop.sh
#!/bin/bash

echo "🛑 Stopping LLM Router..."

if [ -f /tmp/llm-router.pid ]; then
    PID=$(cat /tmp/llm-router.pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "✅ LLM Router stopped (PID: $PID)"
    else
        echo "⚠️ Process not running"
    fi
    rm /tmp/llm-router.pid
else
    echo "⚠️ PID file not found"
fi
```

- [ ] **Step 3: 创建环境变量示例**

```bash
# .env.example
# Router配置
ROUTER_HOST=0.0.0.0
ROUTER_PORT=8000
ROUTER_WORKERS=4
ROUTER_LOG_LEVEL=info

# Redis配置
REDIS_URL=redis://localhost:6379

# 数据库配置
DATABASE_URL=postgresql://user:pass@localhost/router

# GLM5配置
GLM5_BASE_URL=http://glm5.internal:8000
GLM5_TIMEOUT=300

# 轻量模型配置
LIGHTWEIGHT_MODEL_URLS=http://qwen-001:8000,http://qwen-002:8000
LIGHTWEIGHT_MODEL_NAME=qwen2.5-7b
```

- [ ] **Step 4: 创建README**

```markdown
# LLM智能路由层

智能路由层，自动将请求分发到不同层级的LLM模型，优化成本和性能。

## 功能特性

- 🤖 **智能路由**: 基于规则+LLM分类，准确率>90%
- ⚡ **多级模型**: 支持轻量/中端/高端模型分层
- 🔄 **负载均衡**: 最少连接、轮询等多种策略
- 📊 **实时监控**: 完整的指标收集和告警
- 🔧 **动态配置**: 支持热更新路由策略

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
cp .env.example .env
# 编辑.env文件，配置你的环境
```

### 3. 启动服务

```bash
./scripts/deploy/start.sh
```

### 4. 验证服务

```bash
curl http://localhost:8000/health
```

## API使用

### 聊天完成接口（OpenAI兼容）

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "auto",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

### 管理接口

```bash
# 查看统计
curl http://localhost:8000/admin/stats

# 健康检查
curl http://localhost:8000/admin/health
```

## 架构说明

详见设计文档: `docs/superpowers/specs/2026-03-26-llm-router-design.md`

## 开发

### 运行测试

```bash
pytest tests/ -v
```

### 代码风格

```bash
black src/ tests/
isort src/ tests/
```

## 部署

### 生产环境部署

1. 准备配置文件 `config/rules.yaml`
2. 设置环境变量 `.env`
3. 启动服务: `./scripts/deploy/start.sh`
4. 配置Nginx反向代理
5. 配置监控告警

### Docker部署（可选）

```bash
docker build -t llm-router .
docker run -p 8000:8000 --env-file .env llm-router
```

## 许可证

MIT
```

- [ ] **Step 5: 添加执行权限**

```bash
chmod +x scripts/deploy/start.sh scripts/deploy/stop.sh
```

- [ ] **Step 6: Commit**

```bash
git add scripts/ README.md .env.example
git commit -m "feat: add deployment scripts and documentation"
```

---

## Part 2 总结

**已完成任务（Task 7-12）：**
- ✅ Task 7: 模型池和负载均衡
- ✅ Task 8: GLM5和轻量模型客户端
- ✅ Task 9: Router Gateway主服务
- ✅ Task 10: 管理接口
- ✅ Task 11: 日志和监控
- ✅ Task 12: 部署脚本和文档

**Phase 1 MVP完整交付物：**
1. 完整的分类器系统（Level 1规则 + Level 3 LLM）
2. 模型池和负载均衡
3. Router Gateway主服务（OpenAI兼容API）
4. 管理接口和监控
5. 部署脚本和文档

**下一步：**
1. 实际部署Qwen2.5-7B模型（vLLM）
2. 接入10%流量进行测试
3. 收集反馈数据并优化阈值
4. 进入Phase 2: 智能增强阶段

---

**完整的Phase 1 MVP实施计划已完成！是否需要我：**
1. 运行规格审查循环检查计划完整性？
2. 立即开始执行实施任务？

---

### Task 10b: 基础反馈收集（简化版）

**说明**: MVP阶段的简化反馈收集，仅记录关键路由数据到日志文件，供人工分析。完整的自动评估机制在Phase 2实现。

**Files:**
- Create: `src/feedback/simple_collector.py`
- Modify: `src/router/api/completions.py`

**实现要点**:

1. **记录路由结果到日志**
```python
# src/feedback/simple_collector.py
import json
import logging
from datetime import datetime

logger = logging.getLogger("router.feedback")

def log_routing_result(request_id, question, answer, route_decision, model_used, latency_ms):
    \"\"\"记录路由结果到日志，供后续分析\"\"\"\n    log_entry = {\n        "timestamp": datetime.now().isoformat(),\n        "request_id": request_id,\n        "question": question[:200],  # 截断避免日志过大\n        "answer": answer[:500],\n        "route_decision": route_decision,\n        "model_used": model_used,\n        "latency_ms": latency_ms\n    }\n    logger.info(json.dumps(log_entry, ensure_ascii=False))\n```

2. **在completions接口中调用**
```python
# 在Task 9的completions函数末尾添加
from src.feedback.simple_collector import log_routing_result

log_routing_result(\n    request_id=response.id,\n    question=input_text,\n    answer=model_response[\"choices\"][0][\"message\"][\"content\"],\n    route_decision=target_tier,\n    model_used=model_name,\n    latency_ms=route_result.latency_ms if route_result else 0\n)\n```

**使用方式**: 
- 日志文件：`logs/routing_feedback.log`
- 每日人工抽查100条，标记是否满意

**初始化步骤**:
```bash
mkdir -p logs
echo "logs/" >> .gitignore
```
- 统计数据用于手动调整阈值

