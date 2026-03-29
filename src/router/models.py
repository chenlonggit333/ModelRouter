from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Literal, Union
from datetime import datetime
import uuid


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str
    name: Optional[str] = None

    @field_validator("content")
    @classmethod
    def content_not_empty(cls, v):
        if not v or not v.strip():
            raise ValueError("Content cannot be empty")
        return v


class ChatCompletionRequest(BaseModel):
    model: str = "auto"  # "auto", "glm5", "light", "medium"
    messages: List[ChatMessage]
    temperature: Optional[float] = Field(default=0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(default=2000, gt=0)
    top_p: Optional[float] = Field(default=1.0, ge=0, le=1)
    stream: Optional[bool] = False
    user: Optional[str] = None

    @field_validator("model")
    @classmethod
    def validate_model(cls, v):
        allowed = {"auto", "glm5", "light", "medium"}
        if v not in allowed:
            raise ValueError(f"Model must be one of {allowed}")
        return v

    @field_validator("messages")
    @classmethod
    def at_least_one_message(cls, v):
        if len(v) == 0:
            raise ValueError("At least one message is required")
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
