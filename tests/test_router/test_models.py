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
            model="auto", messages=[{"role": "user", "content": "Hi"}]
        )
        assert req.model == "auto"

    def test_specific_model(self):
        req = ChatCompletionRequest(
            model="glm5", messages=[{"role": "user", "content": "Hi"}]
        )
        assert req.model == "glm5"

    def test_invalid_model(self):
        with pytest.raises(ValueError):
            ChatCompletionRequest(
                model="invalid_model", messages=[{"role": "user", "content": "Hi"}]
            )


class TestChatCompletionResponse:
    def test_response_structure(self):
        response = ChatCompletionResponse(
            id="test-123",
            model="qwen2.5-7b",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "Hello"},
                    "finish_reason": "stop",
                }
            ],
        )
        assert response.id == "test-123"
        assert response.model == "qwen2.5-7b"
