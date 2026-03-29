import pytest
from unittest.mock import Mock, patch, AsyncMock
import httpx
from src.models.glm5_client import GLM5Client


class TestGLM5Client:
    @pytest.fixture
    def client(self):
        return GLM5Client(base_url="http://localhost:8000", timeout=60)

    @pytest.mark.asyncio
    async def test_chat_completion(self, client):
        # Mock httpx.AsyncClient
        mock_response = Mock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello"}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        mock_response.raise_for_status = Mock()

        with patch("httpx.AsyncClient.post", return_value=mock_response):
            result = await client.chat_completion(
                messages=[{"role": "user", "content": "Hi"}], temperature=0.7
            )

        assert result["choices"][0]["message"]["content"] == "Hello"

    @pytest.mark.asyncio
    async def test_health_check_success(self, client):
        mock_response = Mock()
        mock_response.status_code = 200

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            is_healthy = await client.health_check()

        assert is_healthy is True

    @pytest.mark.asyncio
    async def test_health_check_failure(self, client):
        with patch("httpx.AsyncClient.get", side_effect=Exception("Connection error")):
            is_healthy = await client.health_check()

        assert is_healthy is False
