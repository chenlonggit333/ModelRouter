import httpx
from typing import List, Dict, Any, Optional


class GLM5Client:
    """
    GLM5模型客户端
    对接内部GLM5服务（H200*2*8卡部署）
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._client = None

    async def _get_client(self) -> httpx.AsyncClient:
        """获取或创建HTTP客户端"""
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.timeout)
        return self._client

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 1.0,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        调用GLM5进行聊天完成

        Args:
            messages: 消息列表
            temperature: 温度参数
            max_tokens: 最大token数
            top_p: top-p采样
            stream: 是否流式输出

        Returns:
            GLM5的响应数据
        """
        payload = {
            "model": "glm5",
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream,
        }

        try:
            client = await self._get_client()
            response = await client.post(
                f"{self.base_url}/v1/chat/completions",
                json=payload,
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise Exception(
                f"GLM5 request failed: {e.response.status_code} - {e.response.text}"
            )
        except Exception as e:
            raise Exception(f"GLM5 request error: {str(e)}")

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            client = await self._get_client()
            response = await client.get(f"{self.base_url}/health", timeout=5.0)
            return response.status_code == 200
        except:
            return False

    async def close(self):
        """关闭连接"""
        if self._client:
            await self._client.aclose()
            self._client = None
