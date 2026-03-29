from typing import List, Dict, Any, Optional
import random
from src.models.glm5_client import GLM5Client


class LightweightModelClient:
    """
    轻量模型客户端
    支持多个轻量模型实例的负载均衡
    """

    def __init__(
        self, base_urls: List[str], model_name: str = "qwen2.5-7b", timeout: int = 60
    ):
        self.base_urls = base_urls
        self.model_name = model_name
        self.timeout = timeout
        self._clients = [GLM5Client(base_url=url, timeout=timeout) for url in base_urls]
        self._current_index = 0

    def _select_client(self) -> GLM5Client:
        """轮询选择一个客户端"""
        client = self._clients[self._current_index % len(self._clients)]
        self._current_index += 1
        return client

    async def chat_completion(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
        top_p: float = 1.0,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        调用轻量模型进行聊天完成

        策略：轮询选择实例，失败时自动重试其他实例
        """
        last_error = None

        # 尝试每个客户端，最多重试3次
        for _ in range(min(3, len(self._clients))):
            client = self._select_client()
            try:
                result = await client.chat_completion(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    top_p=top_p,
                    stream=stream,
                )
                # 添加模型信息
                result["model"] = self.model_name
                return result
            except Exception as e:
                last_error = e
                continue

        # 所有客户端都失败
        raise Exception(
            f"All lightweight model clients failed. Last error: {last_error}"
        )

    async def health_check(self) -> Dict[str, bool]:
        """检查所有实例的健康状态"""
        results = {}
        for i, client in enumerate(self._clients):
            results[f"instance_{i}"] = await client.health_check()
        return results

    async def close(self):
        """关闭所有连接"""
        for client in self._clients:
            await client.close()
