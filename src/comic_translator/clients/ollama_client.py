from __future__ import annotations

import asyncio
from typing import AsyncIterator

from httpx import AsyncClient, HTTPStatusError, RequestError

from comic_translator.clients.base import LLMClient


class OllamaClient(LLMClient):
    """Ollama API 客户端

    支持两种 API：
      /api/generate  — 旧版（仍被大多数 Ollama 版本支持）
      /api/chat      — 新版（Ollama 0.3+，支持 messages 格式）

    内置特性：
    - 指数退避重试（应对临时连接失败）
    - 可配置最大重试次数

    count_tokens() 使用本地估算，不再依赖已废弃的 /api/tokenize。
    """

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout: int = 300,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self._max_retries = max_retries
        self._client = AsyncClient(timeout=timeout, trust_env=False)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        last_exception: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._client.post(
                    f"{self.base_url}/api/generate",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("response", "")
            except (RequestError, HTTPStatusError) as e:
                last_exception = e
                status_code = getattr(e.response, "status_code", 0) if hasattr(e, "response") else 0
                # 5xx 或连接错误可重试
                if (
                    isinstance(e, RequestError)
                    or (500 <= status_code < 600)
                ) and attempt < self._max_retries:
                    sleep_time = min(2 ** attempt, 30)
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Ollama request failed (attempt {attempt}/{self._max_retries}, "
                        f"status={status_code}): {e}. Waiting {sleep_time}s..."
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    raise
            except Exception as e:
                last_exception = e
                if attempt < self._max_retries:
                    sleep_time = min(2 ** attempt, 30)
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Ollama request unexpected error (attempt {attempt}/{self._max_retries}): "
                        f"{e}. Waiting {sleep_time}s..."
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    raise

        raise RuntimeError(
            f"Ollama request failed after {self._max_retries} attempts"
        ) from last_exception

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        """使用 /api/chat 接口（新版 Ollama 推荐）"""
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }

        last_exception: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                response = await self._client.post(
                    f"{self.base_url}/api/chat",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()
                return data.get("message", {}).get("content", "")
            except (RequestError, HTTPStatusError) as e:
                last_exception = e
                status_code = getattr(e.response, "status_code", 0) if hasattr(e, "response") else 0
                if (
                    isinstance(e, RequestError)
                    or (500 <= status_code < 600)
                ) and attempt < self._max_retries:
                    sleep_time = min(2 ** attempt, 30)
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Ollama chat failed (attempt {attempt}/{self._max_retries}, "
                        f"status={status_code}): {e}. Waiting {sleep_time}s..."
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    raise
            except Exception as e:
                last_exception = e
                if attempt < self._max_retries:
                    sleep_time = min(2 ** attempt, 30)
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Ollama chat unexpected error (attempt {attempt}/{self._max_retries}): "
                        f"{e}. Waiting {sleep_time}s..."
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    raise

        raise RuntimeError(
            f"Ollama chat failed after {self._max_retries} attempts"
        ) from last_exception

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "system": system_prompt or "",
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True,
        }

        last_exception: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                async with self._client.stream(
                    "POST",
                    f"{self.base_url}/api/generate",
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        import json

                        data = json.loads(line)
                        chunk = data.get("response", "")
                        if chunk:
                            yield chunk
                return

            except (RequestError, HTTPStatusError) as e:
                last_exception = e
                status_code = getattr(e.response, "status_code", 0) if hasattr(e, "response") else 0
                if (
                    isinstance(e, RequestError)
                    or (500 <= status_code < 600)
                ) and attempt < self._max_retries:
                    sleep_time = min(2 ** attempt, 30)
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Ollama stream failed (attempt {attempt}/{self._max_retries}, "
                        f"status={status_code}): {e}. Waiting {sleep_time}s..."
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    raise
            except Exception as e:
                last_exception = e
                if attempt < self._max_retries:
                    sleep_time = min(2 ** attempt, 30)
                    import logging
                    logging.getLogger(__name__).warning(
                        f"Ollama stream unexpected error (attempt {attempt}/{self._max_retries}): "
                        f"{e}. Waiting {sleep_time}s..."
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    raise

        raise RuntimeError(
            f"Ollama stream failed after {self._max_retries} attempts"
        ) from last_exception

    async def count_tokens(self, text: str) -> int:
        """估算 token 数量

        不需要调用 Ollama API（/api/tokenize 已在 Ollama 0.3+ 移除）。
        使用简单估算：中文约 1 token/字，英文约 1 token/4 字符。
        """
        char_count = len(text)
        # Count non-ASCII characters (Chinese, Japanese, etc.)
        non_ascii = sum(1 for c in text if ord(c) > 127)
        ascii_chars = char_count - non_ascii
        # Rough estimate: non-ASCII ~1 token each, ASCII ~1 token per 4 chars
        return int(non_ascii * 1.0 + ascii_chars / 4.0) + 1

    async def list_models(self) -> list[dict]:
        """获取 Ollama 可用模型列表"""
        response = await self._client.get(f"{self.base_url}/api/tags")
        response.raise_for_status()
        data = response.json()
        return data.get("models", [])

    async def is_available(self) -> bool:
        """快速健康检查 —— 不依赖特定模型

        用于 batch_translate.py 的健康检测。
        """
        try:
            # /api/tags 是最轻量的端点，不加载任何模型
            await self._client.get(f"{self.base_url}/api/tags")
            return True
        except Exception:
            return False

    @property
    def model_name(self) -> str:
        return self.model