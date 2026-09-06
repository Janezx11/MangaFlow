from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator

from loguru import logger
from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from comic_translator.clients.base import LLMClient
from comic_translator.utils.rate_limiter import TokenBucketRateLimiter


class OpenAICompatibleClient(LLMClient):
    """OpenAI Compatible API 客户端

    适用于任何兼容 OpenAI Chat Completions API 的服务：
    DeepSeek, Groq, Together AI, vLLM, etc.

    内置特性：
    - 指数退避重试（应对 429/5xx 临时错误）
    - 令牌桶速率限制（避免触发账号级限流）
    - 可配置最大重试次数和速率（RPS）

    Provider 通过 LLMClient 接口使用，切换后端时只需更换 client 配置。
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str = "",
        timeout: int = 120,
        max_retries: int = 5,
        rps: float = 3.0,
        burst: int | None = None,
    ) -> None:
        """初始化 OpenAI 兼容客户端

        Args:
            model: 模型名称
            base_url: API 基础 URL
            api_key: API 密钥
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数（含首次）
            rps: 每秒最大请求数（Requests Per Second），默认 3
            burst: 令牌桶突发容量，默认 rps * 2
        """
        self.model = model
        self.api_key = api_key
        self._max_retries = max_retries
        base_url = base_url.rstrip("/")
        if not base_url.endswith("/v1"):
            base_url += "/v1"
        self._client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
            max_retries=0,  # 我们自己在外部处理重试，避免 SDK 内部重试干扰
        )
        # 速率限制器
        self._rate_limiter = TokenBucketRateLimiter(rps=rps, burst=burst)

    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if not self.api_key:
            raise RuntimeError(
                "API key is not configured — set llm.api.api_key in config.yaml "
                "or export the referenced environment variable"
            )

        last_exception: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            try:
                # 1. 速率限制等待
                wait_time = await self._rate_limiter.acquire()
                if wait_time > 0.1:
                    logger.debug(f"Rate limited, waited {wait_time:.2f}s before request")

                # 2. 发起请求
                response = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                if not response.choices:
                    # 200 但空 choices（内容过滤/上游错误）：确定性失败，重试无意义
                    raise ValueError("API returned 200 with empty choices (content may be filtered)")
                return response.choices[0].message.content or ""

            except RateLimitError as e:
                last_exception = e
                retry_after = self._parse_retry_after(e)
                if attempt < self._max_retries:
                    sleep_time = retry_after or min(2 ** attempt * 2, 60)
                    logger.warning(
                        f"Rate limited (429) on attempt {attempt}/{self._max_retries}. "
                        f"Waiting {sleep_time:.0f}s... "
                        f"({e.message[:100] if hasattr(e, 'message') else str(e)[:100]})"
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(f"Rate limit persisted after {self._max_retries} attempts")

            except APITimeoutError as e:
                last_exception = e
                if attempt < self._max_retries:
                    sleep_time = min(2 ** attempt * 2, 30)
                    logger.warning(
                        f"Timeout on attempt {attempt}/{self._max_retries}. "
                        f"Waiting {sleep_time:.0f}s before retry..."
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(f"Request timed out after {self._max_retries} attempts")

            except APIConnectionError as e:
                # 连接层错误（DNS 失败/连接重置等）——可重试。
                # 注意：APIConnectionError 是 APIError 的子类且没有 status_code，
                # 必须在 except APIError 之前捕获，否则会落到"非 5xx 不重试"分支。
                last_exception = e
                if attempt < self._max_retries:
                    sleep_time = min(2 ** attempt * 2, 30)
                    logger.warning(
                        f"Connection error on attempt {attempt}/{self._max_retries}: {e}. "
                        f"Waiting {sleep_time:.0f}s before retry..."
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(f"Connection failed after {self._max_retries} attempts: {e}")
                    raise

            except APIError as e:
                last_exception = e
                status_code = getattr(e, "status_code", 0)
                # 5xx 错误可重试
                if 500 <= status_code < 600 and attempt < self._max_retries:
                    sleep_time = min(2 ** attempt * 2, 30)
                    logger.warning(
                        f"Server error ({status_code}) on attempt {attempt}/{self._max_retries}. "
                        f"Waiting {sleep_time:.0f}s before retry..."
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    # 4xx 非限流错误（如 400/401/403）不重试
                    logger.error(
                        f"API error ({status_code}) on attempt {attempt}: {e}"
                    )
                    raise

            except ValueError:
                # 确定性响应错误（如空 choices），重试不会改变结果
                raise

            except Exception as e:
                # 网络错误等可重试的异常
                last_exception = e
                if attempt < self._max_retries:
                    sleep_time = min(2 ** attempt * 2, 30)
                    logger.warning(
                        f"Request failed on attempt {attempt}/{self._max_retries}: {e}. "
                        f"Waiting {sleep_time:.0f}s before retry..."
                    )
                    await asyncio.sleep(sleep_time)
                else:
                    logger.error(f"Request failed after {self._max_retries} attempts: {e}")
                    raise

        # 所有重试用尽
        raise RuntimeError(
            f"Request failed after {self._max_retries} attempts"
        ) from last_exception

    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        if not self.api_key:
            raise RuntimeError(
                "API key is not configured — set llm.api.api_key in config.yaml "
                "or export the referenced environment variable"
            )

        last_exception: Exception | None = None

        for attempt in range(1, self._max_retries + 1):
            yielded_any = False
            try:
                # 速率限制等待
                await self._rate_limiter.acquire()

                stream = await self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stream=True,
                )
                async for chunk in stream:
                    delta = chunk.choices[0].delta if chunk.choices else None
                    if delta and delta.content:
                        yielded_any = True
                        yield delta.content
                # 流式成功完成，退出重试循环
                return

            except Exception as e:
                # 已经向下游输出过内容时不能重试：重试会从头重发全部内容，
                # 导致下游拼接出重复文本。此时只能把错误抛给调用方。
                if yielded_any:
                    logger.error(f"Stream interrupted after partial output: {e}")
                    raise
                last_exception = e
                retry_after = self._parse_retry_after(e) if isinstance(e, RateLimitError) else None
                is_rate_limit = isinstance(e, RateLimitError)
                if isinstance(e, (APIConnectionError, APITimeoutError)) or (
                    is_rate_limit
                ) or (isinstance(e, APIError) and 500 <= getattr(e, "status_code", 0) < 600):
                    if attempt < self._max_retries:
                        if is_rate_limit:
                            sleep_time = retry_after or min(2 ** attempt * 2, 60)
                        else:
                            sleep_time = min(2 ** attempt * 2, 30)
                        logger.warning(
                            f"Stream failed on attempt {attempt}/{self._max_retries}: {e}. "
                            f"Waiting {sleep_time:.0f}s..."
                        )
                        await asyncio.sleep(sleep_time)
                        continue
                    logger.error(f"Stream failed after {self._max_retries} attempts: {e}")
                else:
                    logger.error(f"Stream failed with non-retryable error: {e}")
                raise

        raise RuntimeError(
            f"Stream request failed after {self._max_retries} attempts"
        ) from last_exception

    async def count_tokens(self, text: str) -> int:
        # 近似估算
        return len(text) // 4

    @property
    def model_name(self) -> str:
        return self.model

    @staticmethod
    def _parse_retry_after(error: RateLimitError) -> float | None:
        """从 RateLimitError 中解析 retry-after 头或消息中的等待时间"""
        # 1. 检查 response headers 中的 retry-after
        try:
            if hasattr(error, "response") and error.response is not None:
                retry_after = error.response.headers.get("retry-after")
                if retry_after:
                    return float(retry_after)
        except (AttributeError, ValueError, TypeError):
            pass

        # 2. 从错误消息中提取秒数（如 "Please retry after 20 seconds"）
        message = getattr(error, "message", str(error))
        import re
        match = re.search(r"retry after (\d+)", message, re.IGNORECASE)
        if match:
            return float(match.group(1))
        match = re.search(r"after (\d+) seconds", message, re.IGNORECASE)
        if match:
            return float(match.group(1))

        return None
