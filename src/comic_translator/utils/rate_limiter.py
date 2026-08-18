from __future__ import annotations

import asyncio
import time


class TokenBucketRateLimiter:
    """异步令牌桶速率限制器

    控制每秒最大请求数（RPS），突发（burst）允许短时间内超过 RPS 限制。
    适用于 API 调用场景：在请求前调用 acquire()，如果桶中无令牌则等待。

    Args:
        rps: 每秒允许的请求数（令牌 replenish 速率）
        burst: 桶的最大容量，即允许的突发请求数。默认为 rps。
    """

    def __init__(self, rps: float = 10.0, burst: int | None = None) -> None:
        if rps <= 0:
            raise ValueError("rps must be > 0")
        self._rps = rps
        self._burst = burst if burst is not None else max(1, int(rps * 2))
        self._tokens = float(self._burst)
        self._last_refill = time.monotonic()
        self._lock = asyncio.Lock()

    async def acquire(self) -> float:
        """获取一个令牌，等待直到有可用令牌。

        Returns:
            float: 等待的秒数。
        """
        wait_time = 0.0
        while True:
            async with self._lock:
                self._refill()
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return wait_time
                # 下一个令牌需要等待的时间
                wait_time = (1.0 - self._tokens) / self._rps

            # 在锁外等待，避免阻塞其他 acquire() 调用
            await asyncio.sleep(wait_time)
            wait_time += wait_time  # 累积等待时间

    def _refill(self) -> None:
        """补充令牌"""
        now = time.monotonic()
        elapsed = now - self._last_refill
        self._tokens = min(
            float(self._burst),
            self._tokens + elapsed * self._rps,
        )
        self._last_refill = now

    @property
    def rps(self) -> float:
        return self._rps

    @property
    def burst(self) -> int:
        return self._burst


class NoopRateLimiter:
    """无操作速率限制器（用于本地 Ollama 等无需限速的场景）"""

    async def acquire(self) -> float:
        return 0.0

    @property
    def rps(self) -> float:
        return float("inf")

    @property
    def burst(self) -> int:
        return 0