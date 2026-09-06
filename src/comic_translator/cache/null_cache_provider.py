from typing import Any

from comic_translator.cache.base import CacheProvider


class NullCacheProvider(CacheProvider):
    """空缓存 —— 禁用缓存时使用"""

    async def get(self, key: str) -> Any | None:
        return None

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        pass

    async def exists(self, key: str) -> bool:
        return False

    async def delete(self, key: str) -> None:
        pass

    async def clear(self) -> None:
        pass
