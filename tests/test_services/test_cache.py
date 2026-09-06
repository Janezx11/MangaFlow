"""Tests for cache service"""

from comic_translator.cache.null_cache_provider import NullCacheProvider
from comic_translator.services.cache_service import CacheService


class TestNullCacheProvider:
    def test_get_returns_none(self):
        cache = NullCacheProvider()
        result = None

        async def run():
            nonlocal result
            result = await cache.get("any_key")

        import asyncio
        asyncio.run(run())
        assert result is None

    def test_set_does_not_error(self):
        cache = NullCacheProvider()

        async def run():
            await cache.set("key", {"data": "test"})

        import asyncio
        asyncio.run(run())  # should not raise


class TestCacheService:
    def test_null_cache_service(self):
        cache_service = CacheService(provider=NullCacheProvider())

        async def run():
            result = await cache_service.get_ocr_cache("proj", 1)
            assert result is None

        import asyncio
        asyncio.run(run())
