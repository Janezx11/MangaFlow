from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from PIL import Image
    from comic_translator.models.ocr_block import OCRBlock
    from comic_translator.providers.ocr.base import OCRProvider
    from comic_translator.services.cache_service import CacheService


class OCRService:
    """OCR 服务 —— 封装 OCR 业务逻辑 + 缓存"""

    def __init__(self, provider: OCRProvider, cache: CacheService) -> None:
        self._provider = provider
        self._cache = cache

    async def process_page(
        self,
        image: Image.Image,
        page: int,
        project_id: str = "",
        use_cache: bool = True,
    ) -> list[OCRBlock]:
        if use_cache and project_id:
            cached = await self._cache.get_ocr_cache(project_id, page)
            if cached is not None:
                logger.debug(f"OCR cache hit for page {page}")
                return cached

        logger.info(f"Running OCR on page {page}")
        blocks = await self._provider.detect_text(image)

        if use_cache and project_id:
            await self._cache.set_ocr_cache(project_id, page, blocks)

        return blocks