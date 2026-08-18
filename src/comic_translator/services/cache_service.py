from __future__ import annotations

import json
from typing import TYPE_CHECKING

from comic_translator.cache.base import CacheProvider

if TYPE_CHECKING:
    from comic_translator.models.ocr_block import OCRBlock
    from comic_translator.models.translation_block import TranslationBlock


class CacheService:
    """缓存服务 —— 管理 OCR / Translation / Render 三层缓存

    缓存键规则: {project_id}:{stage}:{page}
    文件格式:
      page_0001.ocr.json       → list[OCRBlock]
      page_0001.translation.json → list[TranslationBlock]
      page_0001.render.json    → str (output_path)
    """

    def __init__(self, provider: CacheProvider) -> None:
        self._provider = provider

    # --- OCR Cache ---

    async def get_ocr_cache(self, project_id: str, page: int) -> list[OCRBlock] | None:
        data = await self._provider.get(f"{project_id}:ocr:{page:06d}")
        if data is None:
            return None
        from comic_translator.models.ocr_block import OCRBlock

        return [OCRBlock(**b) for b in data]

    async def set_ocr_cache(self, project_id: str, page: int, blocks: list[OCRBlock]) -> None:
        data = [b.model_dump() for b in blocks]
        await self._provider.set(f"{project_id}:ocr:{page:06d}", data)

    # --- Translation Cache ---

    async def get_translation_cache(
        self, project_id: str, page: int
    ) -> list[TranslationBlock] | None:
        data = await self._provider.get(f"{project_id}:translation:{page:06d}")
        if data is None:
            return None
        from comic_translator.models.translation_block import TranslationBlock

        return [TranslationBlock(**b) for b in data]

    async def set_translation_cache(
        self, project_id: str, page: int, blocks: list[TranslationBlock]
    ) -> None:
        data = [b.model_dump() for b in blocks]
        await self._provider.set(f"{project_id}:translation:{page:06d}", data)

    # --- Render Cache ---

    async def get_render_cache(self, project_id: str, page: int) -> str | None:
        return await self._provider.get(f"{project_id}:render:{page:06d}")

    async def set_render_cache(self, project_id: str, page: int, output_path: str) -> None:
        await self._provider.set(f"{project_id}:render:{page:06d}", output_path)