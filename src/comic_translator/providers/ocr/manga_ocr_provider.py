from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from comic_translator.providers.ocr.base import OCRProvider

if TYPE_CHECKING:
    from PIL import Image
    from comic_translator.models.ocr_block import OCRBlock


class MangaOCRProvider(OCRProvider):
    """MangaOCR 引擎 —— 日语漫画 OCR"""

    async def detect_text(self, image: Image.Image) -> list[OCRBlock]:
        raise NotImplementedError

    async def detect_text_batch(
        self, images: list[Image.Image]
    ) -> list[list[OCRBlock]]:
        return [await self.detect_text(img) for img in images]