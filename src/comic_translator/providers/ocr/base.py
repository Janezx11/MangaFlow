from abc import ABC, abstractmethod
from comic_translator.models.ocr_block import OCRBlock
from PIL import Image


class OCRProvider(ABC):
    """OCR 能力抽象 —— 所有 OCR 引擎需实现此接口"""

    @abstractmethod
    async def detect_text(self, image: Image.Image) -> list[OCRBlock]:
        ...

    async def detect_text_batch(
        self, images: list[Image.Image]
    ) -> list[list[OCRBlock]]:
        return [await self.detect_text(img) for img in images]