from pydantic import Field

from comic_translator.models.ocr_block import OCRBlock


class NormalizedOCRBlock(OCRBlock):
    """标准化后的 OCR 块 —— 增加阅读顺序和合并标记"""

    reading_order: int = Field(default=0, ge=0)
    is_merged: bool = False
    merged_from_ids: list[str] = Field(default_factory=list)
