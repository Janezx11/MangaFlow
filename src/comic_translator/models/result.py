from typing import Any

from pydantic import BaseModel, Field

from comic_translator.models.context import QAReport
from comic_translator.models.glossary import GlossaryEntry
from comic_translator.models.ocr_block import OCRBlock
from comic_translator.models.translation_block import TranslationBlock


class PageResult(BaseModel):
    """单页完整处理结果"""

    page: int
    page_path: str
    ocr_blocks: list[OCRBlock] = Field(default_factory=list)
    translations: list[TranslationBlock] = Field(default_factory=list)
    glossary_used: list[GlossaryEntry] = Field(default_factory=list)
    qa_report: QAReport | None = None
    render_output_path: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RenderResult(BaseModel):
    """渲染结果"""

    page: int
    source_path: str
    output_path: str
    blocks_rendered: int
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProcessingError(BaseModel):
    """处理错误"""

    node: str
    page: int
    message: str
    exception: str | None = None
    recoverable: bool = True
