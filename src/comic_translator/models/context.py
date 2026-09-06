from typing import Any

from pydantic import BaseModel, Field

from comic_translator.models.enums import ComicStyle
from comic_translator.models.glossary import GlossaryEntry, TranslationMemory


class TranslationContext(BaseModel):
    """翻译上下文（传递给 Provider 的完整语境）"""

    glossary: list[GlossaryEntry] = Field(default_factory=list)
    translation_memory: list[TranslationMemory] = Field(default_factory=list)
    page_summary: str | None = None
    comic_style: ComicStyle = ComicStyle.WESTERN_COMIC
    source_language: str = "en"
    target_language: str = "zh"
    metadata: dict[str, Any] = Field(default_factory=dict)


class QAReport(BaseModel):
    """QA 检查报告"""

    page: int
    passed: bool
    blocks_checked: int
    blocks_failed: int
    issues: list[str] = Field(default_factory=list)
    provider: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
