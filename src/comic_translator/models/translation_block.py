from typing import Any

from pydantic import BaseModel, Field


class TranslationBlock(BaseModel):
    """翻译结果块"""

    id: str
    ocr_block_id: str
    source_text: str
    translated_text: str = ""
    provider: str = ""
    model: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_qa_passed: bool | None = None
    qa_issues: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
