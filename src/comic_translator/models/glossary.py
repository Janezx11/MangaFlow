from typing import Any

from pydantic import BaseModel, Field


class GlossaryEntry(BaseModel):
    """词条条目"""

    source: str
    target: str
    context: str | None = None
    category: str = "general"
    priority: int = Field(default=0, ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class TranslationMemory(BaseModel):
    """翻译记忆"""

    source_text: str
    translated_text: str
    context_hash: str
    similarity: float = Field(ge=0.0, le=1.0)
    provider: str
    model: str
    usage_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)
