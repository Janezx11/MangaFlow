from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from comic_translator.models.enums import ComicStyle, ProcessingStatus


class ComicMetadata(BaseModel):
    """漫画元数据"""

    title: str
    chapter: int | None = None
    volume: int | None = None
    total_pages: int
    comic_style: ComicStyle = ComicStyle.WESTERN_COMIC
    source_language: str = "en"
    target_language: str = "zh"
    extra: dict[str, Any] = Field(default_factory=dict)


class ComicProject(BaseModel):
    """漫画项目（一个 chapter 或 volume）"""

    id: str
    name: str
    metadata: ComicMetadata
    input_dir: str
    output_dir: str
    cache_dir: str
    status: ProcessingStatus = ProcessingStatus.PENDING
    pages_completed: int = 0
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    extra: dict[str, Any] = Field(default_factory=dict)
