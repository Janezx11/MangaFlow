from typing import Any

from pydantic import BaseModel, Field


class PageSummary(BaseModel):
    """页面摘要 —— 每页翻译完成后生成

    用于跨页上下文传递。
    存储到: data/cache/{project_id}/summaries/page_{page:06d}_summary.json
    """

    page_number: int
    summary: str
    characters: list[str] = Field(default_factory=list)
    locations: list[str] = Field(default_factory=list)
    events: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
