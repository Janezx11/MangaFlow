from pydantic import BaseModel, Field
from typing import Any


class CharacterMemory(BaseModel):
    """角色记忆 —— 维护统一译名"""

    character_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    description: str = ""
    speaking_style: str = ""
    first_seen_page: int = 0
    last_seen_page: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class CharacterMemoryEntry(BaseModel):
    """角色记忆条目（含匹配信息）"""

    character: CharacterMemory
    match_type: str = "exact"  # exact | alias | fuzzy
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)