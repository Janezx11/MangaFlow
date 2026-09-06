from typing import Any

from pydantic import BaseModel, Field


class CharacterIdentity(BaseModel):
    """角色身份 —— Sprint 3 升级版

    比 CharacterMemory 增加了:
    - titles: 头衔/称号 (Iron Man, CEO of Stark Industries)
    - nicknames: 昵称 (Shellhead, Point Break)
    - aliases: 化名/变体 (Tony, Mr. Stark)
    - related_character_ids: 关联角色 (Pepper Potts, Happy Hogan)
    """

    character_id: str
    canonical_name: str
    aliases: list[str] = Field(default_factory=list)
    titles: list[str] = Field(default_factory=list)
    nicknames: list[str] = Field(default_factory=list)
    description: str = ""
    speaking_style: str = ""
    related_character_ids: list[str] = Field(default_factory=list)
    first_seen_page: int = 0
    last_seen_page: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class IdentityMatchResult(BaseModel):
    """身份匹配结果"""

    identity: CharacterIdentity
    match_type: str = "exact"  # exact | alias | title | nickname | fuzzy
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
