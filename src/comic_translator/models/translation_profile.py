from pydantic import BaseModel, Field
from comic_translator.models.enums import ComicStyle
from typing import Any


class TranslationProfile(BaseModel):
    """翻译配置 —— 控制 Prompt 生成风格

    Version 2 新增:
    - forbidden_patterns: 禁止使用的表达模式
    - preferred_patterns: 优先使用的表达模式
    """

    profile_name: str
    comic_style: ComicStyle = ComicStyle.WESTERN_COMIC
    translation_rules: list[str] = Field(default_factory=list)
    forbidden_patterns: list[str] = Field(default_factory=list)
    preferred_patterns: list[str] = Field(default_factory=list)
    tone: str = "natural"
    localization_level: str = "moderate"  # minimal | moderate | aggressive
    metadata: dict[str, Any] = Field(default_factory=dict)


# 默认配置
WESTERN_COMIC_PROFILE = TranslationProfile(
    profile_name="western_comic",
    comic_style=ComicStyle.WESTERN_COMIC,
    translation_rules=[
        "Keep proper names in original unless they have standard translations",
        "Translate sound effects as onomatopoeia",
        "Preserve emphasis markers (bold, italics)",
        "Keep it concise — match speech bubble space constraints",
    ],
    forbidden_patterns=[
        "Avoid literal word-for-word translation",
        "Avoid formal/书面 language in casual dialogue",
    ],
    preferred_patterns=[
        "Use natural spoken Chinese",
        "Use shorter sentences for action scenes",
        "Preserve character personality in dialogue",
    ],
    tone="natural_conversation",
    localization_level="moderate",
)

MANGA_PROFILE = TranslationProfile(
    profile_name="manga",
    comic_style=ComicStyle.MANGA,
    translation_rules=[
        "Preserve Japanese honorifics (-san, -kun, -chan, -sama)",
        "Sound effects: keep original in parentheses with Chinese reading",
        "Keep name order: family name first for Japanese names",
        "Match manga dialogue tone",
    ],
    forbidden_patterns=[
        "Avoid removing honorifics",
        "Avoid Westernizing Japanese cultural terms",
    ],
    preferred_patterns=[
        "Match character speech patterns (polite, rough, childish)",
        "Use manga-appropriate sentence-ending particles (呢, 啊, 哟)",
    ],
    tone="matches_manga_dialogue",
    localization_level="moderate",
)

WEBTOON_PROFILE = TranslationProfile(
    profile_name="webtoon",
    comic_style=ComicStyle.WEBTOON,
    translation_rules=[
        "Minimal line breaks — optimize for mobile vertical scrolling",
        "Natural conversational style",
        "Keep text length suitable for narrow reading pane",
    ],
    forbidden_patterns=[
        "Avoid long sentences that span multiple lines",
    ],
    preferred_patterns=[
        "Use modern internet-friendly expressions",
        "Keep text short and punchy",
    ],
    tone="conversational",
    localization_level="moderate",
)