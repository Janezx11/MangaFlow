from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comic_translator.models.glossary import GlossaryEntry


class GlossaryMatcher:
    """词条匹配器 —— 从文本中查找匹配的 glossary 条目"""

    async def match(self, text: str, glossary: list[GlossaryEntry]) -> list[GlossaryEntry]:
        """返回 text 中匹配到的所有 glossary 条目，按优先级排序"""
        matched: list[GlossaryEntry] = []
        for entry in glossary:
            if entry.source.lower() in text.lower():
                matched.append(entry)
        matched.sort(key=lambda e: e.priority, reverse=True)
        return matched