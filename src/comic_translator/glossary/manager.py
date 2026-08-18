from __future__ import annotations

from typing import TYPE_CHECKING

from comic_translator.models.glossary import GlossaryEntry

if TYPE_CHECKING:
    from comic_translator.models.context import TranslationContext
    from comic_translator.models.project import ComicProject


class GlossaryManager:
    """词条管理 —— 加载/保存 glossary 条目"""

    def __init__(self, glossary_dir: str = "data/glossary") -> None:
        self._dir = glossary_dir

    async def load_all(self) -> list[GlossaryEntry]:
        raise NotImplementedError

    async def save_entry(self, entry: GlossaryEntry) -> None:
        raise NotImplementedError

    async def import_from_file(self, path: str) -> int:
        raise NotImplementedError


class GlossaryMatcher:
    """词条匹配 —— 从文本中匹配 glossary 条目"""

    async def match(self, text: str, glossary: list[GlossaryEntry]) -> list[GlossaryEntry]:
        raise NotImplementedError


class GlossaryService:
    """术语表服务"""

    def __init__(self, manager: GlossaryManager, matcher: GlossaryMatcher) -> None:
        self._manager = manager
        self._matcher = matcher

    async def lookup(self, text: str) -> list[GlossaryEntry]:
        glossary = await self._manager.load_all()
        return await self._matcher.match(text, glossary)

    async def build_context(self, project: ComicProject) -> TranslationContext:
        raise NotImplementedError