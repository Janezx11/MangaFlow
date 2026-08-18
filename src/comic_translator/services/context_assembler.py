from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from comic_translator.models.enums import ComicStyle

if TYPE_CHECKING:
    from comic_translator.models.context import TranslationContext
    from comic_translator.services.glossary_service import GlossaryService
    from comic_translator.services.character_identity_service import (
        CharacterIdentityService,
    )
    from comic_translator.services.semantic_tm_service import (
        SemanticTranslationMemoryService,
    )
    from comic_translator.services.page_summary_service import PageSummaryService
    from comic_translator.prompts.base import PromptBuilder


class ContextAssembler:
    """上下文组装器 —— 翻译前统一拼装完整上下文

    Sprint 3: 所有上下文拼装逻辑集中在这里。
    TranslationService 不再手动拼接，直接调用 ContextAssembler。

    输入:
    - Glossary
    - CharacterIdentity
    - PageSummary (最近 N 页)
    - TranslationMemory
    - TranslationProfile

    输出:
    - TranslationContext (传递给 Provider 的完整语境)
    """

    def __init__(
        self,
        glossary_service: GlossaryService,
        character_identity_service: CharacterIdentityService,
        semantic_tm_service: SemanticTranslationMemoryService,
        page_summary_service: PageSummaryService,
        prompt_builder: PromptBuilder,
    ) -> None:
        self._glossary = glossary_service
        self._char_identity = character_identity_service
        self._tm = semantic_tm_service
        self._page_summary = page_summary_service
        self._prompt_builder = prompt_builder

    def assemble(
        self,
        project_id: str,
        page: int,
        source_language: str = "en",
        target_language: str = "zh",
        comic_style: ComicStyle | None = None,
    ) -> TranslationContext:
        """组装翻译上下文"""
        from comic_translator.models.context import TranslationContext

        context = TranslationContext(
            comic_style=comic_style or ComicStyle.WESTERN_COMIC,
            source_language=source_language,
            target_language=target_language,
        )

        # 1. Glossary
        glossary_entries = self._glossary.get_all_entries()
        if glossary_entries:
            context.glossary = glossary_entries
            logger.debug(f"ContextAssembler: loaded {len(glossary_entries)} glossary entries")

        # 2. Character Identity
        characters = self._char_identity.get_all_identities()
        if characters:
            context.metadata["characters"] = [
                {
                    "canonical_name": c.canonical_name,
                    "aliases": c.aliases,
                    "titles": c.titles,
                    "speaking_style": c.speaking_style,
                }
                for c in characters
            ]
            logger.debug(f"ContextAssembler: loaded {len(characters)} character identities")

        # 3. Page Summary (最近 3 页)
        recent_summaries = self._page_summary.get_recent_summaries(
            project_id, page, count=3
        )
        if recent_summaries:
            summary_texts = [
                f"[Page {s.page_number}] {s.summary}" for s in recent_summaries
            ]
            context.page_summary = "\n".join(summary_texts)
            context.metadata["page_summaries"] = [
                s.model_dump() for s in recent_summaries
            ]
            logger.debug(
                f"ContextAssembler: loaded {len(recent_summaries)} previous summaries"
            )

        # 4. Translation Profile (from prompt builder)
        profile = getattr(self._prompt_builder, "profile", None)
        if profile:
            context.metadata["translation_rules"] = profile.translation_rules
            context.metadata["forbidden_patterns"] = profile.forbidden_patterns
            context.metadata["preferred_patterns"] = profile.preferred_patterns
            logger.debug(f"ContextAssembler: applied {profile.profile_name} profile")

        return context

    def assemble_with_tm(
        self,
        project_id: str,
        page: int,
        blocks_source_texts: list[str],
        source_language: str = "en",
        target_language: str = "zh",
        comic_style: ComicStyle | None = None,
    ) -> TranslationContext:
        """组装上下文 + 查询 TM

        比 assemble() 多一步 TM 查询，用于实际翻译流程。
        """
        context = self.assemble(
            project_id=project_id,
            page=page,
            source_language=source_language,
            target_language=target_language,
            comic_style=comic_style,
        )

        # 5. Translation Memory
        tm_entries = []
        for source_text in blocks_source_texts:
            similar = self._tm.find_similar(source_text, limit=2)
            tm_entries.extend(similar)
        if tm_entries:
            context.translation_memory = tm_entries
            logger.debug(
                f"ContextAssembler: found {len(tm_entries)} TM matches for {len(blocks_source_texts)} blocks"
            )

        return context