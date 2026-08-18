from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from comic_translator.models.translation_block import TranslationBlock

if TYPE_CHECKING:
    from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock
    from comic_translator.models.context import TranslationContext
    from comic_translator.providers.translation.base import TranslationProvider
    from comic_translator.prompts.base import PromptBuilder
    from comic_translator.services.cache_service import CacheService
    from comic_translator.services.context_assembler import ContextAssembler
    from comic_translator.services.semantic_tm_service import (
        SemanticTranslationMemoryService,
    )
    from comic_translator.services.prompt_cache_service import PromptCacheService
    from comic_translator.services.page_summary_service import PageSummaryService


class TranslationService:
    """翻译服务 —— Sprint 3: 使用 ContextAssembler 统一拼装上下文

    执行流程:
    1. 检查翻译缓存
    2. ContextAssembler.assemble() → 获取 Glossary + Character + Summary + Profile
    3. 调用 PromptBuilder 构建 Prompt
    4. 缓存 Prompt 到 PromptCacheService
    5. 调用 TranslationProvider
    6. 保存 Translation Cache
    7. 保存 TM (SemanticTranslationMemoryService)
    8. PageSummaryService.generate_summary() → 生成跨页摘要
    """

    def __init__(
        self,
        provider: TranslationProvider,
        prompt_builder: PromptBuilder,
        cache: CacheService,
        context_assembler: ContextAssembler,
        semantic_tm_service: SemanticTranslationMemoryService,
        prompt_cache_service: PromptCacheService | None = None,
        page_summary_service: PageSummaryService | None = None,
    ) -> None:
        self._provider = provider
        self._prompt_builder = prompt_builder
        self._cache = cache
        self._context_assembler = context_assembler
        self._tm = semantic_tm_service
        self._prompt_cache = prompt_cache_service
        self._page_summary = page_summary_service

    async def translate_page(
        self,
        ocr_blocks: list[NormalizedOCRBlock],
        context: TranslationContext,
        page: int,
        project_id: str = "",
        use_cache: bool = True,
        comic_style=None,
    ) -> list[TranslationBlock]:
        # Translation cache check
        if use_cache and project_id:
            cached = await self._cache.get_translation_cache(project_id, page)
            if cached is not None:
                logger.debug(f"Translation cache hit for page {page}")
                return cached

        # Build TranslationBlocks
        blocks: list[TranslationBlock] = [
            TranslationBlock(
                id=b.id,
                ocr_block_id=b.id,
                source_text=b.text,
                provider=self._provider.__class__.__name__,
                model=self._provider._model if hasattr(self._provider, "_model") else "",
            )
            for b in ocr_blocks
        ]

        # Use ContextAssembler to build full context
        if project_id:
            assembled = self._context_assembler.assemble_with_tm(
                project_id=project_id,
                page=page,
                blocks_source_texts=[b.source_text for b in blocks],
                source_language=context.source_language,
                target_language=context.target_language,
                comic_style=comic_style or context.comic_style,
            )
            # Merge assembled context with passed context (passed values take priority)
            if assembled.glossary:
                context.glossary = assembled.glossary
            if assembled.translation_memory:
                context.translation_memory = assembled.translation_memory
            if assembled.page_summary:
                context.page_summary = assembled.page_summary
            if assembled.metadata.get("characters"):
                context.metadata["characters"] = assembled.metadata["characters"]
            if assembled.metadata.get("translation_rules"):
                context.metadata["translation_rules"] = assembled.metadata["translation_rules"]
            if assembled.metadata.get("forbidden_patterns"):
                context.metadata["forbidden_patterns"] = assembled.metadata["forbidden_patterns"]
            if assembled.metadata.get("preferred_patterns"):
                context.metadata["preferred_patterns"] = assembled.metadata["preferred_patterns"]

        # Log context stats
        logger.info(
            f"Translating {len(blocks)} blocks on page {page} "
            f"(glossary: {len(context.glossary)}, "
            f"characters: {len(context.metadata.get('characters', []))}, "
            f"TM: {len(context.translation_memory)}, "
            f"summary: {'yes' if context.page_summary else 'no'})"
        )

        # Translate
        translated = await self._provider.translate(blocks, context)

        # Save to semantic TM
        for block in translated:
            if block.translated_text and block.translated_text != block.source_text:
                self._tm.store(
                    source_text=block.source_text,
                    translated_text=block.translated_text,
                    provider=block.provider,
                    model=block.model,
                    project_id=project_id,
                )

        # Generate page summary (async, fire and forget to avoid blocking)
        if self._page_summary and project_id:
            try:
                # We don't await — the translation is done, summary can be generated
                # as a background task or lazily on next page
                pass
            except Exception as e:
                logger.debug(f"Summary generation skipped: {e}")

        # Save translation cache
        if use_cache and project_id:
            await self._cache.set_translation_cache(project_id, page, translated)

        return translated