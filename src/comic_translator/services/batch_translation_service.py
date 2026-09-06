from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from loguru import logger

from comic_translator.models.batch_result import BatchInput, BatchOutput, PageTranslation
from comic_translator.models.translation_block import TranslationBlock

if TYPE_CHECKING:
    from comic_translator.clients.base import LLMClient
    from comic_translator.services.batch_cache import BatchCacheService
    from comic_translator.services.batch_ocr_service import BatchOCRService
    from comic_translator.services.batch_prompt_builder import BatchPromptBuilder
    from comic_translator.services.glossary_service import GlossaryService


class BatchTranslationService:
    """Batch 翻译编排 —— 核心 orchestrator

    流程:
    1. BatchOCRService.ocr_all() → 全部页面的 OCR 结果
    2. 按 batch_size 切分
    3. 逐 Batch:
       a. 检查 BatchCache 命中 → skip
       b. 从 GlossaryService 读取已有 Glossary
       c. BatchPromptBuilder.build_user_prompt()
       d. LLMClient.generate()
       e. BatchPromptBuilder.parse_response()
       f. 提取 Glossary 写入 GlossaryService
       g. 翻译结果写入 TranslatedPage 表
       h. 写入 BatchCache
    4. 全部完成后合并输出
    """

    def __init__(
        self,
        llm_client: LLMClient,
        ocr_service: BatchOCRService,
        prompt_builder: BatchPromptBuilder,
        cache: BatchCacheService,
        glossary_service: GlossaryService,
        db_path: str = "data/output/project.db",
    ) -> None:
        self._llm = llm_client
        self._ocr = ocr_service
        self._prompt_builder = prompt_builder
        self._cache = cache
        self._glossary = glossary_service
        self._db_path = db_path

    def _paginate(
        self, pages: list, batch_size: int
    ) -> list[tuple[int, list]]:
        """按 batch_size 切分页面列表"""
        batches = []
        for i in range(0, len(pages), batch_size):
            batch_idx = i // batch_size + 1
            batches.append((batch_idx, pages[i : i + batch_size]))
        return batches

    def _load_glossary(self) -> list:
        """从 GlossaryService 加载当前所有词条"""
        return self._glossary.get_all_entries()

    def _save_glossary(self, glossary: dict[str, str], category: str = "general") -> None:
        """将 Batch 提取的 Glossary 写入 SQLite"""
        from comic_translator.models.glossary import GlossaryEntry

        for source, target in glossary.items():
            existing = self._glossary.search_terms(source)
            if not existing:
                self._glossary.add_term(GlossaryEntry(
                    source=source, target=target, category=category,
                ))
                logger.debug(f"Glossary: {source} -> {target}")

    def _save_page_to_db(
        self,
        project_id: str,
        batch_index: int,
        page_num: int,
        page_translation: PageTranslation,
        ocr_source: list,
    ) -> None:
        """将翻译结果持久化到 translated_pages 表"""
        from sqlalchemy import select

        from comic_translator.storage.database import get_engine, get_session
        from comic_translator.storage.models import TranslatedPageORM

        page_data = json.dumps(
            [b.model_dump(mode="json") for b in page_translation.blocks],
            ensure_ascii=False,
        )
        raw_source = json.dumps(
            [b.model_dump(mode="json") for b in ocr_source],
            ensure_ascii=False,
        )

        get_engine(self._db_path)
        session_generator = get_session(self._db_path)
        session = next(session_generator)
        try:
            existing = session.execute(
                select(TranslatedPageORM).where(
                    TranslatedPageORM.project_id == project_id,
                    TranslatedPageORM.page == page_num,
                )
            ).scalar_one_or_none()

            if existing:
                existing.page_data = page_data
                existing.raw_source = raw_source
                existing.status = "completed"
            else:
                session.add(TranslatedPageORM(
                    project_id=project_id,
                    batch_index=batch_index,
                    page=page_num,
                    page_data=page_data,
                    raw_source=raw_source,
                    status="completed",
                ))
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"DB save failed for page {page_num}: {e}")
        finally:
            session.close()

    async def run(
        self,
        project_name: str,
        comic_dir: str = "data/comics",
        batch_size: int = 20,
        source_lang: str = "en",
        target_lang: str = "zh",
        comic_style: str = "western_comic",
        use_cache: bool = True,
        batch_delay: float = 0.0,
        start_page: int = 1,
        end_page: int | None = None,
    ) -> list[BatchOutput]:
        """运行完整 Batch 翻译管线

        Args:
            batch_delay: 每个批次 API 调用之间的间隔秒数。默认 0（不人为等待，
                节奏由 LLM 客户端的令牌桶限速 + 429 退避控制）；仅在确实需要
                规避账号级限流（429 AccountRateLimitExceeded）时手动调大。
            start_page/end_page: 页码范围（1 基、含端点），与 CLI --start/--end 对应。
        """

        logger.info("=== Batch Translation Pipeline ===")
        logger.info(f"Project: {project_name}, Batch size: {batch_size}")

        # Step 1: OCR all pages
        all_pages = await self._ocr.process(
            project_id=project_name,
            comic_dir=comic_dir,
            source_lang=source_lang,
            use_cache=use_cache,
        )

        # 页码范围过滤（CLI --start/--end）
        if start_page > 1 or end_page is not None:
            before = len(all_pages)
            all_pages = [
                p for p in all_pages
                if p.page >= start_page and (end_page is None or p.page <= end_page)
            ]
            logger.info(
                f"Page range filter [{start_page}, {end_page or '∞'}]: "
                f"{before} -> {len(all_pages)} pages"
            )

        if not all_pages:
            logger.error("No pages to translate")
            return []

        logger.info(f"Total pages: {len(all_pages)}, Total blocks: {sum(len(p.blocks) for p in all_pages)}")

        # Step 2: Paginate
        batches = self._paginate(all_pages, batch_size)
        logger.info(f"Split into {len(batches)} batches")

        batch_outputs: list[BatchOutput] = []

        for batch_index, batch_pages in batches:
            logger.info(f"\n{'=' * 60}")
            logger.info(f"Batch {batch_index}/{len(batches)} ({len(batch_pages)} pages)")
            logger.info(f"{'=' * 60}")

            # Cache check
            if use_cache and self._cache.has_batch_cache(project_name, batch_index):
                cached = self._cache.get_batch_output(project_name, batch_index)
                if cached:
                    logger.info("Batch cache hit")
                    batch_outputs.append(BatchOutput(**cached))
                    continue

            # Load existing glossary
            glossary_entries = self._load_glossary()
            logger.info(f"Glossary entries: {len(glossary_entries)}")

            # Build input
            batch_input = BatchInput(
                batch_index=batch_index,
                pages=batch_pages,
                glossary=glossary_entries,
                project_name=project_name,
                comic_style=comic_style,
                source_language=source_lang,
                target_language=target_lang,
            )

            # Build prompt
            system_prompt = self._prompt_builder.SYSTEM_PROMPT
            user_prompt = self._prompt_builder.build_user_prompt(batch_input)

            # API call
            logger.info(f"Calling API for batch {batch_index} ({len(batch_pages)} pages)...")
            response = await self._llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
                temperature=0.0,
                max_tokens=8192,
            )
            logger.info(f"API response received ({len(response)} chars)")

            # Parse response
            batch_output = self._prompt_builder.parse_response(response, batch_input)
            batch_outputs.append(batch_output)

            # Save glossary
            if batch_output.glossary:
                self._save_glossary(batch_output.glossary)
                logger.info(f"Extracted {len(batch_output.glossary)} glossary entries")

            # Save to DB and cache
            for pt in batch_output.pages:
                matching_page = next(
                    (p for p in batch_pages if p.page == pt.page), None
                )
                ocr_source = matching_page.blocks if matching_page else []
                self._save_page_to_db(project_name, batch_index, pt.page, pt, ocr_source)

            if use_cache:
                self._cache.set_batch_output(project_name, batch_index, batch_output.model_dump(mode="json"))
                logger.info("Batch cache saved")

            # 批次间间隔：避免连续请求触发账号级限流
            if batch_index < len(batches) and batch_delay > 0:
                logger.debug(f"Waiting {batch_delay:.1f}s before next batch...")
                await asyncio.sleep(batch_delay)

        # Summary
        total_blocks = sum(
            len(pt.blocks) for bo in batch_outputs for pt in bo.pages
        )
        total_glossary = self._glossary.count()
        logger.info(f"\n{'=' * 60}")
        logger.info("Pipeline complete!")
        logger.info(f"Batches: {len(batch_outputs)}")
        logger.info(f"Pages: {len(all_pages)}")
        logger.info(f"Blocks translated: {total_blocks}")
        logger.info(f"Glossary entries: {total_glossary}")
        logger.info(f"{'=' * 60}")

        return batch_outputs

    def get_translated_blocks(self, project_id: str) -> list[tuple[int, list[TranslationBlock]]]:
        """从 DB 读取翻译结果（用于最终输出/渲染）"""
        from sqlalchemy import select

        from comic_translator.storage.database import get_session
        from comic_translator.storage.models import TranslatedPageORM

        result = []
        with get_session(self._db_path) as session:
            rows = session.execute(
                select(TranslatedPageORM)
                .where(TranslatedPageORM.project_id == project_id)
                .order_by(TranslatedPageORM.page)
            ).scalars().all()

            for row in rows:
                blocks_data = json.loads(row.page_data or "[]")
                blocks = [TranslationBlock(**b) for b in blocks_data]
                result.append((row.page, blocks))
        return result
