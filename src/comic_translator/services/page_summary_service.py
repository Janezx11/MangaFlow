from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from comic_translator.models.page_summary import PageSummary
    from comic_translator.models.translation_block import TranslationBlock
    from comic_translator.models.context import TranslationContext
    from comic_translator.clients.base import LLMClient
    from comic_translator.prompts.base import PromptBuilder


class PageSummaryService:
    """页面摘要服务

    职责:
    1. 翻译完成后生成页面摘要
    2. 保存到文件缓存
    3. 下一页翻译时加载最近 N 页摘要
    """

    def __init__(
        self,
        cache_dir: str = "data/cache",
        llm_client: LLMClient | None = None,
        prompt_builder: PromptBuilder | None = None,
    ) -> None:
        self._cache_dir = Path(cache_dir)
        self._llm_client = llm_client
        self._prompt_builder = prompt_builder

    async def generate_summary(
        self,
        blocks: list[TranslationBlock],
        context: TranslationContext,
        page: int,
        project_id: str = "",
    ) -> PageSummary:
        """生成页面摘要"""
        from comic_translator.models.page_summary import PageSummary

        # If LLM is available, generate a real summary
        if self._llm_client and self._prompt_builder:
            try:
                prompt = self._prompt_builder.build_summary_prompt(blocks, context)
                response = await self._llm_client.generate(prompt=prompt)
                summary = self._parse_summary_response(response, page)
            except Exception as e:
                logger.warning(f"LLM summary failed, using fallback: {e}")
                summary = self._fallback_summary(blocks, page)
        else:
            summary = self._fallback_summary(blocks, page)

        # Extract character names and locations from context
        summary.characters = self._extract_characters(context)
        summary.locations = self._extract_locations(context)

        # Save to cache
        if project_id:
            self._save_summary(project_id, summary)

        return summary

    def get_previous_summary(self, project_id: str, page: int) -> PageSummary | None:
        """获取上一页摘要"""
        if page <= 1:
            return None
        return self._load_summary(project_id, page - 1)

    def get_recent_summaries(
        self, project_id: str, current_page: int, count: int = 3
    ) -> list[PageSummary]:
        """获取最近 N 页摘要"""
        summaries: list[PageSummary] = []
        for page in range(max(1, current_page - count), current_page):
            s = self._load_summary(project_id, page)
            if s is not None:
                summaries.append(s)
        return summaries

    def summary_exists(self, project_id: str, page: int) -> bool:
        return self._summary_path(project_id, page).exists()

    # --- Internal ---

    def _summary_path(self, project_id: str, page: int) -> Path:
        return (
            self._cache_dir
            / project_id
            / "summaries"
            / f"page_{page:06d}_summary.json"
        )

    def _save_summary(self, project_id: str, summary: PageSummary) -> None:
        path = self._summary_path(project_id, summary.page_number)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(summary.model_dump(), ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.debug(f"Summary saved: {path}")

    def _load_summary(self, project_id: str, page: int) -> PageSummary | None:
        from comic_translator.models.page_summary import PageSummary

        path = self._summary_path(project_id, page)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return PageSummary(**data)

    def _parse_summary_response(
        self, response: str, page: int
    ) -> PageSummary:
        from comic_translator.models.page_summary import PageSummary

        # Try JSON first
        response = response.strip()
        try:
            data = json.loads(response)
            return PageSummary(
                page_number=page,
                summary=data.get("summary", response),
                characters=data.get("characters", []),
                locations=data.get("locations", []),
                events=data.get("events", []),
            )
        except json.JSONDecodeError:
            return PageSummary(
                page_number=page,
                summary=response,
            )

    def _fallback_summary(
        self, blocks: list[TranslationBlock], page: int
    ) -> PageSummary:
        from comic_translator.models.page_summary import PageSummary

        summary = "; ".join(
            f"{b.source_text} → {b.translated_text}"
            for b in blocks[:5]
            if b.translated_text
        )
        if len(blocks) > 5:
            summary += f" (... and {len(blocks) - 5} more)"
        return PageSummary(page_number=page, summary=summary[:500])

    def _extract_characters(self, context: TranslationContext) -> list[str]:
        chars = context.metadata.get("characters", [])
        if chars and isinstance(chars[0], dict):
            return [c.get("canonical_name", "") for c in chars if c.get("canonical_name")]
        return []

    def _extract_locations(self, context: TranslationContext) -> list[str]:
        glossary_locations = [
            g.source for g in context.glossary if g.category == "location"
        ]
        return glossary_locations