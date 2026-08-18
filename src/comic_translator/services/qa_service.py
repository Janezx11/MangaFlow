from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from comic_translator.models.translation_block import TranslationBlock
    from comic_translator.models.context import TranslationContext, QAReport
    from comic_translator.providers.qa.base import QAProvider
    from comic_translator.prompts.base import PromptBuilder


class QAService:
    """QA 检查服务"""

    def __init__(self, provider: QAProvider, prompt_builder: PromptBuilder) -> None:
        self._provider = provider
        self._prompt_builder = prompt_builder

    async def check_page(
        self,
        translations: list[TranslationBlock],
        context: TranslationContext,
    ) -> QAReport:
        logger.info(f"Running QA check on {len(translations)} translations")
        report = await self._provider.check(translations, context)

        # Update block-level QA flags
        if not report.passed:
            for block in translations:
                if block.id in report.issues:
                    block.is_qa_passed = False
                    block.qa_issues = [report.issues]

        return report