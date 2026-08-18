from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from comic_translator.clients.base import LLMClient
from comic_translator.providers.qa.base import QAProvider

if TYPE_CHECKING:
    from comic_translator.models.translation_block import TranslationBlock
    from comic_translator.models.context import TranslationContext, QAReport


class GemmaQAProvider(QAProvider):
    """Gemma QA 检查引擎 —— 通过 LLMClient 调用"""

    def __init__(self, client: LLMClient, model: str = "gemma4:e4b") -> None:
        self._client = client
        self._model = model

    async def check(
        self,
        translations: list[TranslationBlock],
        context: TranslationContext,
    ) -> QAReport:
        # TODO: implement QA check via self._client.generate()
        raise NotImplementedError