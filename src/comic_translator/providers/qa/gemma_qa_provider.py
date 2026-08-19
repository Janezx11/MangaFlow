from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from comic_translator.clients.base import LLMClient
from comic_translator.providers.qa.base import QAProvider

if TYPE_CHECKING:
    from comic_translator.models.translation_block import TranslationBlock
    from comic_translator.models.context import TranslationContext, QAReport


class GemmaQAProvider(QAProvider):
    """⚠️ 占位实现（未完成）——`check()` 会抛 NotImplementedError。

    当前质检走 render_qa_service / typography_qa 的规则校验，本 Provider
    为 DI 容器预留的 LLM 质检接口；需要时在 `check()` 中接入
    `self._client.generate()` 即可。
    """

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