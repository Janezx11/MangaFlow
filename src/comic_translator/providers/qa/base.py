from abc import ABC, abstractmethod
from comic_translator.models.translation_block import TranslationBlock
from comic_translator.models.context import TranslationContext, QAReport


class QAProvider(ABC):
    """QA 检查能力抽象"""

    @abstractmethod
    async def check(
        self,
        translations: list[TranslationBlock],
        context: TranslationContext,
    ) -> QAReport:
        ...