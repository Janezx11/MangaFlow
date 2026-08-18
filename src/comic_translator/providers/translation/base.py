from abc import ABC, abstractmethod
from comic_translator.models.translation_block import TranslationBlock
from comic_translator.models.context import TranslationContext


class TranslationProvider(ABC):
    """翻译能力抽象 —— 所有翻译引擎需实现此接口"""

    @abstractmethod
    async def translate(
        self,
        blocks: list[TranslationBlock],
        context: TranslationContext,
    ) -> list[TranslationBlock]:
        ...

    @abstractmethod
    async def translate_single(
        self,
        block: TranslationBlock,
        context: TranslationContext,
    ) -> TranslationBlock:
        ...