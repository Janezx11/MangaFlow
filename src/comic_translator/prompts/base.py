from abc import ABC, abstractmethod
from comic_translator.models.translation_block import TranslationBlock
from comic_translator.models.context import TranslationContext


class PromptBuilder(ABC):
    """Prompt 构建抽象 —— 按漫画风格切换

    禁止把 Prompt 写死在 Provider 中。
    新增漫画风格时只需新增 PromptBuilder 子类，修改 config 中 comic_style。
    """

    @abstractmethod
    def build_translation_prompt(
        self,
        block: TranslationBlock,
        context: TranslationContext,
    ) -> str:
        ...

    @abstractmethod
    def build_qa_prompt(
        self,
        translations: list[TranslationBlock],
        context: TranslationContext,
    ) -> str:
        ...

    @abstractmethod
    def build_glossary_extraction_prompt(
        self,
        text: str,
        context: TranslationContext,
    ) -> str:
        ...

    @abstractmethod
    def build_summary_prompt(
        self,
        blocks: list[TranslationBlock],
        context: TranslationContext,
    ) -> str:
        """生成页面摘要的 Prompt

        Sprint 3: 用于 PageSummaryService 生成跨页上下文。
        """
        ...