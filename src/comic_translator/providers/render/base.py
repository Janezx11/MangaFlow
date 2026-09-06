from abc import ABC, abstractmethod

from PIL import Image

from comic_translator.models.result import RenderResult
from comic_translator.models.translation_block import TranslationBlock


class RenderProvider(ABC):
    """渲染能力抽象 —— 将翻译结果回填到图片"""

    @abstractmethod
    async def render(
        self,
        image: Image.Image,
        blocks: list[TranslationBlock],
        output_path: str,
    ) -> RenderResult:
        ...
