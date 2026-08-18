from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from comic_translator.providers.render.base import RenderProvider

if TYPE_CHECKING:
    from PIL import Image
    from comic_translator.models.translation_block import TranslationBlock
    from comic_translator.models.result import RenderResult


class OpenCVRenderProvider(RenderProvider):
    """OpenCV 渲染引擎 —— 将翻译文本回填到图片"""

    def __init__(
        self,
        font_path: str = "data/fonts/NotoSansSC-Regular.ttf",
        font_size: int = 16,
    ) -> None:
        self.font_path = font_path
        self.font_size = font_size

    async def render(
        self,
        image: Image.Image,
        blocks: list[TranslationBlock],
        output_path: str,
    ) -> RenderResult:
        # TODO: implement actual rendering
        raise NotImplementedError