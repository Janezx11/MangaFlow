"""Provider ABC 基类汇总 re-export"""

from comic_translator.providers.ocr.base import OCRProvider
from comic_translator.providers.qa.base import QAProvider
from comic_translator.providers.render.base import RenderProvider
from comic_translator.providers.translation.base import TranslationProvider

__all__ = [
    "OCRProvider",
    "QAProvider",
    "RenderProvider",
    "TranslationProvider",
]
