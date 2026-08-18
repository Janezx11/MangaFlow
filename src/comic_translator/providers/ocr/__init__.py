"""OCR providers package"""

from comic_translator.providers.ocr.base import OCRProvider
from comic_translator.providers.ocr.easyocr_provider import EasyOCRProvider

__all__ = ["OCRProvider", "EasyOCRProvider"]