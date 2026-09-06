"""Text Detection providers package"""

from comic_translator.providers.text_detection.base import TextDetectorProvider
from comic_translator.providers.text_detection.ctbd_text_detector import CTBDTextDetectorProvider

__all__ = ["CTBDTextDetectorProvider", "TextDetectorProvider"]
