"""Vision Detection Provider 导出"""

from comic_translator.providers.vision.adapters import CTBDVisionAdapter, YOLOVisionAdapter
from comic_translator.providers.vision.base_vision_provider import VisionDetectionProvider
from comic_translator.providers.vision.locate_anything_provider import LocateAnythingProvider

__all__ = [
    "CTBDVisionAdapter",
    "LocateAnythingProvider",
    "VisionDetectionProvider",
    "YOLOVisionAdapter",
]
