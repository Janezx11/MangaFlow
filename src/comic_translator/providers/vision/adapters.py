"""现有 Provider → VisionDetectionProvider 适配器

将 CTBDTextDetectorProvider 和 YOLO11BubbleProvider 适配为统一接口，
不修改原有 Provider 代码。
"""

from __future__ import annotations

from PIL import Image

from comic_translator.models.detection_region import DetectionRegion
from comic_translator.providers.bubble.base import BubbleProvider
from comic_translator.providers.text_detection.base import TextDetectorProvider
from comic_translator.providers.vision.base_vision_provider import VisionDetectionProvider


class CTBDVisionAdapter(VisionDetectionProvider):
    """将 CTBDTextDetectorProvider 适配为 VisionDetectionProvider"""

    def __init__(self, ctbd: TextDetectorProvider):
        self._ctbd = ctbd

    async def detect(self, image: Image.Image) -> list[DetectionRegion]:
        text_regions = await self._ctbd.detect(image)
        return [DetectionRegion.from_text_region(tr) for tr in text_regions]

    def get_model_name(self) -> str:
        return "CTBD (RT-DETR)"

    def is_available(self) -> bool:
        return True


class YOLOVisionAdapter(VisionDetectionProvider):
    """将 YOLO11BubbleProvider 适配为 VisionDetectionProvider"""

    def __init__(self, yolo: BubbleProvider):
        self._yolo = yolo

    async def detect(self, image: Image.Image) -> list[DetectionRegion]:
        bubbles = await self._yolo.detect(image)
        return [DetectionRegion.from_bubble(b) for b in bubbles]

    def get_model_name(self) -> str:
        return self._yolo.get_model_name()

    def is_available(self) -> bool:
        return True
