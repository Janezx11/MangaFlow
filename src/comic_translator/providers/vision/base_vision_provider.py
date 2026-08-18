"""VisionDetectionProvider — 视觉检测统一接口"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from comic_translator.models.detection_region import DetectionRegion


class VisionDetectionProvider(ABC):
    """视觉检测能力抽象 —— 所有视觉检测引擎需实现此接口

    统一接口，支持：
      - CTBD（文本检测）
      - YOLO（气泡检测）
      - LocateAnything3B（视觉语言模型检测）
    """

    @abstractmethod
    async def detect(self, image: Image.Image) -> list[DetectionRegion]:
        """检测图片中的区域

        Args:
            image: PIL Image

        Returns:
            检测到的 DetectionRegion 列表
        """
        ...

    async def detect_batch(
        self, images: list[Image.Image]
    ) -> list[list[DetectionRegion]]:
        """批量检测（默认串行实现）"""
        return [await self.detect(img) for img in images]

    @abstractmethod
    def get_model_name(self) -> str:
        """返回模型名称"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """检查模型是否可用（依赖/GPU/模型文件）"""
        ...
