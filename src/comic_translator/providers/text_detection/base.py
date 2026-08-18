"""Text Detection Provider 抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from comic_translator.models.text_region import TextRegion


class TextDetectorProvider(ABC):
    """文本检测能力抽象 —— 所有 Text Detector 引擎需实现此接口

    与 BubbleProvider 的区别:
      - BubbleProvider 检测气泡（图形区域）
      - TextDetectorProvider 检测文字区域（文本内容区域）
      - Text Detection 是主要入口，Bubble Detection 是辅助信息
    """

    @abstractmethod
    async def detect(self, image: Image.Image) -> list[TextRegion]:
        """检测图片中的文本区域

        Args:
            image: PIL Image

        Returns:
            检测到的 TextRegion 列表
        """
        ...

    async def detect_batch(
        self, images: list[Image.Image]
    ) -> list[list[TextRegion]]:
        """批量检测（默认串行实现）"""
        return [await self.detect(img) for img in images]
