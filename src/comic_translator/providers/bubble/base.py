from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL import Image

    from comic_translator.models.bubble import Bubble


class BubbleProvider(ABC):
    """气泡检测抽象接口

    所有气泡检测模型（YOLO11-seg / RT-DETR / SAM2 等）实现此接口。
    切换模型时无需修改 Render / Translation / Matching。
    """

    @abstractmethod
    async def detect(self, image: Image.Image) -> list[Bubble]:
        """检测漫画页面中的所有气泡

        Args:
            image: PIL Image (RGB)

        Returns:
            按 y 坐标排序的 Bubble 列表
        """
        ...

    @abstractmethod
    def get_model_name(self) -> str:
        """返回当前使用的模型名称（用于日志）"""
        ...
