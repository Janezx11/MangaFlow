"""Mask Provider 接口 — 定义如何从漫画图生成文字擦除 mask"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class MaskProvider(ABC):
    """渲染质量 Sprint — Mask Pipeline 抽象接口

    输入: 原图 + 文本区域信息
    输出: 单通道 uint8 mask,255 表示需要擦除的区域
    """

    @abstractmethod
    def generate(
        self,
        image: Image.Image | np.ndarray,
        dialogue_blocks: list,
    ) -> np.ndarray:
        """生成文字擦除 mask

        Args:
            image: PIL Image 或 numpy array (RGB)
            dialogue_blocks: DialogueBlock 列表

        Returns:
            uint8 单通道 numpy array,形状 (H, W)
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Provider 名称"""
        ...

    @staticmethod
    def _to_numpy(image: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(image, np.ndarray):
            return image
        return np.array(image.convert("RGB"))
