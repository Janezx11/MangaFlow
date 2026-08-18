"""Inpaint Provider 接口 — 定义如何修复被 mask 的区域"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
from PIL import Image


class InpaintProvider(ABC):
    """渲染质量 Sprint — Inpaint Pipeline 抽象接口"""

    @abstractmethod
    def inpaint(
        self,
        image: Image.Image | np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        """修复被 mask 标记的区域

        Args:
            image: PIL Image 或 numpy array (RGB/BGR 均可)
            mask: uint8 单通道 numpy array,255 表示待修复区域

        Returns:
            numpy array,格式与输入 image 一致
        """
        ...

    @abstractmethod
    def get_name(self) -> str:
        """Provider 名称"""
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """当前环境是否可用"""
        ...

    @staticmethod
    def _to_numpy(image: Image.Image | np.ndarray, rgb: bool = True) -> np.ndarray:
        if isinstance(image, np.ndarray):
            arr = image.copy()
        else:
            arr = np.array(image.convert("RGB"))
        if not rgb and len(arr.shape) == 3:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr

    @staticmethod
    def _to_bgr(image: Image.Image | np.ndarray) -> np.ndarray:
        arr = InpaintProvider._to_numpy(image, rgb=True)
        if len(arr.shape) == 3:
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        return arr


# 延迟导入 cv2，避免基类 import 时强依赖
import cv2  # noqa: E402
