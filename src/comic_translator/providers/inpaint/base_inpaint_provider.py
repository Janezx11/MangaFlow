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
            image: PIL Image (RGB) 或 numpy array (BGR —— 与 cv2.imread 一致)
            mask: uint8 单通道 numpy array,255 表示待修复区域

        Returns:
            numpy array,格式与输入 image 一致（ndarray 输入返回 BGR）
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
        """统一转成 BGR ndarray。

        约定：ndarray 输入已经是 BGR（调用方都来自 cv2.imread / COLOR_RGB2BGR
        之后的管线），直接返回副本；PIL 输入是 RGB，转 BGR。
        """
        if isinstance(image, np.ndarray):
            return image.copy()
        return cv2.cvtColor(np.array(image.convert("RGB")), cv2.COLOR_RGB2BGR)


# 延迟导入 cv2，避免基类 import 时强依赖
import cv2
