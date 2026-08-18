"""Mask Generation V2 —— 只 mask 文本区域，保留气泡背景"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock


class InpaintService:
    """生成 inpaint mask —— V2: 只 mask 文本行区域而非整个 bbox

    策略：取 bbox 中间 40-70% 高度区域作为文本行区域。
    这样只擦除文字，保留气泡边框和背景纹理。
    """

    def __init__(self, padding: int = 4, text_ratio: float = 0.45) -> None:
        self._padding = padding
        self._text_ratio = text_ratio  # 文本区域占 bbox 高度的比例

    def create_mask(
        self,
        blocks: list[NormalizedOCRBlock],
        image: np.ndarray,
    ) -> np.ndarray:
        """为所有 OCR block 生成文本区域 mask"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for block in blocks:
            if not block.text.strip():
                continue
            x1, y1, x2, y2 = map(int, block.bbox)
            block_h = y2 - y1
            if block_h <= 0:
                continue
            # 文本区域：bbox 中间 _text_ratio 高度
            text_h = max(int(block_h * self._text_ratio), 8)
            text_y1 = y1 + (block_h - text_h) // 2
            text_y2 = text_y1 + text_h
            # 添加 padding
            x1 = max(0, x1 - self._padding)
            text_y1 = max(0, text_y1 - self._padding)
            x2 = min(w - 1, x2 + self._padding)
            text_y2 = min(h - 1, text_y2 + self._padding)
            mask[text_y1:text_y2, x1:x2] = 255
        return mask

    def create_block_mask(
        self,
        block: NormalizedOCRBlock,
        image: np.ndarray,
    ) -> np.ndarray:
        """为单个 block 生成文本区域 mask"""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        x1, y1, x2, y2 = map(int, block.bbox)
        block_h = y2 - y1
        if block_h <= 0:
            return mask
        text_h = max(int(block_h * self._text_ratio), 8)
        text_y1 = y1 + (block_h - text_h) // 2
        text_y2 = text_y1 + text_h
        x1 = max(0, x1 - self._padding)
        text_y1 = max(0, text_y1 - self._padding)
        x2 = min(w - 1, x2 + self._padding)
        text_y2 = min(h - 1, text_y2 + self._padding)
        mask[text_y1:text_y2, x1:x2] = 255
        return mask