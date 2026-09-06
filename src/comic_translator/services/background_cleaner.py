"""Background Cleaner —— 擦除 OCR 区域，为中文文本留出空白"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from loguru import logger

if TYPE_CHECKING:
    from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock
    from comic_translator.services.inpaint_service import InpaintService


class BackgroundCleaner:
    """背景擦除 —— 使用 OpenCV inpainting 去除原文

    优先: cv2.INPAINT_TELEA
    备用: cv2.INPAINT_NS

    未来预留: LaMa / SDXL Inpaint
    """

    def __init__(self, inpaint_service: InpaintService, method: int = cv2.INPAINT_TELEA) -> None:
        self._inpaint = inpaint_service
        self._method = method
        self._fallback = cv2.INPAINT_NS

    def clean_background(
        self,
        image: np.ndarray,
        blocks: list[NormalizedOCRBlock],
        keep_mask: bool = False,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """擦除所有 OCR block 区域

        Args:
            image: BGR numpy array
            blocks: OCR blocks to erase
            keep_mask: 是否同时返回 mask

        Returns:
            擦除后的图像 (或 (图像, mask) 元组)
        """
        mask = self._inpaint.create_mask(blocks, image)
        if mask.sum() == 0:
            logger.debug("No blocks to inpaint")
            return (image.copy(), mask) if keep_mask else image.copy()

        try:
            result = cv2.inpaint(image, mask, inpaintRadius=3, flags=self._method)
            logger.debug(f"Inpainted {len(blocks)} blocks using TELEA")
        except Exception as e:
            logger.warning(f"TELEA failed ({e}), falling back to NS")
            result = cv2.inpaint(image, mask, inpaintRadius=3, flags=self._fallback)

        if keep_mask:
            return result, mask
        return result

    def clean_block(
        self,
        image: np.ndarray,
        block: NormalizedOCRBlock,
    ) -> np.ndarray:
        """擦除单个 block 区域"""
        mask = self._inpaint.create_block_mask(block, image)
        if mask.sum() == 0:
            return image.copy()
        return cv2.inpaint(image, mask, inpaintRadius=3, flags=self._method)
