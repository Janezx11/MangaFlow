"""OCR Mask Provider — 基于 OCR bbox 生成文字 mask

策略:
- 不直接使用整个 bbox，而是取 bbox 中心文本行区域
- 可配置 padding
- 支持边缘平滑（高斯模糊 + 阈值）
- 输出 binary mask
"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from comic_translator.services.mask.base_mask_provider import MaskProvider


class OCRMaskProvider(MaskProvider):
    """基于 OCR block 的 mask 生成器

    改进：
    - 使用完整 OCR bbox 高度（而非裁切到中间 50%）
    - 较大 padding 确保文字完全覆盖
    - 多余部分由 MaskRefinementService 清理
    """

    def __init__(
        self,
        padding: int = 8,
        text_ratio: float = 1.0,
        smooth_kernel: int = 5,
    ):
        self.padding = padding
        self.text_ratio = text_ratio
        self.smooth_kernel = smooth_kernel

    def get_name(self) -> str:
        return "ocr_mask"

    def generate(
        self,
        image: Image.Image | np.ndarray,
        dialogue_blocks: list,
    ) -> np.ndarray:
        img_np = self._to_numpy(image)
        h, w = img_np.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        for db in dialogue_blocks:
            for block in getattr(db, "ocr_blocks", []) or []:
                x1, y1, x2, y2 = map(int, block.bbox)
                block_h = y2 - y1
                if block_h <= 0:
                    continue

                # 使用完整 bbox 高度（text_ratio 控制，默认 1.0 = 完整）
                text_h = max(int(block_h * self.text_ratio), 8)
                text_y1 = y1 + (block_h - text_h) // 2
                text_y2 = text_y1 + text_h

                # padding（比原来翻倍，确保文字完全覆盖）
                pad = max(self.padding, 8)
                x1 = max(0, x1 - pad)
                text_y1 = max(0, text_y1 - pad)
                x2 = min(w, x2 + pad)
                text_y2 = min(h, text_y2 + pad)

                if x2 > x1 and text_y2 > text_y1:
                    mask[text_y1:text_y2, x1:x2] = 255

        # 边缘平滑
        if self.smooth_kernel > 0:
            k = max(3, self.smooth_kernel | 1)  # 确保奇数
            mask = cv2.GaussianBlur(mask, (k, k), 0)
            _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        return mask
