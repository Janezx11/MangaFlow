"""Mask Refinement Service — 对 Mask 进行后处理优化

优化项:
- 去除过大区域 (占图面积 > threshold)
- 平滑边缘 (morphology close/open)
- 连通区域过滤 (面积过滤)
- 保留气泡边界 (可选): 与 bubble polygon 做交集避免超出气泡
"""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger


class MaskRefinementService:
    """Mask 后处理服务"""

    def __init__(
        self,
        enabled: bool = True,
        max_area_ratio: float = 0.45,
        min_area: int = 80,
        morph_close: int = 3,
        morph_open: int = 3,
        preserve_bubble_boundary: bool = True,
    ):
        self.enabled = enabled
        self.max_area_ratio = max_area_ratio
        self.min_area = min_area
        self.morph_close = morph_close
        self.morph_open = morph_open
        self.preserve_bubble_boundary = preserve_bubble_boundary

    def refine(
        self,
        mask: np.ndarray,
        image_shape: tuple[int, int],
        bubble_polygons: list[list[tuple[float, float]]] | None = None,
    ) -> np.ndarray:
        """优化 mask

        Args:
            mask: uint8 单通道 numpy array
            image_shape: (H, W)
            bubble_polygons: 可选，用于限制 mask 不超出气泡边界

        Returns:
            优化后的 uint8 mask
        """
        if not self.enabled:
            return mask.copy()

        if mask.dtype != np.uint8:
            mask = (mask > 0).astype(np.uint8) * 255

        h, w = image_shape
        refined = mask.copy()

        # 1. 形态学闭运算填充小孔 + 开运算去噪
        if self.morph_close > 0:
            kc = max(1, self.morph_close)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kc, kc))
            refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel)
        if self.morph_open > 0:
            ko = max(1, self.morph_open)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ko, ko))
            refined = cv2.morphologyEx(refined, cv2.MORPH_OPEN, kernel)

        # 2. 连通区域过滤
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            (refined > 0).astype(np.uint8), connectivity=8,
        )
        total_area = h * w
        cleaned = np.zeros_like(refined)
        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            area_ratio = area / total_area
            if area_ratio > self.max_area_ratio:
                logger.debug(f"Refinement: drop component area={area}, ratio={area_ratio:.2f}")
                continue
            if area < self.min_area:
                continue
            cleaned[labels == i] = 255

        refined = cleaned

        # 3. 保留气泡边界
        if self.preserve_bubble_boundary and bubble_polygons:
            boundary_mask = np.zeros((h, w), dtype=np.uint8)
            for poly in bubble_polygons:
                if len(poly) < 3:
                    continue
                pts = np.array(poly, dtype=np.int32).reshape(-1, 1, 2)
                cv2.fillPoly(boundary_mask, [pts], 255)
            refined = cv2.bitwise_and(refined, boundary_mask)

        return refined
