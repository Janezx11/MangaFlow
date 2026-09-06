"""BubbleLayoutService — 气泡几何计算

负责:
  - Max inscribed rectangle（从 polygon 计算最大内接矩形）
  - Bubble interior mask（rasterize polygon → binary mask）
  - Distance transform（到边界的距离）
  - Safe area（安全文本区域）
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from comic_translator.models.typography_models import SafeArea

if TYPE_CHECKING:
    pass


class BubbleLayoutService:
    """气泡几何计算"""

    def max_inscribed_rect(self, polygon: list) -> tuple[int, int, int, int]:
        """计算 polygon 的外接矩形（向后兼容，新代码请用 safe_area）"""
        if not polygon or len(polygon) < 3:
            return (0, 0, 0, 0)
        cnt = np.array(polygon, dtype=np.int32)
        x, y, w, h = cv2.boundingRect(cnt)
        return (x, y, x + w, y + h)

    def bubble_interior_mask(
        self, polygon: list, image_shape: tuple[int, int]
    ) -> np.ndarray:
        """将 polygon rasterize 为 binary mask"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        if len(polygon) < 3:
            return mask
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)
        return mask

    def distance_to_border(self, mask: np.ndarray) -> np.ndarray:
        """距离变换：每个像素到 mask 边界的距离"""
        return cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    def _max_inscribed_rect_in_mask(self, mask: np.ndarray) -> tuple[int, int, int, int]:
        """在 binary mask 中寻找面积最大的内接矩形

        扫描线算法：对每对行 (y1, y2)，维护该区间内的最小公共宽度，
        面积 = 宽度 × 高度，取最大值。

        Returns:
            (x1, y1, x2, y2) — 最大内接矩形
        """
        h, w = mask.shape[:2]

        # 每行的左右边界：left[y]=最左白像素, right[y]=最右白像素
        left = np.full(h, w, dtype=np.int32)
        right = np.full(h, -1, dtype=np.int32)
        for y in range(h):
            row = mask[y, :] > 0
            if row.any():
                indices = np.where(row)[0]
                left[y] = int(indices[0])
                right[y] = int(indices[-1])

        best_area = 0
        best_rect = (0, 0, 0, 0)

        for y1 in range(h):
            if left[y1] >= right[y1]:
                continue
            cur_left = int(left[y1])
            cur_right = int(right[y1])
            for y2 in range(y1, h):
                if left[y2] >= right[y2]:
                    break
                cur_left = max(cur_left, int(left[y2]))
                cur_right = min(cur_right, int(right[y2]))
                if cur_left >= cur_right:
                    break
                area = (cur_right - cur_left) * (y2 - y1 + 1)
                if area > best_area:
                    best_area = area
                    best_rect = (cur_left, y1, cur_right, y2 + 1)

        return best_rect

    def _auto_min_distance(self, mask: np.ndarray) -> int:
        """根据气泡尺寸自动计算 min_distance

        取距离变换最大值的 15%，但至少 4px，最多 35px。
        比原来(10%/3px/30px)留出更多边距，避免文字贴边。
        """
        dist = self.distance_to_border(mask)
        max_dist = dist.max()
        if max_dist <= 0:
            return 5
        return max(4, min(int(max_dist * 0.15), 35))

    def safe_area(
        self, polygon: list, image_shape: tuple[int, int],
        min_distance: int = 0,
        mask: np.ndarray | None = None,
    ) -> SafeArea:
        """计算气泡内安全文本区域

        Args:
            polygon: 气泡多边形顶点
            image_shape: (h, w)
            min_distance: 到边界的最小距离（px），0=自动计算
            mask: 可选预计算 mask（来自 YOLO Bubble.mask），
                  避免重复 rasterize

        Returns:
            SafeArea 包含安全区域多边形、bbox、最大内接矩形、中心
        """
        if mask is not None and isinstance(mask, np.ndarray) and mask.size > 0:
            interior_mask = mask
        elif len(polygon) >= 3:
            interior_mask = self.bubble_interior_mask(polygon, image_shape)
        else:
            return SafeArea(
                polygon=polygon,
                bbox=self.max_inscribed_rect(polygon),
                max_inscribed_rect=self.max_inscribed_rect(polygon),
                center=(0.0, 0.0),
            )

        # 自动计算 min_distance
        if min_distance <= 0:
            min_distance = self._auto_min_distance(interior_mask)

        dist = self.distance_to_border(interior_mask)

        # 提取距离 >= min_distance 的区域
        safe_mask = (dist >= min_distance).astype(np.uint8) * 255
        contours, _ = cv2.findContours(
            safe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            # fallback: 直接用内接矩形
            rect = self._max_inscribed_rect_in_mask(interior_mask)
            return SafeArea(
                polygon=polygon,
                bbox=rect,
                max_inscribed_rect=rect,
                center=(float((rect[0] + rect[2]) // 2), float((rect[1] + rect[3]) // 2)),
            )

        # 取最大轮廓
        cnt = max(contours, key=cv2.contourArea)
        safe_poly = [(int(pt[0][0]), int(pt[0][1])) for pt in cnt]

        # ★ 关键修复：在 safe mask 中找真正的最大内接矩形，而非 boundingRect
        x1, y1, x2, y2 = self._max_inscribed_rect_in_mask(safe_mask)

        # 计算中心（距离变换峰值，即离边界最远的点）
        _, max_val, _, max_loc = cv2.minMaxLoc(dist, mask=safe_mask)
        if max_val > 0:
            center = (float(max_loc[0]), float(max_loc[1]))
        else:
            center = (float((x1 + x2) // 2), float((y1 + y2) // 2))

        return SafeArea(
            polygon=safe_poly,
            bbox=self.max_inscribed_rect(polygon),
            max_inscribed_rect=(x1, y1, x2, y2),
            center=center,
        )

    def line_width_at_y(
        self, polygon: list, y: int, image_shape: tuple[int, int]
    ) -> int | None:
        """计算 polygon 在指定 y 水平线上的宽度（用于 shape-aware layout）"""
        if len(polygon) < 3:
            return None
        mask = self.bubble_interior_mask(polygon, image_shape)
        row = mask[y, :]
        indices = np.where(row > 0)[0]
        if len(indices) < 2:
            return None
        return int(indices[-1] - indices[0])
