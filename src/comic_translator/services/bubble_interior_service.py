"""BubbleInteriorService — 气泡内部安全区域计算

独立于排版引擎，仅负责从 Bubble 多边形和 mask 计算 SafeArea。

输入: Bubble (polygon, mask)
输出: SafeArea (safe_polygon, safe_bbox, max_inscribed_rect, center)

Safe area 向内缩进约 8-12%，确保文字不贴边。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from comic_translator.models.typography_models import SafeArea

if TYPE_CHECKING:
    from comic_translator.models.bubble import Bubble


class BubbleInteriorService:
    """气泡内部安全区域计算"""

    def compute(self, bubble: Bubble, image_shape: tuple[int, int]) -> SafeArea:
        """计算单个气泡的安全区域

        Args:
            bubble: 气泡（使用 polygon + mask）
            image_shape: (h, w)

        Returns:
            SafeArea
        """
        polygon = bubble.polygon
        mask = bubble.mask

        # 优先使用 mask
        if mask and len(mask) > 0:
            mask_np = self._mask_to_ndarray(mask)
            if mask_np.shape[:2] == image_shape[:2]:
                safe = self._compute_from_mask(mask_np, polygon)
                if safe is not None:
                    return safe

        # fallback: 从 polygon 计算
        if polygon and len(polygon) >= 3:
            return self._compute_from_polygon(polygon, image_shape)

        # 最后 fallback: bbox
        x1, y1, x2, y2 = bubble.bbox
        return SafeArea(
            polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
            bbox=(x1, y1, x2, y2),
            max_inscribed_rect=(x1, y1, x2, y2),
            center=(float((x1 + x2) // 2), float((y1 + y2) // 2)),
        )

    def compute_batch(
        self,
        bubbles: list[Bubble],
        image_shape: tuple[int, int],
    ) -> dict[str, SafeArea]:
        """批量计算所有气泡的 safe area"""
        return {b.id: self.compute(b, image_shape) for b in bubbles}

    def _mask_to_ndarray(self, mask: list) -> np.ndarray:
        """将 YOLO mask (list) 转换为 np.ndarray"""
        arr = np.array(mask, dtype=np.float32)
        if arr.ndim == 3:
            arr = arr.squeeze()
        return (arr > 0.5).astype(np.uint8) * 255

    def _compute_from_mask(self, mask: np.ndarray, polygon: list) -> SafeArea | None:
        """从 binary mask 计算 safe area"""
        _h, _w = mask.shape[:2]
        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

        # 向内缩进 10%
        max_dist = dist.max()
        if max_dist <= 0:
            return None
        max_dist * 0.9  # 保留 90% 的半径
        # 但至少 3px
        min_dist = max(3, int(max_dist * 0.1))

        safe_mask = (dist >= min_dist).astype(np.uint8) * 255
        contours, _ = cv2.findContours(safe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        cnt = max(contours, key=cv2.contourArea)
        safe_poly = [(int(pt[0][0]), int(pt[0][1])) for pt in cnt]

        x, y, w_box, h_box = cv2.boundingRect(cnt)
        safe_bbox = (x, y, x + w_box, y + h_box)
        max_inscribed = (x, y, x + w_box, y + h_box)

        # 中心：距离变换峰值
        _, _, _, max_loc = cv2.minMaxLoc(dist, mask=safe_mask)
        center = (float(max_loc[0]), float(max_loc[1]))

        return SafeArea(
            polygon=safe_poly,
            bbox=safe_bbox,
            max_inscribed_rect=max_inscribed,
            center=center,
        )

    def _compute_from_polygon(self, polygon: list, image_shape: tuple[int, int]) -> SafeArea:
        """从 polygon 计算 safe area"""
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)

        dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        max_dist = dist.max()
        if max_dist <= 0:
            x1, y1, x2, y2 = cv2.boundingRect(pts)
            return SafeArea(
                polygon=polygon,
                bbox=(x1, y1, x2, y2),
                max_inscribed_rect=(x1, y1, x2, y2),
                center=(float((x1 + x2) // 2), float((y1 + y2) // 2)),
            )

        min_dist = max(3, int(max_dist * 0.1))
        safe_mask = (dist >= min_dist).astype(np.uint8) * 255
        contours, _ = cv2.findContours(safe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            cnt = max(contours, key=cv2.contourArea)
            safe_poly = [(int(pt[0][0]), int(pt[0][1])) for pt in cnt]
            x, y, wb, hb = cv2.boundingRect(cnt)
        else:
            safe_poly = polygon
            cnt = pts
            x, y, wb, hb = cv2.boundingRect(cnt)

        safe_bbox = (x, y, x + wb, y + hb)
        max_inscribed = (x, y, x + wb, y + hb)

        _, _, _, max_loc = cv2.minMaxLoc(dist, mask=safe_mask)
        center = (float(max_loc[0]), float(max_loc[1]))

        return SafeArea(
            polygon=safe_poly,
            bbox=safe_bbox,
            max_inscribed_rect=max_inscribed,
            center=center,
        )
