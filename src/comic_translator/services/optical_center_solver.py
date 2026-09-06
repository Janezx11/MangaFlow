"""OpticalCenterSolver — 视觉中心计算（基于 SafeArea）

视觉中心略高于几何中心（约 5-10%）。
使用 SafeArea 的 max_inscribed_rect，而非 raw bubble bbox。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comic_translator.models.typography_models import SafeArea


class OpticalCenterSolver:
    """视觉中心求解器（基于 SafeArea）"""

    CENTER_FACTOR = 0.42

    def solve(
        self,
        safe_area: SafeArea,
        total_height: int,
    ) -> int:
        """计算视觉中心 Y 坐标

        Args:
            safe_area: 气泡安全区域
            total_height: 文本块总高度

        Returns:
            start_y: 文本起始 Y 坐标
        """
        _, y1, _, y2 = safe_area.max_inscribed_rect
        box_h = y2 - y1

        if total_height >= box_h:
            return y1 + 2

        offset = int((box_h - total_height) * self.CENTER_FACTOR)
        return y1 + offset

    def get_optical_center(self, safe_area: SafeArea) -> tuple[float, float]:
        """返回视觉中心点"""
        x1, y1, x2, y2 = safe_area.max_inscribed_rect
        cx = (x1 + x2) / 2
        box_h = y2 - y1
        cy = y1 + box_h * self.CENTER_FACTOR
        return (cx, cy)
