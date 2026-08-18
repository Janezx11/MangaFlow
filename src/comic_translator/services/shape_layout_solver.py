"""ShapeLayoutSolver — 根据 Bubble Polygon 计算每行最大宽度

使用多边形与水平扫描线的交点，确定每行在气泡内的可用宽度。
这样文字布局可以贴合气泡形状，而不是矩形。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from comic_translator.services.bubble_layout_service import BubbleLayoutService

if TYPE_CHECKING:
    pass


class ShapeLayoutSolver:
    """形状感知的布局计算

    对于每一行文本，计算其在 bubble polygon Y 位置的最大可用宽度。
    """

    def __init__(self, bubble_layout: BubbleLayoutService) -> None:
        self._layout = bubble_layout

    def per_line_max_widths(
        self,
        polygon: list,
        start_y: int,
        line_count: int,
        line_height: int,
        image_shape: tuple[int, int],
        padding_x: int = 0,
    ) -> list[int]:
        """计算每行可用宽度

        Args:
            polygon: 气泡多边形顶点
            start_y: 起始 Y 坐标
            line_count: 行数
            line_height: 行高（含行间距）
            image_shape: (h, w)
            padding_x: 水平 padding

        Returns:
            list[int] 每行可用宽度
        """
        widths: list[int] = []
        for i in range(line_count):
            y = start_y + i * line_height
            w = self._layout.line_width_at_y(polygon, y, image_shape)
            if w is None:
                # fallback: 用 bounding rect 宽度
                x1, _, x2, _ = self._layout.max_inscribed_rect(polygon)
                w = x2 - x1
            widths.append(max(w - padding_x * 2, 20))
        return widths

    def shape_constrained_width(
        self,
        polygon: list,
        y_center: int,
        image_shape: tuple[int, int],
        padding_x: int = 0,
    ) -> int:
        """计算 polyon 在 y_center 处的宽度"""
        w = self._layout.line_width_at_y(polygon, y_center, image_shape)
        if w is None:
            x1, _, x2, _ = self._layout.max_inscribed_rect(polygon)
            w = x2 - x1
        return max(w - padding_x * 2, 20)