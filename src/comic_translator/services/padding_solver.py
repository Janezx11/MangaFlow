"""PaddingSolver — 动态 Padding（基于 SafeArea）

Padding 由 SafeArea 的 max_inscribed_rect 尺寸和 Font Size 共同决定。
气泡越大 → padding 越大。
字号越小 → padding 越小。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comic_translator.models.typography_models import SafeArea


@dataclass
class Padding:
    x: int
    y: int


class PaddingSolver:
    """动态 Padding 计算（基于 SafeArea）"""

    AREA_SMALL = 10000
    AREA_MEDIUM = 50000

    def compute(self, safe_area: SafeArea, font_size: int) -> Padding:
        """根据 SafeArea 的计算 safe padding

        Args:
            safe_area: 气泡安全区域
            font_size: 字号

        Returns:
            Padding(x, y)
        """
        x1, y1, x2, y2 = safe_area.max_inscribed_rect
        w = x2 - x1
        h = y2 - y1
        area = w * h

        if area < self.AREA_SMALL:
            base_pad = 6
        elif area < self.AREA_MEDIUM:
            base_pad = 10
        else:
            base_pad = 16

        font_scale = max(0.7, min(1.3, font_size / 24))
        px = max(4, int(base_pad * font_scale))
        py = max(3, int(base_pad * 0.6 * font_scale))

        return Padding(x=px, y=py)
