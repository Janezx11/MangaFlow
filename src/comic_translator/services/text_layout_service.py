"""Layout Engine V2 —— 自动换行（支持中文）、字号搜索、垂直居中"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from PIL import ImageFont

from comic_translator.models.render_models import TextLayout

if TYPE_CHECKING:
    from comic_translator.models.render_models import FontProfile


class TextLayoutService:
    """文本布局引擎 V2

    1. 支持中文换行（按字符拆分）
    2. 从大到小遍历字号，先换行再判断高度
    3. 垂直居中
    """

    def __init__(self, padding_x: int = 8, padding_y: int = 4) -> None:
        self._pad_x = padding_x
        self._pad_y = padding_y

    @lru_cache(maxsize=512)
    def _measure_text(self, text: str, font_path: str, size: int) -> tuple[int, int]:
        """缓存文本测量结果"""
        try:
            font = ImageFont.truetype(font_path, size)
        except Exception:
            font = ImageFont.load_default()
        bbox = font.getbbox(text)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        return (w, h)

    def _wrap_text(self, text: str, max_width: int, font_path: str, size: int) -> list[str]:
        """自动换行：先尝试按单词拆分，长词/中文按字符拆分"""
        char_count = sum(1 for c in text if ord(c) > 0x2E80)  # CJK 字符
        if char_count > len(text) * 0.3:
            return self._wrap_cjk(text, max_width, font_path, size)
        return self._wrap_latin(text, max_width, font_path, size)

    def _wrap_latin(self, text: str, max_width: int, font_path: str, size: int) -> list[str]:
        """英文按单词换行"""
        words = text.split()
        if not words:
            return []
        lines: list[str] = []
        current = words[0]
        for word in words[1:]:
            test_w, _ = self._measure_text(current + " " + word, font_path, size)
            if test_w <= max_width:
                current += " " + word
            else:
                lines.append(current)
                current = word
        lines.append(current)
        return lines

    def _wrap_cjk(self, text: str, max_width: int, font_path: str, size: int) -> list[str]:
        """中文按字符拆分换行"""
        lines: list[str] = []
        current = ""
        for char in text:
            test_w, _ = self._measure_text(current + char, font_path, size)
            if test_w <= max_width:
                current += char
            else:
                if current:
                    lines.append(current)
                current = char
        if current:
            lines.append(current)
        return lines

    def find_best_font_size(
        self,
        text: str,
        bbox: tuple[int, int, int, int],
        font_path: str,
        min_size: int,
        max_size: int,
    ) -> int:
        """从大到小遍历字号，先换行再判断是否放下"""
        x1, y1, x2, y2 = bbox
        box_w = (x2 - x1) - self._pad_x * 2
        box_h = (y2 - y1) - self._pad_y * 2
        if box_w <= 0 or box_h <= 0:
            return min_size

        for size in range(max_size, min_size - 1, -1):
            lines = self._wrap_text(text, box_w, font_path, size)
            if not lines:
                continue
            line_h = self._measure_text("A", font_path, size)[1]
            if line_h <= 0:
                continue
            spacing = int(line_h * 1.2)  # use fixed 1.2 for measurement
            total_h = len(lines) * spacing
            if total_h <= box_h:
                return size

        return min_size

    def layout(
        self,
        text: str,
        bbox: tuple[int, int, int, int],
        font_profile: FontProfile,
    ) -> TextLayout:
        """对文本进行完整布局"""
        size = self.find_best_font_size(
            text, bbox, font_profile.font_path,
            font_profile.min_size, font_profile.max_size,
        )
        x1, _y1, x2, _y2 = bbox
        box_w = (x2 - x1) - self._pad_x * 2
        lines = self._wrap_text(text, box_w, font_profile.font_path, size)

        return TextLayout(
            text=text,
            bbox=bbox,
            font_size=size,
            line_count=len(lines),
            lines=lines,
        )

    def get_draw_positions(
        self,
        layout: TextLayout,
        font_profile: FontProfile,
    ) -> list[tuple[int, int]]:
        """计算每行绘制位置（水平居中 + 垂直居中）"""
        x1, y1, x2, y2 = layout.bbox
        box_h = y2 - y1
        line_h = self._measure_text("A", font_profile.font_path, layout.font_size)[1]
        spacing = int(line_h * font_profile.line_spacing)
        total_h = layout.line_count * spacing
        start_y = y1 + (box_h - total_h) // 2

        positions = []
        for line in layout.lines:
            w, _ = self._measure_text(line, font_profile.font_path, layout.font_size)
            cx = x1 + ((x2 - x1) - w) // 2
            positions.append((cx, start_y))
            start_y += spacing
        return positions
