"""TypographyServiceV2 — Comic Typography Engine Orchestrator

全部排版基于 Bubble Polygon 的 SafeArea（最大内接矩形 + 距离变换），
不再使用原始 bbox 做任何尺寸计算。

流程:
  Bubble Polygon
  ↓
  Safe Area (BubbleLayoutService: mask → distance transform → max inscribed rect)
  ↓
  Padding (PaddingSolver: based on safe area size + font size)
  ↓
  Font Size (FontSizeSolver: safe area dimensions + per-line polygon widths)
  ↓
  Balanced Wrap (BalancedLineBreaker: Knuth-Plass DP)
  ↓
  Optical Center (OpticalCenterSolver: safe area + text height → 42% offset)
  ↓
  Draw Positions (safe area center, not bbox center)
  ↓
  ParagraphLayout
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger

from comic_translator.models.typography_models import LineInfo, ParagraphLayout

if TYPE_CHECKING:
    from comic_translator.services.font_manager import FontManager
    from comic_translator.services.bubble_layout_service import BubbleLayoutService
    from comic_translator.services.shape_layout_solver import ShapeLayoutSolver
    from comic_translator.services.balanced_line_breaker import BalancedLineBreaker
    from comic_translator.services.padding_solver import PaddingSolver
    from comic_translator.services.font_size_solver import FontSizeSolver
    from comic_translator.services.optical_center_solver import OpticalCenterSolver


class TypographyServiceV2:
    """排版引擎 V2 — 所有排版基于 SafeArea"""

    def __init__(
        self,
        font_manager: FontManager,
        bubble_layout: BubbleLayoutService,
        shape_solver: ShapeLayoutSolver,
        breaker: BalancedLineBreaker,
        padding_solver: PaddingSolver,
        font_size_solver: FontSizeSolver,
        optical_center: OpticalCenterSolver,
    ) -> None:
        self._fm = font_manager
        self._bl = bubble_layout
        self._shape = shape_solver
        self._breaker = breaker
        self._padding = padding_solver
        self._fss = font_size_solver
        self._oc = optical_center

    def _layout_singleton(
        self,
        text: str,
        bbox: tuple[int, int, int, int],
        font_path: str,
        min_size: int,
        max_size: int,
        line_spacing: float,
        region_type: str,
        profile,
    ) -> ParagraphLayout | None:
        """Singleton 块排版：没有 polygon 时，直接在 bbox 内居中

        不经过 SafeArea / FontSizeSolver / OpticalCenterSolver 等复杂流程，
        直接根据 bbox 尺寸计算合适字号，在 bbox 内水平垂直居中。
        """
        x1, y1, x2, y2 = bbox
        box_w = x2 - x1
        box_h = y2 - y1

        if box_w <= 4 or box_h <= 4:
            return None

        # 最小 padding 2px，确保文字不贴边
        pad = 2
        avail_w = box_w - pad * 2
        avail_h = box_h - pad * 2

        # 从 max_size 向下找合适的字号（单行）
        best_size = min_size
        for size in range(max_size, min_size - 1, -1):
            _, line_h = self._fm.measure_text("A", font_path, size)
            if line_h <= 0:
                continue
            line_h_spaced = int(line_h * line_spacing)
            # 检查高度：单行文字必须 ≤ 可用高度
            if line_h_spaced > avail_h:
                continue
            # 检查宽度：文字必须 ≤ 可用宽度
            lw, _ = self._fm.measure_text(text, font_path, size)
            if lw <= avail_w:
                best_size = size
                break

        # 如果 max_size 都不行，用 min_size（允许换行）
        _, line_h = self._fm.measure_text("A", font_path, best_size)
        line_h_spaced = int(line_h * line_spacing) if line_h > 0 else int(avail_h * 0.8)

        # 最多允许多少行
        max_lines = max(1, avail_h // line_h_spaced) if line_h_spaced > 0 else 1

        # 尝试单行，超宽则换行
        lw, _ = self._fm.measure_text(text, font_path, best_size)
        if lw <= avail_w or max_lines <= 1:
            # 单行，在 bbox 内垂直居中
            lines = [text]
            total_h = line_h_spaced
            start_y = y1 + (box_h - total_h) // 2
        else:
            # 多行：按字符拆分，每行不超过 avail_w
            chars = list(text)
            lines = []
            current_line = ""
            for ch in chars:
                test_line = current_line + ch
                tw, _ = self._fm.measure_text(test_line, font_path, best_size)
                if tw <= avail_w:
                    current_line = test_line
                else:
                    if current_line:
                        lines.append(current_line)
                    current_line = ch
            if current_line:
                lines.append(current_line)
            # 限制行数
            lines = lines[:max_lines]
            total_h = len(lines) * line_h_spaced
            start_y = y1 + (box_h - total_h) // 2

        # 每行在 bbox 内水平居中
        line_infos: list[LineInfo] = []
        max_line_w = 0
        for i, line_text in enumerate(lines):
            lw, _ = self._fm.measure_text(line_text, font_path, best_size)
            cx = x1 + (box_w - lw) // 2
            ly = start_y + i * line_h_spaced
            line_infos.append(LineInfo(
                text=line_text,
                width=lw,
                max_width=box_w - pad * 2,
                x=cx,
                y=ly,
            ))
            max_line_w = max(max_line_w, lw)

        logger.debug(
            f"Singleton: region={region_type}, size={best_size}, "
            f"lines={len(lines)}, bbox=({box_w}x{box_h})"
        )

        from comic_translator.models.typography_models import SafeArea
        dummy_safe = SafeArea(
            polygon=[], bbox=bbox,
            max_inscribed_rect=bbox, center=(0.0, 0.0),
        )
        return ParagraphLayout(
            lines=line_infos,
            font_size=best_size,
            font_name=profile.name,
            line_count=len(lines),
            total_height=total_h,
            total_width=max_line_w,
            safe_area=dummy_safe,
            optical_center_y=start_y,
            score=1.0,
        )

    def layout(
        self,
        polygon: list,
        bbox: tuple[int, int, int, int],
        text: str,
        region_type: str,
        image_shape: tuple[int, int],
        mask: np.ndarray | None = None,
    ) -> ParagraphLayout | None:
        """对单个气泡进行完整排版

        - 有 polygon 时：基于 SafeArea（距离变换 + 最大内接矩形）
        - 无 polygon 时（singleton OCR 块）：直接基于 bbox 简单居中
        """
        if not text.strip():
            return None

        profile = self._fm.get_profile_by_name(region_type)
        font_path = profile.font_path
        min_size = profile.min_size
        max_size = profile.max_size
        line_spacing = profile.line_spacing

        # === 无 polygon → singleton 块，走简单居中排版 ===
        if not polygon or len(polygon) < 3:
            return self._layout_singleton(text, bbox, font_path, min_size, max_size,
                                          line_spacing, region_type, profile)

        # === 有 polygon → 基于 SafeArea 的完整排版 ===
        safe = self._bl.safe_area(polygon, image_shape, min_distance=0, mask=mask)
        rect = safe.max_inscribed_rect  # (x1, y1, x2, y2)

        # === 2. 字号求解 + 平衡换行（基于 safe area 的尺寸和 polygon 扫描线） ===
        font_size, lines, score = self._fss.solve(
            text=text,
            polygon=polygon,
            safe_area=safe,
            font_path=font_path,
            min_size=min_size,
            max_size=max_size,
            image_shape=image_shape,
            line_spacing=line_spacing,
        )

        if not lines:
            return None

        # === 3. 行高 + 总高度 ===
        _, line_h = self._fm.measure_text("A", font_path, font_size)
        line_h_spaced = int(line_h * line_spacing)
        total_h = len(lines) * line_h_spaced

        # === 4. 视觉中心（在 safe area 内居中） ===
        start_y = self._oc.solve(safe, total_h)

        # === 5. 每行可用宽度（基于 polygon 扫描线） ===
        pad = self._padding.compute(safe, font_size)
        per_line_widths = self._shape.per_line_max_widths(
            polygon,
            start_y,
            len(lines),
            line_h_spaced,
            image_shape,
            pad.x,
        )

        # === 6. 每行绘制位置（在 safe area 内水平居中） ===
        rect_x1, rect_y1, rect_x2, rect_y2 = rect
        rect_w = rect_x2 - rect_x1
        rect_h = rect_y2 - rect_y1

        # 安全裁剪：如果文本超出 safe area，重新调整起始 Y
        if total_h > rect_h:
            # 文本比 safe area 高，缩小字号或从顶部开始
            start_y = rect_y1 + pad.y
            logger.warning(f"Text height ({total_h}) > safe area ({rect_h}), clamping start_y")
        else:
            # 确保文本底部不超出 safe area
            bottom_y = start_y + total_h
            if bottom_y > rect_y2 - pad.y:
                start_y = rect_y2 - pad.y - total_h
                if start_y < rect_y1 + pad.y:
                    start_y = rect_y1 + pad.y

        line_infos: list[LineInfo] = []
        max_line_w = 0

        for i, line_text in enumerate(lines):
            lw, _ = self._fm.measure_text(line_text, font_path, font_size)
            max_w = per_line_widths[i] if i < len(per_line_widths) else lw
            # 在 safe area 内水平居中（非 bbox）
            cx = rect_x1 + (rect_w - lw) // 2
            lx = max(rect_x1 + pad.x, cx)
            # 安全裁剪：确保文字不超出 safe area 左右边界
            if lx + lw > rect_x2 - pad.x:
                lx = rect_x2 - pad.x - lw
            if lx < rect_x1 + pad.x:
                lx = rect_x1 + pad.x
            ly = start_y + i * line_h_spaced

            line_infos.append(LineInfo(
                text=line_text,
                width=lw,
                max_width=max_w,
                x=lx,
                y=ly,
            ))
            max_line_w = max(max_line_w, lw)

        logger.debug(
            f"Typography: region={region_type}, font={profile.name}, "
            f"size={font_size}, lines={len(lines)}, score={score:.2f}"
        )

        return ParagraphLayout(
            lines=line_infos,
            font_size=font_size,
            font_name=profile.name,
            line_count=len(lines),
            total_height=total_h,
            total_width=max_line_w,
            safe_area=safe,
            optical_center_y=start_y,
            score=score,
        )