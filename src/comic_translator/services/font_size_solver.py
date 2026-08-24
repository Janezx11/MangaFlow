"""FontSizeSolver — 动态字号计算（基于 Polygon Safe Area）

所有字号和换行计算使用 SafeArea 的 max_inscribed_rect，
不再使用 raw bubble bbox。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from comic_translator.services.balanced_line_breaker import BalancedLineBreaker
from comic_translator.services.shape_layout_solver import ShapeLayoutSolver

if TYPE_CHECKING:
    from comic_translator.models.typography_models import SafeArea
    from comic_translator.services.font_manager import FontManager
    from comic_translator.services.padding_solver import PaddingSolver


class FontSizeSolver:
    """字号求解器（基于 SafeArea）"""

    def __init__(
        self,
        font_manager: FontManager,
        breaker: BalancedLineBreaker,
        shape_solver: ShapeLayoutSolver,
        padding_solver: PaddingSolver,
    ) -> None:
        self._fm = font_manager
        self._breaker = breaker
        self._shape = shape_solver
        self._padding = padding_solver

    def solve(
        self,
        text: str,
        polygon: list,
        safe_area: SafeArea,
        font_path: str,
        min_size: int,
        max_size: int,
        image_shape: tuple[int, int],
        line_spacing: float = 1.3,
        target_rect: tuple[int, int, int, int] | None = None,
    ) -> tuple[int, list[str], float]:
        """求解最佳字号和换行

        Args:
            text: 输入文本
            polygon: 气泡多边形（用于 per-line width）
            safe_area: 安全区域（来自 BubbleLayoutService）
            font_path: 字体路径
            min_size: 最小字号
            max_size: 最大字号
            image_shape: (h, w)
            line_spacing: 行间距
            target_rect: 可选 (x1,y1,x2,y2) —— 原文 OCR 框占位。
                提供时把可用宽/高限制在原文占位内（带下限），
                让译文字号/换行贴合原文，避免溢出或过大。

        Returns:
            (font_size, lines, score)
        """
        x1, y1, x2, y2 = safe_area.max_inscribed_rect
        box_h = y2 - y1
        box_w = x2 - x1

        if box_h <= 0 or box_w <= 0:
            return min_size, [text], 0.0

        # 计算 padding（基于 safe area 的尺寸）
        pad = self._padding.compute(safe_area, max_size)
        avail_h = box_h - pad.y * 2
        avail_w = box_w - pad.x * 2

        # ★ OCR 框指导：把可用区域限制在原文占位内（留 8px 内边距，下限 40px）
        if target_rect is not None:
            tx1, ty1, tx2, ty2 = target_rect
            t_w = (tx2 - tx1) - 8
            t_h = (ty2 - ty1) - 8
            avail_w = min(avail_w, max(t_w, 40))
            avail_h = min(avail_h, max(t_h, 40))

        if avail_h <= 0 or avail_w <= 0:
            return min_size, [text], 0.0

        # 从最大字号向下搜索；若最小字号仍放不下，允许低于 min_size（硬下限 8），
        # 避免退化成长文本整段单行渲染导致溢出气泡。
        hard_floor = min(8, min_size)
        best_size = min_size
        best_lines = [text]
        best_score = 0.0

        for size in range(max_size, hard_floor - 1, -2):
            _, line_h = self._fm.measure_text("A", font_path, size)
            if line_h <= 0:
                continue
            line_h_spaced = int(line_h * line_spacing)

            max_lines = max(1, avail_h // line_h_spaced)

            # 计算每行可用宽度（基于 polygon 扫描线）
            # 起始 Y 使用 safe_area 的顶部 + padding
            per_line_widths = self._shape.per_line_max_widths(
                polygon,
                y1 + pad.y,
                max_lines,
                line_h_spaced,
                image_shape,
                pad.x,
            )

            # ★ 关键修复：限制每行宽度不超过 safe area 可用宽度
            # 防止 polygon 扫描线比 safe area 宽时文字溢出
            per_line_widths = [min(w, avail_w) for w in per_line_widths]

            # 平衡换行
            lines, cost = self._breaker.break_text(text, per_line_widths, font_path, size)
            if not lines:
                continue

            # ★ 检查每行宽度是否在 safe area 允许范围内
            # BalancedLineBreaker 可能返回超宽的单行（单词太长）
            width_ok = True
            for i, line_text in enumerate(lines):
                lw, _ = self._fm.measure_text(line_text, font_path, size)
                max_w = per_line_widths[i] if i < len(per_line_widths) else lw
                if lw > max_w + 2:  # 2px 容差
                    width_ok = False
                    break
            if not width_ok:
                continue

            # 检查高度
            total_h = len(lines) * line_h_spaced
            if total_h > avail_h:
                continue

            # 得分：优先大字号 + 低成本
            size_score = (size - min_size) / max(max_size - min_size, 1)
            cost_score = max(0, 1.0 - cost / 100.0)
            score = size_score * 0.7 + cost_score * 0.3

            if score > best_score:
                best_size = size
                best_lines = lines
                best_score = score
                break  # 从最大向下，找到即最优

        if best_size == min_size and best_lines == [text] and min_size < max_size:
            _, line_h = self._fm.measure_text("A", font_path, min_size)
            line_h_spaced = int(line_h * line_spacing) if line_h > 0 else 20
            max_lines = max(1, avail_h // line_h_spaced)
            per_line_widths = self._shape.per_line_max_widths(
                polygon, y1 + pad.y, max_lines, line_h_spaced, image_shape, pad.x
            )
            per_line_widths = [min(w, avail_w) for w in per_line_widths]
            best_lines, cost = self._breaker.break_text(text, per_line_widths, font_path, min_size)
            # 保持按行换行结果：即使个别行略超宽，也优于整段单行渲染（后者溢出更严重）

        logger.debug(f"FontSizeSolver: size={best_size}, lines={len(best_lines)}, score={best_score:.2f}")
        return best_size, best_lines, best_score