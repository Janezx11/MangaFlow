"""TypographyQA — 排版质量检查

检查项:
  - Font size too small / too large
  - Bubble overflow
  - Visual center deviation
  - Paragraph balance
  - Line variance
  - Padding appropriateness
  - English residue
  - Text outside bubble

输出: typography_score (0-100)
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import cv2
import numpy as np

from comic_translator.models.typography_models import TypographyScore

if TYPE_CHECKING:
    from comic_translator.models.typography_models import ParagraphLayout


class TypographyQA:
    """排版质量检查"""

    MIN_FONT_SIZE = 10
    ENGLISH_PATTERN = re.compile(r"[a-zA-Z]{4,}")

    # 权重
    W_BALANCE = 0.20
    W_FONT_SIZE = 0.20
    W_OVERFLOW = 0.25
    W_CENTER = 0.15
    W_PADDING = 0.10
    W_ENGLISH = 0.10

    def check(
        self,
        layout: ParagraphLayout,
        bubble_polygon: list,
        image_shape: tuple[int, int],
    ) -> TypographyScore:
        """对单气泡排版结果进行质量检查"""
        issues: list[str] = []

        # 1. Font size score
        font_score, font_issues = self._check_font_size(layout)
        issues.extend(font_issues)

        # 2. Overflow score
        overflow_score, overflow_issues = self._check_overflow(layout, bubble_polygon, image_shape)
        issues.extend(overflow_issues)

        # 3. Balance score
        balance_score, balance_issues = self._check_balance(layout)
        issues.extend(balance_issues)

        # 4. Center score
        center_score, center_issues = self._check_center(layout)
        issues.extend(center_issues)

        # 5. Padding score
        padding_score, padding_issues = self._check_padding(layout)
        issues.extend(padding_issues)

        # 6. English residue
        english_score = 1.0
        for line in layout.lines:
            matches = self.ENGLISH_PATTERN.findall(line.text)
            if len(matches) >= 3:
                english_score = 0.3
                issues.append(f"English residue: {matches[:3]}")
                break

        # Overall score (0-100)
        overall = (
            balance_score * self.W_BALANCE +
            font_score * self.W_FONT_SIZE +
            overflow_score * self.W_OVERFLOW +
            center_score * self.W_CENTER +
            padding_score * self.W_PADDING +
            english_score * self.W_ENGLISH
        ) * 100

        return TypographyScore(
            overall=round(min(100, max(0, overall)), 1),
            balance=round(balance_score, 2),
            font_size_score=round(font_score, 2),
            overflow_score=round(overflow_score, 2),
            center_score=round(center_score, 2),
            padding_score=round(padding_score, 2),
            issues=issues,
        )

    def _check_font_size(self, layout: ParagraphLayout) -> tuple[float, list[str]]:
        issues = []
        if layout.font_size < self.MIN_FONT_SIZE:
            issues.append(f"Font too small: {layout.font_size}px")
            return (0.2, issues)
        if layout.font_size >= 40:
            return (0.7, issues)
        if layout.font_size >= 20:
            return (1.0, issues)
        return (0.8, issues)

    def _check_overflow(
        self,
        layout: ParagraphLayout,
        polygon: list,
        image_shape: tuple[int, int],
    ) -> tuple[float, list[str]]:
        issues = []
        if len(polygon) < 3:
            return (0.5, issues)

        # 检查每行绘制位置是否在 polygon 内
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(mask, [pts], 255)

        outside_count = 0
        total_chars = 0
        for line in layout.lines:
            total_chars += len(line.text)
            # 检查终点
            if line.y >= h or line.x >= w:
                outside_count += len(line.text)
                continue
            # 粗略检查：行中心点是否在 mask 内
            cx = min(w - 1, max(0, line.x + line.width // 2))
            cy = min(h - 1, max(0, line.y + 5))
            if mask[cy, cx] == 0:
                outside_count += len(line.text)

        if outside_count > 0 and total_chars > 0:
            ratio = outside_count / total_chars
            if ratio > 0.3:
                issues.append(f"Text overflow: {outside_count}/{total_chars} chars outside bubble")
                return (0.3, issues)
            return (0.7 - ratio * 0.4, issues)
        return (1.0, issues)

    def _check_balance(self, layout: ParagraphLayout) -> tuple[float, list[str]]:
        issues = []
        if layout.line_count <= 1:
            return (1.0, issues)

        widths = [l.width for l in layout.lines]
        if not widths:
            return (0.5, issues)

        max_w = max(widths)
        if max_w == 0:
            return (0.5, issues)

        # 方差
        ratios = [w / max_w for w in widths]
        mean_r = sum(ratios) / len(ratios)
        variance = sum((r - mean_r) ** 2 for r in ratios) / len(ratios)

        # 孤行检测
        if len(widths) > 1 and widths[-1] < max_w * 0.4:
            issues.append(f"Widow/orphan line: last line width={widths[-1]}/{max_w}")

        # 得分：方差越小越好
        balance = max(0, 1.0 - variance * 2)
        return (balance, issues)

    def _check_center(self, layout: ParagraphLayout) -> tuple[float, list[str]]:
        issues = []
        if not layout.lines:
            return (0.5, issues)

        # 计算每行的居中偏差
        total_dev = 0.0
        for line in layout.lines:
            x1, _, x2, _ = layout.safe_area.bbox
            box_w = x2 - x1
            if box_w <= 0:
                continue
            # 期望居中的 x
            expected_cx = x1 + box_w // 2
            actual_cx = line.x + line.width // 2
            dev = abs(actual_cx - expected_cx) / box_w
            total_dev += dev

        avg_dev = total_dev / len(layout.lines)
        if avg_dev > 0.2:
            issues.append(f"Center deviation: {avg_dev:.2f}")
            return (max(0, 1.0 - avg_dev * 2), issues)
        return (1.0, issues)

    def _check_padding(self, layout: ParagraphLayout) -> tuple[float, list[str]]:
        issues = []
        x1, _, x2, _ = layout.safe_area.bbox
        box_w = x2 - x1
        if box_w <= 0:
            return (0.5, issues)

        # text_to_box ratio
        text_w = layout.total_width
        ratio = text_w / box_w

        # 太挤 (< 0.5) 或太空 (> 0.95) 都不好
        if ratio < 0.5:
            issues.append(f"Padding too large: text/box ratio={ratio:.2f}")
            return (0.5, issues)
        if ratio > 0.95:
            issues.append(f"Padding too small: text/box ratio={ratio:.2f}")
            return (0.5, issues)
        return (1.0, issues)