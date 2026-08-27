"""回归测试：OCR 框指导排版 + 渲染命名继承（近期修复，防止退化）"""

from __future__ import annotations

from unittest import mock

import pytest

from comic_translator.models.bubble import Bubble, DialogueBlock
from comic_translator.services.font_size_solver import FontSizeSolver
from comic_translator.services.render_service import merge_ocr_rect


def _make_block(ocr_rects=None):
    b = Bubble(
        id="b1", bbox=(0, 0, 100, 100),
        polygon=[(0, 0), (100, 0), (100, 100), (0, 100)],
        mask=[], class_name="speech",
    )
    db = DialogueBlock(id="d1", bubble=b, merged_text="Hello", translated_text="你好")
    db.ocr_blocks = []
    for (x1, y1, x2, y2) in (ocr_rects or []):
        ob = mock.Mock()
        ob.bbox = (x1, y1, x2, y2)
        db.ocr_blocks.append(ob)
    return db


def test_merge_ocr_rect_union():
    """多个 OCR 框应合并为外接矩形"""
    db = _make_block([(10, 20, 30, 40), (15, 25, 50, 60)])
    assert merge_ocr_rect(db) == (10, 20, 50, 60)


def test_merge_ocr_rect_empty():
    """无 OCR 块时返回 None（退回气泡安全区排版）"""
    assert merge_ocr_rect(_make_block([])) is None
    assert merge_ocr_rect(_make_block(None)) is None


def test_merge_ocr_rect_invalid():
    """空/退化 bbox 返回 None"""
    db = _make_block([(5, 5, 5, 5)])  # x2<=x1
    assert merge_ocr_rect(db) is None


class _FakeSafe:
    def __init__(self, rect):
        self.max_inscribed_rect = rect


def _make_solver():
    from comic_translator.services.balanced_line_breaker import BalancedLineBreaker
    from comic_translator.services.bubble_layout_service import BubbleLayoutService
    from comic_translator.services.font_manager import FontManager
    from comic_translator.services.padding_solver import PaddingSolver
    from comic_translator.services.shape_layout_solver import ShapeLayoutSolver

    fm = FontManager()
    bl = BubbleLayoutService()
    sl = ShapeLayoutSolver(bl)
    brk = BalancedLineBreaker(fm)
    ps = PaddingSolver()
    return FontSizeSolver(fm, brk, sl, ps)


def test_target_rect_clamps_font_size():
    """给定较小 target_rect 时，字号应缩小（贴合原文占位）"""
    fss = _make_solver()
    safe = _FakeSafe((0, 0, 300, 100))
    polygon = [(0, 0), (300, 0), (300, 100), (0, 100)]
    fp = fss._fm.get_profile_by_name("speech").font_path
    text = "这是一段比较长的中文测试文本用于验证OCR框指导的排版效果是否生效"

    size_big, _, _ = fss.solve(text, polygon, safe, fp, 12, 64, (300, 100))
    size_small, _, _ = fss.solve(
        text, polygon, safe, fp, 12, 64, (300, 100), target_rect=(0, 0, 150, 40)
    )
    assert size_small <= size_big


def test_target_rect_none_unchanged():
    """target_rect=None 时不改变行为"""
    fss = _make_solver()
    safe = _FakeSafe((0, 0, 300, 100))
    polygon = [(0, 0), (300, 0), (300, 100), (0, 100)]
    fp = fss._fm.get_profile_by_name("speech").font_path
    text = "短文本"
    a, la, _ = fss.solve(text, polygon, safe, fp, 12, 64, (300, 100))
    b, lb, _ = fss.solve(text, polygon, safe, fp, 12, 64, (300, 100), target_rect=None)
    assert (a, la) == (b, lb)


@pytest.mark.skipif(True, reason="命名继承走 RenderQualityService.render_page，需完整渲染栈；此处仅验证 helper")
def test_placeholder():
    pass
