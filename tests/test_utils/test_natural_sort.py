"""回归测试：图片自然排序 → 渲染/翻译页码 = 实际阅读顺序

覆盖：
- natural_key 数字块按数值比较（"2.jpg" < "10.jpg"）
- GuiPipelineService.collect_images 页码顺序
- BatchOCRService.collect_images 页码顺序

背景：原实现用字典序 sorted()，导致 "10.jpg" < "2.jpg"，
带编号的漫画被错误排序，渲染页码与实际阅读顺序不符。
"""

from pathlib import Path

import pytest

from comic_translator.gui.services.gui_pipeline_service import GuiPipelineService
from comic_translator.services.batch_ocr_service import BatchOCRService
from comic_translator.utils.natural_sort import natural_key


def test_natural_key_numeric_order():
    """数字块按数值比较，而非字典序"""
    names = ["10.jpg", "2.jpg", "1.jpg"]
    assert sorted(names, key=natural_key) == ["1.jpg", "2.jpg", "10.jpg"]


def test_natural_key_mixed_prefix():
    """带前缀的编号名同样正确排序"""
    names = ["page_10.png", "page_2.png", "page_1.png"]
    assert sorted(names, key=natural_key) == ["page_1.png", "page_2.png", "page_10.png"]


def test_natural_key_no_numbers_falls_back_lexical():
    """无数字的文件退化为字典序（行为与之前一致）"""
    names = ["b.png", "a.png"]
    assert sorted(names, key=natural_key) == ["a.png", "b.png"]


def test_gui_collect_images_uses_reading_order(tmp_path: Path):
    """GUI 管线：页码 = 自然排序后的阅读顺序（2.jpg 是第 2 页，10.jpg 是第 3 页）"""
    for name in ["10.jpg", "2.jpg", "1.jpg"]:
        (tmp_path / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    images = GuiPipelineService.collect_images(str(tmp_path))
    assert [p.name for p in images] == ["1.jpg", "2.jpg", "10.jpg"]

    # 页码即列表索引+1：1.jpg→1, 2.jpg→2, 10.jpg→3
    names = [p.name for p in images]
    assert names.index("2.jpg") + 1 == 2
    assert names.index("10.jpg") + 1 == 3


def test_batch_collect_images_uses_reading_order(tmp_path: Path):
    """CLI 批处理：页码同样按自然排序"""
    for name in ["10.png", "2.png", "1.png"]:
        (tmp_path / name).write_bytes(b"\x89PNG\r\n\x1a\n")
    service = BatchOCRService(provider=None, normalize=None, cache=None)
    pages = service.collect_images(str(tmp_path))
    assert [Path(p).name for _, p in pages] == ["1.png", "2.png", "10.png"]
    assert [page for page, _ in pages] == [1, 2, 3]
