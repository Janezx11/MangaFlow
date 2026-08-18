"""BubbleAwareOCRService 验收测试

Case 1: 两个相邻气泡，OCR 合并 → 必须拆成 2 个 TextRegion
Case 2: 三个相邻气泡，OCR 合并 → 必须拆成 3 个 TextRegion
Case 3: 单气泡，OCR 正常 → 不应触发重新 OCR
Case 4: 旁白框 + 气泡 → 不应错误合并
Case 5: SFX + 气泡 → 不应错误合并
"""

import pytest
from PIL import Image

from comic_translator.models.bubble import Bubble
from comic_translator.models.ocr_block import OCRBlock
from comic_translator.services.bubble_aware_ocr_service import (
    SPLIT_CROP_OCR,
    SPLIT_DIRECT,
    BubbleAwareOCRService,
)


def make_bubble(bid: str, x1, y1, x2, y2, cls: str = "speech") -> Bubble:
    poly = [(float(x1), float(y1)), (float(x2), float(y1)),
            (float(x2), float(y2)), (float(x1), float(y2))]
    return Bubble(
        id=bid, bbox=(x1, y1, x2, y2), polygon=poly,
        mask=[], class_name=cls, confidence=0.9,
    )


class FakeOCR:
    """假 OCR：记录调用次数，返回预设文本"""

    def __init__(self, text: str = "TEXT") -> None:
        self.text = text
        self.calls = 0

    async def detect_text(self, image):
        self.calls += 1
        w, h = image.size
        return [OCRBlock(
            id=f"fake_{self.calls}", page=0,
            bbox=(w * 0.2, h * 0.2, w * 0.8, h * 0.8),
            text=self.text, confidence=0.9,
        )]


def make_service(fake: FakeOCR, threshold: float = 0.15) -> BubbleAwareOCRService:
    return BubbleAwareOCRService(
        ocr_provider=fake, threshold=threshold,
        strategy="auto", enabled=True,
    )


async def test_case1_two_adjacent_bubbles_merged():
    """两个相邻气泡，OCR 合并成一个大 block → 必须拆成 2 个 TextRegion"""
    img = Image.new("RGB", (600, 400), "white")
    a = make_bubble("A", 100, 100, 300, 300)
    b = make_bubble("B", 320, 100, 520, 300)
    merged = OCRBlock(id="big", page=0, bbox=(100, 100, 520, 300),
                      text="Hello How are you?", confidence=0.9)

    fake = FakeOCR("Hello")
    result = await make_service(fake).process(img, [merged], [a, b])

    assert len(result.blocks) == 2, f"应拆成 2 个，实际 {len(result.blocks)}"
    ids = sorted(x.metadata["bubble_id"] for x in result.blocks)
    assert ids == ["A", "B"]
    assert all(x.metadata["cross_bubble"] is True for x in result.blocks)
    assert all(x.metadata["split_method"] == SPLIT_CROP_OCR for x in result.blocks)
    assert result.cross_bubble_count == 1
    assert fake.calls == 2, "应对 A、B 各触发一次 crop OCR"


async def test_case2_three_adjacent_bubbles_merged():
    """三个相邻气泡，OCR 合并 → 必须拆成 3 个 TextRegion"""
    img = Image.new("RGB", (800, 400), "white")
    a = make_bubble("A", 100, 100, 300, 300)
    b = make_bubble("B", 320, 100, 520, 300)
    c = make_bubble("C", 540, 100, 740, 300)
    merged = OCRBlock(id="big3", page=0, bbox=(100, 100, 740, 300),
                      text="A B C", confidence=0.9)

    fake = FakeOCR("T")
    result = await make_service(fake).process(img, [merged], [a, b, c])

    assert len(result.blocks) == 3, f"应拆成 3 个，实际 {len(result.blocks)}"
    ids = sorted(x.metadata["bubble_id"] for x in result.blocks)
    assert ids == ["A", "B", "C"]
    assert fake.calls == 3


async def test_case3_single_bubble_no_reocr():
    """单气泡，OCR 正常 → 不触发重新 OCR"""
    img = Image.new("RGB", (600, 400), "white")
    a = make_bubble("A", 100, 100, 300, 300)
    normal = OCRBlock(id="normal", page=0, bbox=(120, 120, 280, 280),
                      text="Hello", confidence=0.95)

    fake = FakeOCR()
    result = await make_service(fake).process(img, [normal], [a])

    assert len(result.blocks) == 1
    assert result.blocks[0].metadata["bubble_id"] == "A"
    assert result.blocks[0].metadata["split_method"] == SPLIT_DIRECT
    assert result.blocks[0].metadata["cross_bubble"] is False
    assert result.cross_bubble_count == 0
    assert fake.calls == 0, "单气泡不应触发重新 OCR"


async def test_case4_narration_plus_bubble_not_merged():
    """旁白框 + 气泡 → 不应错误合并（各自保留，不触发拆分）"""
    img = Image.new("RGB", (600, 400), "white")
    a = make_bubble("A", 100, 100, 300, 300, cls="speech")
    bubble_block = OCRBlock(id="sp", page=0, bbox=(120, 120, 280, 280),
                            text="Hello", confidence=0.9)
    # 旁白文字位于气泡下方，不与气泡 polygon 重叠
    narration_block = OCRBlock(id="nar", page=0, bbox=(100, 320, 300, 380),
                               text="Meanwhhile...", confidence=0.9)

    fake = FakeOCR()
    result = await make_service(fake).process(img, [bubble_block, narration_block], [a])

    assert len(result.blocks) == 2
    bubble_out = next(x for x in result.blocks if x.id == "sp")
    nar_out = next(x for x in result.blocks if x.id == "nar")
    assert bubble_out.metadata["bubble_id"] == "A"
    assert nar_out.metadata["bubble_id"] is None
    assert result.cross_bubble_count == 0
    assert fake.calls == 0


async def test_case5_sfx_plus_bubble_not_merged():
    """SFX + 气泡 → 不应错误合并（各自保留，不触发拆分）"""
    img = Image.new("RGB", (600, 400), "white")
    a = make_bubble("A", 100, 100, 300, 300, cls="speech")
    bubble_block = OCRBlock(id="sp", page=0, bbox=(120, 120, 280, 280),
                            text="Hello", confidence=0.9)
    # SFX 文字在气泡右侧，不与气泡 polygon 重叠
    sfx_block = OCRBlock(id="sfx", page=0, bbox=(330, 110, 380, 200),
                         text="BOOM!", confidence=0.8)

    fake = FakeOCR()
    result = await make_service(fake).process(img, [bubble_block, sfx_block], [a])

    assert len(result.blocks) == 2
    sfx_out = next(x for x in result.blocks if x.id == "sfx")
    assert sfx_out.metadata["bubble_id"] is None
    assert sfx_out.metadata["split_method"] == SPLIT_DIRECT
    assert result.cross_bubble_count == 0
    assert fake.calls == 0


async def test_duplicate_bubbles_merged_before_split():
    """YOLO 重复检测同一气泡（松/紧两个框）→ 先合并，不误触发拆分"""
    img = Image.new("RGB", (600, 400), "white")
    tight = make_bubble("T", 100, 100, 300, 300)
    loose = make_bubble("L", 95, 95, 305, 305)  # 包含率 ~0.94 → 合并
    normal = OCRBlock(id="n", page=0, bbox=(120, 120, 280, 280),
                      text="Hello", confidence=0.9)

    fake = FakeOCR()
    result = await make_service(fake).process(img, [normal], [tight, loose])

    # 合并后只剩 1 个气泡 → 不跨气泡
    assert len(result.blocks) == 1
    assert result.blocks[0].metadata["bubble_id"] in ("T", "L")
    assert result.cross_bubble_count == 0
    assert fake.calls == 0
