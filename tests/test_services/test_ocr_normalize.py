"""Tests for OCRNormalizeService"""

from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock
from comic_translator.models.ocr_block import OCRBlock
from comic_translator.services.ocr_normalize_service import OCRNormalizeService


class TestOCRNormalizeService:
    def setup_method(self):
        self.service = OCRNormalizeService()

    def test_sort_blocks_top_to_bottom(self):
        blocks = [
            OCRBlock(id="b2", page=1, bbox=(10, 100, 100, 130), text="Bottom"),
            OCRBlock(id="b1", page=1, bbox=(10, 10, 100, 40), text="Top"),
        ]
        sorted_blocks = self.service.sort_blocks(blocks)
        assert sorted_blocks[0].id == "b1"
        assert sorted_blocks[1].id == "b2"

    def test_merge_vertical_adjacent(self):
        """Blocks with x-overlap and close vertical gap should merge"""
        blocks = [
            OCRBlock(id="b1", page=1, bbox=(10, 10, 100, 30), text="Hello"),
            OCRBlock(id="b2", page=1, bbox=(10, 32, 100, 50), text="World"),
        ]
        merged = self.service.merge_blocks(blocks)
        assert len(merged) == 1
        assert merged[0].is_merged is True
        assert merged[0].text == "Hello World"

    def test_no_merge_when_x_no_overlap(self):
        """Blocks without x-overlap should not merge"""
        blocks = [
            OCRBlock(id="b1", page=1, bbox=(10, 10, 50, 30), text="Left"),
            OCRBlock(id="b2", page=1, bbox=(100, 10, 150, 30), text="Right"),
        ]
        merged = self.service.merge_blocks(blocks)
        assert len(merged) == 2

    def test_no_merge_when_vertical_gap_large(self):
        blocks = [
            OCRBlock(id="b1", page=1, bbox=(10, 10, 100, 30), text="Top"),
            OCRBlock(id="b2", page=1, bbox=(10, 200, 100, 230), text="Far"),
        ]
        merged = self.service.merge_blocks(blocks)
        assert len(merged) == 2

    def test_clean_text(self):
        blocks = [
            NormalizedOCRBlock(id="b1", page=1, bbox=(0, 0, 10, 10), text="  Hello   World!  "),
            NormalizedOCRBlock(id="b2", page=1, bbox=(0, 10, 10, 20), text="."),
            NormalizedOCRBlock(id="b3", page=1, bbox=(0, 20, 10, 30), text="Clean text."),
        ]
        cleaned = self.service.clean_text(blocks)
        assert len(cleaned) == 2  # b2 removed (just ".")
        assert cleaned[0].text == "Hello World!"

    def test_process_full_pipeline(self):
        blocks = [
            OCRBlock(id="b1", page=1, bbox=(10, 100, 100, 130), text="  Bottom  "),
            OCRBlock(id="b2", page=1, bbox=(10, 10, 100, 30), text="Top"),
            OCRBlock(id="b3", page=1, bbox=(10, 32, 100, 50), text="Middle"),
        ]
        result = self.service.process(blocks, page=1)
        assert len(result) >= 2  # b1 bottom + merged(b2+b3 top)
        # Check reading order assigned
        assert all(b.reading_order > 0 for b in result)
        # Top merged block should have lower reading order
        top_blocks = [b for b in result if b.bbox[1] < 50]
        bottom_blocks = [b for b in result if b.bbox[1] > 50]
        assert top_blocks[0].reading_order < bottom_blocks[0].reading_order
