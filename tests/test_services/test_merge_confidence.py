"""Tests for MergeConfidenceService"""

from comic_translator.models.ocr_block import OCRBlock
from comic_translator.services.merge_confidence_service import MergeConfidenceService


class TestMergeConfidenceService:
    def setup_method(self):
        self.service = MergeConfidenceService(threshold=0.6)

    def test_high_confidence_merge(self):
        a = OCRBlock(id="a", page=1, bbox=(10, 10, 100, 30), text="Hello", confidence=0.95)
        b = OCRBlock(id="b", page=1, bbox=(10, 32, 100, 50), text="World", confidence=0.95)
        decision = self.service.evaluate(a, b, line_height=20)
        assert decision.score >= 0.6
        assert decision.should_merge is True

    def test_low_confidence_no_merge(self):
        a = OCRBlock(id="a", page=1, bbox=(10, 10, 100, 30), text="Hello.", confidence=0.95)
        b = OCRBlock(id="b", page=1, bbox=(10, 200, 100, 220), text="Far below", confidence=0.95)
        decision = self.service.evaluate(a, b, line_height=20)
        assert decision.score < 0.6
        assert decision.should_merge is False

    def test_no_x_overlap(self):
        a = OCRBlock(id="a", page=1, bbox=(10, 10, 50, 30), text="Left")
        b = OCRBlock(id="b", page=1, bbox=(100, 10, 150, 30), text="Right")
        decision = self.service.evaluate(a, b, line_height=20)
        assert decision.x_overlap_score == 0.0

    def test_punctuation_score_high(self):
        block = OCRBlock(id="a", page=1, bbox=(0, 0, 10, 10), text="and")
        score = self.service._calc_punctuation(block)
        assert score >= 0.7

    def test_punctuation_score_sentence_end(self):
        block = OCRBlock(id="a", page=1, bbox=(0, 0, 10, 10), text="Hello.")
        score = self.service._calc_punctuation(block)
        assert score <= 0.4

    def test_batch_evaluate(self):
        blocks = [
            OCRBlock(id="a", page=1, bbox=(10, 10, 100, 30), text="Hello"),
            OCRBlock(id="b", page=1, bbox=(10, 32, 100, 50), text="World"),
        ]
        decisions = self.service.batch_evaluate(blocks, line_height=20)
        assert len(decisions) == 2
        assert len(decisions[0]) == 2
        # Self-decision should not be a merge
        assert decisions[0][0].should_merge is False
