"""Tests for PaddleOCRProvider _convert_results"""

from comic_translator.providers.ocr.paddle_ocr_provider import PaddleOCRProvider


class TestPaddleOCRProvider:
    def setup_method(self):
        self.provider = PaddleOCRProvider(lang="en", use_gpu=False)

    def test_convert_results_empty(self):
        blocks = self.provider._convert_results([[]], page=1)
        assert blocks == []

    def test_convert_results_single(self):
        raw = [
            [
                [[[10.0, 20.0], [100.0, 20.0], [100.0, 50.0], [10.0, 50.0]], ("Hello", 0.95)],
            ]
        ]
        blocks = self.provider._convert_results(raw, page=1)
        assert len(blocks) == 1
        assert blocks[0].text == "Hello"
        assert blocks[0].confidence == 0.95
        assert blocks[0].bbox == (10.0, 20.0, 100.0, 50.0)
        assert "ocr_000001" in blocks[0].id

    def test_convert_results_multiple(self):
        raw = [
            [
                [[[0.0, 0.0], [50.0, 0.0], [50.0, 20.0], [0.0, 20.0]], ("First", 0.9)],
                [[[60.0, 0.0], [100.0, 0.0], [100.0, 20.0], [60.0, 20.0]], ("Second", 0.85)],
            ]
        ]
        blocks = self.provider._convert_results(raw, page=2)
        assert len(blocks) == 2
        assert blocks[0].text == "First"
        assert blocks[1].text == "Second"

    def test_convert_results_skip_invalid(self):
        raw = [
            [
                None,
                [[[0.0, 0.0], [50.0, 0.0], [50.0, 20.0], [0.0, 20.0]], ("Valid", 0.9)],
                [],
            ]
        ]
        blocks = self.provider._convert_results(raw, page=1)
        assert len(blocks) == 1
        assert blocks[0].text == "Valid"