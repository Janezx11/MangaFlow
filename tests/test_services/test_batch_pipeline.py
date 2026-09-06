"""Tests for Batch Translation Pipeline"""


from comic_translator.models.batch_result import BatchInput, BatchPage
from comic_translator.models.glossary import GlossaryEntry
from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock
from comic_translator.services.batch_cache import BatchCacheService
from comic_translator.services.batch_prompt_builder import BatchPromptBuilder


class TestBatchCache:
    def test_cache_roundtrip(self, tmp_path):
        cache = BatchCacheService(cache_dir=str(tmp_path))
        data = [{"page": 1, "blocks": []}]
        cache.set_ocr_cache("test_proj", data)
        assert cache.has_ocr_cache("test_proj") is True
        loaded = cache.get_ocr_cache("test_proj")
        assert loaded == data

    def test_batch_cache_roundtrip(self, tmp_path):
        cache = BatchCacheService(cache_dir=str(tmp_path))
        data = {"batch_index": 1, "pages": [], "glossary": {}}
        cache.set_batch_output("test_proj", 1, data)
        assert cache.has_batch_cache("test_proj", 1) is True
        loaded = cache.get_batch_output("test_proj", 1)
        assert loaded["batch_index"] == 1

    def test_no_cache(self, tmp_path):
        cache = BatchCacheService(cache_dir=str(tmp_path))
        assert cache.get_ocr_cache("nonexistent") is None
        assert cache.get_batch_output("nonexistent", 1) is None


class TestBatchPromptBuilder:
    def setup_method(self):
        self.builder = BatchPromptBuilder()
        self.blocks = [
            NormalizedOCRBlock(id="p1_b1", page=1, bbox=(0, 0, 10, 10), text="Hello Tony."),
            NormalizedOCRBlock(id="p1_b2", page=1, bbox=(0, 10, 10, 20), text="We have a problem."),
        ]

    def test_build_user_prompt_contains_text(self):
        batch_input = BatchInput(
            batch_index=1,
            pages=[BatchPage(page=1, blocks=self.blocks)],
        )
        prompt = self.builder.build_user_prompt(batch_input)
        assert "Hello Tony." in prompt
        assert "PAGE 1" in prompt
        assert "p1_b1" in prompt
        assert "p1_b2" in prompt

    def test_build_user_prompt_with_glossary(self):
        batch_input = BatchInput(
            batch_index=1,
            pages=[BatchPage(page=1, blocks=self.blocks)],
            glossary=[GlossaryEntry(source="Tony Stark", target="托尼·斯塔克")],
        )
        prompt = self.builder.build_user_prompt(batch_input)
        assert "Tony Stark" in prompt
        assert "托尼·斯塔克" in prompt

    def test_parse_valid_response(self):
        response = """
        {
          "glossary": {"Tony Stark": "托尼·斯塔克"},
          "pages": [
            {
              "page": 1,
              "blocks": [
                {"id": "p1_b1", "corrected": "Hello Tony.", "translated": "你好，托尼。"},
                {"id": "p1_b2", "corrected": "We have a problem.", "translated": "我们出问题了。"}
              ]
            }
          ]
        }
        """
        batch_input = BatchInput(
            batch_index=1,
            pages=[BatchPage(page=1, blocks=self.blocks)],
        )
        result = self.builder.parse_response(response, batch_input)
        assert len(result.pages) == 1
        assert len(result.pages[0].blocks) == 2
        assert result.pages[0].blocks[0].translated_text == "你好，托尼。"
        assert result.pages[0].blocks[1].translated_text == "我们出问题了。"
        assert result.glossary["Tony Stark"] == "托尼·斯塔克"

    def test_parse_response_with_markdown(self):
        response = """```json
        {"glossary": {}, "pages": []}
        ```"""
        batch_input = BatchInput(batch_index=1, pages=[])
        result = self.builder.parse_response(response, batch_input)
        assert result.batch_index == 1

    def test_parse_invalid_response(self):
        response = "sorry i cannot translate this"
        batch_input = BatchInput(batch_index=1, pages=[])
        result = self.builder.parse_response(response, batch_input)
        # Should return empty result, not crash
        assert len(result.pages) == 0

    def test_parse_empty_response(self):
        batch_input = BatchInput(batch_index=1, pages=[])
        result = self.builder.parse_response("", batch_input)
        assert len(result.pages) == 0
