"""Tests for QwenTranslator"""

from unittest.mock import AsyncMock

from comic_translator.models.context import TranslationContext
from comic_translator.models.glossary import GlossaryEntry
from comic_translator.models.translation_block import TranslationBlock
from comic_translator.providers.translation.qwen_translator import QwenTranslator


def _make_mock_client(response: str):
    client = AsyncMock()
    client.generate = AsyncMock(return_value=response)
    client.model_name = "qwen3.5:9b"
    return client


class TestQwenTranslationProvider:
    def test_translate_basic(self):
        client = _make_mock_client(
            '[{"id": "b1", "translated_text": "你好"}, {"id": "b2", "translated_text": "世界"}]'
        )
        provider = QwenTranslator(client=client)
        blocks = [
            TranslationBlock(id="b1", ocr_block_id="b1", source_text="Hello", provider="qwen"),
            TranslationBlock(id="b2", ocr_block_id="b2", source_text="World", provider="qwen"),
        ]
        ctx = TranslationContext()

        import asyncio
        result = asyncio.run(provider.translate(blocks, ctx))

        assert result[0].translated_text == "你好"
        assert result[1].translated_text == "世界"

    def test_translate_with_glossary(self):
        client = _make_mock_client('[{"id": "b1", "translated_text": "斯塔克大厦"}]')
        provider = QwenTranslator(client=client)
        blocks = [
            TranslationBlock(id="b1", ocr_block_id="b1", source_text="Stark Tower"),
        ]
        ctx = TranslationContext(
            glossary=[GlossaryEntry(source="Stark Tower", target="斯塔克大厦")]
        )

        import asyncio
        result = asyncio.run(provider.translate(blocks, ctx))

        assert result[0].translated_text == "斯塔克大厦"

    def test_parse_response_markdown_fence(self):
        """Should handle markdown code fences in response"""
        client = _make_mock_client(
            '```json\n[{"id": "b1", "translated_text": "你好"}]\n```'
        )
        provider = QwenTranslator(client=client)
        blocks = [TranslationBlock(id="b1", ocr_block_id="b1", source_text="Hello")]
        ctx = TranslationContext()

        import asyncio
        result = asyncio.run(provider.translate(blocks, ctx))
        assert result[0].translated_text == "你好"

    def test_fallback_on_json_error(self):
        """When JSON parsing fails, fallback to line-by-line matching"""
        client = _make_mock_client("你好\n世界")
        provider = QwenTranslator(client=client)
        blocks = [
            TranslationBlock(id="b1", ocr_block_id="b1", source_text="Hello"),
            TranslationBlock(id="b2", ocr_block_id="b2", source_text="World"),
        ]
        ctx = TranslationContext()

        import asyncio
        result = asyncio.run(provider.translate(blocks, ctx))
        assert result[0].translated_text == "你好"
        assert result[1].translated_text == "世界"

    def test_empty_blocks(self):
        client = _make_mock_client("[]")
        provider = QwenTranslator(client=client)
        ctx = TranslationContext()

        import asyncio
        result = asyncio.run(provider.translate([], ctx))
        assert result == []

    def test_model_name_property(self):
        client = _make_mock_client("")
        provider = QwenTranslator(client=client, model="qwen3.5:9b")
        assert provider._model == "qwen3.5:9b"
