"""Pytest fixtures for comic-translator"""

import pytest

from comic_translator.models.ocr_block import OCRBlock
from comic_translator.models.translation_block import TranslationBlock
from comic_translator.models.context import TranslationContext, QAReport
from comic_translator.models.glossary import GlossaryEntry
from comic_translator.models.enums import ComicStyle

# Disable async warnings in tests
pytest.register_assert_rewrite("comic_translator")


@pytest.fixture
def sample_ocr_block() -> OCRBlock:
    return OCRBlock(
        id="block_001",
        page=1,
        bbox=(10.0, 20.0, 100.0, 50.0),
        text="Hello, world!",
        confidence=0.95,
        language="en",
    )


@pytest.fixture
def sample_translation_block() -> TranslationBlock:
    return TranslationBlock(
        id="trans_001",
        ocr_block_id="block_001",
        source_text="Hello, world!",
        translated_text="你好，世界！",
        provider="qwen",
        model="qwen3.5:9b",
        confidence=0.9,
    )


@pytest.fixture
def sample_context() -> TranslationContext:
    return TranslationContext(
        glossary=[
            GlossaryEntry(source="Tony Stark", target="托尼·斯塔克", category="character"),
            GlossaryEntry(source="Stark Tower", target="斯塔克大厦", category="location"),
        ],
        comic_style=ComicStyle.WESTERN_COMIC,
        source_language="en",
        target_language="zh",
    )


@pytest.fixture
def sample_qa_report() -> QAReport:
    return QAReport(
        page=1,
        passed=True,
        blocks_checked=5,
        blocks_failed=0,
        provider="gemma",
    )