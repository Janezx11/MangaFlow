"""Tests for Pydantic models"""

from comic_translator.models.context import TranslationContext
from comic_translator.models.enums import ComicStyle, ProcessingStatus
from comic_translator.models.glossary import GlossaryEntry, TranslationMemory
from comic_translator.models.ocr_block import OCRBlock
from comic_translator.models.project import ComicMetadata, ComicProject
from comic_translator.models.translation_block import TranslationBlock


class TestOCRPydanticModels:
    """测试 OCR 相关 Model"""

    def test_ocr_block_creation(self):
        block = OCRBlock(
            id="b1", page=1, bbox=(0, 0, 100, 50),
            text="Hello", confidence=0.95,
        )
        assert block.id == "b1"
        assert block.confidence == 0.95
        assert block.metadata == {}

    def test_ocr_block_defaults(self):
        block = OCRBlock(id="b1", page=1, bbox=(0, 0, 10, 10), text="test")
        assert block.language == "en"
        assert block.confidence == 1.0

    def test_ocr_block_confidence_validation(self):
        import pydantic
        try:
            OCRBlock(id="b1", page=1, bbox=(0, 0, 10, 10), text="test", confidence=1.5)
            raise AssertionError("Should have raised")
        except pydantic.ValidationError:
            pass


class TestTranslationPydanticModels:
    def test_translation_block_creation(self, sample_translation_block):
        assert sample_translation_block.source_text == "Hello, world!"
        assert sample_translation_block.translated_text == "你好，世界！"
        assert sample_translation_block.is_qa_passed is None

    def test_translation_block_defaults(self):
        block = TranslationBlock(id="t1", ocr_block_id="b1", source_text="hi")
        assert block.translated_text == ""
        assert block.qa_issues == []
        assert block.confidence == 0.0


class TestGlossaryPydanticModels:
    def test_glossary_entry_creation(self):
        entry = GlossaryEntry(source="Stark", target="斯塔克", category="character")
        assert entry.priority == 0
        assert entry.context is None

    def test_translation_memory_creation(self):
        tm = TranslationMemory(
            source_text="hello", translated_text="你好",
            context_hash="abc", similarity=0.95,
            provider="qwen", model="qwen3.5:9b",
        )
        assert tm.usage_count == 0


class TestContextModel:
    def test_context_defaults(self):
        ctx = TranslationContext()
        assert ctx.glossary == []
        assert ctx.translation_memory == []
        assert ctx.comic_style == ComicStyle.WESTERN_COMIC
        assert ctx.source_language == "en"
        assert ctx.target_language == "zh"


class TestProjectModel:
    def test_project_creation(self):
        meta = ComicMetadata(title="Test Comic", total_pages=10)
        project = ComicProject(
            id="proj_001", name="test", metadata=meta,
            input_dir="/in", output_dir="/out", cache_dir="/cache",
        )
        assert project.status == ProcessingStatus.PENDING
        assert project.pages_completed == 0

    def test_metadata_defaults(self):
        meta = ComicMetadata(title="Test", total_pages=1)
        assert meta.comic_style == ComicStyle.WESTERN_COMIC
        assert meta.source_language == "en"
