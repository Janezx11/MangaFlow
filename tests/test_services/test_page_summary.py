"""Tests for PageSummaryService"""

from comic_translator.models.page_summary import PageSummary
from comic_translator.services.page_summary_service import PageSummaryService


class TestPageSummaryService:
    def test_save_and_load_summary(self, tmp_path):
        svc = PageSummaryService(cache_dir=str(tmp_path))
        summary = PageSummary(page_number=1, summary="Test page", characters=["Tony"])
        svc._save_summary("proj1", summary)

        loaded = svc._load_summary("proj1", 1)
        assert loaded is not None
        assert loaded.summary == "Test page"
        assert loaded.characters == ["Tony"]

    def test_get_previous_summary(self, tmp_path):
        svc = PageSummaryService(cache_dir=str(tmp_path))
        svc._save_summary("proj1", PageSummary(page_number=1, summary="Page 1"))
        prev = svc.get_previous_summary("proj1", 2)
        assert prev is not None
        assert prev.summary == "Page 1"

    def test_get_previous_summary_page1(self, tmp_path):
        svc = PageSummaryService(cache_dir=str(tmp_path))
        prev = svc.get_previous_summary("proj1", 1)
        assert prev is None

    def test_get_recent_summaries(self, tmp_path):
        svc = PageSummaryService(cache_dir=str(tmp_path))
        for i in range(1, 6):
            svc._save_summary("proj1", PageSummary(page_number=i, summary=f"Page {i}"))

        recent = svc.get_recent_summaries("proj1", current_page=5, count=3)
        assert len(recent) == 3
        assert recent[0].page_number == 2
        assert recent[-1].page_number == 4

    def test_summary_exists(self, tmp_path):
        svc = PageSummaryService(cache_dir=str(tmp_path))
        assert svc.summary_exists("p", 1) is False
        svc._save_summary("p", PageSummary(page_number=1, summary="x"))
        assert svc.summary_exists("p", 1) is True

    def test_fallback_summary(self, tmp_path):
        from comic_translator.models.translation_block import TranslationBlock
        from comic_translator.models.context import TranslationContext
        svc = PageSummaryService(cache_dir=str(tmp_path))
        blocks = [TranslationBlock(id="b1", ocr_block_id="b1", source_text="Hello", translated_text="你好")]
        ctx = TranslationContext()

        import asyncio
        summary = asyncio.run(svc.generate_summary(blocks, ctx, page=1))
        assert summary.page_number == 1
        assert "Hello" in summary.summary