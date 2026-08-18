"""Tests for TranslationMemoryService"""

import pytest

from comic_translator.services.translation_memory_service import TranslationMemoryService


@pytest.fixture
def tms(tmp_path):
    _counter = [0]

    def _make():
        _counter[0] += 1
        db = str(tmp_path / f"test_tm_{_counter[0]}.db")
        return TranslationMemoryService(db_path=db)
    return _make


class TestTranslationMemoryService:
    def test_store_and_find_exact(self, tms):
        s = tms()
        s.store("Hello, world!", "你好，世界！", provider="qwen", model="qwen3.5:9b")
        results = s.find_similar("Hello, world!")
        assert len(results) >= 1
        assert results[0].translated_text == "你好，世界！"
        assert results[0].similarity == 1.0

    def test_find_like_match(self, tms):
        s = tms()
        s.store("Hello, world!", "你好，世界！")
        results = s.find_similar("Hello")
        assert len(results) >= 1

    def test_empty_tm(self, tms):
        s = tms()
        results = s.find_similar("Unknown text")
        assert results == []

    def test_get_stats(self, tms):
        s = tms()
        stats = s.get_stats()
        assert "total_entries" in stats
        assert stats["total_entries"] == 0

        s.store("test", "测试")
        stats = s.get_stats()
        assert stats["total_entries"] == 1