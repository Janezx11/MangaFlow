"""Tests for SemanticTranslationMemoryService"""

import pytest

from comic_translator.services.semantic_tm_service import SemanticTranslationMemoryService


@pytest.fixture
def stm(tmp_path):
    _counter = [0]
    def _make():
        _counter[0] += 1
        return SemanticTranslationMemoryService(db_path=str(tmp_path / f"test_stm_{_counter[0]}.db"))
    return _make


class TestSemanticTMService:
    def test_store_and_find_exact(self, stm):
        s = stm()
        s.store("Hello, world!", "你好，世界！", provider="qwen", model="qwen3.5:9b")
        results = s.find_similar("Hello, world!")
        assert len(results) >= 1
        assert results[0].translated_text == "你好，世界！"
        assert results[0].similarity == 1.0

    def test_dedup_on_store(self, stm):
        s = stm()
        id1 = s.store("Hello", "你好")
        id2 = s.store("Hello", "你好")  # Same text
        assert id2 == id1  # Should return existing ID (dedup)

    def test_like_match(self, stm):
        s = stm()
        s.store("Hello, world!", "你好，世界！")
        results = s.find_similar("Hello")
        assert len(results) >= 1

    def test_empty(self, stm):
        s = stm()
        results = s.find_similar("Nonexistent")
        assert results == []

    def test_get_stats(self, stm):
        s = stm()
        assert s.get_stats()["total_entries"] == 0
        s.store("test", "测试")
        assert s.get_stats()["total_entries"] == 1

    def test_get_recent(self, stm):
        s = stm()
        s.store("first", "第一")
        s.store("second", "第二")
        recent = s.get_recent(limit=10)
        assert len(recent) >= 2
