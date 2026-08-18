"""Tests for GlossaryService (SQLite-backed)"""

import pytest

from comic_translator.models.glossary import GlossaryEntry
from comic_translator.services.glossary_service import GlossaryService


@pytest.fixture
def gs(tmp_path):
    """Unique DB per test call"""
    _counter = [0]

    def _make():
        _counter[0] += 1
        db = str(tmp_path / f"test_glossary_{_counter[0]}.db")
        return GlossaryService(db_path=db)
    return _make


class TestGlossaryService:
    def test_add_and_get_term(self, gs):
        s = gs()
        entry = GlossaryEntry(source="Stark", target="斯塔克", category="character")
        term_id = s.add_term(entry)
        assert term_id > 0

        retrieved = s.get_term(term_id)
        assert retrieved is not None
        assert retrieved.source == "Stark"
        assert retrieved.target == "斯塔克"
        assert retrieved.category == "character"

    def test_update_term(self, gs):
        s = gs()
        entry = GlossaryEntry(source="Tony", target="托尼")
        term_id = s.add_term(entry)

        updated = GlossaryEntry(source="Tony", target="托尼·斯塔克", category="character")
        result = s.update_term(term_id, updated)
        assert result is True

        retrieved = s.get_term(term_id)
        assert retrieved is not None
        assert retrieved.target == "托尼·斯塔克"

    def test_delete_term(self, gs):
        s = gs()
        entry = GlossaryEntry(source="Stark Tower", target="斯塔克大厦")
        term_id = s.add_term(entry)

        result = s.delete_term(term_id)
        assert result is True

        retrieved = s.get_term(term_id)
        assert retrieved is None

    def test_search_terms(self, gs):
        s = gs()
        s.add_term(GlossaryEntry(source="Stark", target="斯塔克", category="character"))
        s.add_term(GlossaryEntry(source="Stark Tower", target="斯塔克大厦", category="location"))
        s.add_term(GlossaryEntry(source="Hello", target="你好", category="general"))

        results = s.search_terms("Stark")
        assert len(results) == 2

        results = s.search_terms("Stark", category="character")
        assert len(results) == 1

    def test_get_all_entries(self, gs):
        s = gs()
        s.add_term(GlossaryEntry(source="A", target="甲"))
        s.add_term(GlossaryEntry(source="B", target="乙"))
        all_entries = s.get_all_entries()
        assert len(all_entries) == 2

    def test_get_all_entries_by_category(self, gs):
        s = gs()
        s.add_term(GlossaryEntry(source="A", target="甲", category="character"))
        s.add_term(GlossaryEntry(source="B", target="乙", category="location"))
        chars = s.get_all_entries(category="character")
        assert len(chars) == 1

    def test_count(self, gs):
        s = gs()
        assert s.count() == 0
        s.add_term(GlossaryEntry(source="A", target="甲"))
        assert s.count() == 1