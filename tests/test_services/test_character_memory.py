"""Tests for CharacterMemoryService"""

import pytest

from comic_translator.models.character_memory import CharacterMemory
from comic_translator.services.character_memory_service import CharacterMemoryService


@pytest.fixture
def cs(tmp_path):
    """Unique DB per test call"""
    _counter = [0]

    def _make():
        _counter[0] += 1
        db = str(tmp_path / f"test_char_{_counter[0]}.db")
        return CharacterMemoryService(db_path=db)
    return _make


class TestCharacterMemoryService:
    def test_register_and_resolve_exact(self, cs):
        s = cs()
        char = CharacterMemory(
            character_id="tony_stark",
            canonical_name="Tony Stark",
            aliases=["Tony", "Mr. Stark", "Iron Man"],
            description="Billionaire inventor, Iron Man",
            speaking_style="sarcastic, confident",
        )
        s.register_character(char)

        result = s.resolve_name("Tony Stark")
        assert result is not None
        assert result.match_type == "exact"
        assert result.character.canonical_name == "Tony Stark"

    def test_resolve_by_alias(self, cs):
        s = cs()
        char = CharacterMemory(
            character_id="tony_stark",
            canonical_name="Tony Stark",
            aliases=["Tony", "Mr. Stark", "Iron Man"],
        )
        s.register_character(char)

        result = s.resolve_name("Mr. Stark")
        assert result is not None
        assert result.match_type == "alias"
        assert result.character.canonical_name == "Tony Stark"

    def test_resolve_by_partial(self, cs):
        s = cs()
        char = CharacterMemory(
            character_id="tony_stark",
            canonical_name="Tony Stark",
            aliases=["Tony", "Mr. Stark"],
        )
        s.register_character(char)

        result = s.resolve_name("Tony")
        assert result is not None
        assert result.match_type == "alias"
        assert result.character.canonical_name == "Tony Stark"

    def test_resolve_unknown(self, cs):
        s = cs()
        result = s.resolve_name("Unknown")
        assert result is None

    def test_batch_register(self, cs):
        s = cs()
        chars = [
            CharacterMemory(character_id="c1", canonical_name="Char One"),
            CharacterMemory(character_id="c2", canonical_name="Char Two"),
        ]
        count = s.batch_register(chars)
        assert count == 2

        all_chars = s.get_all_characters()
        assert len(all_chars) == 2

    def test_update_last_seen(self, cs):
        s = cs()
        char = CharacterMemory(character_id="c1", canonical_name="Char One")
        s.register_character(char)
        s.update_last_seen("c1", 10)

        result = s.resolve_name("Char One")
        assert result is not None
        assert result.character.last_seen_page == 10
