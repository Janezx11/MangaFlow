"""Tests for new models"""

from comic_translator.models.character_memory import CharacterMemory, CharacterMemoryEntry
from comic_translator.models.translation_profile import (
    TranslationProfile,
    WESTERN_COMIC_PROFILE,
    MANGA_PROFILE,
    WEBTOON_PROFILE,
)
from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock
from comic_translator.models.enums import ComicStyle


class TestCharacterMemoryModel:
    def test_character_memory_defaults(self):
        cm = CharacterMemory(character_id="c1", canonical_name="Tony Stark")
        assert cm.aliases == []
        assert cm.description == ""
        assert cm.first_seen_page == 0
        assert cm.last_seen_page == 0

    def test_character_memory_entry(self):
        cm = CharacterMemory(character_id="c1", canonical_name="Tony Stark", aliases=["Tony"])
        entry = CharacterMemoryEntry(character=cm, match_type="alias", confidence=0.8)
        assert entry.match_type == "alias"
        assert entry.character.canonical_name == "Tony Stark"


class TestTranslationProfileModel:
    def test_default_profile(self):
        profile = TranslationProfile(profile_name="test")
        assert profile.comic_style == ComicStyle.WESTERN_COMIC
        assert profile.tone == "natural"
        assert profile.localization_level == "moderate"

    def test_western_profile(self):
        assert WESTERN_COMIC_PROFILE.profile_name == "western_comic"
        assert len(WESTERN_COMIC_PROFILE.translation_rules) == 4

    def test_manga_profile(self):
        assert MANGA_PROFILE.profile_name == "manga"
        assert MANGA_PROFILE.comic_style == ComicStyle.MANGA

    def test_webtoon_profile(self):
        assert WEBTOON_PROFILE.profile_name == "webtoon"
        assert WEBTOON_PROFILE.comic_style == ComicStyle.WEBTOON


class TestNormalizedOCRBlock:
    def test_defaults(self):
        block = NormalizedOCRBlock(
            id="b1", page=1, bbox=(0, 0, 10, 10), text="test"
        )
        assert block.reading_order == 0
        assert block.is_merged is False
        assert block.merged_from_ids == []

    def test_merged_block(self):
        block = NormalizedOCRBlock(
            id="b1", page=1, bbox=(0, 0, 10, 10), text="test",
            reading_order=1, is_merged=True, merged_from_ids=["b2", "b3"],
        )
        assert block.reading_order == 1
        assert block.is_merged is True
        assert block.merged_from_ids == ["b2", "b3"]