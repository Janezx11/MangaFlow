"""Tests for CharacterIdentityService"""

import pytest

from comic_translator.models.character_identity import CharacterIdentity
from comic_translator.services.character_identity_service import CharacterIdentityService


@pytest.fixture
def cis(tmp_path):
    _counter = [0]
    def _make():
        _counter[0] += 1
        return CharacterIdentityService(db_path=str(tmp_path / f"test_ci_{_counter[0]}.db"))
    return _make


class TestCharacterIdentityService:
    def test_create_and_get(self, cis):
        s = cis()
        sid = s.create_identity(CharacterIdentity(
            character_id="tony_stark", canonical_name="Tony Stark"
        ))
        assert sid > 0
        identity = s.get_identity("tony_stark")
        assert identity is not None
        assert identity.canonical_name == "Tony Stark"

    def test_resolve_exact(self, cis):
        s = cis()
        s.create_identity(CharacterIdentity(character_id="t1", canonical_name="Tony Stark", aliases=["Tony"]))
        result = s.resolve_name("Tony Stark")
        assert result is not None
        assert result.match_type == "exact"

    def test_resolve_alias(self, cis):
        s = cis()
        s.create_identity(CharacterIdentity(character_id="t1", canonical_name="Tony Stark", aliases=["Tony", "Mr. Stark"]))
        result = s.resolve_name("Mr. Stark")
        assert result is not None
        assert result.match_type == "alias"
        assert result.identity.canonical_name == "Tony Stark"

    def test_resolve_title(self, cis):
        s = cis()
        s.create_identity(CharacterIdentity(character_id="t1", canonical_name="Tony Stark", titles=["Iron Man"]))
        result = s.resolve_name("Iron Man")
        assert result is not None
        assert result.match_type == "title"

    def test_resolve_nickname(self, cis):
        s = cis()
        s.create_identity(CharacterIdentity(character_id="t1", canonical_name="Tony Stark", nicknames=["Shellhead"]))
        result = s.resolve_name("Shellhead")
        assert result is not None
        assert result.match_type == "nickname"

    def test_resolve_fuzzy(self, cis):
        s = cis()
        s.create_identity(CharacterIdentity(character_id="t1", canonical_name="Tony Stark", aliases=["Tony"]))
        result = s.resolve_name("To")
        assert result is not None
        assert result.match_type == "fuzzy"

    def test_resolve_unknown(self, cis):
        s = cis()
        result = s.resolve_name("Unknown")
        assert result is None

    def test_link_alias(self, cis):
        s = cis()
        s.create_identity(CharacterIdentity(character_id="t1", canonical_name="Tony Stark"))
        assert s.link_alias("t1", "Tony") is True
        result = s.resolve_name("Tony")
        assert result is not None
        assert result.match_type == "alias"

    def test_search_identity(self, cis):
        s = cis()
        s.create_identity(CharacterIdentity(character_id="t1", canonical_name="Tony Stark", aliases=["Iron Man"]))
        s.create_identity(CharacterIdentity(character_id="p1", canonical_name="Pepper Potts"))
        results = s.search_identity("Tony")
        assert len(results) == 1
        results = s.search_identity("Iron")
        assert len(results) == 1

    def test_update_identity(self, cis):
        s = cis()
        s.create_identity(CharacterIdentity(character_id="t1", canonical_name="Tony"))
        updated = CharacterIdentity(character_id="t1", canonical_name="Tony Stark", titles=["Iron Man"])
        assert s.update_identity("t1", updated) is True
        identity = s.get_identity("t1")
        assert identity.canonical_name == "Tony Stark"
