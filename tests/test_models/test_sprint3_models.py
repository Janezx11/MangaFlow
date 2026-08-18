"""Tests for Sprint 3 models"""

from comic_translator.models.character_identity import CharacterIdentity, IdentityMatchResult
from comic_translator.models.page_summary import PageSummary
from comic_translator.models.merge_decision import MergeDecision


class TestCharacterIdentityModel:
    def test_defaults(self):
        ci = CharacterIdentity(character_id="tony", canonical_name="Tony Stark")
        assert ci.aliases == []
        assert ci.titles == []
        assert ci.nicknames == []
        assert ci.related_character_ids == []

    def test_full_fields(self):
        ci = CharacterIdentity(
            character_id="tony",
            canonical_name="Tony Stark",
            aliases=["Tony", "Mr. Stark"],
            titles=["Iron Man", "CEO of Stark Industries"],
            nicknames=["Shellhead", "Point Break"],
            speaking_style="sarcastic",
            related_character_ids=["pepper", "happy"],
        )
        assert len(ci.aliases) == 2
        assert "Iron Man" in ci.titles
        assert "Shellhead" in ci.nicknames
        assert "pepper" in ci.related_character_ids

    def test_match_result(self):
        ci = CharacterIdentity(character_id="tony", canonical_name="Tony Stark")
        result = IdentityMatchResult(identity=ci, match_type="alias", confidence=0.95)
        assert result.match_type == "alias"
        assert result.identity.canonical_name == "Tony Stark"


class TestPageSummaryModel:
    def test_defaults(self):
        ps = PageSummary(page_number=1, summary="Test summary")
        assert ps.characters == []
        assert ps.locations == []
        assert ps.events == []

    def test_full(self):
        ps = PageSummary(
            page_number=1,
            summary="Tony arrived at the tower",
            characters=["Tony Stark"],
            locations=["Stark Tower"],
            events=["arrival"],
        )
        assert len(ps.characters) == 1


class TestMergeDecisionModel:
    def test_defaults(self):
        md = MergeDecision()
        assert md.score == 0.0
        assert md.should_merge is False
        assert md.reason == ""

    def test_good_merge(self):
        md = MergeDecision(
            score=0.85,
            x_overlap_score=0.9,
            y_distance_score=0.95,
            punctuation_score=0.8,
            reading_order_score=0.9,
            should_merge=True,
            reason="all criteria met",
        )
        assert md.score > 0.6
        assert md.should_merge is True