"""Tests for ContextAssembler"""

from comic_translator.models.enums import ComicStyle
from comic_translator.models.character_identity import CharacterIdentity
from comic_translator.models.glossary import GlossaryEntry
from comic_translator.models.page_summary import PageSummary
from comic_translator.prompts.western_comic_prompt_builder import WesternComicPromptBuilder
from comic_translator.services.context_assembler import ContextAssembler
from comic_translator.services.glossary_service import GlossaryService
from comic_translator.services.character_identity_service import CharacterIdentityService
from comic_translator.services.semantic_tm_service import SemanticTranslationMemoryService
from comic_translator.services.page_summary_service import PageSummaryService


def _make_assembler(tmp_path, db_path):
    gs = GlossaryService(db_path=db_path)
    cs = CharacterIdentityService(db_path=db_path)
    stm = SemanticTranslationMemoryService(db_path=db_path)
    ps = PageSummaryService(cache_dir=str(tmp_path / "cache"))
    pb = WesternComicPromptBuilder()
    return ContextAssembler(
        glossary_service=gs,
        character_identity_service=cs,
        semantic_tm_service=stm,
        page_summary_service=ps,
        prompt_builder=pb,
    ), gs, cs, ps


class TestContextAssembler:
    def test_assemble_basic(self, tmp_path):
        db = str(tmp_path / "ctx.db")
        assembler, _, _, _ = _make_assembler(tmp_path, db)
        ctx = assembler.assemble(project_id="p1", page=1)
        assert ctx.source_language == "en"
        assert ctx.target_language == "zh"
        assert ctx.comic_style == ComicStyle.WESTERN_COMIC

    def test_assemble_with_glossary(self, tmp_path):
        db = str(tmp_path / "ctx2.db")
        assembler, gs, _, _ = _make_assembler(tmp_path, db)
        gs.add_term(GlossaryEntry(source="Stark", target="斯塔克", category="character"))
        ctx = assembler.assemble(project_id="p1", page=1)
        assert len(ctx.glossary) == 1
        assert ctx.glossary[0].source == "Stark"

    def test_assemble_with_characters(self, tmp_path):
        db = str(tmp_path / "ctx3.db")
        assembler, _, cs, _ = _make_assembler(tmp_path, db)
        cs.create_identity(CharacterIdentity(
            character_id="tony", canonical_name="Tony Stark",
            aliases=["Tony"], speaking_style="sarcastic",
        ))
        ctx = assembler.assemble(project_id="p1", page=1)
        chars = ctx.metadata.get("characters", [])
        assert len(chars) == 1
        assert chars[0]["canonical_name"] == "Tony Stark"

    def test_assemble_with_summaries(self, tmp_path):
        db = str(tmp_path / "ctx4.db")
        assembler, _, _, ps = _make_assembler(tmp_path, db)
        ps._save_summary("p1", PageSummary(page_number=1, summary="Page 1 summary"))
        ps._save_summary("p1", PageSummary(page_number=2, summary="Page 2 summary"))
        ctx = assembler.assemble(project_id="p1", page=3)
        assert ctx.page_summary is not None
        assert "Page 1" in ctx.page_summary or "Page 2" in ctx.page_summary

    def test_assemble_with_profile(self, tmp_path):
        db = str(tmp_path / "ctx5.db")
        assembler, _, _, _ = _make_assembler(tmp_path, db)
        ctx = assembler.assemble(project_id="p1", page=1)
        assert "translation_rules" in ctx.metadata
        assert len(ctx.metadata["translation_rules"]) > 0

    def test_assemble_with_tm(self, tmp_path):
        db = str(tmp_path / "ctx6.db")
        assembler, _, _, _ = _make_assembler(tmp_path, db)
        assembler._tm.store("Hello", "你好")
        ctx = assembler.assemble_with_tm(
            project_id="p1", page=1,
            blocks_source_texts=["Hello"],
        )
        assert len(ctx.translation_memory) >= 1