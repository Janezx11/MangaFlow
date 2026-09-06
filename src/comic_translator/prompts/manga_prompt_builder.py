from __future__ import annotations

from typing import TYPE_CHECKING

from comic_translator.models.translation_profile import MANGA_PROFILE
from comic_translator.prompts.base import PromptBuilder

if TYPE_CHECKING:
    from comic_translator.models.context import TranslationContext
    from comic_translator.models.translation_block import TranslationBlock


class MangaPromptBuilder(PromptBuilder):
    """Manga (Japanese comic) style prompt builder — Sprint 3 升级版"""

    profile = MANGA_PROFILE

    def build_translation_prompt(
        self,
        block: TranslationBlock,
        context: TranslationContext,
    ) -> str:
        rules = "\n".join(f"  - {r}" for r in context.metadata.get("translation_rules", self.profile.translation_rules))
        forbidden = context.metadata.get("forbidden_patterns", self.profile.forbidden_patterns)
        preferred = context.metadata.get("preferred_patterns", self.profile.preferred_patterns)

        parts = [
            f"Translate the following Japanese manga text to Simplified Chinese.\n{rules}",
        ]
        if forbidden:
            parts.append("Forbidden:\n" + "\n".join(f"  - {f}" for f in forbidden))
        if preferred:
            parts.append("Preferred:\n" + "\n".join(f"  - {p}" for p in preferred))

        glossary_section = ""
        if context.glossary:
            lines = [f"  {g.source} → {g.target}" for g in context.glossary]
            glossary_section = "Preserve these translations:\n" + "\n".join(lines)
            parts.append(f"\n{glossary_section}")

        if context.page_summary:
            parts.append(f"\nPrevious page context:\n{context.page_summary}")

        parts.append(f"\nText: {block.source_text}")
        parts.append("\nTranslated text (Chinese only):")
        return "\n".join(parts)

    def build_qa_prompt(
        self,
        translations: list[TranslationBlock],
        context: TranslationContext,
    ) -> str:
        texts = "\n".join(
            f"[{t.id}] {t.source_text} → {t.translated_text}"
            for t in translations
            if t.translated_text
        )
        return (
            f"Review the following manga translations.\n"
            f"Check: honorifics preserved, SFX handled correctly, natural Chinese.\n"
            f"Respond ONLY with PASS or FAIL for each block.\n\n{texts}"
        )

    def build_glossary_extraction_prompt(
        self,
        text: str,
        context: TranslationContext,
    ) -> str:
        return (
            f"Extract named entities from the following manga text:\n\n{text}\n\n"
            f"Format: source | category\n"
            f"Categories: character, location, term\n"
            f"Include honorific usage if present."
        )

    def build_summary_prompt(
        self,
        blocks: list[TranslationBlock],
        context: TranslationContext,
    ) -> str:
        texts = "\n".join(f"  {b.source_text} → {b.translated_text}" for b in blocks if b.translated_text)
        return (
            f"Generate a brief summary (1-2 sentences in English) of this manga page.\n"
            f"Return JSON.\n\n{texts}"
        )
