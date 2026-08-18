from __future__ import annotations

from typing import TYPE_CHECKING

from comic_translator.models.translation_profile import WEBTOON_PROFILE
from comic_translator.prompts.base import PromptBuilder

if TYPE_CHECKING:
    from comic_translator.models.translation_block import TranslationBlock
    from comic_translator.models.context import TranslationContext


class WebtoonPromptBuilder(PromptBuilder):
    """Webtoon (vertical scroll) style prompt builder — Sprint 3 升级版"""

    profile = WEBTOON_PROFILE

    def build_translation_prompt(
        self,
        block: TranslationBlock,
        context: TranslationContext,
    ) -> str:
        rules = "\n".join(f"  - {r}" for r in context.metadata.get("translation_rules", self.profile.translation_rules))
        forbidden = context.metadata.get("forbidden_patterns", self.profile.forbidden_patterns)
        preferred = context.metadata.get("preferred_patterns", self.profile.preferred_patterns)

        parts = [
            f"Translate the following webtoon text to Simplified Chinese.\n{rules}",
        ]
        if forbidden:
            parts.append("Forbidden:\n" + "\n".join(f"  - {f}" for f in forbidden))
        if preferred:
            parts.append("Preferred:\n" + "\n".join(f"  - {p}" for p in preferred))

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
            f"Review the following webtoon translations.\n"
            f"Check: correct length for vertical bubbles, natural Chinese.\n"
            f"Respond ONLY with PASS or FAIL for each block.\n\n{texts}"
        )

    def build_glossary_extraction_prompt(
        self,
        text: str,
        context: TranslationContext,
    ) -> str:
        return (
            f"Extract named entities from the following webtoon text:\n\n{text}\n\n"
            f"Format: source | category\nCategories: character, location, term"
        )

    def build_summary_prompt(
        self,
        blocks: list[TranslationBlock],
        context: TranslationContext,
    ) -> str:
        texts = "\n".join(f"  {b.source_text} → {b.translated_text}" for b in blocks if b.translated_text)
        return (
            f"Generate a brief summary (1-2 sentences in English) of this webtoon episode page.\n"
            f"Return JSON.\n\n{texts}"
        )