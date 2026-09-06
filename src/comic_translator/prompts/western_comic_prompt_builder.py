from __future__ import annotations

from typing import TYPE_CHECKING

from comic_translator.models.translation_profile import WESTERN_COMIC_PROFILE
from comic_translator.prompts.base import PromptBuilder

if TYPE_CHECKING:
    from comic_translator.models.context import TranslationContext
    from comic_translator.models.translation_block import TranslationBlock


class WesternComicPromptBuilder(PromptBuilder):
    """Western comic style prompt builder — Sprint 3 升级版

    使用 TranslationProfile V2 (forbidden_patterns, preferred_patterns)。
    新增 build_summary_prompt() 支持跨页上下文。
    """

    profile = WESTERN_COMIC_PROFILE

    def build_translation_prompt(
        self,
        block: TranslationBlock,
        context: TranslationContext,
    ) -> str:
        glossary_section = ""
        if context.glossary:
            lines = [
                f"  {g.source} → {g.target} ({g.category})"
                for g in context.glossary
            ]
            glossary_section = "Preserve these established translations:\n" + "\n".join(lines)

        character_section = ""
        chars = context.metadata.get("characters", [])
        if chars:
            lines = []
            for c in chars:
                name = c.get("canonical_name", c.get("name", "?"))
                style = c.get("speaking_style", "")
                if style:
                    lines.append(f"  {name} — speaking style: {style}")
                else:
                    lines.append(f"  {name}")
            character_section = "Characters in this scene:\n" + "\n".join(lines)

        summary_section = ""
        if context.page_summary:
            summary_section = f"Previous page context:\n{context.page_summary}"

        tm_section = ""
        if context.translation_memory:
            lines = [
                f"  {tm.source_text} → {tm.translated_text}"
                for tm in context.translation_memory[:3]
            ]
            tm_section = "Previous translations for reference:\n" + "\n".join(lines)

        rules_section = ""
        rules = context.metadata.get("translation_rules", self.profile.translation_rules)
        forbidden = context.metadata.get("forbidden_patterns", self.profile.forbidden_patterns)
        preferred = context.metadata.get("preferred_patterns", self.profile.preferred_patterns)
        if rules:
            rules_section = "Rules:\n" + "\n".join(f"  - {r}" for r in rules)
        if forbidden:
            rules_section += "\nForbidden:\n" + "\n".join(f"  - {f}" for f in forbidden)
        if preferred:
            rules_section += "\nPreferred:\n" + "\n".join(f"  - {p}" for p in preferred)

        parts = [
            "Translate the following English comic text to Simplified Chinese.",
            "Style: natural conversational Chinese, suitable for speech bubbles.",
            "Keep it concise. DO NOT add explanations.",
        ]
        for section in [rules_section, glossary_section, character_section, summary_section, tm_section]:
            if section:
                parts.append(f"\n{section}")

        parts.append(f"\nText: {block.source_text}")
        parts.append("\nTranslated text (Chinese only, one line):")

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
            f"Review the following comic translations for accuracy and naturalness.\n"
            f"Flag any: mistranslations, unnatural Chinese, missing context.\n"
            f"Respond ONLY with PASS or FAIL for each block.\n\n{texts}"
        )

    def build_glossary_extraction_prompt(
        self,
        text: str,
        context: TranslationContext,
    ) -> str:
        rules = "\n".join(f"- {r}" for r in self.profile.translation_rules)
        return (
            f"Extract named entities (characters, locations, special terms) from the "
            f"following comic text:\n\n{text}\n\n"
            f"Rules:\n{rules}\n\n"
            f"Format: one entry per line: source | category\n"
            f"Categories: character, location, term"
        )

    def build_summary_prompt(
        self,
        blocks: list[TranslationBlock],
        context: TranslationContext,
    ) -> str:
        """生成页面摘要的 Prompt"""
        texts = "\n".join(
            f"  {b.source_text} → {b.translated_text}"
            for b in blocks
            if b.translated_text
        )
        chars = context.metadata.get("characters", [])
        char_names = ", ".join(c.get("canonical_name", "?") for c in chars) if chars else "None"

        return (
            f"Generate a brief summary (1-2 sentences in English) of what happens on this comic page.\n"
            f"Characters present: {char_names}\n"
            f"Dialogue summary:\n{texts}\n\n"
            f"Return JSON: {{\"summary\": \"...\", \"characters\": [...], \"locations\": [...], \"events\": [...]}}"
        )
