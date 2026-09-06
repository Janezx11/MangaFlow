from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING

from loguru import logger

from comic_translator.models.batch_result import BatchOutput, PageTranslation
from comic_translator.models.translation_block import TranslationBlock

if TYPE_CHECKING:
    from comic_translator.models.batch_result import BatchInput


class BatchPromptBuilder:
    """构建和解析 Batch 翻译的 Prompt

    职责:
    1. 构建 System Prompt
    2. 构建 User Prompt（含多页 OCR 文本 + Glossary）
    3. 解析模型返回的 JSON → BatchOutput
    """

    SYSTEM_PROMPT = (
        "You are a professional comic translator. Your tasks:\n"
        "1. Correct OCR errors in the source text (e.g. 'ANDPEw woodS' -> 'Andrew Woods')\n"
        "2. Translate the corrected text to Simplified Chinese\n"
        "3. Maintain consistency of character names, locations, and titles\n"
        "4. Preserve speaking style and emotional tone\n\n"
        "Output ONLY valid JSON. No explanations, no markdown fences."
    )

    def build_user_prompt(self, batch_input: BatchInput) -> str:
        """构建 User Prompt"""
        parts = [
            f"Comic: \"{batch_input.project_name}\"",
            f"Style: {batch_input.comic_style}",
            f"Language: {batch_input.source_language} -> {batch_input.target_language}",
            f"Batch: {batch_input.batch_index} ({len(batch_input.pages)} pages)",
            "",
        ]

        # Glossary section
        if batch_input.glossary:
            parts.append("Glossary (use these translations consistently):")
            for g in batch_input.glossary:
                parts.append(f"  {g.source} -> {g.target}")
            parts.append("")

        # Entity context section
        entity_context = getattr(batch_input, "entity_context", "")
        if entity_context:
            parts.append(entity_context)
            parts.append("")

        # Pages section
        parts.append("Pages to translate:")
        for page in batch_input.pages:
            parts.append(f"--- PAGE {page.page} ---")
            for block in page.blocks:
                parts.append(f"[id={block.id}] {block.text}")
        parts.append("")

        # Output format section
        parts.append(
            "Output JSON format:\n"
            '{\n'
            '  "glossary": {"CharacterName": "角色名", ...},\n'
            '  "pages": [\n'
            '    {\n'
            '      "page": 1,\n'
            '      "blocks": [\n'
            '        {"id": "...", "corrected": "...", "translated": "..."}\n'
            '      ]\n'
            '    }\n'
            '  ]\n'
            '}'
        )

        return "\n".join(parts)

    def build_retry_prompt(self, batch_input: BatchInput, retry_reason: str = "") -> str:
        """构建带失败上下文的重跑 prompt

        与普通 prompt 的区别：
        1. 开头标注 RETRY 标记
        2. 注入失败原因作为额外指令
        3. 更强调"必须完整翻译所有文本"
        """
        parts = [
            "⚠️ RETRY - Previous translation had quality issues",
            f"Comic: \"{batch_input.project_name}\"",
            f"Style: {batch_input.comic_style}",
            f"Language: {batch_input.source_language} -> {batch_input.target_language}",
            "",
        ]

        if retry_reason:
            parts.append(f"Failure reason: {retry_reason}")
            parts.append("")

        # Glossary section
        if batch_input.glossary:
            parts.append("Glossary (use these translations consistently):")
            for g in batch_input.glossary:
                parts.append(f"  {g.source} -> {g.target}")
            parts.append("")

        # Entity context section
        entity_context = getattr(batch_input, "entity_context", "")
        if entity_context and "RETRY" not in entity_context:
            parts.append(entity_context)
            parts.append("")

        # Pages section
        parts.append("Page to translate (retry):")
        for page in batch_input.pages:
            parts.append(f"--- PAGE {page.page} ---")
            for block in page.blocks:
                parts.append(f"[id={block.id}] {block.text}")
        parts.append("")

        # Strong retry instruction
        parts.append(
            "IMPORTANT RETRY INSTRUCTIONS:\n"
            "1. This is a retry because the previous translation was unsatisfactory.\n"
            "2. You MUST translate EVERY text block completely to Simplified Chinese.\n"
            "3. Do NOT leave any text in the original language.\n"
            f"4. Specific issue to fix: {retry_reason}\n"
            "5. Output ONLY valid JSON. No explanations, no markdown fences."
        )
        parts.append("")

        # Output format section
        parts.append(
            "Output JSON format:\n"
            '{\n'
            '  "glossary": {"CharacterName": "角色名", ...},\n'
            '  "pages": [\n'
            '    {\n'
            '      "page": 1,\n'
            '      "blocks": [\n'
            '        {"id": "...", "corrected": "...", "translated": "..."}\n'
            '      ]\n'
            '    }\n'
            '  ]\n'
            '}'
        )

        return "\n".join(parts)

    def parse_response(self, response: str, batch_input: BatchInput) -> BatchOutput:
        """解析模型返回的 JSON 为 BatchOutput"""
        response = response.strip()

        # Strip markdown fences
        if response.startswith("```"):
            response = re.sub(r"^```\w*\n?", "", response)
            response = re.sub(r"\n```$", "", response)
            response = response.strip()

        # Find JSON
        start = response.find("{")
        end = response.rfind("}") + 1
        if start >= 0 and end > start:
            response = response[start:end]

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Batch JSON parse failed at char {e.pos}")
            return BatchOutput(batch_index=batch_input.batch_index, pages=[])

        raw_glossary = parsed.get("glossary", {})
        raw_pages = parsed.get("pages", [])

        # Parse glossary
        glossary = {}
        if isinstance(raw_glossary, dict):
            glossary = {str(k): str(v) for k, v in raw_glossary.items()}

        # 原文 id→text 映射：模型漏掉 corrected 时回退原始 OCR 文本
        original_text_by_id = {
            block.id: block.text
            for page in batch_input.pages
            for block in page.blocks
        }

        # Parse pages
        pages: list[PageTranslation] = []
        for raw_page in raw_pages:
            page_num = int(raw_page.get("page", 0))
            raw_blocks = raw_page.get("blocks", [])
            blocks = []
            for raw_block in raw_blocks:
                bid = str(raw_block.get("id", ""))
                corrected = str(raw_block.get("corrected", ""))
                translated = str(raw_block.get("translated", ""))
                # Use corrected text as source if available, else original
                source = corrected or original_text_by_id.get(bid, "")
                blocks.append(TranslationBlock(
                    id=bid,
                    ocr_block_id=bid,
                    source_text=source,
                    translated_text=translated,
                    provider="batch_api",
                    model="",
                ))
            if blocks:
                pages.append(PageTranslation(page=page_num, blocks=blocks))

        logger.info(f"Parsed batch {batch_input.batch_index}: {len(pages)} pages, {len(glossary)} glossary entries")
        return BatchOutput(
            batch_index=batch_input.batch_index,
            pages=pages,
            glossary=glossary,
            raw_response=response,
        )
