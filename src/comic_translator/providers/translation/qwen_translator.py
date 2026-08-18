from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loguru import logger

from comic_translator.clients.base import LLMClient
from comic_translator.providers.translation.base import TranslationProvider

if TYPE_CHECKING:
    from comic_translator.models.translation_block import TranslationBlock
    from comic_translator.models.context import TranslationContext


class QwenTranslator(TranslationProvider):
    """Qwen 翻译引擎 —— 通过 LLMClient 调用

    Provider 只依赖 LLMClient 抽象，不直接调用 Ollama SDK。
    未来切换到 OpenAI API 时，只需更换 LLMClient 实现。

    翻译策略：将所有文本块打包到一次 LLM 调用中批量翻译，
    返回 JSON 格式结果，按 id 映射到每个 TranslationBlock。
    """

    def __init__(self, client: LLMClient, model: str = "qwen3.5:9b") -> None:
        self._client = client
        self._model = model
        self._system_prompt = (
            "You are a professional comic translator specializing in English-to-Chinese translation. "
            "Your task is to translate comic dialogue and narration naturally while preserving:\n"
            "- Character voice and personality\n"
            "- Emotional tone (anger, surprise, whisper, etc.)\n"
            "- Cultural context (use standard Chinese localizations)\n\n"
            "Rules:\n"
            "1. Keep proper names untranslated unless they have standard Chinese translations\n"
            "2. Translate sound effects as Chinese onomatopoeia\n"
            "3. Keep text concise — match the speech bubble space\n"
            "4. DO NOT add explanations or notes\n"
            "5. Return ONLY valid JSON"
        )

    async def translate(
        self,
        blocks: list[TranslationBlock],
        context: TranslationContext,
    ) -> list[TranslationBlock]:
        if not blocks:
            return blocks

        # Build glossary context text
        glossary_text = ""
        if context.glossary:
            lines = [f"  {g.source} → {g.target} ({g.category})" for g in context.glossary]
            glossary_text = "Glossary:\n" + "\n".join(lines)

        # Build character context text
        character_text = ""
        if context.metadata.get("characters"):
            chars = context.metadata["characters"]
            char_lines = []
            for c in chars:
                name = c.get("canonical_name", c.get("name", "?"))
                style = c.get("speaking_style", "")
                if style:
                    char_lines.append(f"  {name} (speaking style: {style})")
                else:
                    char_lines.append(f"  {name}")
            if char_lines:
                character_text = "Characters:\n" + "\n".join(char_lines)

        # Build the text blocks section
        block_lines = [
            f"[id={b.id}] {b.source_text}" for b in blocks
        ]
        blocks_text = "\n".join(block_lines)

        # Build user prompt
        user_prompt_parts = [
            "Translate the following English comic text blocks to Simplified Chinese.",
        ]
        if glossary_text:
            user_prompt_parts.append(f"\n{glossary_text}")
        if character_text:
            user_prompt_parts.append(f"\n{character_text}")
        user_prompt_parts.append(
            f"\nText blocks to translate:\n{blocks_text}\n\n"
            f"Respond ONLY with a JSON array, no other text:\n"
            f'[{{"id": "...", "translated_text": "..."}}, ...]'
        )
        user_prompt = "\n".join(user_prompt_parts)

        logger.info(f"Translating {len(blocks)} blocks via {self._model}")

        try:
            response = await self._client.generate(
                prompt=user_prompt,
                system_prompt=self._system_prompt,
            )
            return self._parse_response(response, blocks)
        except Exception as e:
            logger.exception(f"Translation failed")
            logger.error(f"  Message: {e}")
            # Fallback: return blocks with source text as translated text
            for block in blocks:
                block.translated_text = block.source_text
            return blocks

    async def translate_single(
        self,
        block: TranslationBlock,
        context: TranslationContext,
    ) -> TranslationBlock:
        result = await self.translate([block], context)
        return result[0] if result else block

    def _parse_response(
        self, response: str, blocks: list[TranslationBlock]
    ) -> list[TranslationBlock]:
        """解析模型返回的 JSON 并映射到 blocks"""
        response = response.strip()

        # Remove markdown code fences if present
        if response.startswith("```"):
            response = response.split("\n", 1)[-1]
            response = response.rsplit("```", 1)[0]
            response = response.strip()

        # Try to extract JSON array
        start = response.find("[")
        end = response.rfind("]") + 1
        if start >= 0 and end > start:
            response = response[start:end]

        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON parse failed at char {e.pos}: {response[max(0,e.pos-20):e.pos+20]!r}")
            logger.warning(f"Full response preview: {response[:500]!r}")
            return self._fallback_parse(response, blocks)

        logger.debug(f"Model returned {len(parsed)} items: {parsed}")

        # Warn about duplicate IDs (model might return wrong IDs)
        seen_ids = {}
        for item in parsed:
            item_id = str(item.get("id", ""))
            if item_id in seen_ids:
                logger.warning(f"Duplicate ID '{item_id}' in model response, using positional fallback")
            seen_ids[item_id] = seen_ids.get(item_id, 0) + 1

        # Map parsed results to blocks
        id_map = {str(item.get("id")): item.get("translated_text", "") for item in parsed}

        for block in blocks:
            translated = id_map.get(block.id) or id_map.get(block.ocr_block_id, "")
            if not translated:
                logger.warning(f"Block {block.id}: no translation found in model response")
            block.translated_text = translated if translated else block.source_text

        # If any blocks still have empty translated_text (shouldn't happen with the fallback above),
        # use positional fallback
        for i, block in enumerate(blocks):
            if not block.translated_text:
                if i < len(parsed):
                    logger.warning(f"Block {block.id}: positional fallback to item {i}")
                    block.translated_text = parsed[i].get("translated_text", "") or block.source_text
                else:
                    block.translated_text = block.source_text

        return blocks

    def _fallback_parse(
        self, response: str, blocks: list[TranslationBlock]
    ) -> list[TranslationBlock]:
        """JSON 解析失败时的回退方案：按行匹配"""
        lines = [l.strip() for l in response.split("\n") if l.strip()]
        for i, block in enumerate(blocks):
            if i < len(lines):
                block.translated_text = lines[i]
            else:
                block.translated_text = block.source_text
        return blocks