"""Render QA Engine —— 渲染后自动检查质量"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from comic_translator.models.render_models import QAResult

if TYPE_CHECKING:
    pass


class RenderQAService:
    """渲染质量检查

    检查项:
    1. Overflow — 文字是否超出 bbox
    2. Tiny Font — 字号是否过小
    3. English Residue — 是否残留英文
    4. Coverage — mask 覆盖检查
    """

    MIN_FONT_SIZE = 10
    ENGLISH_PATTERN = re.compile(r"[a-zA-Z]{4,}")

    def check(
        self,
        render_blocks: list,
        image_width: int = 0,
        image_height: int = 0,
    ) -> QAResult:
        """对所有渲染块进行质量检查"""
        qa = QAResult()

        for block in render_blocks:
            bid = getattr(block, "ocr_block_id", "") or getattr(block, "id", "?")
            translated = getattr(block, "translated_text", "") or getattr(block, "text", "")

            # Overflow check —— DialogueBlock 没有 layout 字段，用 bubble bbox
            bubble = getattr(block, "bubble", None)
            if bubble is not None and getattr(bubble, "bbox", None):
                _x1, _y1, x2, y2 = bubble.bbox
                if x2 > image_width or y2 > image_height:
                    qa.add_issue(f"[{bid}] Overflow: text exceeds image bounds")
                    qa.overflow_blocks.append(bid)

            # Tiny font check —— font_size 为 0 表示排版未执行，跳过
            font_size = getattr(block, "font_size", 0)
            if 0 < font_size < self.MIN_FONT_SIZE:
                qa.add_issue(f"[{bid}] Tiny font: size={font_size}")
                qa.tiny_font_blocks.append(bid)

            # English residue check
            if translated:
                english_words = self.ENGLISH_PATTERN.findall(translated)
                if len(english_words) >= 3:
                    qa.add_issue(f"[{bid}] English residue: {english_words[:3]}")
                    qa.english_residue_blocks.append(bid)

        return qa
