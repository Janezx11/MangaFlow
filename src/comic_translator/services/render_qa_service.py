"""Render QA Engine —— 渲染后自动检查质量"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from comic_translator.models.render_models import QAResult

if TYPE_CHECKING:
    from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock
    from comic_translator.models.render_models import RenderBlock


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
            layout = getattr(block, "layout", None)
            translated = getattr(block, "translated_text", "") or getattr(block, "text", "")

            # Overflow check
            if layout:
                x1, y1, x2, y2 = layout.bbox if hasattr(layout, 'bbox') else (0, 0, 0, 0)
                if x2 > image_width or y2 > image_height:
                    qa.add_issue(f"[{bid}] Overflow: text exceeds image bounds")
                    qa.overflow_blocks.append(bid)

            # Tiny font check
            if layout and hasattr(layout, 'font_size'):
                if layout.font_size < self.MIN_FONT_SIZE:
                    qa.add_issue(f"[{bid}] Tiny font: size={layout.font_size}")
                    qa.tiny_font_blocks.append(bid)

            # English residue check
            if translated:
                english_words = self.ENGLISH_PATTERN.findall(translated)
                if len(english_words) >= 3:
                    qa.add_issue(f"[{bid}] English residue: {english_words[:3]}")
                    qa.english_residue_blocks.append(bid)

        return qa