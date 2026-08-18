from __future__ import annotations

import math
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from comic_translator.models.ocr_block import OCRBlock
    from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock


class OCRNormalizeService:
    """OCR 结果标准化服务

    职责:
    1. sort_blocks — 按阅读顺序排序（从上到下，从左到右）
    2. merge_blocks — 合并被错误拆分的相邻文本块
    3. clean_text — 清理 OCR 噪音
    4. process — 完整 pipeline
    """

    # 垂直合并阈值（行高比例）
    VERTICAL_MERGE_THRESHOLD = 1.5

    def process(
        self,
        blocks: list[OCRBlock],
        page: int = 0,
        language: str = "en",
        reading_order: str = "western",
    ) -> list[NormalizedOCRBlock]:
        """完整处理 pipeline"""
        # Step 1: 排序
        sorted_blocks = self.sort_blocks(blocks, reading_order)
        # Step 2: 合并
        merged = self.merge_blocks(sorted_blocks, reading_order)
        # Step 3: 清理
        cleaned = self.clean_text(merged)
        # Step 4: 分配阅读顺序
        for i, block in enumerate(cleaned):
            block.reading_order = i + 1
        return cleaned

    def sort_blocks(
        self,
        blocks: list[OCRBlock],
        reading_order: str = "western",
    ) -> list[OCRBlock]:
        """按阅读顺序排序

        英语漫画：从上到下 → 从左到右
        日漫：从右到左
        """
        if not blocks:
            return []

        # 估计行高（取所有 bbox 高度的中位数）
        heights = [b.bbox[3] - b.bbox[1] for b in blocks if b.bbox[3] > b.bbox[1]]
        if not heights:
            return blocks
        line_height = sorted(heights)[len(heights) // 2]

        def sort_key(block: OCRBlock) -> tuple[int, int]:
            x1, y1, x2, y2 = block.bbox
            row = int(y1 / max(line_height, 1))
            col = int(x1)
            return (row, col)

        return sorted(blocks, key=sort_key)

    def merge_blocks(
        self,
        blocks: list[OCRBlock],
        reading_order: str = "western",
    ) -> list[NormalizedOCRBlock]:
        """合并被错误拆分的相邻文本块

        例如:
        Hello
        World
        ↓
        Hello World

        合并条件:
        - 水平方向: x 轴投影重叠
        - 垂直方向: y 轴距离 < VERTICAL_MERGE_THRESHOLD * line_height
        """
        if not blocks:
            return []

        from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock

        heights = [b.bbox[3] - b.bbox[1] for b in blocks if b.bbox[3] > b.bbox[1]]
        line_height = sorted(heights)[len(heights) // 2] if heights else 16

        result: list[NormalizedOCRBlock] = []
        used = [False] * len(blocks)

        for i, block in enumerate(blocks):
            if used[i]:
                continue

            merged_text = block.text
            merged_bbox = list(block.bbox)
            merged_ids = [block.id]

            # 尝试合并后面的块
            for j in range(i + 1, len(blocks)):
                if used[j]:
                    continue
                other = blocks[j]

                # 检查水平投影是否重叠
                x_overlap = min(merged_bbox[2], other.bbox[2]) - max(merged_bbox[0], other.bbox[0])
                if x_overlap <= 0:
                    continue

                # 检查垂直距离
                vertical_gap = other.bbox[1] - merged_bbox[3]
                if vertical_gap < 0 or vertical_gap > line_height * self.VERTICAL_MERGE_THRESHOLD:
                    continue

                # 合并
                merged_text = merged_text.rstrip() + " " + other.text.strip()
                merged_bbox[2] = max(merged_bbox[2], other.bbox[2])
                merged_bbox[3] = max(merged_bbox[3], other.bbox[3])
                merged_ids.append(other.id)
                used[j] = True

            nb = NormalizedOCRBlock(
                id=block.id,
                page=block.page,
                bbox=tuple(merged_bbox),
                text=merged_text.strip(),
                confidence=block.confidence,
                language=block.language,
                is_merged=len(merged_ids) > 1,
                merged_from_ids=merged_ids[1:] if len(merged_ids) > 1 else [],
            )
            result.append(nb)
            used[i] = True

        return result

    def clean_text(self, blocks: list[NormalizedOCRBlock]) -> list[NormalizedOCRBlock]:
        """清理 OCR 噪音"""
        import re
        for block in blocks:
            text = block.text
            # 替换多余空白
            text = re.sub(r"\s+", " ", text)
            # 替换异常换行
            text = text.replace("\n", " ")
            # 去掉首尾空白
            text = text.strip()
            # 去掉空文本
            if text in (".", "|", "-", "—", "..."):
                text = ""
            block.text = text
        return [b for b in blocks if b.text.strip()]

    def estimate_line_height(self, blocks: list[OCRBlock]) -> int:
        """估算行高"""
        heights = [
            b.bbox[3] - b.bbox[1]
            for b in blocks
            if b.bbox[3] > b.bbox[1]
        ]
        if not heights:
            return 16
        return int(sorted(heights)[len(heights) // 4])