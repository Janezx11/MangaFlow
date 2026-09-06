from __future__ import annotations

from typing import TYPE_CHECKING

from comic_translator.models.merge_decision import MERGE_THRESHOLD, MergeDecision

if TYPE_CHECKING:
    from comic_translator.models.ocr_block import OCRBlock


class MergeConfidenceService:
    """合并置信度引擎 —— 多维打分决定是否合并两个 OCR 块

    评分维度:
    - x_overlap_score: 水平投影重叠比例 [0, 1]
    - y_distance_score: 垂直距离越近分越高 [0, 1]
    - punctuation_score: 行尾标点判断 [0 or 1]
    - reading_order_score: 阅读顺序连续性 [0 or 1]

    总分 = 加权平均, score > threshold 才允许合并。
    """

    def __init__(self, threshold: float = MERGE_THRESHOLD) -> None:
        self.threshold = threshold

    def evaluate(self, block_a: OCRBlock, block_b: OCRBlock, line_height: int = 16) -> MergeDecision:
        """评估两个 OCR 块是否应该合并"""
        x_score = self._calc_x_overlap(block_a, block_b)
        y_score = self._calc_y_distance(block_a, block_b, line_height)
        p_score = self._calc_punctuation(block_a)
        r_score = self._calc_reading_order(block_a, block_b)

        # Weighted score
        score = (
            x_score * 0.35
            + y_score * 0.35
            + p_score * 0.15
            + r_score * 0.15
        )

        # Determine reason
        reasons: list[str] = []
        if x_score < 0.3:
            reasons.append(f"low x_overlap ({x_score:.2f})")
        if y_score < 0.3:
            reasons.append(f"large y_gap ({y_score:.2f})")
        if p_score == 1.0:
            reasons.append("ends with punctuation")
        if r_score == 0.0:
            reasons.append("reading order discontinuity")

        should_merge = score >= self.threshold

        return MergeDecision(
            score=round(score, 4),
            x_overlap_score=round(x_score, 4),
            y_distance_score=round(y_score, 4),
            punctuation_score=round(p_score, 4),
            reading_order_score=round(r_score, 4),
            should_merge=should_merge,
            reason="; ".join(reasons) if reasons else "all criteria met",
        )

    def batch_evaluate(
        self, blocks: list[OCRBlock], line_height: int = 16
    ) -> list[list[MergeDecision]]:
        """评估所有相邻块组合的合并可能性"""
        decisions: list[list[MergeDecision]] = []
        for i in range(len(blocks)):
            row: list[MergeDecision] = []
            for j in range(len(blocks)):
                if i == j:
                    row.append(MergeDecision(score=1.0, should_merge=False, reason="self"))
                elif j > i:
                    row.append(self.evaluate(blocks[i], blocks[j], line_height))
                else:
                    row.append(self.evaluate(blocks[j], blocks[i], line_height))
            decisions.append(row)
        return decisions

    # --- Scoring dimensions ---

    def _calc_x_overlap(self, a: OCRBlock, b: OCRBlock) -> float:
        """水平重叠比例"""
        overlap = min(a.bbox[2], b.bbox[2]) - max(a.bbox[0], b.bbox[0])
        if overlap <= 0:
            return 0.0
        width_a = a.bbox[2] - a.bbox[0]
        width_b = b.bbox[2] - b.bbox[0]
        min_width = min(width_a, width_b)
        if min_width <= 0:
            return 0.0
        return min(1.0, overlap / min_width)

    def _calc_y_distance(self, a: OCRBlock, b: OCRBlock, line_height: int) -> float:
        """垂直距离评分: 越近越好"""
        gap = b.bbox[1] - a.bbox[3]
        if gap < 0:
            return 0.0  # overlapping vertically
        if line_height <= 0:
            return 0.5
        ratio = gap / line_height
        if ratio <= 0.5:
            return 1.0
        if ratio <= 1.5:
            return 1.0 - (ratio - 0.5) / 1.0
        if ratio <= 3.0:
            return 0.3
        return 0.0

    def _calc_punctuation(self, block: OCRBlock) -> float:
        """行尾标点: 以标点结尾 = 不太可能继续 (高分 = 应合并)"""
        text = block.text.strip()
        if not text:
            return 0.5
        last_char = text[-1]
        # Sentences ending with punctuation are less likely to need merging
        sentence_enders = {".", "!", "?", "。", "！", "？", "…", "\""}
        if last_char in sentence_enders:
            return 0.3  # Low merge need
        # Comma/连词结尾 = 很可能需要合并
        continuers = {",", "，", "、", "and", "or", "the", "a", "an"}
        last_word = text.split()[-1].lower() if text.split() else ""
        if last_char in {",", "，", "、"}:
            return 0.9
        if last_word in continuers:
            return 0.8
        return 0.6

    def _calc_reading_order(self, a: OCRBlock, b: OCRBlock) -> float:
        """阅读顺序连续性: 自然顺序 = 1.0, 乱序 = 0.0"""
        # If b is below and to the right of a, it's natural reading order
        y_ok = b.bbox[1] >= a.bbox[3]  # b is below a
        x_ok = abs(b.bbox[0] - a.bbox[0]) < max(a.bbox[2] - a.bbox[0], 10) * 2
        if y_ok and x_ok:
            return 1.0
        if y_ok:
            return 0.5
        return 0.0
