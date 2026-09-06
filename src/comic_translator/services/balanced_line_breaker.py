"""BalancedLineBreaker — Knuth-Plass 平衡换行算法

采用 TeX 的经典 Knuth-Plass 动态规划换行算法。
代价函数考虑：
  1. Line length variance（行长度方差）
  2. Widow/orphan penalty（孤行惩罚）
  3. Center balance（居中对称性）

DP 递推: dp[i] = min(dp[j] + cost(j, i)) for all valid breaks j→i
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comic_translator.services.font_manager import FontManager


class BalancedLineBreaker:
    """Knuth-Plass 平衡换行

    break(text, max_widths, font_manager, font_path, font_size) → (lines, cost)

    max_widths: 每行的最大可用宽度（来自 ShapeLayoutSolver）
    """

    # 惩罚系数
    WIDOW_PENALTY = 50.0  # 孤行（最后一行只有一个词）
    OVERFLOW_PENALTY = 200.0  # 超宽行
    VARIANCE_PENALTY = 1.0  # 行长度方差
    BALANCE_PENALTY = 0.5  # 居中对称性

    def __init__(self, font_manager: FontManager) -> None:
        self._fm = font_manager

    @lru_cache(maxsize=2048)
    def _word_width(self, word: str, font_path: str, font_size: int) -> int:
        w, _ = self._fm.measure_text(word, font_path, font_size)
        return w

    def _text_width(self, text: str, font_path: str, font_size: int) -> int:
        w, _ = self._fm.measure_text(text, font_path, font_size)
        return w

    def break_text(
        self,
        text: str,
        max_widths: list[int],
        font_path: str,
        font_size: int,
    ) -> tuple[list[str], float]:
        """使用 Knuth-Plass DP 进行平衡换行

        Args:
            text: 输入文本
            max_widths: 每行可用宽度列表
            font_path: 字体路径
            font_size: 字号

        Returns:
            (lines, cost) — 换行结果和总代价
        """
        if not text.strip():
            return [], 0.0

        # 判断是否含大量 CJK → 按字符拆分
        char_count = sum(1 for c in text if ord(c) > 0x2E80)
        if char_count > len(text) * 0.3:
            return self._break_cjk(text, max_widths, font_path, font_size)

        return self._break_latin(text, max_widths, font_path, font_size)

    def _break_cjk(
        self,
        text: str,
        max_widths: list[int],
        font_path: str,
        font_size: int,
    ) -> tuple[list[str], float]:
        """CJK 字符的 Knuth-Plass 换行（按字符拆分）"""
        chars = list(text)
        n = len(chars)
        if n == 0:
            return [], 0.0

        # DP 数组: dp[i] = 前 i 个字符的最小代价
        dp = [float("inf")] * (n + 1)
        breaks = [0] * (n + 1)
        dp[0] = 0.0

        for i in range(1, n + 1):
            for j in range(max(0, i - 30), i):  # 最多看 30 个字符
                line_len = i - j
                # 确定当前行索引（最后一行单独处理）
                # 用 dp 状态回溯估算当前在第几行比较困难，
                # 简化：统一用最大可用宽度的一半基准
                max_w = max_widths[min(line_len - 1, len(max_widths) - 1)]
                segment = "".join(chars[j:i])
                seg_w = self._text_width(segment, font_path, font_size)

                if seg_w > max_w * 1.2:
                    continue  # 超宽太多，跳过

                cost = self._line_cost(segment, seg_w, max_w, is_last=(i == n))
                if dp[j] + cost < dp[i]:
                    dp[i] = dp[j] + cost
                    breaks[i] = j

        # 回溯
        lines = []
        i = n
        while i > 0:
            j = breaks[i]
            lines.insert(0, "".join(chars[j:i]))
            i = j

        return lines, dp[n]

    def _break_latin(
        self,
        text: str,
        max_widths: list[int],
        font_path: str,
        font_size: int,
    ) -> tuple[list[str], float]:
        """英文单词的 Knuth-Plass 换行"""
        words = text.split()
        n = len(words)
        if n == 0:
            return [], 0.0

        # 预计算单词宽度
        word_widths = [self._word_width(w, font_path, font_size) for w in words]
        space_w = self._word_width(" ", font_path, font_size)
        # 累计宽度 prefix[i] = sum(word_widths[0..i-1]) + i * space_w
        prefix = [0.0] * (n + 1)
        for i in range(1, n + 1):
            prefix[i] = prefix[i - 1] + word_widths[i - 1] + (space_w if i < n else 0)

        def line_width(j: int, i: int) -> float:
            """words[j:i] 的总宽度（含单词间空格）"""
            if j >= i:
                return 0.0
            w = sum(word_widths[j:i])
            w += space_w * (i - j - 1)
            return w

        dp = [float("inf")] * (n + 1)
        breaks = [0] * (n + 1)
        dp[0] = 0.0

        for i in range(1, n + 1):
            max_lookback = min(i, 15)  # 最多看 15 个词（性能保护）
            for j in range(max(0, i - max_lookback), i):
                lw = line_width(j, i)
                line_idx = 0
                # 估算当前行号
                # 简化: 根据 dp 状态回溯估算
                cur = j
                temp_lines = 0
                while cur > 0:
                    cur = breaks[cur]
                    temp_lines += 1
                line_idx = temp_lines

                max_w = max_widths[min(line_idx, len(max_widths) - 1)]
                if lw > max_w * 1.3:
                    continue  # 超宽太多

                segment = " ".join(words[j:i])
                cost = self._line_cost(segment, lw, max_w, is_last=(i == n))
                if dp[j] + cost < dp[i]:
                    dp[i] = dp[j] + cost
                    breaks[i] = j

        # 回溯
        lines = []
        i = n
        while i > 0:
            j = breaks[i]
            lines.insert(0, " ".join(words[j:i]))
            i = j

        return lines, dp[n]

    def _line_cost(
        self,
        text: str,
        line_w: float,
        max_w: int,
        is_last: bool = False,
    ) -> float:
        """单行代价计算"""
        if max_w <= 0:
            return self.OVERFLOW_PENALTY

        # 宽度比例
        ratio = line_w / max_w
        if ratio > 1.0:
            # 超宽
            return self.OVERFLOW_PENALTY * (ratio ** 2)

        # 填充度（越接近 1 越好）
        fill = 1.0 - ratio
        cost = fill * fill * self.VARIANCE_PENALTY

        # 孤行惩罚（最后一行太短）
        if is_last and len(text.split()) <= 2 and len(text) < 8:
            cost += self.WIDOW_PENALTY

        # 居中偏差（行太长或太短都不好）
        balance = abs(ratio - 0.85) * self.BALANCE_PENALTY
        cost += balance

        return cost
