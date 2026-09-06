"""Typography Models V2 — SafeArea, ParagraphLayout, TypographyScore"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


@dataclass
class SafeArea:
    """气泡内的安全区域（距离变换后得到的可用文本区）"""
    polygon: list[tuple[int, int]]  # 安全区域多边形
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    max_inscribed_rect: tuple[int, int, int, int]  # 最大内接矩形
    center: tuple[float, float]  # 视觉中心


@dataclass
class LineInfo:
    """单行排版信息"""
    text: str
    width: int
    max_width: int  # 该行在 bubble 中的最大可用宽度
    x: int  # 绘制起点 x
    y: int  # 基线 y


@dataclass
class ParagraphLayout:
    """段落排版结果"""
    lines: list[LineInfo]
    font_size: int
    font_name: str
    line_count: int
    total_height: int
    total_width: int  # 最宽行宽度
    safe_area: SafeArea
    optical_center_y: int
    score: float  # 排版质量 0-1


@dataclass
class TypographyScore:
    """排版质量评分"""
    overall: float  # 0-100
    balance: float  # 行平衡
    font_size_score: float
    padding_score: float
    overflow_score: float
    center_score: float
    issues: list[str] = field(default_factory=list)
