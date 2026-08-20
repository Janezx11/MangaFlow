"""TextRegion — 文本区域检测结果模型"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TextRegion(BaseModel):
    """Text Detection Layer 输出的文本区域

    与 Bubble 的区别:
      - Bubble 是气泡（图形区域）
      - TextRegion 是文字（文本内容区域）
      - 一个 Bubble 可能包含多个 TextRegion（如多行对白）
      - TextRegion 也可以独立于 Bubble 存在（旁白、标题、SFX）
    """

    id: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    polygon: list[tuple[float, float]] | None = None  # 可选，从 mask 提取
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    region_type: str = "unknown"  # "text" / "title" / "sfx" / "unknown"
    source: str = ""  # 产生此区域的检测器名称
