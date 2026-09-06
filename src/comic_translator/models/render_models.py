from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RegionType(str, Enum):
    DIALOGUE = "dialogue"
    NARRATION = "narration"
    TITLE = "title"
    SFX = "sfx"
    UNKNOWN = "unknown"


@dataclass
class RegionInfo:
    region_type: RegionType
    confidence: float = 1.0
    reason: str = ""


@dataclass
class FontProfile:
    name: str
    font_path: str
    min_size: int = 12
    max_size: int = 64
    default_size: int = 24
    line_spacing: float = 1.2
    stroke_width: int = 1
    stroke_color: tuple[int, int, int] = (0, 0, 0)
    color: tuple[int, int, int] = (255, 255, 255)


@dataclass
class TextLayout:
    text: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    font_size: int
    line_count: int
    lines: list[str] = field(default_factory=list)


@dataclass
class RenderBlock:
    """渲染块 —— 包含分类、布局、最终绘制信息"""

    ocr_text: str
    translated_text: str
    bbox: tuple[int, int, int, int]
    region_type: RegionType = RegionType.UNKNOWN
    region_confidence: float = 0.0
    layout: TextLayout | None = None
    mask: Any = None  # np.ndarray mask
    rendered: bool = False


@dataclass
class QAResult:
    passed: bool = True
    issues: list[str] = field(default_factory=list)
    overflow_blocks: list[str] = field(default_factory=list)
    tiny_font_blocks: list[str] = field(default_factory=list)
    english_residue_blocks: list[str] = field(default_factory=list)
    coverage_issues: list[str] = field(default_factory=list)

    def add_issue(self, msg: str) -> None:
        self.issues.append(msg)
        self.passed = False


class RenderPageResult(BaseModel):
    """单页渲染结果"""

    page: int
    source_path: str = ""
    output_path: str = ""
    blocks_rendered: int = 0
    qa: QAResult | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
