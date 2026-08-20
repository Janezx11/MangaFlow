"""DetectionRegion — 多源检测统一模型

将 TextRegion、Bubble 等不同检测模型的输出统一为相同格式，
用于 Detection Fusion 和下游 Pipeline。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from comic_translator.models.bubble import Bubble
    from comic_translator.models.text_region import TextRegion


class DetectionRegion(BaseModel):
    """统一检测区域 —— 所有 Detection Provider 的标准输出格式"""

    id: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    polygon: list[tuple[float, float]] | None = None
    label: str = "unknown"  # speech / narration / title / sfx / text / unknown
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    source: str = ""  # "ctbd" / "yolo11" / "locate_anything3b"

    @classmethod
    def from_text_region(cls, tr: TextRegion) -> DetectionRegion:
        return cls(
            id=tr.id,
            bbox=tr.bbox,
            polygon=tr.polygon,
            label=tr.region_type,
            confidence=tr.confidence,
            source=tr.source,
        )

    @classmethod
    def from_bubble(cls, b: Bubble) -> DetectionRegion:
        return cls(
            id=b.id,
            bbox=b.bbox,
            polygon=b.polygon,
            label=b.class_name,
            confidence=b.confidence,
            source="yolo11",
        )

    def to_text_region(self) -> TextRegion:
        from comic_translator.models.text_region import TextRegion

        return TextRegion(
            id=self.id,
            bbox=self.bbox,
            polygon=self.polygon,
            confidence=self.confidence,
            region_type=self.label,
            source=self.source,
        )

    def to_bubble(self) -> Bubble:
        from comic_translator.models.bubble import Bubble

        x1, y1, x2, y2 = self.bbox
        polygon = self.polygon or [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return Bubble(
            id=self.id,
            bbox=self.bbox,
            polygon=polygon,
            mask=[],
            class_name=self.label,
            confidence=self.confidence,
        )
