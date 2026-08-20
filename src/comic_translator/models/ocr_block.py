from pydantic import BaseModel, Field
from typing import Any


class OCRBlock(BaseModel):
    """单个 OCR 识别结果块"""

    id: str
    page: int
    bbox: tuple[float, float, float, float]  # (x1, y1, x2, y2)
    text: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    language: str = "en"
    metadata: dict[str, Any] = Field(default_factory=dict)