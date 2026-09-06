from pydantic import BaseModel, Field


class MergeDecision(BaseModel):
    """合并决策 —— 多维打分决定是否合并两个 OCR 块"""

    score: float = Field(default=0.0, ge=0.0, le=1.0)
    x_overlap_score: float = Field(default=0.0, ge=0.0, le=1.0)
    y_distance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    punctuation_score: float = Field(default=0.0, ge=0.0, le=1.0)
    reading_order_score: float = Field(default=0.0, ge=0.0, le=1.0)
    should_merge: bool = False
    reason: str = ""


MERGE_THRESHOLD: float = 0.6  # 默认合并阈值
