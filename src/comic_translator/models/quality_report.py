from __future__ import annotations

from pydantic import BaseModel, Field


class RetryRecord(BaseModel):
    """单次重跑记录"""
    attempt: int
    reason: str
    translation_status: str
    render_quality: float
    prompt_snippet: str = ""


class PageQualityReport(BaseModel):
    """单页质检报告"""

    page: int
    image_path: str = ""

    # OCR 质量
    ocr_block_count: int = 0
    ocr_avg_confidence: float = 0.0

    # 翻译质量
    translation_status: str = "PASS"       # PASS / WARNING / FAIL
    translation_issues: list[str] = Field(default_factory=list)
    translated_block_count: int = 0
    block_count_diff: int = 0               # 正=OCR block 多于翻译 block
    length_ratio: float = 0.0               # 中文字数/英文字数
    english_residue: bool = False

    # 渲染质量
    render_quality: float = 0.0             # 0-100
    render_blur: float = 0.0
    render_text_residual: float = 0.0

    # 排版质量
    typography_score: float = 0.0

    # 重跑记录
    retry_count: int = 0
    retry_history: list[RetryRecord] = Field(default_factory=list)

    # 综合决策
    overall_score: float = 0.0
    needs_retry: bool = False
    retry_reason: str = ""


class QualityReportSummary(BaseModel):
    """全局质检报告汇总"""
    project_name: str = ""
    total_pages: int = 0
    passed: int = 0
    retry_passed: int = 0
    needs_manual_review: int = 0
    overall_pass_rate: float = 0.0
    avg_render_quality: float = 0.0
    avg_translation_status: str = "PASS"
    pages: list[PageQualityReport] = Field(default_factory=list)
    manual_review_pages: list[PageQualityReport] = Field(default_factory=list)
