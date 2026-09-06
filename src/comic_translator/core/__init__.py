"""核心翻译管线（CLI 与 GUI 共用）

`ComicPipelineService` 编排完整流程：检测（CTBD+YOLO）→ 气泡归属 →
OCR → 翻译 → 擦除 → 渲染，逐页产出 `PipelineProgress`。
本包保持框架中立：不 import gradio，也不依赖 GUI 命名空间。
"""

from comic_translator.core.pipeline import (
    ComicPipelineService,
    GuiPipelineService,
    PageResult,
    PipelineProgress,
)
from comic_translator.core.translation import PipelineTranslationMixin

__all__ = [
    "ComicPipelineService",
    "GuiPipelineService",
    "PageResult",
    "PipelineProgress",
    "PipelineTranslationMixin",
]
