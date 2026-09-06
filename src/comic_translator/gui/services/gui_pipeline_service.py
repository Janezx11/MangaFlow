"""兼容层 —— 核心管线已迁移到 :mod:`comic_translator.core.pipeline`

`GuiPipelineService` 现在是 `ComicPipelineService` 的别名：CLI（run/resume）
与 GUI 消费同一条完整管线（检测 → OCR → 翻译 → 擦除 → 渲染）。
所有历史导入路径继续有效。
"""

from comic_translator.core.pipeline import (
    ComicPipelineService,
    GuiPipelineService,
    PageResult,
    PipelineProgress,
)

__all__ = [
    "ComicPipelineService",
    "GuiPipelineService",
    "PageResult",
    "PipelineProgress",
]
