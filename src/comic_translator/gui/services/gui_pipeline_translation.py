"""兼容层 —— 翻译阶段逻辑已迁移到 :mod:`comic_translator.core.translation`"""

from comic_translator.core.translation import (
    PipelineTranslationMixin,
    _bump_global,
    _get_global_counts,
    _parse_glossary_json,
)

# 旧类名（历史代码/测试引用）
GuiPipelineTranslationMixin = PipelineTranslationMixin

__all__ = [
    "GuiPipelineTranslationMixin",
    "PipelineTranslationMixin",
    "_bump_global",
    "_get_global_counts",
    "_parse_glossary_json",
]
