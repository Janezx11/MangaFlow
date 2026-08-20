from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock
    from comic_translator.models.typography_models import SafeArea


@dataclass
class Bubble:
    """YOLO/Provider 输出的气泡检测结果

    mask 保留为 list（来自 YOLO seg），便于序列化。
    使用时通过 np.array() 转换为 ndarray。
    blocks 在 PostProcess 阶段填充。
    safe_area 在 InteriorService 中计算。
    """

    id: str
    bbox: tuple[int, int, int, int]  # x1, y1, x2, y2
    polygon: list[tuple[float, float]]  # 精确轮廓点 (用于 Layout)
    mask: list  # segmentation mask pixels (list for serialization)
    class_name: str  # speech / narration / title / sfx
    confidence: float = 0.0
    blocks: list = field(default_factory=list)  # 关联的 OCR blocks
    safe_area: object | None = None  # SafeArea (延迟计算，避免导入循环)


@dataclass
class DialogueBlock:
    """对话块 —— Bubble + 内部 OCR 文本的完整语义单元

    从这一层开始，OCR Block 不再单独使用。
    翻译、Layout、Render 全部基于 DialogueBlock。
    """

    id: str
    bubble: Bubble
    region_type: str = "speech"  # speech / narration / title / sfx
    ocr_blocks: list = field(default_factory=list)
    merged_text: str = ""
    translated_text: str = ""
    translation_status: str = "pending"   # pending / success / failed / unchanged / skipped
    render_text: str = ""
    render_status: str = "pending"        # pending / success / failed / fallback
    font_size: int = 0
    line_count: int = 0
    lines: list[str] = field(default_factory=list)