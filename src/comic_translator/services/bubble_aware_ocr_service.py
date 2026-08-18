"""BubbleAwareOCRService — "哪些文字属于哪个气泡"

职责边界（重要架构原则）：
- YOLO 负责：哪里是气泡
- OCR 负责：哪里有文字、文字是什么
- **本服务负责：哪些文字属于哪个气泡**
- Translation 负责：文字翻译成什么
- Typography 负责：怎么排版

核心逻辑：
1. 检测每个 OCR block 是否"跨气泡"（与 2+ 个气泡 polygon 明显重叠，
   不用 bbox center，用 polygon 重叠面积比例）
2. 优先使用 OCR 细粒度信息（word/line bbox，来自 block.metadata）
   做逐词/逐行气泡分配
3. 若无细粒度信息 → fallback 到 bubble-guided OCR：
   对相关气泡分别 crop 并抹除干扰文字后单独 OCR
4. 只有 cross_bubble == True 的 block 才触发二次识别，普通 block 原样保留
"""

from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass, field
from loguru import logger
from PIL import Image, ImageDraw

from comic_translator.models.bubble import Bubble
from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock
from comic_translator.models.ocr_block import OCRBlock
from comic_translator.providers.ocr.base import OCRProvider

# 拆分方法
SPLIT_DIRECT = "direct"          # 只与一个气泡重叠，直接归属
SPLIT_WORD = "word_split"        # 用 OCR 细粒度 word bbox 分配
SPLIT_LINE = "line_split"        # 用 OCR 细粒度 line bbox 分配
SPLIT_CROP_OCR = "bubble_crop_ocr"  # 无细粒度 → 气泡裁剪 + 单独 OCR


@dataclass
class BubbleAwareSplitInfo:
    """一次跨气泡拆分的调试记录"""

    source_ocr_block_id: str
    original_text: str
    original_bbox: tuple
    bubbles: list[str]
    split_method: str
    produced: list[str]  # 产出 block id
    split_confidence: float


@dataclass
class BubbleAwareResult:
    """BubbleAwareOCRService 处理结果"""

    blocks: list[NormalizedOCRBlock] = field(default_factory=list)
    splits: list[BubbleAwareSplitInfo] = field(default_factory=list)

    @property
    def cross_bubble_count(self) -> int:
        return len(self.splits)


# ---------------- 几何工具 ----------------

def _bbox_area(bbox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, (x2 - x1) * (y2 - y1))


def _bbox_iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _bubble_contain_ratio(bbox1, bbox2) -> float:
    """较小 bbox 被较大 bbox 包含的面积比例（识别松/紧两个重复框）"""
    ax1, ay1, ax2, ay2 = map(int, bbox1)
    bx1, by1, bx2, by2 = map(int, bbox2)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    small = min(a_area, b_area)
    return inter / small if small > 0 else 0.0


def _polygon_overlap_ratio(bbox, polygon) -> float:
    """OCR block bbox 与气泡 polygon 的重叠面积占 block 面积的比例"""
    x1, y1, x2, y2 = map(int, bbox)
    bw, bh = x2 - x1, y2 - y1
    if bw <= 0 or bh <= 0 or not polygon or len(polygon) < 3:
        return 0.0
    block_mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.rectangle(block_mask, (0, 0), (bw, bh), 1, -1)
    poly = np.array([(px - x1, py - y1) for px, py in polygon], dtype=np.int32)
    bubble_mask = np.zeros((bh, bw), dtype=np.uint8)
    cv2.fillPoly(bubble_mask, [poly], 1)
    overlap = int((block_mask & bubble_mask).sum())
    return overlap / (bw * bh)


def _merge_overlapping_bubbles(
    bubbles: list[Bubble],
    iou_threshold: float = 0.5,
    contain_ratio: float = 0.7,
) -> list[Bubble]:
    """合并 YOLO 重复检测的气泡（bbox 高度重叠或松/紧两个框）"""
    bubbles = list(bubbles)
    merged: list[Bubble] = []
    used: set[int] = set()

    for i, b in enumerate(bubbles):
        if i in used:
            continue
        group = [b]
        used.add(i)
        for j in range(i + 1, len(bubbles)):
            if j in used:
                continue
            if (
                _bbox_iou(b.bbox, bubbles[j].bbox) > iou_threshold
                or _bubble_contain_ratio(b.bbox, bubbles[j].bbox) > contain_ratio
            ):
                group.append(bubbles[j])
                used.add(j)
        best = max(group, key=lambda x: _bbox_area(x.bbox))
        merged.append(best)

    if len(merged) != len(bubbles):
        logger.info(f"bubble-aware: merged overlapping bubbles {len(bubbles)} -> {len(merged)}")
    return merged


def _to_normalized(block: OCRBlock, page: int | None = None) -> NormalizedOCRBlock:
    return NormalizedOCRBlock(
        id=block.id,
        page=page if page is not None else block.page,
        bbox=block.bbox,
        text=block.text,
        confidence=block.confidence,
        language=block.language,
        metadata=dict(block.metadata or {}),
    )


def _tag(
    block: OCRBlock,
    bubble_id: str | None,
    split_method: str,
    cross_bubble: bool,
    split_confidence: float,
    source_ocr_block_id: str,
    page: int | None = None,
) -> NormalizedOCRBlock:
    """给 OCR block 打上气泡归属 / 拆分元信息"""
    nb = _to_normalized(block, page)
    nb.metadata.update({
        "cross_bubble": cross_bubble,
        "source_ocr_block_id": source_ocr_block_id,
        "bubble_id": bubble_id,
        "split_method": split_method,
        "split_confidence": split_confidence,
    })
    return nb


class BubbleAwareOCRService:
    """跨气泡文字归属服务

    Args:
        ocr_provider: OCR 引擎（用于 bubble-guided 二次识别）
        threshold: 跨气泡判定阈值（intersection ratio 默认 0.15）
        strategy: auto / line_split / word_split / bubble_crop_ocr
        enabled: 总开关（False 时仅做直接归属标记，不拆分）
    """

    def __init__(
        self,
        ocr_provider: OCRProvider,
        threshold: float = 0.15,
        strategy: str = "auto",
        enabled: bool = True,
    ) -> None:
        self.ocr_provider = ocr_provider
        self.threshold = max(0.0, min(0.9, float(threshold)))
        self.strategy = strategy
        self.enabled = enabled

    # ---------------- 主入口 ----------------

    async def process(
        self,
        image: Image.Image,
        ocr_blocks: list[OCRBlock],
        bubbles: list[Bubble],
    ) -> BubbleAwareResult:
        """处理一批 OCR block，返回拆分后的 block 与调试信息"""
        result = BubbleAwareResult()

        if not self.enabled or not ocr_blocks:
            return result
        if not bubbles:
            # 无气泡信息：全部保留，标记为无归属
            for b in ocr_blocks:
                result.blocks.append(
                    _tag(b, None, SPLIT_DIRECT, False, 1.0, b.id)
                )
            return result

        # 合并重复气泡，避免误触发
        bubbles = _merge_overlapping_bubbles(bubbles)

        for block in ocr_blocks:
            overlaps = self._bubble_overlaps(block, bubbles)

            if len(overlaps) < 2:
                # 正常块：直接归属（若有唯一气泡）
                target = overlaps[0][0] if overlaps else None
                result.blocks.append(
                    _tag(
                        block,
                        target.id if target else None,
                        SPLIT_DIRECT,
                        False,
                        float(overlaps[0][1]) if overlaps else 1.0,
                        block.id,
                    )
                )
                continue

            # ------- 跨气泡 block：触发拆分 -------
            overlaps.sort(key=lambda x: -x[1])
            top_bubbles = [b for b, _ in overlaps[:3]]
            info = BubbleAwareSplitInfo(
                source_ocr_block_id=block.id,
                original_text=block.text,
                original_bbox=block.bbox,
                bubbles=[b.id for b in top_bubbles],
                split_method=SPLIT_CROP_OCR,
                produced=[],
                split_confidence=0.0,
            )

            sub_blocks = await self._split_block(image, block, top_bubbles, bubbles, info)
            info.produced = [s.id for s in sub_blocks]
            if sub_blocks:
                info.split_confidence = float(np.mean([s.confidence for s in sub_blocks]))
            result.splits.append(info)
            result.blocks.extend(sub_blocks)

        if result.cross_bubble_count:
            logger.info(
                f"BubbleAwareOCR: {result.cross_bubble_count} cross-bubble blocks resolved "
                f"({len(ocr_blocks)} -> {len(result.blocks)} blocks)"
            )
        return result

    # ---------------- 跨气泡检测 ----------------

    def _bubble_overlaps(
        self, block: OCRBlock, bubbles: list[Bubble]
    ) -> list[tuple[Bubble, float]]:
        """返回与 block 有明显 polygon 重叠的气泡及比例"""
        overlaps: list[tuple[Bubble, float]] = []
        for b in bubbles:
            if not b.polygon or len(b.polygon) < 3:
                continue
            ratio = _polygon_overlap_ratio(block.bbox, b.polygon)
            if ratio > self.threshold:
                overlaps.append((b, ratio))
        return overlaps

    # ---------------- 拆分 ----------------

    async def _split_block(
        self,
        image: Image.Image,
        block: OCRBlock,
        top_bubbles: list[Bubble],
        bubbles: list[Bubble],
        info: BubbleAwareSplitInfo,
    ) -> list[NormalizedOCRBlock]:
        """根据策略拆分跨气泡 block"""
        fine = self._extract_fine_boxes(block)

        # 自动策略：细粒度优先；显式策略：按用户指定
        use_fine = (self.strategy in ("auto", SPLIT_LINE, SPLIT_WORD)) and bool(fine)
        force_crop = self.strategy == SPLIT_CROP_OCR or not use_fine

        if force_crop:
            info.split_method = SPLIT_CROP_OCR
            return await self._crop_split(image, block, top_bubbles, bubbles)

        # 细粒度拆分
        info.split_method = SPLIT_LINE if any(k == "lines" for k in block.metadata) else SPLIT_WORD
        return self._fine_split(block, top_bubbles, fine)

    # ---- 细粒度（word/line bbox）分配 ----

    @staticmethod
    def _extract_fine_boxes(block: OCRBlock) -> list[tuple[str, tuple, float]]:
        """从 metadata 提取细粒度子框：[("text", bbox, confidence), ...]"""
        md = block.metadata or {}
        fine: list[tuple[str, tuple, float]] = []
        for key in ("words", "lines"):
            items = md.get(key)
            if not items:
                continue
            for it in items:
                if not isinstance(it, dict):
                    continue
                text = str(it.get("text", ""))
                bb = it.get("bbox")
                if not bb or len(bb) != 4:
                    continue
                fine.append((text, tuple(float(v) for v in bb), float(it.get("confidence", 1.0))))
        return fine

    def _fine_split(
        self,
        block: OCRBlock,
        top_bubbles: list[Bubble],
        fine: list[tuple[str, tuple, float]],
    ) -> list[NormalizedOCRBlock]:
        """按细粒度子框分配到气泡，重组为每气泡一个 block"""
        per_bubble: dict[str, list[tuple[str, tuple, float]]] = {}
        for text, bb, conf in fine:
            target = None
            best = 0.0
            for b in top_bubbles:
                if not b.polygon or len(b.polygon) < 3:
                    continue
                ratio = _polygon_overlap_ratio(bb, b.polygon)
                if ratio > best:
                    best = ratio
                    target = b
            if target is None:
                continue
            per_bubble.setdefault(target.id, []).append((text, bb, conf))

        out: list[NormalizedOCRBlock] = []
        for bid, items in per_bubble.items():
            items.sort(key=lambda x: (x[1][1], x[1][0]))  # Y→X
            text = " ".join(t for t, _, _ in items if t)
            conf = float(np.mean([c for _, _, c in items]))
            bboxes = [bb for _, bb, _ in items]
            min_x = min(bb[0] for bb in bboxes)
            min_y = min(bb[1] for bb in bboxes)
            max_x = max(bb[2] for bb in bboxes)
            max_y = max(bb[3] for bb in bboxes)
            out.append(_tag(
                block, bid, SPLIT_LINE, True, conf, block.id,
            ).model_copy(update={
                "text": text,
                "bbox": (min_x, min_y, max_x, max_y),
            }))
        return out

    # ---- bubble-guided OCR（无细粒度 fallback） ----

    async def _crop_split(
        self,
        image: Image.Image,
        block: OCRBlock,
        top_bubbles: list[Bubble],
        bubbles: list[Bubble],
    ) -> list[NormalizedOCRBlock]:
        """对每个气泡 crop 并抹除干扰文字后单独 OCR"""
        out: list[NormalizedOCRBlock] = []
        for target in top_bubbles:
            sub = await self._crop_ocr_in_bubble(
                image, block.bbox, target, source_id=block.id
            )
            out.extend(sub)
        return out

    async def _crop_ocr_in_bubble(
        self,
        image: Image.Image,
        block_bbox,
        target_bubble: Bubble,
        source_id: str = "",
        dilate: int = 8,
    ) -> list[NormalizedOCRBlock]:
        """在 block bbox 范围内，只保留 target_bubble 内文字（polygon 膨胀保护），
        其余像素涂白（含其他气泡文字），再 OCR。"""
        x1, y1, x2, y2 = map(int, block_bbox)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return []

        crop = image.crop((x1, y1, x2, y2))

        keep = Image.new("L", (w, h), 0)
        if target_bubble.polygon and len(target_bubble.polygon) >= 3:
            pts = [(px - x1, py - y1) for px, py in target_bubble.polygon]
            ImageDraw.Draw(keep).polygon(pts, fill=255)
        else:
            bx1, by1, bx2, by2 = map(int, target_bubble.bbox)
            ImageDraw.Draw(keep).rectangle(
                [(bx1 - x1, by1 - y1), (bx2 - x1, by2 - y1)], fill=255
            )
        if dilate > 0:
            keep_np = np.array(keep)
            kernel = np.ones((dilate, dilate), np.uint8)
            keep_np = cv2.dilate(keep_np, kernel)
            keep = Image.fromarray(keep_np)

        white = Image.new("RGB", (w, h), (255, 255, 255))
        masked = Image.composite(crop, white, keep)

        raw_blocks = await self.ocr_provider.detect_text(masked)
        out: list[NormalizedOCRBlock] = []
        for rb in raw_blocks:
            rx1, ry1, rx2, ry2 = map(int, rb.bbox)
            out.append(_tag(
                rb,
                target_bubble.id,
                SPLIT_CROP_OCR,
                True,
                float(rb.confidence),
                source_ocr_block_id=source_id,
                page=0,
            ).model_copy(update={
                "id": f"{target_bubble.id}_crop_{rb.id}",
                "bbox": (rx1 + x1, ry1 + y1, rx2 + x1, ry2 + y1),
            }))
        return out
