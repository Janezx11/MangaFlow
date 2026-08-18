"""TextRegionAdapter — TextRegion → DialogueBlock 转换

将 Text Detection Layer 的输出转换为下游 Pipeline 可用的 DialogueBlock。
可选：与 Bubble Detection 结果融合，获得精确的 polygon/mask。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from loguru import logger

from comic_translator.models.bubble import Bubble, DialogueBlock
from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock
from comic_translator.models.text_region import TextRegion

if TYPE_CHECKING:
    pass

# region_type → DialogueBlock.region_type 映射
REGION_TYPE_MAP = {
    "text": "speech",
    "title": "title",
    "sfx": "sfx",
    "unknown": "speech",
}

# Bubble IoU 阈值：超过此值则用真实 bubble 替换 synthetic bubble
BUBE_MERGE_IOU = 0.3


class TextRegionAdapter:
    """将 TextRegion + OCR blocks 转换为 DialogueBlock

    支持两种模式:
      1. 无 Bubble: 直接用 TextRegion 生成 synthetic Bubble
      2. 有 Bubble: 尝试与真实 Bubble 融合，获得精确 polygon/mask
    """

    def to_dialogue_blocks(
        self,
        regions: list[TextRegion],
        ocr_blocks: list[NormalizedOCRBlock],
        bubbles: list[Bubble] | None = None,
    ) -> list[DialogueBlock]:
        """转换 TextRegion 列表为 DialogueBlock 列表

        Args:
            regions: Text Detection 输出的文本区域
            ocr_blocks: 区域内 OCR 结果（原图坐标系）
            bubbles: 可选的 Bubble Detection 结果（用于 polygon/mask 增强）

        Returns:
            DialogueBlock 列表，可直接传给 Translation + Render
        """
        # 1. 将 OCR blocks 分配到各 region
        region_blocks = self._assign_ocr_to_regions(regions, ocr_blocks)

        # 2. 为每个 region 生成 synthetic Bubble
        synthetic_bubbles = {
            r.id: self._region_to_bubble(r) for r in regions
        }

        # 3. 可选：与真实 Bubble 融合
        if bubbles:
            synthetic_bubbles = self._enhance_with_bubbles(
                regions, synthetic_bubbles, bubbles
            )

        # 4. 构建 DialogueBlock
        dialogues: list[DialogueBlock] = []
        for region in regions:
            bubble = synthetic_bubbles[region.id]
            assigned = region_blocks.get(region.id, [])

            merged_text = " ".join(
                b.text.strip() for b in assigned if b.text.strip()
            )

            dialogues.append(DialogueBlock(
                id=region.id,
                bubble=bubble,
                region_type=REGION_TYPE_MAP.get(region.region_type, "speech"),
                ocr_blocks=assigned,
                merged_text=merged_text,
            ))

        # 5. 处理未分配的 OCR blocks（成为 singleton）
        assigned_ids = set()
        for blocks in region_blocks.values():
            assigned_ids.update(b.id for b in blocks)

        singletons = [
            b for b in ocr_blocks if b.id not in assigned_ids
        ]
        for block in singletons:
            x1, y1, x2, y2 = map(int, block.bbox)
            fake_bubble = Bubble(
                id=f"singleton_{block.id}",
                bbox=(x1, y1, x2, y2),
                polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                mask=[],
                class_name="speech",
            )
            dialogues.append(DialogueBlock(
                id=fake_bubble.id,
                bubble=fake_bubble,
                region_type="speech",
                ocr_blocks=[block],
                merged_text=block.text.strip(),
            ))

        dialogues.sort(key=lambda d: (d.bubble.bbox[1], d.bubble.bbox[0]))

        matched = len(dialogues) - len(singletons)
        logger.info(
            f"TextRegionAdapter: {len(regions)} regions → "
            f"{matched} dialogues + {len(singletons)} singletons"
        )
        return dialogues

    def to_dialogue_blocks_by_bubble(
        self,
        bubbles: list[Bubble],
        ocr_blocks: list[NormalizedOCRBlock],
    ) -> list[DialogueBlock]:
        """按气泡直接构建 DialogueBlock（逐气泡 OCR 后无需匹配）

        Args:
            bubbles: YOLO 检测到的气泡
            ocr_blocks: 逐气泡 OCR 结果（metadata["bubble_id"] 关联到气泡）

        Returns:
            DialogueBlock 列表，每个气泡一个 block
        """
        # 按 bubble_id 分组 OCR blocks
        by_bubble: dict[str, list[NormalizedOCRBlock]] = {}
        for block in ocr_blocks:
            bid = block.metadata.get("bubble_id")
            if bid:
                by_bubble.setdefault(bid, []).append(block)

        dialogues: list[DialogueBlock] = []
        for bubble in bubbles:
            blocks = by_bubble.get(bubble.id, [])
            if not blocks:
                continue
            # 按 Y→X 排序
            blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
            merged_text = " ".join(
                b.text.strip() for b in blocks if b.text.strip()
            )
            dialogues.append(DialogueBlock(
                id=bubble.id,
                bubble=bubble,
                region_type=bubble.class_name,
                ocr_blocks=blocks,
                merged_text=merged_text,
            ))

        # 未关联到气泡的 OCR blocks → singleton
        assigned = {b.id for bl in by_bubble.values() for b in bl}
        for block in ocr_blocks:
            if block.id in assigned:
                continue
            x1, y1, x2, y2 = map(int, block.bbox)
            fake_bubble = Bubble(
                id=f"singleton_{block.id}",
                bbox=(x1, y1, x2, y2),
                polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                mask=[],
                class_name="speech",
            )
            dialogues.append(DialogueBlock(
                id=fake_bubble.id,
                bubble=fake_bubble,
                region_type="speech",
                ocr_blocks=[block],
                merged_text=block.text.strip(),
            ))

        dialogues.sort(key=lambda d: (d.bubble.bbox[1], d.bubble.bbox[0]))
        logger.info(
            f"TextRegionAdapter (bubble): {len(bubbles)} bubbles → "
            f"{len(dialogues)} dialogues"
        )
        return dialogues

    def _assign_ocr_to_regions(
        self,
        regions: list[TextRegion],
        ocr_blocks: list[NormalizedOCRBlock],
    ) -> dict[str, list[NormalizedOCRBlock]]:
        """将 OCR blocks 分配到各 TextRegion（bbox 包含 + 最近距离 fallback）"""
        assignments: dict[str, list[NormalizedOCRBlock]] = {
            r.id: [] for r in regions
        }

        for block in ocr_blocks:
            best_region = None
            best_score = 0.0

            bx1, by1, bx2, by2 = map(int, block.bbox)
            bcx = (bx1 + bx2) / 2
            bcy = (by1 + by2) / 2

            for region in regions:
                rx1, ry1, rx2, ry2 = region.bbox

                # 中心点在 region 内 → 直接分配
                if rx1 <= bcx <= rx2 and ry1 <= bcy <= ry2:
                    best_region = region.id
                    best_score = 1.0
                    break

                # 否则计算 IoU
                iou = self._bbox_iou(
                    (bx1, by1, bx2, by2), (rx1, ry1, rx2, ry2)
                )
                if iou > best_score:
                    best_score = iou
                    best_region = region.id

            if best_region and best_score > 0.05:
                assignments[best_region].append(block)

        # 每个 region 内按 Y→X 排序
        for rid in assignments:
            assignments[rid].sort(key=lambda b: (b.bbox[1], b.bbox[0]))

        return assignments

    def _region_to_bubble(self, region: TextRegion) -> Bubble:
        """将 TextRegion 转换为 synthetic Bubble"""
        x1, y1, x2, y2 = region.bbox
        polygon = region.polygon or [
            (x1, y1), (x2, y1), (x2, y2), (x1, y2)
        ]
        return Bubble(
            id=region.id,
            bbox=region.bbox,
            polygon=polygon,
            mask=[],
            class_name=REGION_TYPE_MAP.get(region.region_type, "speech"),
            confidence=region.confidence,
        )

    def _enhance_with_bubbles(
        self,
        regions: list[TextRegion],
        synthetic: dict[str, Bubble],
        real_bubbles: list[Bubble],
    ) -> dict[str, Bubble]:
        """用真实 Bubble 的 polygon/mask 增强 synthetic Bubble"""
        enhanced = dict(synthetic)

        for region in regions:
            rx1, ry1, rx2, ry2 = region.bbox
            best_bubble = None
            best_iou = 0.0

            for bubble in real_bubbles:
                bx1, by1, bx2, by2 = bubble.bbox
                iou = self._bbox_iou(
                    (rx1, ry1, rx2, ry2), (bx1, by1, bx2, by2)
                )
                if iou > best_iou:
                    best_iou = iou
                    best_bubble = bubble

            if best_bubble and best_iou > BUBE_MERGE_IOU:
                # 用真实 bubble 的 polygon/mask 替换
                syn = enhanced[region.id]
                enhanced[region.id] = Bubble(
                    id=syn.id,
                    bbox=syn.bbox,
                    polygon=best_bubble.polygon if best_bubble.polygon else syn.polygon,
                    mask=best_bubble.mask if best_bubble.mask else syn.mask,
                    class_name=best_bubble.class_name,
                    confidence=max(syn.confidence, best_bubble.confidence),
                )
                logger.debug(
                    f"Enhanced {region.id} with bubble {best_bubble.id} "
                    f"(IoU={best_iou:.3f})"
                )

        return enhanced

    @staticmethod
    def _bbox_iou(
        a: tuple[int, int, int, int],
        b: tuple[int, int, int, int],
    ) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        inter = (ix2 - ix1) * (iy2 - iy1)
        area_a = (ax2 - ax1) * (ay2 - ay1)
        area_b = (bx2 - bx1) * (by2 - by1)
        union = area_a + area_b - inter

        return inter / union if union > 0 else 0.0
