"""TextRegionService — 文本区域内 OCR 编排

流程:
  TextRegion list → 逐区域裁剪 → OCR → 坐标偏移 → 合并去重 → NormalizedOCRBlock

与旧 Pipeline 的区别:
  旧: 全图 OCR → 匹配 Bubble
  新: Text Detection → 区域内 OCR → 合并
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from PIL import Image, ImageDraw

from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock
from comic_translator.models.ocr_block import OCRBlock
from comic_translator.models.text_region import TextRegion

if TYPE_CHECKING:
    from comic_translator.models.bubble import Bubble
    from comic_translator.providers.ocr.base import OCRProvider

# 区域内 OCR 时的边距扩展（确保不切断文字边缘）
REGION_PADDING = 8

# 去重 IoU 阈值（两个 OCR block 重叠超过此值则去重）
DEDUP_IOU_THRESHOLD = 0.7


class TextRegionService:
    """文本区域服务 —— 在 TextRegion 内执行 OCR 并合并结果"""

    async def extract_text(
        self,
        image: Image.Image,
        regions: list[TextRegion],
        ocr_provider: OCRProvider,
    ) -> list[NormalizedOCRBlock]:
        """在每个 TextRegion 内执行 OCR，返回合并后的 OCR blocks

        Args:
            image: 原始图片
            regions: 检测到的文本区域
            ocr_provider: OCR 引擎

        Returns:
            合并去重后的 NormalizedOCRBlock 列表（坐标为原图坐标系）
        """
        if not regions:
            return []

        img_rgb = image.convert("RGB")
        img_np = np.array(img_rgb)
        all_blocks: list[NormalizedOCRBlock] = []

        for region in regions:
            crop = self._crop_region(img_rgb, region)
            if crop is None:
                continue

            raw_blocks = await ocr_provider.detect_text(crop)
            offset_blocks = [
                self._offset_block(b, region.bbox) for b in raw_blocks
            ]
            all_blocks.extend(offset_blocks)

        # 去重
        deduped = self._deduplicate(all_blocks)

        # 排序: 先按区域 Y→X，再按区域内 Y→X
        deduped.sort(key=lambda b: (b.bbox[1], b.bbox[0]))

        # 赋 reading_order
        for i, block in enumerate(deduped):
            block.reading_order = i

        logger.info(
            f"TextRegionService: {len(regions)} regions → "
            f"{len(all_blocks)} raw OCR blocks → "
            f"{len(deduped)} after dedup"
        )
        return deduped

    async def extract_text_by_bubbles(
        self,
        image: Image.Image,
        bubbles: list[Bubble],
        ocr_provider: OCRProvider,
        padding: int = 8,
    ) -> list[NormalizedOCRBlock]:
        """按气泡区域执行 OCR，每个气泡只识别自己区域内的文字

        解决"两个相邻气泡被 OCR 识别成一个 block"的问题。
        关键：用 YOLO 气泡的 polygon 掩码把气泡外像素涂成背景色，
        OCR 引擎只能看到当前气泡内部的文字，从根源上避免跨气泡合并。

        Args:
            image: 原始图片
            bubbles: YOLO 检测到的气泡列表
            ocr_provider: OCR 引擎
            padding: 裁剪时额外扩大的像素

        Returns:
            NormalizedOCRBlock 列表，每个 block 已关联到对应气泡
        """
        if not bubbles:
            return []

        img_rgb = image.convert("RGB")
        w, h = image.size
        all_blocks: list[NormalizedOCRBlock] = []

        for bubble in bubbles:
            bx1, by1, bx2, by2 = map(int, bubble.bbox)

            # 给气泡 bbox 加 padding，确保不切断文字边缘
            px1 = max(0, bx1 - padding)
            py1 = max(0, by1 - padding)
            px2 = min(w, bx2 + padding)
            py2 = min(h, by2 + padding)

            if px2 <= px1 or py2 <= py1:
                continue

            crop = img_rgb.crop((px1, py1, px2, py2))

            # polygon 掩码：气泡外涂成背景色，OCR 只能看到气泡内文字
            if bubble.polygon and len(bubble.polygon) >= 3:
                crop = self._apply_polygon_mask(crop, bubble.polygon, (px1, py1))

            raw_blocks = await ocr_provider.detect_text(crop)

            for block in raw_blocks:
                # 坐标偏移回原图
                x1, y1, x2, y2 = block.bbox
                offset_block = NormalizedOCRBlock(
                    id=f"{bubble.id}_ocr_{block.id}",
                    page=block.page,
                    bbox=(x1 + px1, y1 + py1, x2 + px1, y2 + py1),
                    text=block.text,
                    confidence=block.confidence,
                    language=block.language,
                    metadata={**block.metadata, "bubble_id": bubble.id},
                )
                all_blocks.append(offset_block)

        # 按 Y→X 排序
        all_blocks.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        for i, block in enumerate(all_blocks):
            block.reading_order = i

        logger.info(
            f"TextRegionService (bubble OCR): {len(bubbles)} bubbles → "
            f"{len(all_blocks)} OCR blocks"
        )
        return all_blocks

    @staticmethod
    def _apply_polygon_mask(
        image: Image.Image,
        polygon: list[tuple[float, float]],
        origin: tuple[int, int],
    ) -> Image.Image:
        """把 polygon 外的像素涂成白色背景，只保留多边形内部

        Args:
            image: 裁剪后的图片（在气泡 bbox 坐标系内）
            polygon: 气泡 polygon（原图坐标系）
            origin: 裁剪左上角在原图中的坐标 (px1, py1)

        Returns:
            掩码后的图片（气泡内保留原图，气泡外为白色）
        """
        ox, oy = origin
        # 平移 polygon 到裁剪图坐标系
        pts = [(x - ox, y - oy) for x, y in polygon]
        # 生成 mask：内部 255，外部 0
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).polygon(pts, fill=255)
        # 外部涂白
        white = Image.new("RGB", image.size, (255, 255, 255))
        return Image.composite(image, white, mask)

    def _crop_region(
        self,
        image: Image.Image,
        region: TextRegion,
    ) -> Image.Image | None:
        """裁剪文本区域（带 padding）"""
        x1, y1, x2, y2 = region.bbox
        w, h = image.size

        px1 = max(0, x1 - REGION_PADDING)
        py1 = max(0, y1 - REGION_PADDING)
        px2 = min(w, x2 + REGION_PADDING)
        py2 = min(h, y2 + REGION_PADDING)

        if px2 <= px1 or py2 <= py1:
            return None

        return image.crop((px1, py1, px2, py2))

    def _offset_block(
        self,
        block: OCRBlock,
        region_bbox: tuple[int, int, int, int],
    ) -> NormalizedOCRBlock:
        """将区域内 OCR block 的坐标偏移回原图坐标系"""
        rx1, ry1 = region_bbox[0], region_bbox[1]
        # 注意: crop 加了 REGION_PADDING，需要减回去
        rx1 -= REGION_PADDING
        ry1 -= REGION_PADDING

        x1, y1, x2, y2 = block.bbox
        return NormalizedOCRBlock(
            id=block.id,
            page=block.page,
            bbox=(
                x1 + rx1,
                y1 + ry1,
                x2 + rx1,
                y2 + ry1,
            ),
            text=block.text,
            confidence=block.confidence,
            language=block.language,
            metadata=block.metadata,
        )

    def _deduplicate(
        self,
        blocks: list[NormalizedOCRBlock],
    ) -> list[NormalizedOCRBlock]:
        """基于 bbox IoU 去重（重叠区域可能产生重复 OCR 结果）"""
        if len(blocks) <= 1:
            return blocks

        keep: list[bool] = [True] * len(blocks)

        for i in range(len(blocks)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(blocks)):
                if not keep[j]:
                    continue
                iou = self._bbox_iou(blocks[i].bbox, blocks[j].bbox)
                if iou > DEDUP_IOU_THRESHOLD:
                    # 保留置信度更高的
                    if blocks[i].confidence >= blocks[j].confidence:
                        keep[j] = False
                    else:
                        keep[i] = False
                        break

        return [b for b, k in zip(blocks, keep) if k]

    @staticmethod
    def _bbox_iou(
        a: tuple[float, float, float, float],
        b: tuple[float, float, float, float],
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
