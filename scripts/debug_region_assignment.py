#!/usr/bin/env python3
"""Bubble ↔ OCR Region 归属诊断脚本（Phase 1：纯诊断，不改生产代码）

输入一张漫画图片，输出：
  1. YOLO Bubble 详情（id/class/conf/bbox/polygon/area/center）
  2. CTBD TextRegion 详情
  3. OCR blocks（extract_text 后）
  4. BubbleAwareOCRService 处理后的 blocks（确认拆分是否发生）
  5. TextRegionAdapter 匹配结果（每个 block → region/bubble，含评分）
  6. multi-bubble 判断（intersects_multiple_bubbles）
  7. 诊断指标
  8. 可视化 overlay（debug_region_assignment.png）

用法:
  python scripts/debug_region_assignment.py <image_path> [--out xxx.png]
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from PIL import Image, ImageDraw

from comic_translator.providers.bubble.yolo_bubble_provider import YOLO11BubbleProvider
from comic_translator.providers.ocr.easyocr_provider import EasyOCRProvider
from comic_translator.providers.text_detection.ctbd_text_detector import CTBDTextDetectorProvider
from comic_translator.services.text_region_adapter import TextRegionAdapter
from comic_translator.services.text_region_service import TextRegionService
from comic_translator.services.bubble_aware_ocr_service import (
    BubbleAwareOCRService,
    _polygon_overlap_ratio,
)

# 气泡颜色
BUBBLE_COLORS = [
    (255, 0, 0), (0, 200, 0), (0, 80, 255), (255, 200, 0),
    (0, 200, 200), (200, 0, 200), (255, 120, 0), (120, 120, 255),
]
MULTI_COLOR = (255, 0, 255)   # 品红：multi-bubble OCR
NONE_COLOR = (120, 120, 120)  # 灰：未分配


def _center(bbox):
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) / 2, (y1 + y2) / 2)


def _area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, (x2 - x1) * (y2 - y1))


def _fmt_bbox(bbox):
    return f"({bbox[0]:.0f},{bbox[1]:.0f},{bbox[2]:.0f},{bbox[3]:.0f})"


def intersects_multiple_bubbles(bbox, bubbles, threshold=0.15):
    """判断 OCR bbox 是否横跨多个气泡（polygon 重叠比例 > threshold）"""
    hits = []
    for b in bubbles:
        if not b.polygon or len(b.polygon) < 3:
            continue
        ratio = _polygon_overlap_ratio(bbox, b.polygon)
        if ratio > threshold:
            hits.append((b.id, ratio))
    return hits


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image", help="漫画图片路径")
    parser.add_argument("--out", default="debug_region_assignment.png")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--fusion", action="store_true", help="启用检测融合（复现真实 pipeline）")
    args = parser.parse_args()

    img = Image.open(args.image).convert("RGB")
    print(f"\n=== 诊断: {args.image} ({img.width}x{img.height}) ===\n")

    # 初始化 providers
    ctbd = CTBDTextDetectorProvider(confidence=args.conf, device="auto")
    yolo = YOLO11BubbleProvider(model_path="models/bubble/yolo11n-seg-mixed-bubble.pt")
    ocr = EasyOCRProvider(lang="en", use_gpu=True)
    trs = TextRegionService()
    adapter = TextRegionAdapter()

    # ---------- 1. YOLO Bubble ----------
    bubbles = await yolo.detect(img)
    print("===== 1. YOLO Bubble =====")
    for i, b in enumerate(bubbles):
        cx, cy = _center(b.bbox)
        print(f"Bubble {i} [{b.id}] class={b.class_name} conf={b.confidence:.2f}")
        print(f"   bbox={_fmt_bbox(b.bbox)}  center=({cx:.0f},{cy:.0f})  area={_area(b.bbox):.0f}  polygon_pts={len(b.polygon)}")

    # ---------- 2. CTBD TextRegion ----------
    text_regions = await ctbd.detect(img)
    print("\n===== 2. CTBD TextRegion =====")
    for i, r in enumerate(text_regions):
        cx, cy = _center(r.bbox)
        print(f"Region {i} [{r.id}] type={r.region_type} conf={r.confidence:.2f}")
        print(f"   bbox={_fmt_bbox(r.bbox)}  center=({cx:.0f},{cy:.0f})  area={_area(r.bbox):.0f}")

    # ---------- 2b. Fusion（可选，复现真实 pipeline）----------
    if args.fusion:
        from comic_translator.models.detection_region import DetectionRegion
        from comic_translator.services.detection_fusion_service import DetectionFusionService
        det_lists = [
            [DetectionRegion.from_text_region(tr) for tr in text_regions],
            [DetectionRegion.from_bubble(b) for b in bubbles],
        ]
        fusion = DetectionFusionService()
        fused = fusion.fuse(det_lists)
        text_regions = [r.to_text_region() for r in fused]
        print("\n===== 2b. Fused TextRegion =====")
        for i, r in enumerate(text_regions):
            cx, cy = _center(r.bbox)
            print(f"FusedRegion {i} [{r.id}] type={r.region_type} conf={r.confidence:.2f}")
            print(f"   bbox={_fmt_bbox(r.bbox)}  center=({cx:.0f},{cy:.0f})  area={_area(r.bbox):.0f}")

    # ---------- 3. OCR blocks（extract_text 后）----------
    ocr_blocks = await trs.extract_text(img, text_regions, ocr)
    print("\n===== 3. OCR blocks (extract_text) =====")
    print(f"total OCR blocks: {len(ocr_blocks)}")
    for i, blk in enumerate(ocr_blocks):
        cx, cy = _center(blk.bbox)
        print(f"OCR {i} [{blk.id}] conf={blk.confidence:.2f}")
        print(f"   bbox={_fmt_bbox(blk.bbox)}  center=({cx:.0f},{cy:.0f})  area={_area(blk.bbox):.0f}")
        print(f"   text: '{blk.text[:60]}'")
        # multi-bubble 判断
        hits = intersects_multiple_bubbles(blk.bbox, bubbles)
        if len(hits) >= 2:
            print(f"   ⚠ MULTI-BUBBLE: {hits}")

    # ---------- 4. BubbleAwareOCRService 处理 ----------
    ba = BubbleAwareOCRService(ocr_provider=ocr, threshold=0.15, strategy="auto", enabled=True)
    ba_result = await ba.process(img, ocr_blocks, bubbles)
    print("\n===== 4. BubbleAwareOCRService =====")
    print(f"in: {len(ocr_blocks)} blocks → out: {len(ba_result.blocks)} blocks, cross_bubble_splits: {ba_result.cross_bubble_count}")
    for i, blk in enumerate(ba_result.blocks):
        md = blk.metadata
        print(f"  [{i}] bubble={md.get('bubble_id')} method={md.get('split_method')} "
              f"cross={md.get('cross_bubble')} '{blk.text[:40]}'")
    for s in ba_result.splits:
        print(f"  SPLIT {s.source_ocr_block_id}: '{s.original_text[:30]}' → {s.split_method} → {s.produced}")

    # ---------- 5. Adapter 匹配（复现 _assign_ocr_to_regions）----------
    print("\n===== 5. Adapter 匹配（block → region）=====")
    assignments = adapter._assign_ocr_to_regions(text_regions, ba_result.blocks)
    for rid, blks in assignments.items():
        region = next((r for r in text_regions if r.id == rid), None)
        if not blks:
            continue
        print(f"Region [{rid}] (bbox={_fmt_bbox(region.bbox)}) <- {len(blks)} blocks")
        for blk in blks:
            # 计算与每个 region 的评分
            scores = []
            bx1, by1, bx2, by2 = map(int, blk.bbox)
            bcx, bcy = (bx1 + bx2) / 2, (by1 + by2) / 2
            for r in text_regions:
                rx1, ry1, rx2, ry2 = r.bbox
                if rx1 <= bcx <= rx2 and ry1 <= bcy <= ry2:
                    scores.append((r.id, 1.0, "center"))
                else:
                    iou = adapter._bbox_iou((bx1, by1, bx2, by2), (rx1, ry1, rx2, ry2))
                    scores.append((r.id, iou, "iou"))
            scores.sort(key=lambda x: -x[1])
            score_str = ", ".join(f"{sid}:{v:.2f}({k})" for sid, v, k in scores[:3])
            bid = blk.metadata.get("bubble_id", "?")
            print(f"    [{blk.id}] bubble_id={bid} | region_scores: {score_str} | '{blk.text[:35]}'")

    # ---------- 6. Dialogues ----------
    dialogues = adapter.to_dialogue_blocks(text_regions, ba_result.blocks, bubbles=bubbles if bubbles else None)
    print("\n===== 6. DialogueBlock（最终）=====")
    for d in dialogues:
        print(f"Dialogue [{d.id}] bubble={d.bubble.id} type={d.region_type}")
        print(f"    ocr_blocks={len(d.ocr_blocks)}  text='{d.merged_text[:60]}'")

    # ---------- 7. 诊断指标 ----------
    print("\n===== 7. 诊断指标 =====")
    single = sum(1 for blk in ba_result.blocks if len(intersects_multiple_bubbles(blk.bbox, bubbles)) == 1)
    multi = sum(1 for blk in ba_result.blocks if len(intersects_multiple_bubbles(blk.bbox, bubbles)) >= 2)
    none = sum(1 for blk in ba_result.blocks if not intersects_multiple_bubbles(blk.bbox, bubbles))
    bubbles_with_text = sum(1 for d in dialogues if d.merged_text.strip())
    bubbles_without = len(bubbles) - bubbles_with_text
    print(f"Bubbles: {len(bubbles)}")
    print(f"TextRegions (CTBD): {len(text_regions)}")
    print(f"OCR blocks (after BubbleAwareOCR): {len(ba_result.blocks)}")
    print(f"Single-bubble OCR: {single}")
    print(f"Multi-bubble OCR: {multi}")
    print(f"Unassigned: {none}")
    print(f"Bubbles with text: {bubbles_with_text}")
    print(f"Bubbles without text: {bubbles_without}")
    print(f"Avg OCR blocks per dialogue: {len(ba_result.blocks)/max(len(dialogues),1):.2f}")

    # ---------- 8. 可视化 ----------
    out_img = img.copy()
    draw = ImageDraw.Draw(out_img)

    # 画气泡 polygon
    for i, b in enumerate(bubbles):
        color = BUBBLE_COLORS[i % len(BUBBLE_COLORS)]
        if b.polygon and len(b.polygon) >= 3:
            draw.polygon([tuple(p) for p in b.polygon], outline=color, width=3)
        draw.rectangle([b.bbox[0], b.bbox[1], b.bbox[2], b.bbox[3]], outline=color, width=1)
        draw.text((b.bbox[0], b.bbox[1] - 14), f"B{i}:{b.id[:18]}", fill=color)

    # 画 OCR blocks
    for i, blk in enumerate(ba_result.blocks):
        bid = blk.metadata.get("bubble_id")
        hits = intersects_multiple_bubbles(blk.bbox, bubbles)
        if len(hits) >= 2:
            color = MULTI_COLOR
            label = f"OCR-{i}->MULTI"
        elif bid:
            # 用气泡颜色
            idx = next((k for k, b in enumerate(bubbles) if b.id == bid), -1)
            color = BUBBLE_COLORS[idx % len(BUBBLE_COLORS)] if idx >= 0 else NONE_COLOR
            label = f"OCR-{i}->B{idx}"
        else:
            color = NONE_COLOR
            label = f"OCR-{i}->NONE"
        x1, y1, x2, y2 = blk.bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        draw.text((x1, y2 + 2), label, fill=color)

    out_img.save(args.out)
    print(f"\n✓ 可视化已保存: {args.out}")


if __name__ == "__main__":
    asyncio.run(main())
