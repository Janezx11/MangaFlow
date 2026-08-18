#!/usr/bin/env python
"""
Bubble Matching Debug Overlay

输出 page_xxxx_matching_debug.png

颜色:
  红色: Bubble polygon
  蓝色: OCR bbox
  绿色: 成功匹配 (score >= 0.65)
  黄色: 候选匹配 (0.45 <= score < 0.65)
  白色: Singleton (未匹配)

用法:
  python demo/debug_bubble_matching.py
  python demo/debug_bubble_matching.py --page 1
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
from loguru import logger
from PIL import Image

logger.remove()
logger.add(sys.stderr, level="INFO")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from comic_translator.providers.bubble import YOLO11BubbleProvider
    from comic_translator.providers.ocr.easyocr_provider import EasyOCRProvider
    from comic_translator.services.bubble_matching_service import BubbleMatchingService

    provider = YOLO11BubbleProvider(device=args.device)
    ocr = EasyOCRProvider(lang="en", use_gpu=False)
    matcher = BubbleMatchingService()

    images = sorted([f for f in Path("data/comics").iterdir()
                     if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not images:
        logger.error("No images in data/comics/")
        return

    pages_to_process = [(i + 1, img) for i, img in enumerate(images)]
    if args.page > 0:
        pages_to_process = [p for p in pages_to_process if p[0] == args.page]

    output_dir = Path("data/output/matching_debug")
    output_dir.mkdir(parents=True, exist_ok=True)

    for page, img_path in pages_to_process:
        logger.info(f"\n=== Page {page}: {img_path.name} ===")
        img = Image.open(img_path)
        img_np = np.array(img.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # YOLO + OCR + Matching
        bubbles = await provider.detect(img)
        raw_blocks = await ocr.detect_text(img)
        dialogues = matcher.match(bubbles, raw_blocks)

        # 绘制 overlay
        overlay = img_bgr.copy()

        # 1. Bubble polygon (红色)
        for b in bubbles:
            if len(b.polygon) >= 3:
                pts = np.array(b.polygon, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(overlay, [pts], True, (0, 0, 255), 2)
            x1, y1, x2, y2 = b.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 255), 1)
            cv2.putText(overlay, f"B:{b.id[-8:]}", (x1 + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

        # 2. OCR blocks + 匹配结果
        for d in dialogues:
            is_singleton = d.id.startswith("singleton_")
            for ocr_b in d.ocr_blocks:
                x1, y1, x2, y2 = map(int, ocr_b.bbox)
                if is_singleton:
                    color = (255, 255, 255)  # 白色: singleton
                else:
                    color = (0, 255, 0)  # 绿色: 匹配
                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
                label = f"OCR:{ocr_b.id[-8:]}"
                cv2.putText(overlay, label, (x1 + 2, y1 - 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        # 3. 图例
        legend_y = 30
        for text, color in [
            ("Bubble polygon", (0, 0, 255)),
            ("Matched OCR", (0, 255, 0)),
            ("Singleton OCR", (255, 255, 255)),
        ]:
            cv2.putText(overlay, text, (10, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            legend_y += 20

        # 4. 统计信息
        matched = sum(1 for d in dialogues if not d.id.startswith("singleton_"))
        singletons = len(dialogues) - matched
        total_ocr = sum(len(d.ocr_blocks) for d in dialogues)
        stats = [
            f"Page: {page}",
            f"Bubbles: {len(bubbles)}",
            f"OCR blocks: {total_ocr}",
            f"Dialogues: {len(dialogues)}",
            f"Matched: {matched}",
            f"Singletons: {singletons}",
            f"Rate: {matched / max(total_ocr, 1) * 100:.1f}%",
        ]
        for i, s in enumerate(stats):
            cv2.putText(overlay, s, (img.width - 250, 30 + i * 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        out_path = output_dir / f"page_{page:04d}_matching_debug.png"
        cv2.imwrite(str(out_path), overlay)
        logger.info(f"Saved: {out_path}")

        # 打印每页统计
        logger.info(
            f"  Bubbles={len(bubbles)}, OCR={total_ocr}, "
            f"Dialogues={len(dialogues)}, Matched={matched}, "
            f"Singletons={singletons}, Rate={matched / max(total_ocr, 1) * 100:.1f}%"
        )

    logger.info(f"\nDone! Check {output_dir}/")


if __name__ == "__main__":
    asyncio.run(main())