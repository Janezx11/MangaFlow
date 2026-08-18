#!/usr/bin/env python
"""
Bubble Detection + Assignment 测试

用法:
  python scripts/test_bubble_detection.py
  python scripts/test_bubble_detection.py --page 1
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
from loguru import logger
from PIL import Image

logger.remove()
logger.add(sys.stderr, level="INFO")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Bubble Detection 测试")
    parser.add_argument("--page", type=int, default=0, help="只测试第 N 页")
    args = parser.parse_args()

    from comic_translator.services.bubble_detector import BubbleDetector
    from comic_translator.services.bubble_assignment_service import BubbleAssignmentService
    from comic_translator.services.batch_cache import BatchCacheService
    from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock

    detector = BubbleDetector()
    assigner = BubbleAssignmentService()
    cache = BatchCacheService()

    # Load OCR data
    ocr_pages = cache.get_ocr_cache("batch_project")
    if not ocr_pages:
        logger.error("No OCR cache found")
        return

    images = sorted(
        [f for f in Path("data/comics").iterdir()
         if f.suffix.lower() in (".png", ".jpg", ".jpeg")]
    )

    pages_to_check = ocr_pages
    if args.page > 0:
        pages_to_check = [p for p in ocr_pages if p["page"] == args.page]

    for page_data in pages_to_check:
        page = page_data["page"]
        img_path = images[page - 1] if page <= len(images) else None
        if not img_path or not img_path.exists():
            continue

        img_bgr = cv2.cvtColor(np.array(Image.open(img_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        blocks = [NormalizedOCRBlock(**b) for b in page_data["blocks"]]

        logger.info(f"=== Page {page} ===")

        # Detect bubbles
        bubbles = detector.detect(img_bgr)
        logger.info(f"Bubbles: {len(bubbles)}")
        for b in bubbles:
            x1, y1, x2, y2 = b.bbox
            logger.info(f"  {b.id}: ({x1},{y1})-({x2},{y2}) type={b.bubble_type} area={b.area}")

        # Assign OCR blocks to bubbles
        regions = assigner.assign(blocks, bubbles)
        logger.info(f"ConversationRegions: {len(regions)}")
        for r in regions:
            logger.info(f"  {r.id}: combined_text='{r.combined_text[:60]}...' ")

        # Draw debug output
        bubble_img = detector.draw_bubbles(img_bgr, bubbles)
        out_dir = Path("data/output/rendered")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"page_{page:04d}_bubbles.png"
        cv2.imwrite(str(out_path), bubble_img)
        logger.info(f"Bubble debug: {out_path}")


if __name__ == "__main__":
    main()