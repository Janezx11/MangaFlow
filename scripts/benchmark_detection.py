#!/usr/bin/env python
"""
Detection Layer Benchmark — 比较新旧 Pipeline

旧 Pipeline: YOLO Bubble Detection → OCR → Bubble Matching
新 Pipeline: Text Detection (CTBD) → OCR → TextRegionAdapter

用法:
  python scripts/benchmark_detection.py
  python scripts/benchmark_detection.py --page 1
  python scripts/benchmark_detection.py --conf 0.3
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
from loguru import logger
from PIL import Image

logger.remove()
logger.add(sys.stderr, level="INFO")


async def main():
    parser = argparse.ArgumentParser(description="Detection Layer Benchmark")
    parser.add_argument("--page", type=int, default=0, help="Page number (0=all)")
    parser.add_argument("--conf", type=float, default=0.3, help="CTBD confidence threshold")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    parser.add_argument("--output", default="data/output/benchmark_detection.csv")
    args = parser.parse_args()

    # --- Providers ---
    from comic_translator.providers.text_detection import CTBDTextDetectorProvider
    from comic_translator.providers.bubble import YOLO11BubbleProvider
    from comic_translator.providers.ocr.easyocr_provider import EasyOCRProvider
    from comic_translator.services.text_region_service import TextRegionService
    from comic_translator.services.text_region_adapter import TextRegionAdapter
    from comic_translator.services.bubble_matching_service import BubbleMatchingService
    from comic_translator.services.bubble_postprocess_service import BubblePostProcessService
    from comic_translator.services.bubble_interior_service import BubbleInteriorService
    from comic_translator.services.region_classifier import RegionClassifier

    # New pipeline
    text_detector = CTBDTextDetectorProvider(confidence=args.conf, device=args.device)
    text_region_service = TextRegionService()
    text_region_adapter = TextRegionAdapter()

    # Old pipeline
    bubble_provider = YOLO11BubbleProvider(
        model_path="models/bubble/yolo11n-seg-mixed-bubble.pt",
        device=args.device if args.device == "cuda" else "cpu",
    )
    ocr_provider = EasyOCRProvider(lang="en", use_gpu=False)
    matcher = BubbleMatchingService()
    postproc = BubblePostProcessService()
    interior = BubbleInteriorService()
    classifier = RegionClassifier()

    # Collect images
    images = sorted([
        f for f in Path("data/comics").iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg")
    ])
    if not images:
        logger.error("No images found in data/comics/")
        return

    pages = [(i + 1, img) for i, img in enumerate(images)]
    if args.page > 0:
        pages = [p for p in pages if p[0] == args.page]

    results = []
    output_dir = Path("data/output/benchmark_detection")
    output_dir.mkdir(parents=True, exist_ok=True)

    for page, img_path in pages:
        logger.info(f"\n=== Page {page}: {img_path.name} ===")
        img = Image.open(img_path)
        img_np = np.array(img.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # ==================== NEW PIPELINE ====================
        logger.info("--- New Pipeline (Text Detection) ---")

        # Text Detection
        t0 = time.time()
        text_regions = await text_detector.detect(img)
        text_detect_time = time.time() - t0

        # OCR within regions
        t0 = time.time()
        new_ocr_blocks = await text_region_service.extract_text(
            img, text_regions, ocr_provider
        )
        new_ocr_time = time.time() - t0

        # Optional: Bubble Detection
        t0 = time.time()
        bubbles_aux = await bubble_provider.detect(img)
        bubble_time = time.time() - t0

        # Text-Bubble Matching
        t0 = time.time()
        new_dialogues = text_region_adapter.to_dialogue_blocks(
            text_regions, new_ocr_blocks, bubbles=bubbles_aux,
        )
        new_match_time = time.time() - t0

        new_matched = sum(
            1 for d in new_dialogues if not d.id.startswith("singleton_")
        )
        new_singletons = len(new_dialogues) - new_matched
        new_total_time = text_detect_time + new_ocr_time + bubble_time + new_match_time

        # Region type distribution
        new_type_counts = {}
        for d in new_dialogues:
            new_type_counts[d.region_type] = new_type_counts.get(d.region_type, 0) + 1

        logger.info(
            f"  Regions={len(text_regions)}, OCR={len(new_ocr_blocks)}, "
            f"Dialogues={len(new_dialogues)} (matched={new_matched}, singleton={new_singletons})"
        )

        # ==================== OLD PIPELINE ====================
        logger.info("--- Old Pipeline (Bubble Detection) ---")

        # YOLO Bubble Detection
        t0 = time.time()
        bubbles_old = await bubble_provider.detect(img)
        old_yolo_time = time.time() - t0

        # Post-process
        t0 = time.time()
        bubbles_old = postproc.process(bubbles_old, img.size[::-1])
        old_post_time = time.time() - t0

        # Full-page OCR
        t0 = time.time()
        old_ocr_blocks = await ocr_provider.detect_text(img)
        old_ocr_time = time.time() - t0

        # Bubble Matching
        t0 = time.time()
        old_dialogues = matcher.match(bubbles_old, old_ocr_blocks)
        old_match_time = time.time() - t0

        old_matched = sum(
            1 for d in old_dialogues if not d.id.startswith("singleton_")
        )
        old_singletons = len(old_dialogues) - old_matched
        old_total_time = old_yolo_time + old_post_time + old_ocr_time + old_match_time

        old_type_counts = {}
        for d in old_dialogues:
            old_type_counts[d.region_type] = old_type_counts.get(d.region_type, 0) + 1

        logger.info(
            f"  Bubbles={len(bubbles_old)}, OCR={len(old_ocr_blocks)}, "
            f"Dialogues={len(old_dialogues)} (matched={old_matched}, singleton={old_singletons})"
        )

        # ==================== COMPARISON ====================
        row = {
            "page": page,
            "file": img_path.name,
            # New pipeline
            "new_regions": len(text_regions),
            "new_ocr_blocks": len(new_ocr_blocks),
            "new_dialogues": len(new_dialogues),
            "new_matched": new_matched,
            "new_singletons": new_singletons,
            "new_singleton_rate": round(new_singletons / max(len(new_ocr_blocks), 1), 3),
            "new_avg_conf": round(
                sum(r.confidence for r in text_regions) / max(len(text_regions), 1), 3
            ),
            "new_time_s": round(new_total_time, 2),
            "new_text_detect_s": round(text_detect_time, 2),
            "new_ocr_s": round(new_ocr_time, 2),
            # Old pipeline
            "old_bubbles": len(bubbles_old),
            "old_ocr_blocks": len(old_ocr_blocks),
            "old_dialogues": len(old_dialogues),
            "old_matched": old_matched,
            "old_singletons": old_singletons,
            "old_singleton_rate": round(old_singletons / max(len(old_ocr_blocks), 1), 3),
            "old_time_s": round(old_total_time, 2),
            "old_yolo_s": round(old_yolo_time, 2),
            "old_ocr_s": round(old_ocr_time, 2),
            # Comparison
            "detection_diff": len(text_regions) - len(bubbles_old),
            "matched_diff": new_matched - old_matched,
        }
        results.append(row)

        logger.info(
            f"  Comparison: regions={len(text_regions)} vs bubbles={len(bubbles_old)}, "
            f"matched={new_matched} vs {old_matched}, "
            f"singleton_rate={row['new_singleton_rate']} vs {row['old_singleton_rate']}"
        )

        # ==================== VISUALIZATION ====================
        # Side-by-side comparison
        h, w = img_bgr.shape[:2]
        combined = np.zeros((h, w * 2 + 20, 3), dtype=np.uint8)
        combined[:, :w] = img_bgr
        combined[:, w + 20:] = img_bgr

        # Left: New pipeline (green)
        for d in new_dialogues:
            color = {
                "speech": (0, 255, 0),
                "narration": (0, 128, 255),
                "title": (255, 0, 0),
                "sfx": (255, 255, 0),
            }.get(d.region_type, (128, 128, 128))
            x1, y1, x2, y2 = d.bubble.bbox
            cv2.rectangle(combined, (x1, y1), (x2, y2), color, 2)
            label = f"NEW [{d.region_type}]"
            cv2.putText(combined, label, (x1 + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Right: Old pipeline (blue)
        for d in old_dialogues:
            color = {
                "speech": (0, 255, 0),
                "narration": (0, 128, 255),
                "title": (255, 0, 0),
                "sfx": (255, 255, 0),
            }.get(d.region_type, (128, 128, 128))
            x1, y1, x2, y2 = d.bubble.bbox
            cv2.rectangle(combined, (x1 + w + 20, y1), (x2 + w + 20, y2), color, 2)
            label = f"OLD [{d.region_type}]"
            cv2.putText(combined, label, (x1 + w + 22, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # Labels
        cv2.putText(combined, "NEW: Text Detection", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        cv2.putText(combined, "OLD: Bubble Detection", (w + 30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 128, 0), 2)

        viz_path = output_dir / f"page_{page:04d}_comparison.png"
        cv2.imwrite(str(viz_path), combined)
        logger.info(f"  Comparison: {viz_path}")

    # ==================== CSV OUTPUT ====================
    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # ==================== SUMMARY ====================
    if results:
        logger.info(f"\n{'=' * 70}")
        logger.info(f"Detection Benchmark Summary ({len(results)} pages)")
        logger.info(f"{'=' * 70}")

        total_new_regions = sum(r["new_regions"] for r in results)
        total_old_bubbles = sum(r["old_bubbles"] for r in results)
        total_new_matched = sum(r["new_matched"] for r in results)
        total_old_matched = sum(r["old_matched"] for r in results)
        total_new_singletons = sum(r["new_singletons"] for r in results)
        total_old_singletons = sum(r["old_singletons"] for r in results)
        avg_new_time = sum(r["new_time_s"] for r in results) / len(results)
        avg_old_time = sum(r["old_time_s"] for r in results) / len(results)

        logger.info(f"  New pipeline (Text Detection):")
        logger.info(f"    Total regions:    {total_new_regions}")
        logger.info(f"    Total matched:    {total_new_matched}")
        logger.info(f"    Total singletons: {total_new_singletons}")
        logger.info(f"    Avg time:         {avg_new_time:.2f}s")
        logger.info(f"  Old pipeline (Bubble Detection):")
        logger.info(f"    Total bubbles:    {total_old_bubbles}")
        logger.info(f"    Total matched:    {total_old_matched}")
        logger.info(f"    Total singletons: {total_old_singletons}")
        logger.info(f"    Avg time:         {avg_old_time:.2f}s")
        logger.info(f"  CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
