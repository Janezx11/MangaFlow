#!/usr/bin/env python
"""
Bubble Detection Benchmark

统计每页的 Bubble Detection 质量，输出 CSV 和可视化 Overlay。

用法:
  python demo/benchmark_bubble.py
  python demo/benchmark_bubble.py --page 1
  python demo/benchmark_bubble.py --output benchmark_results.csv
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--model", default="", help="Path to YOLO .pt model (default: yolo11n-seg.pt)")
    parser.add_argument("--conf", type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="YOLO IoU threshold")
    parser.add_argument("--output", default="data/output/benchmark_bubble.csv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from comic_translator.providers.bubble import YOLO11BubbleProvider
    from comic_translator.providers.ocr.easyocr_provider import EasyOCRProvider
    from comic_translator.services.bubble_matching_service import BubbleMatchingService
    from comic_translator.services.bubble_postprocess_service import BubblePostProcessService
    from comic_translator.services.bubble_interior_service import BubbleInteriorService
    from comic_translator.services.region_classifier import RegionClassifier

    bubble_provider = YOLO11BubbleProvider(
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
    )
    ocr_provider = EasyOCRProvider(lang="en", use_gpu=False)
    matcher = BubbleMatchingService()
    postproc = BubblePostProcessService()
    interior = BubbleInteriorService()
    classifier = RegionClassifier()

    images = sorted([f for f in Path("data/comics").iterdir()
                     if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not images:
        logger.error("No images found in data/comics/")
        return

    pages_to_process = [(i + 1, img) for i, img in enumerate(images)]
    if args.page > 0:
        pages_to_process = [p for p in pages_to_process if p[0] == args.page]

    results = []
    output_dir = Path("data/output/benchmark")
    output_dir.mkdir(parents=True, exist_ok=True)

    for page, img_path in pages_to_process:
        logger.info(f"\n=== Page {page}: {img_path.name} ===")
        img = Image.open(img_path)
        img_np = np.array(img.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        # YOLO detection
        t0 = time.time()
        bubbles = await bubble_provider.detect(img)
        yolo_time = time.time() - t0
        raw_count = len(bubbles)

        # Post-process
        t1 = time.time()
        bubbles = postproc.process(bubbles, img.size[::-1])
        post_time = time.time() - t1
        post_count = len(bubbles)

        # OCR
        t2 = time.time()
        raw_blocks = await ocr_provider.detect_text(img)
        ocr_time = time.time() - t2
        ocr_count = len(raw_blocks)

        # Matching
        t3 = time.time()
        dialogues = matcher.match(bubbles, raw_blocks)
        match_time = time.time() - t3

        # Interior + Region
        img_shape = (img.height, img.width)
        safe_areas = interior.compute_batch(bubbles, img_shape)
        for b in bubbles:
            b.safe_area = safe_areas.get(b.id)
        for d in dialogues:
            d.region_type = classifier.classify(d.bubble, img.height, img.width)

        # Stats
        matched_bubbles = sum(1 for d in dialogues if not d.id.startswith("singleton_"))
        singleton_count = len(dialogues) - matched_bubbles
        total_area = img.width * img.height
        bubble_area_ratio = sum(
            (b.bbox[2] - b.bbox[0]) * (b.bbox[3] - b.bbox[1])
            for b in bubbles
        ) / total_area if total_area > 0 else 0

        # Region distribution
        region_counts = {}
        for d in dialogues:
            rt = d.region_type
            region_counts[rt] = region_counts.get(rt, 0) + 1

        row = {
            "page": page,
            "file": img_path.name,
            "width": img.width,
            "height": img.height,
            "yolo_time_s": round(yolo_time, 2),
            "ocr_time_s": round(ocr_time, 2),
            "post_time_s": round(post_time, 2),
            "match_time_s": round(match_time, 2),
            "total_time_s": round(yolo_time + post_time + ocr_time + match_time, 2),
            "raw_bubbles": raw_count,
            "post_bubbles": post_count,
            "filtered_removed": raw_count - post_count,
            "ocr_blocks": ocr_count,
            "dialogues": len(dialogues),
            "matched_bubbles": matched_bubbles,
            "singletons": singleton_count,
            "bubble_area_ratio": round(bubble_area_ratio, 3),
            "speech": region_counts.get("speech", 0),
            "narration": region_counts.get("narration", 0),
            "title": region_counts.get("title", 0),
            "sfx": region_counts.get("sfx", 0),
        }
        results.append(row)

        logger.info(
            f"  YOLO={raw_count} → Post={post_count} | "
            f"OCR={ocr_count} | Dialogues={len(dialogues)} "
            f"(matched={matched_bubbles}, singleton={singleton_count}) | "
            f"Regions: {region_counts}"
        )

        # 可视化 Overlay
        overlay = img_bgr.copy()
        for d in dialogues:
            color = {
                "speech": (0, 255, 0),
                "narration": (0, 128, 255),
                "title": (255, 0, 0),
                "sfx": (255, 255, 0),
            }.get(d.region_type, (128, 128, 128))

            x1, y1, x2, y2 = d.bubble.bbox
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
            if d.bubble.polygon and len(d.bubble.polygon) >= 3:
                pts = np.array(d.bubble.polygon, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(overlay, [pts], True, color, 1)

            label = f"{d.id[:16]} [{d.region_type}]"
            cv2.putText(overlay, label, (x1 + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imwrite(str(output_dir / f"page_{page:04d}_benchmark.png"), overlay)
        logger.info(f"  Overlay: {output_dir / f'page_{page:04d}_benchmark.png'}")

    # 输出 CSV
    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # 汇总
    if results:
        avg_yolo = sum(r["yolo_time_s"] for r in results) / len(results)
        avg_ocr = sum(r["ocr_time_s"] for r in results) / len(results)
        total_bubbles = sum(r["raw_bubbles"] for r in results)
        total_post = sum(r["post_bubbles"] for r in results)
        total_matched = sum(r["matched_bubbles"] for r in results)
        total_singletons = sum(r["singletons"] for r in results)
        logger.info(f"\n{'=' * 50}")
        logger.info(f"Benchmark Summary ({len(results)} pages)")
        logger.info(f"{'=' * 50}")
        logger.info(f"  Avg YOLO time: {avg_yolo:.2f}s")
        logger.info(f"  Avg OCR time:  {avg_ocr:.2f}s")
        logger.info(f"  Total raw bubbles:     {total_bubbles}")
        logger.info(f"  Total post-processed:  {total_post}")
        logger.info(f"  Total matched:         {total_matched}")
        logger.info(f"  Total singletons:      {total_singletons}")
        logger.info(f"  CSV: {csv_path.resolve()}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())