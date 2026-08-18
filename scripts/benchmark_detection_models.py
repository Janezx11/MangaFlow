#!/usr/bin/env python
"""
Detection Models Benchmark — 多模型检测对比

比较 CTBD / YOLO / LocateAnything3B / Fusion 的检测效果。

用法:
  python scripts/benchmark_detection_models.py
  python scripts/benchmark_detection_models.py --models ctbd yolo
  python scripts/benchmark_detection_models.py --models ctbd yolo locate --page 2
"""

import argparse
import asyncio
import json
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

MODEL_COLORS = {
    "ctbd": (0, 255, 0),           # green
    "yolo11": (255, 128, 0),       # blue
    "locate_anything3b": (0, 128, 255),  # orange
    "fused": (255, 0, 255),        # magenta
}


async def main():
    parser = argparse.ArgumentParser(description="Detection Models Benchmark")
    parser.add_argument("--page", type=int, default=0, help="Page number (0=all)")
    parser.add_argument("--models", nargs="+", default=["ctbd", "yolo"],
                        choices=["ctbd", "yolo", "locate"],
                        help="Models to benchmark")
    parser.add_argument("--conf", type=float, default=0.3)
    parser.add_argument("--output", default="data/output/benchmark_models.json")
    args = parser.parse_args()

    from comic_translator.providers.vision import (
        CTBDVisionAdapter,
        YOLOVisionAdapter,
        LocateAnythingProvider,
    )
    from comic_translator.providers.text_detection.ctbd_text_detector import CTBDTextDetectorProvider
    from comic_translator.providers.bubble.yolo_bubble_provider import YOLO11BubbleProvider
    from comic_translator.services.detection_fusion_service import DetectionFusionService

    # Initialize providers
    providers = {}
    if "ctbd" in args.models:
        ctbd = CTBDTextDetectorProvider(confidence=args.conf)
        providers["ctbd"] = CTBDVisionAdapter(ctbd)
    if "yolo" in args.models:
        yolo = YOLO11BubbleProvider(
            model_path="models/bubble/yolo11n-seg-mixed-bubble.pt",
        )
        providers["yolo11"] = YOLOVisionAdapter(yolo)
    if "locate" in args.models:
        providers["locate_anything3b"] = LocateAnythingProvider(confidence=args.conf)

    fusion = DetectionFusionService()

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

    output_dir = Path("data/output/benchmark_models")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = []

    for page, img_path in pages:
        logger.info(f"\n=== Page {page}: {img_path.name} ===")
        img = Image.open(img_path)
        img_np = np.array(img.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        page_result = {"page": page, "file": img_path.name, "models": {}}

        # Run each model
        detection_lists = []
        for name, provider in providers.items():
            t0 = time.time()
            regions = await provider.detect(img)
            elapsed = time.time() - t0

            model_data = {
                "regions": len(regions),
                "avg_confidence": round(
                    sum(r.confidence for r in regions) / max(len(regions), 1), 3
                ),
                "time_s": round(elapsed, 2),
                "available": provider.is_available(),
                "labels": {},
            }
            for r in regions:
                model_data["labels"][r.label] = model_data["labels"].get(r.label, 0) + 1

            page_result["models"][name] = model_data
            detection_lists.append(regions)

            logger.info(
                f"  {name}: {len(regions)} regions, "
                f"avg_conf={model_data['avg_confidence']}, "
                f"time={elapsed:.2f}s"
            )

        # Fusion
        t0 = time.time()
        fused = fusion.fuse(detection_lists)
        fusion_time = time.time() - t0

        fused_data = {
            "regions": len(fused),
            "avg_confidence": round(
                sum(r.confidence for r in fused) / max(len(fused), 1), 3
            ),
            "time_s": round(fusion_time, 2),
            "labels": {},
            "sources": {},
        }
        for r in fused:
            fused_data["labels"][r.label] = fused_data["labels"].get(r.label, 0) + 1
            src = r.source
            fused_data["sources"][src] = fused_data["sources"].get(src, 0) + 1

        page_result["fused"] = fused_data
        logger.info(
            f"  fused: {len(fused)} regions, "
            f"avg_conf={fused_data['avg_confidence']}"
        )

        all_results.append(page_result)

        # Visualization: side-by-side for each model + fused
        h, w = img_bgr.shape[:2]
        n_panels = len(providers) + 1
        combined = np.zeros((h, w * n_panels + 10 * (n_panels - 1), 3), dtype=np.uint8)

        for idx, (name, provider) in enumerate(providers.items()):
            x_offset = idx * (w + 10)
            combined[:, x_offset:x_offset + w] = img_bgr

            regions = detection_lists[idx]
            color = MODEL_COLORS.get(name, (128, 128, 128))
            for r in regions:
                x1, y1, x2, y2 = r.bbox
                cv2.rectangle(combined, (x1 + x_offset, y1), (x2 + x_offset, y2), color, 2)
                label = f"{r.label} {r.confidence:.2f}"
                cv2.putText(combined, label, (x1 + x_offset + 2, y1 + 14),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            cv2.putText(combined, name.upper(), (x_offset + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # Fused panel
        x_offset = len(providers) * (w + 10)
        combined[:, x_offset:x_offset + w] = img_bgr
        fused_color = MODEL_COLORS["fused"]
        for r in fused:
            x1, y1, x2, y2 = r.bbox
            cv2.rectangle(combined, (x1 + x_offset, y1), (x2 + x_offset, y2), fused_color, 2)
            label = f"{r.label} {r.confidence:.2f} [{r.source}]"
            cv2.putText(combined, label, (x1 + x_offset + 2, y1 + 14),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, fused_color, 1)

        cv2.putText(combined, "FUSED", (x_offset + 10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, fused_color, 2)

        viz_path = output_dir / f"page_{page:04d}_models.png"
        cv2.imwrite(str(viz_path), combined)
        logger.info(f"  Viz: {viz_path}")

    # JSON output
    json_path = Path(args.output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    # Summary
    if all_results:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Detection Models Benchmark ({len(all_results)} pages)")
        logger.info(f"{'=' * 60}")

        for name in list(providers.keys()) + ["fused"]:
            if name == "fused":
                entries = [r.get("fused", {}) for r in all_results]
            else:
                entries = [r.get("models", {}).get(name, {}) for r in all_results]

            total_regions = sum(e.get("regions", 0) for e in entries)
            avg_conf = np.mean([e.get("avg_confidence", 0) for e in entries])
            avg_time = np.mean([e.get("time_s", 0) for e in entries])
            logger.info(f"  {name:20s}: regions={total_regions}, avg_conf={avg_conf:.3f}, avg_time={avg_time:.2f}s")

        logger.info(f"  JSON: {json_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())
