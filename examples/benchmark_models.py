#!/usr/bin/env python
"""
YOLO Model Benchmark — 比较不同 YOLO 模型在漫画气泡检测上的表现

用法:
  python demo/benchmark_models.py
  python demo/benchmark_models.py --models yolo11n-seg.pt,yolo11m-seg.pt
  python demo/benchmark_models.py --models yolo11n-seg.pt,yolo11m-seg.pt,yolo11l-seg.pt
"""

import asyncio
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

MODELS = [
    "yolo11n-seg.pt",
    "yolo11m-seg.pt",
    "yolo11l-seg.pt",
]


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS), help="Comma-separated model names")
    parser.add_argument("--output", default="data/output/benchmark_models.csv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]

    from comic_translator.providers.bubble import YOLO11BubbleProvider
    from comic_translator.providers.ocr.easyocr_provider import EasyOCRProvider
    from comic_translator.services.bubble_matching_service import BubbleMatchingService

    images = sorted([f for f in Path("data/comics").iterdir()
                     if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not images:
        logger.error("No images in data/comics/")
        return

    logger.info(f"Images: {len(images)} pages")
    logger.info(f"Models: {model_names}")

    all_rows = []

    for model_name in model_names:
        logger.info(f"\n{'='*60}")
        logger.info(f"Model: {model_name}")
        logger.info(f"{'='*60}")

        provider = YOLO11BubbleProvider(model_path=model_name, device=args.device)
        ocr = EasyOCRProvider(lang="en", use_gpu=False)
        matcher = BubbleMatchingService()

        total_bubbles = 0
        total_ocr = 0
        total_matched = 0
        total_inside = 0
        total_mask_area = 0
        total_bbox_area = 0
        total_conf = 0.0
        page_rows = []

        for page_idx, img_path in enumerate(images):
            page = page_idx + 1
            img = Image.open(img_path)

            # YOLO detection
            t0 = time.time()
            bubbles = await provider.detect(img)
            yolo_time = time.time() - t0

            # OCR
            blocks = await ocr.detect_text(img)

            # Matching
            t1 = time.time()
            dialogues = matcher.match(bubbles, blocks)
            match_time = time.time() - t1

            n_matched_ocr = sum(len(d.ocr_blocks) for d in dialogues if not d.id.startswith("singleton_"))

            # Stats per bubble
            n_bubbles = len(bubbles)
            total_bubbles += n_bubbles
            total_ocr += len(blocks)
            total_matched += n_matched_ocr

            # Mask quality
            mask_area = 0
            bbox_area = 0
            poly_bbox_ratio = 0.0
            total_conf += sum(b.confidence for b in bubbles)

            for b in bubbles:
                bx1, by1, bx2, by2 = b.bbox
                ba = (bx2 - bx1) * (by2 - by1)
                bbox_area += ba

                # Polygon area
                if len(b.polygon) >= 3:
                    poly_pts = np.array(b.polygon, dtype=np.int32)
                    pa = abs(cv2.contourArea(poly_pts))
                    mask_area += pa
                else:
                    pa = 0

                # Mask area (from YOLO mask)
                if b.mask and len(b.mask) > 0:
                    mask_arr = np.array(b.mask, dtype=np.float32)
                    if mask_arr.ndim == 3:
                        mask_arr = mask_arr.squeeze()
                    mask_pixels = int((mask_arr > 0.5).sum())
                    mask_area = max(mask_area, mask_pixels)

            # Count OCR blocks inside bubble bbox
            n_inside = 0
            for ob in blocks:
                ox1, oy1, ox2, oy2 = map(int, ob.bbox)
                for b in bubbles:
                    if ox2 > b.bbox[0] and ox1 < b.bbox[2] and oy2 > b.bbox[1] and oy1 < b.bbox[3]:
                        n_inside += 1
                        break

            total_inside += n_inside

            poly_ratio = mask_area / max(bbox_area, 1)
            inside_rate = n_matched_ocr / max(n_inside, 1)

            row = {
                "model": model_name,
                "page": page,
                "file": img_path.name,
                "bubbles": n_bubbles,
                "ocr_blocks": len(blocks),
                "matched_ocr": n_matched_ocr,
                "inside_bubble": n_inside,
                "inside_rate": round(inside_rate, 3),
                "mask_area_px": mask_area,
                "bbox_area_px": bbox_area,
                "poly_bbox_ratio": round(poly_ratio, 3),
                "avg_conf": round(sum(b.confidence for b in bubbles) / max(n_bubbles, 1), 3),
                "yolo_time_s": round(yolo_time, 2),
                "match_time_s": round(match_time, 2),
            }
            page_rows.append(row)
            all_rows.append(row)

            logger.info(
                f"  Page {page}: {n_bubbles} bubbles, "
                f"OCR={len(blocks)}, matched={n_matched_ocr}/{n_inside}, "
                f"poly/bbox={poly_ratio:.2f}, "
                f"avg_conf={row['avg_conf']:.2f}"
            )

        # 模型汇总
        avg_inside_rate = total_matched / max(total_inside, 1)
        avg_poly_ratio = 0
        avg_conf = total_conf / max(total_bubbles, 1)

        # 计算平均 poly/bbox ratio
        poly_ratios = [r["poly_bbox_ratio"] for r in page_rows]
        if poly_ratios:
            avg_poly_ratio = sum(poly_ratios) / len(poly_ratios)

        logger.info(
            f"\n  [{model_name}] Summary: "
            f"bubbles={total_bubbles}, "
            f"inside_rate={avg_inside_rate*100:.1f}%, "
            f"poly/bbox={avg_poly_ratio:.2f}, "
            f"avg_conf={avg_conf:.2f}"
        )

    # 输出 CSV
    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    if all_rows:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        logger.info(f"\nCSV: {csv_path.resolve()}")

    # 模型对比汇总
    logger.info(f"\n{'='*70}")
    logger.info(f"{'Model':<20} {'Bubbles':<10} {'Mask/BBox':<12} {'Inside%':<10} {'Conf':<8} {'Time':<8}")
    logger.info(f"{'-'*70}")
    for model_name in model_names:
        rows = [r for r in all_rows if r["model"] == model_name]
        if rows:
            total_b = sum(r["bubbles"] for r in rows)
            avg_pb = sum(r["poly_bbox_ratio"] for r in rows) / len(rows)
            total_m = sum(r["matched_ocr"] for r in rows)
            total_i = sum(r["inside_bubble"] for r in rows)
            avg_c = sum(r["avg_conf"] for r in rows) / len(rows)
            avg_t = sum(r["yolo_time_s"] for r in rows) / len(rows)
            ir = total_m / max(total_i, 1) * 100
            logger.info(f"{model_name:<20} {total_b:<10} {avg_pb:<12.3f} {ir:<10.1f} {avg_c:<8.2f} {avg_t:<8.2f}s")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())