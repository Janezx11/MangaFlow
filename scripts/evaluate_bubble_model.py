#!/usr/bin/env python
"""
YOLO 气泡检测模型评估脚本

评估维度:
  - bubble count: 每页检测到的气泡数
  - mask/bbox ratio: polygon 面积 / bbox 面积 (一致性)
  - OCR matching rate: 匹配上的 OCR 块比例
  - average confidence: 平均置信度
  - inference time: 推理时间

用法:
  python scripts/evaluate_bubble_model.py
  python scripts/evaluate_bubble_model.py --model yolo11m-seg.pt
  python scripts/evaluate_bubble_model.py --model models/bubble/yolo11m-seg-bubble.pt
  python scripts/evaluate_bubble_model.py --models yolo11n-seg.pt,yolo11m-seg.pt  # 对比
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


async def evaluate_model(model_name: str, images: list[Path], device: str) -> dict:
    """评估单个模型"""
    from comic_translator.providers.bubble import YOLO11BubbleProvider
    from comic_translator.providers.ocr.easyocr_provider import EasyOCRProvider
    from comic_translator.services.bubble_matching_service import BubbleMatchingService

    provider = YOLO11BubbleProvider(model_path=model_name, device=device)
    ocr = EasyOCRProvider(lang="en", use_gpu=False)
    matcher = BubbleMatchingService()

    total_bubbles = 0
    total_ocr = 0
    total_matched = 0
    total_inside = 0
    total_mask_area = 0
    total_bbox_area = 0
    total_conf = 0.0
    total_time = 0.0
    page_results = []

    for img_path in images:
        img = Image.open(img_path)

        t0 = time.time()
        bubbles = await provider.detect(img)
        elapsed = time.time() - t0
        total_time += elapsed

        blocks = await ocr.detect_text(img)
        dialogues = matcher.match(bubbles, blocks)

        n_matched = sum(len(d.ocr_blocks) for d in dialogues if not d.id.startswith("singleton_"))
        total_bubbles += len(bubbles)
        total_ocr += len(blocks)
        total_matched += n_matched
        total_conf += sum(b.confidence for b in bubbles)

        # mask/bbox ratio
        page_mask_area = 0
        page_bbox_area = 0
        for b in bubbles:
            bx1, by1, bx2, by2 = b.bbox
            ba = (bx2 - bx1) * (by2 - by1)
            page_bbox_area += ba

            if b.mask and len(b.mask) > 0:
                mask_arr = np.array(b.mask, dtype=np.float32)
                if mask_arr.ndim == 3:
                    mask_arr = mask_arr.squeeze()
                page_mask_area += int((mask_arr > 0.5).sum())

        total_mask_area += page_mask_area
        total_bbox_area += page_bbox_area

        # 气泡内 OCR 计数
        n_inside = 0
        for ob in blocks:
            ox1, oy1, ox2, oy2 = map(int, ob.bbox)
            for b in bubbles:
                if ox2 > b.bbox[0] and ox1 < b.bbox[2] and oy2 > b.bbox[1] and oy1 < b.bbox[3]:
                    n_inside += 1
                    break
        total_inside += n_inside

        page_results.append({
            "file": img_path.name,
            "bubbles": len(bubbles),
            "ocr": len(blocks),
            "matched": n_matched,
            "inside": n_inside,
            "mask_ratio": round(page_mask_area / max(page_bbox_area, 1), 3),
            "avg_conf": round(sum(b.confidence for b in bubbles) / max(len(bubbles), 1), 3),
            "time_s": round(elapsed, 2),
        })

    avg_conf = total_conf / max(total_bubbles, 1)
    avg_mask_ratio = total_mask_area / max(total_bbox_area, 1)
    inside_rate = total_matched / max(total_inside, 1)

    return {
        "model": model_name,
        "total_bubbles": total_bubbles,
        "total_ocr": total_ocr,
        "total_matched": total_matched,
        "total_inside": total_inside,
        "inside_rate": inside_rate,
        "avg_mask_ratio": avg_mask_ratio,
        "avg_conf": avg_conf,
        "avg_time": total_time / len(images),
        "pages": page_results,
    }


async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate YOLO bubble detection models")
    parser.add_argument("--models", default="yolo11n-seg.pt,yolo11m-seg.pt",
                        help="逗号分隔的模型列表")
    parser.add_argument("--data", default="data/comics", help="测试图片目录")
    parser.add_argument("--output", default="data/output/eval_bubble.csv")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    model_names = [m.strip() for m in args.models.split(",")]

    data_dir = Path(args.data)
    images = sorted([f for f in data_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not images:
        logger.error(f"No images found in {args.data}")
        return

    logger.info(f"测试图片: {len(images)} 页")
    logger.info(f"模型: {model_names}")

    all_results = []
    for model_name in model_names:
        logger.info(f"\n--- {model_name} ---")
        result = await evaluate_model(model_name, images, args.device)

        # 输出汇总
        logger.info(f"  Bubbles:     {result['total_bubbles']}")
        logger.info(f"  OCR:         {result['total_ocr']}")
        logger.info(f"  Matched:     {result['total_matched']}/{result['total_inside']}")
        logger.info(f"  Inside rate: {result['inside_rate']*100:.1f}%")
        logger.info(f"  Mask/BBox:   {result['avg_mask_ratio']:.3f}")
        logger.info(f"  Avg conf:    {result['avg_conf']:.3f}")
        logger.info(f"  Avg time:    {result['avg_time']:.2f}s")

        all_results.append(result)

    # 输出 CSV
    csv_path = Path(args.output)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "page", "bubbles", "ocr", "matched", "inside",
                         "mask_ratio", "avg_conf", "time_s"])
        for r in all_results:
            for p in r["pages"]:
                writer.writerow([r["model"], p["file"], p["bubbles"], p["ocr"],
                                 p["matched"], p["inside"], p["mask_ratio"],
                                 p["avg_conf"], p["time_s"]])

    # 对比表
    logger.info(f"\n{'='*70}")
    logger.info(f"{'Model':<25} {'Bubbles':<8} {'Mask/BBox':<10} {'Inside%':<10} {'Conf':<8} {'Time':<8}")
    logger.info(f"{'-'*70}")
    for r in all_results:
        logger.info(f"{r['model']:<25} {r['total_bubbles']:<8} "
                    f"{r['avg_mask_ratio']:<10.3f} {r['inside_rate']*100:<10.1f} "
                    f"{r['avg_conf']:<8.2f} {r['avg_time']:<8.2f}s")
    logger.info(f"{'='*70}")
    logger.info(f"CSV saved: {csv_path.resolve()}")


if __name__ == "__main__":
    asyncio.run(main())