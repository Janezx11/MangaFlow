"""Inpaint Provider Benchmark — 对比 OpenCV / LaMa / SD 修复质量

用法:
  python scripts/benchmark_inpaint.py --image data/comics/0002_GREECE0001.png
  python scripts/benchmark_inpaint.py --image data/comics/0002_GREECE0001.png --providers opencv lama

输出:
  data/output/benchmark_inpaint/page_XXXX_{provider}.png
  data/output/benchmark_inpaint.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from comic_translator.providers.inpaint import (
    LaMaInpaintProvider,
    OpenCVInpaintProvider,
    SDInpaintProvider,
)
from comic_translator.providers.inpaint.base_inpaint_provider import InpaintProvider
from comic_translator.services.mask import OCRMaskProvider
from comic_translator.services.mask_refinement_service import MaskRefinementService
from comic_translator.services.render_quality_evaluator import RenderQualityEvaluator


OUT_DIR = Path("data/output/benchmark_inpaint")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def build_mask(image: Image.Image) -> np.ndarray:
    """用简单的 placeholder mask：模拟 OCR bbox"""
    w, h = image.size
    # 画几个模拟文字框
    mask = np.zeros((h, w), dtype=np.uint8)
    bboxes = [
        (int(w*0.25), int(h*0.25), int(w*0.75), int(h*0.35)),
        (int(w*0.20), int(h*0.60), int(w*0.60), int(h*0.70)),
    ]
    for x1, y1, x2, y2 in bboxes:
        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)

    # 用 OCRMaskProvider 风格做收缩/平滑
    provider = OCRMaskProvider(padding=6, text_ratio=0.6, smooth_kernel=5)
    # 伪造一个 DialogueBlock 以便 provider 能处理
    from comic_translator.models.bubble import DialogueBlock, Bubble
    bubble = Bubble(id="b1", bbox=(0,0,w,h), polygon=[], mask=[], class_name="speech")
    db = DialogueBlock(id="d1", bubble=bubble, ocr_blocks=[])
    return provider.generate(image, [db])


def run_provider(provider: InpaintProvider, image: np.ndarray, mask: np.ndarray):
    if not provider.is_available():
        return None
    t0 = time.time()
    try:
        result = provider.inpaint(image, mask)
    except Exception as e:
        logger.error(f"{provider.get_name()} failed: {e}")
        return None
    elapsed = time.time() - t0
    return result, elapsed


def main():
    parser = argparse.ArgumentParser(description="Inpaint Provider Benchmark")
    parser.add_argument("--image", required=True, help="输入图片路径")
    parser.add_argument(
        "--providers",
        nargs="+",
        default=["opencv", "lama", "sd"],
        choices=["opencv", "lama", "sd"],
    )
    args = parser.parse_args()

    img_path = Path(args.image)
    image = Image.open(img_path).convert("RGB")
    img_np = np.array(image)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    mask = build_mask(image)
    refined = MaskRefinementService().refine(mask, mask.shape[:2])

    evaluator = RenderQualityEvaluator()
    results = []

    providers = {
        "opencv": OpenCVInpaintProvider(method="telea", radius=5),
        "lama": LaMaInpaintProvider(),
        "sd": SDInpaintProvider(),
    }

    cv2.imwrite(str(OUT_DIR / "mask.png"), refined)

    for name in args.providers:
        provider = providers[name]
        res = run_provider(provider, img_bgr, refined)
        if res is None:
            score = {"overall": 0, "text_residual": 0, "blur": 0, "edge_preservation": 0}
            elapsed = 0.0
            rendered = img_bgr.copy()
            available = False
        else:
            rendered, elapsed = res
            quality = evaluator.evaluate(img_np, cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB), refined)
            score = quality.__dict__
            available = True

        out_path = OUT_DIR / f"{img_path.stem}_{name}.png"
        cv2.imwrite(str(out_path), rendered)

        results.append({
            "provider": name,
            "available": available,
            "time_s": round(elapsed, 3),
            "quality": score,
            "output": str(out_path),
        })

    summary_path = OUT_DIR / "benchmark_inpaint.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # 打印对比
    print(f"\nInpaint Benchmark: {img_path.name}")
    print(f"{'Provider':<12} {'Available':<10} {'Time(s)':<10} {'Overall':<10} {'Residual':<10} {'Blur':<10} {'Edge':<10}")
    for r in results:
        q = r["quality"]
        print(
            f"{r['provider']:<12} {str(r['available']):<10} {r['time_s']:<10} "
            f"{q['overall']:<10} {q['text_residual']:<10} {q['blur']:<10} {q['edge_preservation']:<10}"
        )
    print(f"\nSaved: {summary_path}")


if __name__ == "__main__":
    main()
