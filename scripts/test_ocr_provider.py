#!/usr/bin/env python
"""
OCR Provider 诊断工具

测试三个步骤，每步独立记录结果，不中断：
  1. Import — paddleocr / paddle 能否导入
  2. Init — PaddleOCR() 初始化是否成功
  3. Inference — ocr.ocr() 实际推理是否成功

用法:
  uv run python scripts/test_ocr_provider.py
  uv run python scripts/test_ocr_provider.py --image sample.png
"""

import sys

print("=" * 60)
print("PYTHON EXECUTABLE")
print(sys.executable)
print()

print("FIRST 10 SYS.PATH")
for p in sys.path[:10]:
    print(p)

print("=" * 60)


import argparse
import platform
import traceback
from pathlib import Path


def check_import() -> dict:
    """Step 1: 测试 paddleocr 能否导入"""
    result = {"ok": False, "msg": "", "traceback": ""}
    try:
        import paddle
        import paddleocr

        result["ok"] = True
        result["msg"] = (
            f"OK (paddleocr={paddleocr.__version__}, "
            f"paddle={paddle.__version__})"
        )
    except Exception:
        result["msg"] = "FAILED"
        result["traceback"] = traceback.format_exc()
    return result


def check_init(lang: str = "en", use_gpu: bool = False) -> dict:
    """Step 2: 测试 PaddleOCR() 初始化"""
    result = {"ok": False, "msg": "", "traceback": ""}
    try:
        from paddleocr import PaddleOCR

        ocr = PaddleOCR(lang=lang, use_gpu=use_gpu, show_log=False)
        result["ok"] = True
        result["msg"] = f"OK (lang={lang}, use_gpu={use_gpu})"
    except Exception:
        result["msg"] = "FAILED"
        result["traceback"] = traceback.format_exc()
    return result


def check_inference(image_path: str, lang: str = "en", use_gpu: bool = False) -> dict:
    """Step 3: 测试 ocr.ocr() 实际推理"""
    result = {"ok": False, "msg": "", "traceback": ""}
    try:
        from paddleocr import PaddleOCR
        import numpy as np
        from PIL import Image

        ocr = PaddleOCR(lang=lang, use_gpu=use_gpu, show_log=False)
        img = np.array(Image.open(image_path).convert("RGB"))
        raw = ocr.ocr(img)
        if raw and raw[0]:
            text_count = sum(1 for r in raw[0] if r and len(r) >= 2)
            result["ok"] = True
            result["msg"] = f"OK (detected {text_count} text blocks)"
        else:
            result["ok"] = True
            result["msg"] = "OK (no text detected)"
    except Exception:
        result["msg"] = "FAILED"
        result["traceback"] = traceback.format_exc()
    return result


def generate_test_image() -> str:
    """生成一张简单的测试图片"""
    from PIL import Image, ImageDraw

    path = Path("data/test_ocr.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (400, 200), color="white")
    draw = ImageDraw.Draw(img)
    draw.text((50, 80), "Hello, World!", fill="black")
    img.save(path)
    return str(path)


def print_step(label: str, result: dict) -> None:
    """格式化输出测试结果"""
    status = "OK" if result["ok"] else "FAILED"
    print(f"  {label}: {status}")
    if result["msg"]:
        print(f"           {result['msg']}")
    if not result["ok"] and result["traceback"]:
        # 缩进 traceback 便于阅读
        for line in result["traceback"].strip().splitlines():
            print(f"           {line}")


def main():
    parser = argparse.ArgumentParser(description="OCR Provider 诊断工具")
    parser.add_argument("--image", default="", help="测试图片路径（默认自动生成）")
    parser.add_argument("--lang", default="en", help="OCR 语言")
    parser.add_argument("--gpu", action="store_true", help="使用 GPU")
    args = parser.parse_args()

    print(f"Python:   {platform.python_version()}")
    print(f"Platform: {platform.platform()}")
    print()

    # Step 1 — Import
    print("=" * 60)
    print("Step 1: Import Test")
    print("=" * 60)
    import_result = check_import()
    print_step("Import", import_result)
    print()

    # Step 2 — Init (不论 Step 1 是否成功都执行)
    print("=" * 60)
    print("Step 2: Init Test")
    print("=" * 60)
    init_result = check_init(lang=args.lang, use_gpu=args.gpu)
    print_step("Init", init_result)
    print()

    # Step 3 — Inference (不论前两步是否成功都执行)
    print("=" * 60)
    print("Step 3: Inference Test")
    print("=" * 60)

    image_path = args.image
    if not image_path:
        image_path = generate_test_image()
        print(f"  Test image: {image_path}")

    if not Path(image_path).exists():
        print(f"  Inference: SKIP (image not found: {image_path})")
        inference_result = {"ok": False, "msg": "SKIP — no test image"}
    else:
        inference_result = check_inference(
            image_path, lang=args.lang, use_gpu=args.gpu,
        )
        print_step("Inference", inference_result)
    print()

    # 汇总
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    results = [
        ("Import", import_result),
        ("Init", init_result),
        ("Inference", inference_result),
    ]
    all_ok = True
    for label, r in results:
        mark = "✓" if r["ok"] else "✗"
        print(f"  [{mark}] {label}")
        all_ok = all_ok and r["ok"]

    print()
    if all_ok:
        print("✓ All checks passed — OCR is working.")
    else:
        failed = [label for label, r in results if not r["ok"]]
        print(f"✗ {len(failed)} check(s) failed: {', '.join(failed)}")
        print("  See details above for error traceback.")


if __name__ == "__main__":
    main()