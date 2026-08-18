#!/usr/bin/env python
"""
OCR 健康检查 CLI

检查系统中可用的 OCR Provider 状态:
  - PaddleOCR (import / init / inference)
  - EasyOCR (import / init / inference)

用法:
  uv run python demo/check_ocr.py
  uv run python demo/check_ocr.py --lang ja
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from comic_translator.services.ocr_health_checker import OCRHealthChecker


def main():
    import argparse

    parser = argparse.ArgumentParser(description="OCR 健康检查")
    parser.add_argument("--lang", default="en", help="OCR 语言代码")
    parser.add_argument("--all", action="store_true", help="检查所有 Provider")
    args = parser.parse_args()

    print("=" * 60)
    print("OCR Health Check")
    print("=" * 60)
    print()

    if args.all:
        results = OCRHealthChecker.check_all(lang=args.lang)
    else:
        results = [
            OCRHealthChecker.check_paddle(lang=args.lang),
            OCRHealthChecker.check_easyocr(lang=args.lang),
        ]

    for result in results:
        print(str(result))
        print()

    # Summary
    working = [r for r in results if r.all_ok]
    if working:
        print(f"✓ {len(working)} provider(s) available: {', '.join(r.provider for r in working)}")
    else:
        print("✗ No OCR provider available.")
        print("  Install:  uv sync --extra paddle")
        print("  Or:       pip install easyocr")


if __name__ == "__main__":
    main()