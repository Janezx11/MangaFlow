#!/usr/bin/env python
"""
真实漫画 OCR 验证

输入:  data/comics/*.png（或其他格式图片）
流程:  Image → PaddleOCRProvider.detect_text() → OCRNormalizeService.process()
输出:  控制台 + data/output/ocr_dump/*.json

用法:
  uv run python scripts/test_real_comic_ocr.py
  uv run python scripts/test_real_comic_ocr.py --image data/comics/sample.png
  uv run python scripts/test_real_comic_ocr.py --limit 3
"""

import argparse
import asyncio
import json
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# Windows GBK 控制台兼容
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]


def main():
    parser = argparse.ArgumentParser(description="真实漫画 OCR 验证")
    parser.add_argument("--image", default="", help="单张图片路径（默认处理 data/comics/ 下全部）")
    parser.add_argument("--limit", type=int, default=0, help="最多处理 N 页（0=全部）")
    parser.add_argument("--lang", default="en", help="OCR 语言")
    parser.add_argument("--gpu", action="store_true", help="使用 GPU")
    parser.add_argument("--no-normalize", action="store_true", help="跳过标准化（只看原始 OCR 结果）")
    args = parser.parse_args()

    # --- 收集图片 ---
    if args.image:
        images = [Path(args.image)]
    else:
        comic_dir = Path("data/comics")
        if not comic_dir.exists():
            print(f"目录不存在: {comic_dir}")
            sys.exit(1)
        images = sorted(
            [f for f in comic_dir.iterdir() if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")]
        )

    if not images:
        print("没有找到图片。请把漫画图片放到 data/comics/ 目录。")
        sys.exit(1)

    if args.limit > 0:
        images = images[: args.limit]

    print(f"共 {len(images)} 页")
    print()

    # --- 初始化 OCR Provider ---
    # 自动检测可用 Provider: EasyOCR → PaddleOCR
    provider = None

    try:
        from comic_translator.providers.ocr.easyocr_provider import EasyOCRProvider

        print("尝试 EasyOCR ...")
        provider = EasyOCRProvider(lang=args.lang, use_gpu=args.gpu)
        print("EasyOCR 初始化成功")
    except Exception as e:
        print(f"  EasyOCR: {e}")

    if provider is None:
        try:
            from comic_translator.providers.ocr.paddle_ocr_provider import PaddleOCRProvider

            print("尝试 PaddleOCR ...")
            provider = PaddleOCRProvider(lang=args.lang, use_gpu=args.gpu)
            print("PaddleOCR 初始化成功")
        except Exception as e:
            print(f"  PaddleOCR: {e}")

    if provider is None:
        print("错误: 没有可用的 OCR Provider")
        print("  安装 EasyOCR: uv pip install easyocr")
        print("  安装 PaddleOCR: uv sync --extra paddle")
        sys.exit(1)

    # --- 初始化标准化服务 ---
    from comic_translator.services.ocr_normalize_service import OCRNormalizeService

    normalize = OCRNormalizeService() if not args.no_normalize else None

    # --- 创建输出目录 ---
    dump_dir = Path("data/output/ocr_dump")
    dump_dir.mkdir(parents=True, exist_ok=True)

    # --- 逐页处理 ---
    total_raw = 0
    total_normalized = 0

    for idx, img_path in enumerate(images):
        print("=" * 60)
        print(f"Page {idx + 1}: {img_path.name}")
        print("=" * 60)

        # ----- Step 1: OCR -----
        from PIL import Image

        try:
            image = Image.open(img_path)
            print(f"  图片尺寸: {image.size}")
        except Exception as e:
            print(f"  打开失败: {e}")
            continue

        try:
            raw_blocks = asyncio.run(provider.detect_text(image))
        except Exception as e:
            print(f"\n  OCR 失败: {e}")
            print(f"  Traceback:\n{traceback.format_exc()}")
            # 即使 OCR 失败也继续（PaddleOCRProvider 已改为返回空列表）
            raw_blocks = []

        # 打印 raw 统计
        print(f"\n  Raw Blocks: {len(raw_blocks)}")
        total_raw += len(raw_blocks)

        # 保存原始结果
        raw_path = dump_dir / f"page_{idx + 1:04d}_raw.json"
        raw_data = [
            {
                "id": b.id,
                "bbox": list(b.bbox),
                "text": b.text,
                "confidence": b.confidence,
                "language": b.language,
            }
            for b in raw_blocks
        ]
        raw_path.write_text(json.dumps(raw_data, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"  Saved: {raw_path}")

        # 打印前 20 条 raw OCR 文本
        if raw_blocks:
            print()
            for i, b in enumerate(raw_blocks[:20]):
                x1, y1, x2, y2 = b.bbox
                print(
                    f"  [{i + 1}] "
                    f"({int(x1)},{int(y1)})-({int(x2)},{int(y2)}) "
                    f"conf={b.confidence:.2f}  "
                    f"{b.text[:80]}"
                )
            if len(raw_blocks) > 20:
                print(f"  ... (还有 {len(raw_blocks) - 20} 块)")

        # ----- Step 2: Normalize -----
        if normalize:
            try:
                normalized = normalize.process(raw_blocks, page=idx + 1)
            except Exception as e:
                print(f"\n  Normalize 失败: {e}")
                normalized = []

            print(f"\n  Normalized Blocks: {len(normalized)}")
            total_normalized += len(normalized)

            # 保存标准化结果
            norm_path = dump_dir / f"page_{idx + 1:04d}_normalized.json"
            norm_data = [
                {
                    "id": b.id,
                    "bbox": list(b.bbox),
                    "text": b.text,
                    "confidence": b.confidence,
                    "reading_order": b.reading_order,
                    "is_merged": b.is_merged,
                    "merged_from_ids": b.merged_from_ids,
                }
                for b in normalized
            ]
            norm_path.write_text(json.dumps(norm_data, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  Saved: {norm_path}")

            # 打印前 20 条标准化文本
            if normalized:
                print()
                for i, b in enumerate(normalized[:20]):
                    x1, y1, x2, y2 = b.bbox
                    merge_tag = " [merged]" if b.is_merged else ""
                    print(
                        f"  [{b.reading_order}] "
                        f"({int(x1)},{int(y1)})-({int(x2)},{int(y2)}) "
                        f"conf={b.confidence:.2f}{merge_tag}  "
                        f"{b.text[:80]}"
                    )
                if len(normalized) > 20:
                    print(f"  ... (还有 {len(normalized) - 20} 块)")

        print()

    # --- 汇总 ---
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  Pages processed:    {len(images)}")
    print(f"  Total raw blocks:   {total_raw}")
    if normalize:
        print(f"  Total normalized:   {total_normalized}")
        if total_raw > 0:
            ratio = (1 - total_normalized / total_raw) * 100
            print(f"  Merge reduction:    {ratio:.1f}%")
    print(f"  Raw dumps:          {dump_dir}/")
    print()

    if total_raw == 0:
        print("(warn) 没有检测到任何文本块。")
        print("  可能原因:")
        print("  - OCR 推理失败")
        print("  - 图片中没有文字")
        print("  - 图片格式不兼容")
    else:
        print("[OK] OCR 完成")


if __name__ == "__main__":
    main()