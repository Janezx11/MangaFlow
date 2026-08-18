#!/usr/bin/env python
"""
labelme 标注 → YOLO seg 格式 转换工具

用法:
  python scripts/convert_labels.py
  python scripts/convert_labels.py --input datasets/comic_bubbles/labelme_raw --output datasets/comic_bubbles

标注方式:
  1. pip install labelme
  2. labelme datasets/comic_bubbles/images/train/
  3. 沿着气泡边缘画 polygon，标签名: speech / narration / title / sfx
  4. 保存后 JSON 文件在同目录下
  5. 运行此脚本转换
"""

import argparse
import json
from pathlib import Path

import numpy as np


# YOLO class id 映射
CLASS_MAP = {
    "speech": 0,
    "narration": 1,
    "title": 2,
    "sfx": 3,
}


def labelme_to_yolo_seg(json_path: Path, output_dir: Path, img_width: int, img_height: int) -> str | None:
    """转换一个 labelme JSON 文件为 YOLO seg .txt 格式

    YOLO seg 格式:
      class_id x1 y1 x2 y2 ... xn yn
    坐标归一化到 [0, 1]
    """
    with open(json_path, encoding="utf-8") as f:
        data = json.load(f)

    # 图片尺寸（优先使用 JSON 中的尺寸）
    if "imageWidth" in data and "imageHeight" in data:
        img_width = data["imageWidth"]
        img_height = data["imageHeight"]

    if img_width <= 0 or img_height <= 0:
        print(f"  WARNING: invalid image size for {json_path.name}")
        return None

    lines = []
    for shape in data.get("shapes", []):
        label = shape["label"].lower()
        if label not in CLASS_MAP:
            print(f"  WARNING: unknown label '{label}' in {json_path.name}, skipping")
            continue

        class_id = CLASS_MAP[label]
        points = shape["points"]

        # 最少 3 个点
        if len(points) < 3:
            continue

        # 归一化坐标
        normalized = []
        for px, py in points:
            nx = px / img_width
            ny = py / img_height
            # 限制在 [0, 1] 范围内
            nx = max(0, min(1, nx))
            ny = max(0, min(1, ny))
            normalized.extend([nx, ny])

        line = f"{class_id} " + " ".join(f"{v:.6f}" for v in normalized)
        lines.append(line)

    if not lines:
        return None

    # 输出 .txt 文件（与图片同名）
    txt_path = output_dir / f"{json_path.stem}.txt"
    txt_path.write_text("\n".join(lines), encoding="utf-8")
    return str(txt_path)


def main():
    parser = argparse.ArgumentParser(description="Convert labelme JSON to YOLO seg format")
    parser.add_argument("--input", default="datasets/comic_bubbles/images",
                        help="labelme JSON 所在目录（默认遍历 images/train 和 images/val）")
    parser.add_argument("--output", default="datasets/comic_bubbles/labels",
                        help="YOLO labels 输出目录")
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)

    # 处理 train/val 子目录
    for subset in ["train", "val"]:
        img_dir = input_dir / subset
        label_dir = output_dir / subset
        label_dir.mkdir(parents=True, exist_ok=True)

        if not img_dir.exists():
            print(f"  {img_dir} 不存在，跳过")
            continue

        json_files = sorted(img_dir.glob("*.json"))
        if not json_files:
            print(f"  {img_dir}/ 没有 JSON 文件")
            continue

        converted = 0
        for jf in json_files:
            # 尝试获取图片尺寸（从同名的图片文件）
            img_w, img_h = 1600, 1200  # 默认值
            for ext in [".png", ".jpg", ".jpeg"]:
                img_path = jf.with_suffix(ext)
                if img_path.exists():
                    from PIL import Image
                    with Image.open(img_path) as img:
                        img_w, img_h = img.size
                    break

            result = labelme_to_yolo_seg(jf, label_dir, img_w, img_h)
            if result:
                converted += 1
                print(f"  OK {jf.name} -> {result}")
            else:
                print(f"  SKIP {jf.name} -> no valid labels")

        print(f"  [{subset}] converted: {converted}/{len(json_files)} files")


if __name__ == "__main__":
    main()