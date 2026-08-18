"""Detection Overlay — 在图片上绘制检测区域"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from comic_translator.models.detection_region import DetectionRegion

SOURCE_COLORS = {
    "ctbd": (0, 255, 0),               # green
    "yolo11": (255, 128, 0),           # blue
    "locate_anything3b": (0, 128, 255), # orange
    "fused": (255, 0, 255),            # magenta
}

LABEL_COLORS = {
    "speech": (0, 255, 0),
    "narration": (0, 128, 255),
    "title": (255, 0, 0),
    "sfx": (255, 255, 0),
    "text": (0, 255, 0),
    "unknown": (128, 128, 128),
}


def draw_detection_overlay(
    image: Image.Image,
    regions: list[DetectionRegion],
    color_by: str = "source",
) -> np.ndarray:
    """在图片上绘制检测区域 overlay

    Args:
        image: PIL Image
        regions: DetectionRegion 列表
        color_by: "source" 或 "label"

    Returns:
        BGR numpy array（可直接用于 cv2 或转 PIL）
    """
    img_np = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    colors = SOURCE_COLORS if color_by == "source" else LABEL_COLORS

    for r in regions:
        key = r.source if color_by == "source" else r.label
        # 融合区域用 "+" 连接多个 source
        if "+" in key:
            color = SOURCE_COLORS.get("fused", (255, 0, 255))
        else:
            color = colors.get(key, (128, 128, 128))

        x1, y1, x2, y2 = r.bbox
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)

        label = f"{r.label} {r.confidence:.2f}"
        if color_by == "source":
            label = f"[{r.source}] {label}"
        cv2.putText(img_bgr, label, (x1 + 2, y1 + 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        # 绘制 polygon
        if r.polygon:
            pts = np.array(r.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(img_bgr, [pts], True, color, 1)

    return img_bgr


def draw_ocr_overlay(
    image: Image.Image,
    ocr_blocks: list,
) -> np.ndarray:
    """在图片上绘制 OCR 结果 overlay"""
    img_np = np.array(image.convert("RGB"))
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    for block in ocr_blocks:
        x1, y1, x2, y2 = map(int, block.bbox)
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 255), 1)

        text = block.text[:30] if block.text else ""
        conf = f"{block.confidence:.2f}" if hasattr(block, 'confidence') else ""
        label = f"{text} ({conf})"
        cv2.putText(img_bgr, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 255, 255), 1)

    return img_bgr


def bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    """BGR numpy → PIL Image"""
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)
