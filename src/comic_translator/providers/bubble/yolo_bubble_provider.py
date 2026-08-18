from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
from loguru import logger

from comic_translator.models.bubble import Bubble
from comic_translator.providers.bubble.base import BubbleProvider

if TYPE_CHECKING:
    from PIL import Image


class YOLO11BubbleProvider(BubbleProvider):
    """YOLO11-seg 气泡检测 Provider

    加载外部 .pt 模型文件，运行 segmentation 推理。
    输出 mask / bbox / class_name / confidence。
    模型路径配置化，不硬编码。

    模型来源（任选一）：
      1. 通用预训练 YOLO11-seg (yolo11n-seg.pt) → 需 fine-tune
      2. 漫画气泡专用模型 (models/bubble/yolo11n-seg.pt)

    用法:
        provider = YOLO11BubbleProvider("models/bubble/yolo11n-seg.pt")
        bubbles = await provider.detect(image)
    """

    # YOLO class id → 气泡类型
    # COCO 默认: 0=person, 41=cup, etc.
    # Fine-tune 后的模型使用自定义 class mapping
    CLASS_MAP: dict[int, str] = {
        0: "speech",
        1: "narration",
        2: "title",
        3: "sfx",
    }

    def __init__(
        self,
        model_path: str = "",
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.45,
        device: str = "",
    ) -> None:
        self._model_path = model_path or ""
        self._conf = conf_threshold
        self._iou = iou_threshold
        self._device = device or ("cuda" if self._has_cuda() else "cpu")
        self._model = None  # lazy init

    def _has_cuda(self) -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    def _ensure_model(self):
        if self._model is not None:
            return
        model_spec = self._model_path if self._model_path else "models/bubble/yolo11n-seg-mixed-bubble.pt"
        try:
            from ultralytics import YOLO
            self._model = YOLO(model_spec)
            logger.info(f"YOLO model loaded: {model_spec} (device={self._device})")
        except ImportError:
            raise ImportError(
                "ultralytics not installed. Run: pip install ultralytics"
            )

    async def detect(self, image: Image.Image) -> list[Bubble]:
        """检测气泡（异步包装，推理在后台线程执行，避免阻塞 event loop）"""
        import asyncio
        return await asyncio.to_thread(self._run_detection, image)

    def _run_detection(self, image: Image.Image) -> list[Bubble]:
        """同步气泡检测推理（在 asyncio.to_thread 中运行）"""
        self._ensure_model()
        import numpy as np

        img_np = np.array(image.convert("RGB"))
        results = self._model(
            img_np,
            conf=self._conf,
            iou=self._iou,
            device=self._device,
            verbose=False,
        )

        bubbles: list[Bubble] = []
        if not results or not results[0].masks:
            logger.warning("YOLO: no masks detected")
            return bubbles

        result = results[0]
        for i in range(len(result.masks.data)):
            cls_id = int(result.boxes.cls[i].item()) if result.boxes is not None else 0
            conf = float(result.boxes.conf[i].item()) if result.boxes is not None else 0
            class_name = self.CLASS_MAP.get(cls_id, "speech")

            # bbox
            x1, y1, x2, y2 = map(int, result.boxes.xyxy[i].tolist()) if result.boxes is not None else (0, 0, 0, 0)

            # mask: resize from YOLO resolution to original image size
            # YOLO 的 mask 在 (640, 640) 分辨率，需要缩放到原图尺寸
            mask_np = result.masks.data[i].cpu().numpy()
            h_orig, w_orig = img_np.shape[:2]
            if mask_np.ndim == 2:
                mask_resized = cv2.resize(mask_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            else:
                mask_resized = mask_np

            # polygon: 从缩放到原图后的 mask 提取，确保坐标在原图空间
            contours, _ = cv2.findContours(
                (mask_resized > 0.5).astype(np.uint8),
                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
            )
            polygon = []
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                polygon = [(int(pt[0][0]), int(pt[0][1])) for pt in cnt]

            mask = mask_resized.tolist()

            bubbles.append(Bubble(
                id=f"bubble_yolo_{i + 1:04d}",
                bbox=(x1, y1, x2, y2),
                polygon=polygon,
                mask=mask,
                class_name=class_name,
                confidence=conf,
            ))

        bubbles.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        logger.info(f"YOLO: {len(bubbles)} bubbles detected")
        return bubbles

    def get_model_name(self) -> str:
        return f"YOLO11-seg ({Path(self._model_path).name})"