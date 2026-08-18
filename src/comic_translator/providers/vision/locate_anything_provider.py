"""LocateAnything-3B Vision Provider

NVIDIA LocateAnything-3B: Vision-Language Grounding Model
- HuggingFace: nvidia/LocateAnything-3B
- 架构: MoonViT-SO-400M + Qwen2.5-3B-Instruct + MLP projector
- 输入: Image + 自然语言 prompt
- 输出: Bounding boxes [x1,y1,x2,y2]，归一化到 0-1000
- 许可证: NVIDIA Non-Commercial License（仅限研究/个人用途）

GPU 需求:
  - BF16: ~12GB VRAM
  - 4-bit 量化: ~4-6GB VRAM（需要 bitsandbytes）
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from comic_translator.models.detection_region import DetectionRegion
from comic_translator.providers.vision.base_vision_provider import VisionDetectionProvider

MODEL_ID = "nvidia/LocateAnything-3B"

# 检测 prompt → label 映射
PROMPT_LABEL_MAP = {
    "speech bubble": "speech",
    "narration box or caption box": "narration",
    "title text or heading text": "title",
    "sound effect text or onomatopoeia": "sfx",
}

# NMS IoU 阈值（同一 prompt 内去重）
NMS_IOU_THRESHOLD = 0.5


class LocateAnythingProvider(VisionDetectionProvider):
    """LocateAnything-3B 视觉语言检测 Provider

    使用自然语言 prompt 检测漫画中的文本区域和气泡。
    需要 NVIDIA GPU + transformers + bitsandbytes。
    """

    def __init__(
        self,
        model_name: str = MODEL_ID,
        confidence: float = 0.3,
        prompts: list[str] | None = None,
        cache_dir: str | None = None,
    ):
        self._model_name = model_name
        self._confidence = confidence
        self._prompts = prompts or list(PROMPT_LABEL_MAP.keys())
        self._cache_dir = cache_dir

        self._model = None
        self._processor = None
        self._available: bool | None = None

    def is_available(self) -> bool:
        if self._available is not None:
            return self._available

        try:
            import torch
            if not torch.cuda.is_available():
                logger.info("LocateAnything3B: CUDA not available")
                self._available = False
                return False

            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            if vram_gb < 4.0:
                logger.info(f"LocateAnything3B: VRAM too low ({vram_gb:.1f}GB < 4GB)")
                self._available = False
                return False

            import transformers  # noqa: F401
            self._available = True
        except ImportError:
            logger.info("LocateAnything3B: transformers/torch not installed")
            self._available = False

        return self._available

    def get_model_name(self) -> str:
        return f"LocateAnything-3B ({self._model_name})"

    async def detect(self, image: Image.Image) -> list[DetectionRegion]:
        if not self.is_available():
            logger.warning("LocateAnything3B: not available, returning empty")
            return []

        self._ensure_model()

        try:
            return await asyncio.to_thread(self._run_detection, image)
        except Exception as e:
            logger.error(f"LocateAnything3B detection failed: {e}")
            return []

    def _ensure_model(self):
        if self._model is not None:
            return

        import torch
        from transformers import AutoModel, AutoProcessor

        vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
        load_kwargs = self._get_load_config(vram_gb)

        logger.info(
            f"LocateAnything3B: loading {self._model_name} "
            f"(VRAM={vram_gb:.1f}GB, config={load_kwargs.get('load_in_4bit', 'bf16')})"
        )

        self._processor = AutoProcessor.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            cache_dir=self._cache_dir,
        )
        self._model = AutoModel.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            cache_dir=self._cache_dir,
            **load_kwargs,
        )
        self._model.eval()

    @staticmethod
    def _get_load_config(vram_gb: float) -> dict:
        import torch

        if vram_gb >= 12:
            return {"torch_dtype": torch.bfloat16, "device_map": "auto"}
        elif vram_gb >= 4:
            return {
                "load_in_4bit": True,
                "device_map": "auto",
                "torch_dtype": torch.float16,
            }
        return {}

    def _run_detection(self, image: Image.Image) -> list[DetectionRegion]:
        """对每个 prompt 执行推理，合并结果"""
        img = image.convert("RGB")
        orig_w, orig_h = img.size
        all_regions: list[DetectionRegion] = []

        for prompt in self._prompts:
            label = PROMPT_LABEL_MAP.get(prompt, "unknown")
            boxes = self._infer_single(img, prompt)

            for i, (box, score) in enumerate(boxes):
                if score < self._confidence:
                    continue

                # 从 0-1000 归一化转换为像素坐标
                x1 = int(box[0] / 1000.0 * orig_w)
                y1 = int(box[1] / 1000.0 * orig_h)
                x2 = int(box[2] / 1000.0 * orig_w)
                y2 = int(box[3] / 1000.0 * orig_h)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_w, x2), min(orig_h, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                all_regions.append(DetectionRegion(
                    id=f"locate_{label}_{i:03d}",
                    bbox=(x1, y1, x2, y2),
                    label=label,
                    confidence=round(score, 4),
                    source="locate_anything3b",
                ))

        # NMS 去重
        all_regions = self._nms(all_regions)
        all_regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))

        # 重排 id
        for i, r in enumerate(all_regions):
            r.id = f"locate_{i:04d}"

        logger.info(f"LocateAnything3B: {len(all_regions)} regions detected")
        return all_regions

    def _infer_single(
        self, image: Image.Image, prompt: str
    ) -> list[tuple[list[float], float]]:
        """对单个 prompt 执行推理，返回 (box_0_1000, score) 列表"""
        import torch

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": f"Detect all {prompt} in this comic image."},
                ],
            }
        ]

        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self._processor(
            text=[text], images=[image], return_tensors="pt",
        ).to(self._model.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=4096,
                do_sample=False,
            )

        output_text = self._processor.batch_decode(
            output_ids[:, inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )[0]

        return self._parse_boxes(output_text)

    @staticmethod
    def _parse_boxes(text: str) -> list[tuple[list[float], float]]:
        """从模型输出文本中解析 bounding boxes

        预期格式: [x1, y1, x2, y2] 或 JSON 格式
        """
        results = []

        # 尝试 JSON 解析
        try:
            import json
            data = json.loads(text)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict) and "bbox" in item:
                        box = item["bbox"]
                        score = item.get("score", item.get("confidence", 1.0))
                        results.append((box, float(score)))
                    elif isinstance(item, (list, tuple)) and len(item) >= 4:
                        results.append((list(item[:4]), 1.0))
                if results:
                    return results
        except (json.JSONDecodeError, TypeError):
            pass

        # 正则匹配 [x1, y1, x2, y2] 格式
        pattern = r"\[?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\]?"
        for match in re.finditer(pattern, text):
            box = [float(match.group(i)) for i in range(1, 5)]
            results.append((box, 1.0))

        return results

    @staticmethod
    def _nms(regions: list[DetectionRegion]) -> list[DetectionRegion]:
        """简单 NMS 去重"""
        if len(regions) <= 1:
            return regions

        regions = sorted(regions, key=lambda r: r.confidence, reverse=True)
        keep: list[DetectionRegion] = []

        for r in regions:
            should_keep = True
            for k in keep:
                if _bbox_iou(r.bbox, k.bbox) > NMS_IOU_THRESHOLD:
                    should_keep = False
                    break
            if should_keep:
                keep.append(r)

        return keep


def _bbox_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0
