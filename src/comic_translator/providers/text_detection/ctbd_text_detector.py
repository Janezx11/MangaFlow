"""CTBD Text Detector — Comic Text and Bubble Detector (RT-DETR-v2)

使用 ogkalu/comic-text-and-bubble-detector ONNX 模型。
检测文本区域（text_bubble, text_free）和气泡（bubble）。

模型: https://huggingface.co/ogkalu/comic-text-and-bubble-detector
架构: RT-DETR-v2 r50vd
训练数据: manga + webtoons + manhua + western comics (~11k images)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from loguru import logger
from PIL import Image

from comic_translator.models.text_region import TextRegion
from comic_translator.providers.text_detection.base import TextDetectorProvider

# 模型类别映射
CTBD_CLASSES = {
    0: "bubble",
    1: "text_bubble",
    2: "text_free",
}

# TextRegion region_type 映射
# bubble → 过滤掉（由 BubbleProvider 处理）
# text_bubble → TEXT（气泡内文字）
# text_free → TEXT（自由文字：旁白、标题、SFX）
CLASS_TO_REGION_TYPE = {
    "bubble": None,       # 过滤
    "text_bubble": "text",
    "text_free": "text",
}

# 默认置信度阈值
DEFAULT_CONFIDENCE = 0.5

# 默认模型文件（int8 量化版，更快更小）
DEFAULT_MODEL = "detector_int8.onnx"
# 备选：detector.onnx（全精度，更大更慢）


class CTBDTextDetectorProvider(TextDetectorProvider):
    """Comic Text and Bubble Detector (CTBD) Provider

    使用 RT-DETR-v2 ONNX 模型检测文本区域。
    支持 text_bubble（气泡内文字）和 text_free（自由文字）。

    用法:
        provider = CTBDTextDetectorProvider()
        regions = await provider.detect(image)
    """

    def __init__(
        self,
        model_path: str = "",
        confidence: float = DEFAULT_CONFIDENCE,
        device: str = "auto",
    ) -> None:
        self._model_path = model_path
        self._confidence = confidence
        self._device = device
        self._session = None
        self._model_dir: Path | None = None

    def _ensure_model(self) -> None:
        """懒加载 ONNX 模型"""
        if self._session is not None:
            return

        # Windows: onnxruntime-gpu 需要 CUDA 12 + cuDNN 9 的 DLL。
        # 系统未安装 CUDA Toolkit，但 torch cu124 自带全套 CUDA DLL，
        # 把 torch/lib 加入 DLL 搜索路径即可让 CUDAExecutionProvider 正常加载。
        if os.name == "nt":
            try:
                import torch
                torch_lib = Path(torch.__file__).parent / "lib"
                if torch_lib.exists():
                    os.add_dll_directory(str(torch_lib))
            except Exception:
                pass

        import onnxruntime as ort

        model_path = self._model_path
        if not model_path:
            model_path = str(self._download_model())

        providers = ["CPUExecutionProvider"]
        use_cuda = False
        if self._device == "cuda":
            use_cuda = True
        elif self._device == "auto":
            # 自动检测：onnxruntime-gpu 且 CUDA 可用才用 GPU
            try:
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    use_cuda = True
            except Exception:
                use_cuda = False

        if use_cuda:
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        self._session = ort.InferenceSession(model_path, providers=providers)
        actual = self._session.get_providers()
        logger.info(
            f"CTBD model loaded: {Path(model_path).name} "
            f"(device={self._device}, providers={actual}, conf={self._confidence})"
        )

    def _download_model(self) -> Path:
        """从 Hugging Face 下载模型"""
        if self._model_dir is not None:
            return self._model_dir / DEFAULT_MODEL

        from huggingface_hub import snapshot_download

        repo_id = "ogkalu/comic-text-and-bubble-detector"
        self._model_dir = Path(snapshot_download(
            repo_id,
            allow_patterns=[f"*{DEFAULT_MODEL}", "*.json"],
        ))
        return self._model_dir / DEFAULT_MODEL

    async def detect(self, image: Image.Image) -> list[TextRegion]:
        """检测文本区域"""
        self._ensure_model()
        import asyncio

        try:
            return await asyncio.to_thread(self._run_detection, image)
        except Exception as e:
            logger.error(f"CTBD detection failed: {e}")
            return []

    def _run_detection(self, image: Image.Image) -> list[TextRegion]:
        """同步推理（在 asyncio.to_thread 中调用）"""
        img_rgb = image.convert("RGB")
        orig_w, orig_h = img_rgb.size

        # 预处理: resize → rescale → NCHW
        img_resized = img_rgb.resize((640, 640), Image.BILINEAR)
        img_array = np.array(img_resized, dtype=np.float32) / 255.0
        img_nchw = img_array.transpose(2, 0, 1)[np.newaxis, ...]  # (1, 3, 640, 640)
        orig_sizes = np.array([[orig_w, orig_h]], dtype=np.int64)

        # 推理
        outputs = self._session.run(
            None,
            {
                "images": img_nchw,
                "orig_target_sizes": orig_sizes,
            },
        )

        labels, boxes, scores = outputs

        # 后处理
        regions: list[TextRegion] = []
        for i in range(labels.shape[-1]):
            label_id = int(labels.flat[i])
            score = float(scores.flat[i])
            box = boxes.flat[i * 4: (i + 1) * 4]

            class_name = CTBD_CLASSES.get(label_id, "unknown")
            region_type = CLASS_TO_REGION_TYPE.get(class_name)

            # 过滤: bubble 类不输出（由 BubbleProvider 处理）
            if region_type is None:
                continue

            # 过滤低置信度
            if score < self._confidence:
                continue

            x1, y1, x2, y2 = map(int, box)

            # 裁剪到图片范围
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))

            if x2 <= x1 or y2 <= y1:
                continue

            regions.append(TextRegion(
                id=f"ctbd_{i:04d}",
                bbox=(x1, y1, x2, y2),
                polygon=None,
                confidence=round(score, 4),
                region_type=region_type,
                source="ctbd",
            ))

        # 按 Y→X 排序
        regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))

        logger.info(f"CTBD: {len(regions)} text regions detected")
        return regions
