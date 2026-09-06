"""Stable Diffusion Inpaint Provider — 预留接口

如果 diffusers/transformers/torch 未安装或模型路径不存在，
返回 Unavailable。
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from comic_translator.providers.inpaint.base_inpaint_provider import InpaintProvider


class SDInpaintProvider(InpaintProvider):
    """Stable Diffusion Inpaint 预留接口"""

    def __init__(
        self,
        model_path: str = "models/sd-inpaint",
        device: str = "auto",
        strength: float = 0.75,
    ):
        self.model_path = Path(model_path)
        self.device = device
        self.strength = strength
        self._pipe = None

    def get_name(self) -> str:
        return "sd_inpaint"

    def is_available(self) -> bool:
        try:
            import importlib
            importlib.import_module("diffusers")
            importlib.import_module("torch")
            return self.model_path.exists()
        except Exception:
            return False

    def _load_pipeline(self):
        if self._pipe is not None:
            return self._pipe
        try:
            import torch
            from diffusers import StableDiffusionInpaintPipeline

            device = self.device
            if device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"

            self._pipe = StableDiffusionInpaintPipeline.from_pretrained(
                str(self.model_path),
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            ).to(device)
            logger.info(f"SD inpaint pipeline loaded on {device}")
        except Exception as e:
            logger.error(f"Failed to load SD inpaint: {e}")
            raise
        return self._pipe

    def inpaint(
        self,
        image: Image.Image | np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError("SD inpaint not available")

        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = image.convert("RGB")

        # resize to SD standard size (multiples of 8/64)
        w, h = pil_img.size
        new_w = (w // 8) * 8
        new_h = (h // 8) * 8
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        mask_pil = Image.fromarray((mask > 0).astype(np.uint8) * 255).resize(
            (new_w, new_h), Image.NEAREST,
        )

        pipe = self._load_pipeline()
        result = pipe(
            prompt="comic book speech bubble, clean background, no text, high quality",
            image=pil_img,
            mask_image=mask_pil,
            strength=self.strength,
            num_inference_steps=20,
        ).images[0]

        if result.size != (w, h):
            result = result.resize((w, h), Image.LANCZOS)
        return cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)


import cv2
