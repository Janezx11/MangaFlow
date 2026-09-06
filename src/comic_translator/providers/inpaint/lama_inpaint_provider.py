"""LaMa Inpaint Provider — 本地模型修复

依赖: simple-lama-inpainting（pip install simple-lama-inpainting）
如果不可用则返回 is_available=False

注意：LaMa 要求输入图片尺寸能被 8 整除（模型内部下采样倍数）。
如果图片尺寸不是 8 的倍数，会自动垫到下一个 8 的倍数，推理完再裁回原尺寸。
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from comic_translator.providers.inpaint.base_inpaint_provider import InpaintProvider

# LaMa 下采样倍率（模型内部有 3 层下采样，2^3 = 8）
LAMA_DOWNSCALE_FACTOR = 8


def _pad_to_multiple(
    img: Image.Image,
    mask: Image.Image,
    factor: int = LAMA_DOWNSCALE_FACTOR,
) -> tuple[Image.Image, Image.Image, tuple[int, int]]:
    """将图片和 mask 同时垫到 factor 的倍数

    Returns:
        (padded_img, padded_mask, original_size)
        original_size = (w, h) 用于后续裁切
    """
    w, h = img.size
    pad_w = (factor - w % factor) % factor
    pad_h = (factor - h % factor) % factor

    if pad_w == 0 and pad_h == 0:
        return img, mask, (w, h)

    # 右/下 padding（用 0 填充，mask 的 0 表示不擦除，不影响结果）
    from PIL import ImageOps
    padded_img = ImageOps.expand(img, border=(0, 0, pad_w, pad_h), fill=0)
    padded_mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=0)

    logger.debug(
        f"LaMa padding: ({w},{h}) -> ({w+pad_w},{h+pad_h}) "
        f"(+{pad_w}x{pad_h})"
    )
    return padded_img, padded_mask, (w, h)


def _crop_to_original(
    img: Image.Image,
    original_size: tuple[int, int],
) -> Image.Image:
    """把垫过的图片裁回原尺寸"""
    w, h = original_size
    return img.crop((0, 0, w, h))


class LaMaInpaintProvider(InpaintProvider):
    """LaMa 模型修复"""

    def __init__(self, model_dir: str = "models/lama", device: str = "auto"):
        self.model_dir = Path(model_dir)
        self.device = device
        self._model = None

    def get_name(self) -> str:
        return "lama"

    def is_available(self) -> bool:
        """检查 simple-lama-inpainting 是否已安装"""
        try:
            import simple_lama_inpainting  # noqa: F401
            return True
        except Exception:
            return False

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            import torch
            from simple_lama_inpainting import SimpleLama

            if self.device == "auto":
                device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                device = self.device

            # 兼容两种目录布局：
            #   model_dir 本身就是 big-lama 权重目录（config.yaml 的 lama_model_path
            #   通常指向 models/lama/big-lama）
            #   model_dir 的下一级 big-lama/ 子目录
            # 都找不到时才回退 SimpleLama 默认（自动下载）。
            candidates = [self.model_dir, self.model_dir / "big-lama"]
            self._model = None
            for cand in candidates:
                if not (cand.exists() and cand.is_dir() and any(cand.iterdir())):
                    continue
                try:
                    self._model = SimpleLama(model_dir=str(cand), device=device)
                    logger.info(f"LaMa model loaded from {cand} on {device}")
                    break
                except Exception as load_err:
                    logger.warning(f"Failed to load LaMa from {cand}: {load_err}")
            if self._model is None:
                self._model = SimpleLama(device=device)
                logger.info(f"LaMa model loaded (default download) on {device}")
        except Exception as e:
            logger.error(f"Failed to load LaMa: {e}")
            raise
        return self._model

    def inpaint(
        self,
        image: Image.Image | np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        if not self.is_available():
            raise RuntimeError("LaMa not available. Install: pip install simple-lama-inpainting")

        t0 = time.time()
        if isinstance(image, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_img = image.convert("RGB")

        mask_pil = Image.fromarray((mask > 0).astype(np.uint8) * 255)

        # 确保图片+mask 尺寸能被 8 整除（LaMa 模型要求）
        pil_img, mask_pil, original_size = _pad_to_multiple(pil_img, mask_pil)

        model = self._load_model()
        try:
            result_pil = model(pil_img, mask_pil)
        except Exception as e:
            # 如果还是失败，给一个更清晰的错误信息
            logger.error(
                f"LaMa inference failed (image size {pil_img.size}, "
                f"mask size {mask_pil.size}): {e}"
            )
            raise

        # 裁回原尺寸
        result_pil = _crop_to_original(result_pil, original_size)

        result = cv2.cvtColor(np.array(result_pil), cv2.COLOR_RGB2BGR)
        logger.info(f"LaMa inpaint done in {time.time()-t0:.2f}s")
        return result
