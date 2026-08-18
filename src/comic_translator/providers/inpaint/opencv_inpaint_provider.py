"""OpenCV Inpaint Provider — 保留现有 baseline 方案"""

from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from comic_translator.providers.inpaint.base_inpaint_provider import InpaintProvider


class OpenCVInpaintProvider(InpaintProvider):
    """OpenCV Telea/NS inpaint，作为 baseline"""

    def __init__(self, method: str = "telea", radius: int = 5):
        self.method = method
        self.radius = radius
        self._flags = {
            "telea": cv2.INPAINT_TELEA,
            "ns": cv2.INPAINT_NS,
        }

    def get_name(self) -> str:
        return f"opencv_{self.method}"

    def is_available(self) -> bool:
        return True

    def inpaint(
        self,
        image: Image.Image | np.ndarray,
        mask: np.ndarray,
    ) -> np.ndarray:
        img_bgr = self._to_bgr(image)
        mask_uint8 = (mask > 0).astype(np.uint8) * 255

        flag = self._flags.get(self.method, cv2.INPAINT_TELEA)
        try:
            result = cv2.inpaint(img_bgr, mask_uint8, inpaintRadius=self.radius, flags=flag)
        except Exception:
            result = cv2.inpaint(img_bgr, mask_uint8, inpaintRadius=self.radius,
                                 flags=cv2.INPAINT_NS)
        return result
