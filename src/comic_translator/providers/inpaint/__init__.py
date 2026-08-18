"""Inpaint Provider exports"""

from __future__ import annotations

from comic_translator.providers.inpaint.base_inpaint_provider import InpaintProvider
from comic_translator.providers.inpaint.opencv_inpaint_provider import OpenCVInpaintProvider
from comic_translator.providers.inpaint.lama_inpaint_provider import LaMaInpaintProvider
from comic_translator.providers.inpaint.sd_inpaint_provider import SDInpaintProvider

__all__ = [
    "InpaintProvider",
    "OpenCVInpaintProvider",
    "LaMaInpaintProvider",
    "SDInpaintProvider",
]


def create_inpaint_provider(config: dict | None = None) -> InpaintProvider:
    """根据配置创建 InpaintProvider"""
    cfg = config or {}
    provider = cfg.get("provider", "opencv")
    if provider == "lama":
        return LaMaInpaintProvider(
            model_dir=cfg.get("lama_model_path", "models/lama"),
            device=cfg.get("device", "auto"),
        )
    if provider == "sd":
        return SDInpaintProvider(
            model_path=cfg.get("sd_model_path", "models/sd-inpaint"),
            device=cfg.get("device", "auto"),
        )
    return OpenCVInpaintProvider(
        method=cfg.get("opencv_method", "telea"),
        radius=cfg.get("opencv_radius", 5),
    )
