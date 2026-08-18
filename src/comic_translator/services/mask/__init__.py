"""Mask Pipeline exports"""

from __future__ import annotations

from comic_translator.services.mask.base_mask_provider import MaskProvider
from comic_translator.services.mask.ocr_mask_provider import OCRMaskProvider

__all__ = ["MaskProvider", "OCRMaskProvider"]


def create_mask_provider(config: dict | None = None) -> MaskProvider:
    """根据配置创建 MaskProvider"""
    cfg = config or {}
    provider = cfg.get("provider", "ocr")
    if provider == "ocr":
        return OCRMaskProvider(
            padding=cfg.get("padding", 4),
            text_ratio=cfg.get("text_ratio", 0.50),
            smooth_kernel=cfg.get("smooth_kernel", 5),
        )
    # 可扩展: bubble_polygon provider
    return OCRMaskProvider()
