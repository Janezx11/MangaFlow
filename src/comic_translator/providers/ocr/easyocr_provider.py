from __future__ import annotations

from typing import Any

from loguru import logger

from comic_translator.providers.ocr.base import OCRProvider

if False:
    from PIL import Image

    from comic_translator.models.ocr_block import OCRBlock


class EasyOCRProvider(OCRProvider):
    """EasyOCR 引擎 —— PaddleOCR 的降级方案

    EasyOCR 在 Windows + Python 3.13 上兼容性更好。
    当 PaddleOCR 推理失败时自动切换到此 Provider。
    接口与 OCRProvider 完全一致，无需修改任何调用方代码。
    """

    def __init__(self, lang: str = "en", use_gpu: bool = False) -> None:
        self.lang = self._normalize_lang(lang)
        self.use_gpu = use_gpu
        self._reader = None

    def _normalize_lang(self, lang: str) -> str:
        """EasyOCR 和 PaddleOCR 语言代码不完全一致"""
        mapping = {
            "en": "en",
            "ja": "ja",
            "ko": "ko",
            "zh": "ch_sim",
            "chinese": "ch_sim",
            "ch_tra": "ch_tra",
        }
        return mapping.get(lang, lang)

    def _ensure_initialized(self) -> None:
        if self._reader is None:
            import easyocr  # type: ignore[import-untyped]

            logger.info(f"EasyOCR: initializing reader (lang={self.lang})")
            self._reader = easyocr.Reader(
                [self.lang],
                gpu=self.use_gpu,
            )

    async def detect_text(self, image: Image.Image) -> list[OCRBlock]:
        import asyncio

        self._ensure_initialized()
        try:
            raw_results = await asyncio.to_thread(self._run_ocr, image)
            return self._convert_results(raw_results, page=0)
        except Exception as e:
            logger.error(f"EasyOCR inference failed: {e}")
            return []

    def _run_ocr(self, image: Image.Image) -> list[Any]:
        import numpy as np

        img_array = np.array(image.convert("RGB"))
        return self._reader.readtext(img_array)

    def _convert_results(self, raw_results: list[Any], page: int) -> list[OCRBlock]:
        """EasyOCR 结果格式:
        [
            ([[x1,y1],[x2,y2],[x3,y3],[x4,y4]], text, confidence),
            ...
        ]
        """
        from comic_translator.models.ocr_block import OCRBlock

        blocks: list[OCRBlock] = []
        for idx, result in enumerate(raw_results):
            if not result or len(result) < 3:
                continue

            bbox_raw, text, confidence = result

            xs = [p[0] for p in bbox_raw]
            ys = [p[1] for p in bbox_raw]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            block = OCRBlock(
                id=f"easyocr_{page:06d}_{idx:04d}",
                page=page,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                text=str(text),
                confidence=float(confidence),
                language=self.lang,
                metadata={
                    # 保留 4 点多边形，供 BubbleAwareOCRService 做精确归属
                    "polygon": [tuple(float(v) for v in p) for p in bbox_raw],
                },
            )
            blocks.append(block)

        logger.debug(f"EasyOCR: {len(blocks)} blocks detected")
        return blocks
