from __future__ import annotations

import sys
from typing import Any

from loguru import logger


class OCRHealthCheckResult:
    """OCR Provider 健康检查结果"""

    def __init__(
        self,
        provider: str,
        import_ok: bool = False,
        init_ok: bool = False,
        inference_ok: bool = False,
        error: str = "",
    ) -> None:
        self.provider = provider
        self.import_ok = import_ok
        self.init_ok = init_ok
        self.inference_ok = inference_ok
        self.error = error

    @property
    def all_ok(self) -> bool:
        return self.import_ok and self.init_ok and self.inference_ok

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "import_ok": self.import_ok,
            "init_ok": self.init_ok,
            "inference_ok": self.inference_ok,
            "error": self.error,
        }

    def __str__(self) -> str:
        status = "✓" if self.all_ok else "✗"
        return (
            f"[{status}] {self.provider}\n"
            f"  Import:    {'OK' if self.import_ok else 'FAILED'}\n"
            f"  Init:      {'OK' if self.init_ok else 'FAILED'}\n"
            f"  Inference: {'OK' if self.inference_ok else 'FAILED'}\n"
            f"  Error:     {self.error}"
        )


class OCRHealthChecker:
    """OCR Provider 健康检查器

    检查指定 OCR Provider 的三个阶段:
    1. Import — 能否导入
    2. Init — 能否初始化
    3. Inference — 能否推理
    """

    @staticmethod
    def check_paddle(lang: str = "en", use_gpu: bool = False) -> OCRHealthCheckResult:
        result = OCRHealthCheckResult(provider="PaddleOCR")

        # 1. Import
        try:
            from paddleocr import PaddleOCR
            import paddle

            result.import_ok = True
        except ImportError as e:
            result.error = f"ImportError: {e}"
            return result
        except Exception as e:
            result.error = f"Import failed: {e}"
            return result

        # 2. Init
        try:
            ocr = PaddleOCR(lang=lang, use_gpu=use_gpu, show_log=False)
            result.init_ok = True
        except Exception as e:
            result.error = f"Init failed: {e}"
            return result

        # 3. Inference
        try:
            import numpy as np
            from PIL import Image

            img = Image.new("RGB", (100, 50), color="white")
            from PIL import ImageDraw

            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "Test", fill="black")

            img_array = np.array(img.convert("RGB"))
            raw = ocr.ocr(img_array)
            result.inference_ok = True
        except Exception as e:
            result.error = f"Inference failed: {e}"

        return result

    @staticmethod
    def check_easyocr(lang: str = "en", use_gpu: bool = False) -> OCRHealthCheckResult:
        result = OCRHealthCheckResult(provider="EasyOCR")

        # 1. Import
        try:
            import easyocr  # type: ignore[import-untyped]
            result.import_ok = True
        except ImportError as e:
            result.error = f"ImportError: {e}"
            return result
        except Exception as e:
            result.error = f"Import failed: {e}"
            return result

        # 2. Init
        try:
            lang_code = "ch_sim" if lang == "zh" else lang
            reader = easyocr.Reader([lang_code], gpu=use_gpu)
            result.init_ok = True
        except Exception as e:
            result.error = f"Init failed: {e}"
            return result

        # 3. Inference
        try:
            import numpy as np
            from PIL import Image, ImageDraw

            img = Image.new("RGB", (100, 50), color="white")
            draw = ImageDraw.Draw(img)
            draw.text((10, 10), "Test", fill="black")

            img_array = np.array(img.convert("RGB"))
            raw = reader.readtext(img_array)
            result.inference_ok = True
        except Exception as e:
            result.error = f"Inference failed: {e}"

        return result

    @staticmethod
    def check_all(lang: str = "en") -> list[OCRHealthCheckResult]:
        """检查所有可用的 OCR Provider"""
        results: list[OCRHealthCheckResult] = []

        paddle_result = OCRHealthChecker.check_paddle(lang=lang)
        results.append(paddle_result)

        easyocr_result = OCRHealthChecker.check_easyocr(lang=lang)
        results.append(easyocr_result)

        return results

    @staticmethod
    def get_best_provider(lang: str = "en") -> tuple[str, Any]:
        """返回 (provider_name, provider_class_or_none)"""
        from comic_translator.providers.ocr.paddle_ocr_provider import (
            PaddleOCRProvider,
        )

        paddle_result = OCRHealthChecker.check_paddle(lang=lang)

        if paddle_result.all_ok:
            logger.info("OCR: using PaddleOCR (primary)")
            return ("paddle", PaddleOCRProvider)

        # Try EasyOCR as fallback
        logger.warning(f"PaddleOCR health check failed: {paddle_result.error}")
        easyocr_result = OCRHealthChecker.check_easyocr(lang=lang)

        if easyocr_result.all_ok:
            from comic_translator.providers.ocr.easyocr_provider import (
                EasyOCRProvider,
            )

            logger.info(f"OCR: PaddleOCR failed, falling back to EasyOCR")
            return ("easyocr", EasyOCRProvider)

        logger.error("No OCR provider available. Install paddleocr or easyocr.")
        return ("none", None)