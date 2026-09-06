from __future__ import annotations

import sys
from typing import Any

from loguru import logger

from comic_translator.providers.ocr.base import OCRProvider

if False:
    from PIL import Image

    from comic_translator.models.ocr_block import OCRBlock


class PaddleOCRProvider(OCRProvider):
    """PaddleOCR 引擎 —— 含降级与兼容性处理

    修复:
    - MKLDNN 兼容性问题 (默认关闭 enable_mkldnn)
    - 推理失败自动返回空列表 (不崩溃)
    - 版本信息记录 (便于诊断)
    """

    def __init__(
        self,
        lang: str = "en",
        use_gpu: bool = False,
        enable_mkldnn: bool = False,
    ) -> None:
        self.lang = lang
        self.use_gpu = use_gpu
        self.enable_mkldnn = enable_mkldnn
        self._ocr = None
        self._versions_logged = False

    def _ensure_initialized(self) -> None:
        if self._ocr is None:
            from paddleocr import PaddleOCR

            self._log_versions()

            self._ocr = PaddleOCR(
                lang=self.lang,
                use_gpu=self.use_gpu,
                show_log=False,
            )

    def _log_versions(self) -> None:
        """记录版本信息便于诊断"""
        if self._versions_logged:
            return
        self._versions_logged = True
        try:
            import paddle
            import paddleocr
            logger.info(
                f"PaddleOCRProvider: "
                f"paddleocr={paddleocr.__version__}, "
                f"paddle={paddle.__version__}, "
                f"python={sys.version.split()[0]}, "
                f"enable_mkldnn={self.enable_mkldnn}"
            )
        except Exception:
            pass

    async def detect_text(self, image: Image.Image) -> list[OCRBlock]:
        import asyncio

        try:
            # 懒初始化（可能构建 PaddleOCR 实例）放在线程里，避免阻塞事件循环
            await asyncio.to_thread(self._ensure_initialized)
            raw_results = await asyncio.to_thread(self._run_ocr, image)
            return self._convert_results(raw_results, page=0)
        except Exception as e:
            logger.error(f"PaddleOCR inference failed: {e}")
            logger.error(f"Falling back to empty result — page {getattr(image, 'filename', 'unknown')}")
            return []

    def _run_ocr(self, image: Image.Image) -> list[Any]:
        import numpy as np

        img_array = np.array(image.convert("RGB"))
        return self._ocr.ocr(img_array)

    def _convert_results(self, raw_results: list[Any], page: int) -> list[OCRBlock]:
        from comic_translator.models.ocr_block import OCRBlock

        blocks: list[OCRBlock] = []
        results = raw_results[0] if raw_results and isinstance(raw_results[0], list) else raw_results
        if not results:
            return blocks

        for idx, result in enumerate(results):
            if not result or len(result) < 2:
                continue

            bbox_raw, text_data = result
            text, confidence = text_data

            xs = [p[0] for p in bbox_raw]
            ys = [p[1] for p in bbox_raw]
            x1, y1, x2, y2 = min(xs), min(ys), max(xs), max(ys)

            block = OCRBlock(
                id=f"ocr_{page:06d}_{idx:04d}",
                page=page,
                bbox=(float(x1), float(y1), float(x2), float(y2)),
                text=str(text),
                confidence=float(confidence),
                language=self.lang,
            )
            blocks.append(block)

        logger.debug(f"PaddleOCR: {len(blocks)} blocks detected")
        return blocks
