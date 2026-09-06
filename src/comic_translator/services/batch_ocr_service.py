from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from comic_translator.models.batch_result import BatchPage

if TYPE_CHECKING:
    from comic_translator.providers.ocr.base import OCRProvider
    from comic_translator.services.batch_cache import BatchCacheService
    from comic_translator.services.ocr_normalize_service import OCRNormalizeService
    from comic_translator.services.ocr_postprocess_service import OCRPostprocessService


class BatchOCRService:
    """一次性 OCR 全部页面

    流程:
    1. 遍历 data/comics/ 下所有图片（按文件名排序）
    2. 用 OCRProvider 逐页 detect_text
    3. 用 OCRNormalizeService 逐页 Normalize
    4. 用 OCRPostprocessService 后处理（5 级修复）
    5. 输出 list[BatchPage]
    6. 缓存到 BatchCacheService (all_pages_ocr.json)
    """

    def __init__(
        self,
        provider: OCRProvider,
        normalize: OCRNormalizeService,
        cache: BatchCacheService,
        postprocess: OCRPostprocessService | None = None,
    ) -> None:
        self._provider = provider
        self._normalize = normalize
        self._cache = cache
        self._postprocess = postprocess

    def collect_images(self, comic_dir: str = "data/comics") -> list[tuple[int, Path]]:
        """收集并按文件名自然排序图片（"2.jpg" < "10.jpg"）"""
        from comic_translator.utils.natural_sort import natural_key

        path = Path(comic_dir)
        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        images = sorted(
            [f for f in path.iterdir() if f.suffix.lower() in exts],
            key=natural_key,
        )
        return [(i + 1, img) for i, img in enumerate(images)]

    async def process(
        self,
        project_id: str,
        comic_dir: str = "data/comics",
        source_lang: str = "en",
        use_cache: bool = True,
    ) -> list[BatchPage]:
        """OCR 全部页面，返回 BatchPage 列表"""

        # Cache check
        if use_cache:
            cached = self._cache.get_ocr_cache(project_id)
            if cached is not None:
                logger.info(f"OCR cache hit for {project_id}: {len(cached)} pages")
                return [BatchPage(**p) for p in cached]

        images = self.collect_images(comic_dir)
        if not images:
            logger.warning(f"No images found in {comic_dir}")
            return []

        logger.info(f"OCR scanning {len(images)} pages...")
        from PIL import Image

        from comic_translator.services.ocr_postprocess_service import OCRPostprocessService

        if self._postprocess is None:
            self._postprocess = OCRPostprocessService()

        pages: list[BatchPage] = []
        all_entities: list[dict] = []
        total_stats = {"raw": 0, "cleaned": 0, "repaired": 0, "merged": 0}

        for page, img_path in images:
            logger.info(f"  OCR page {page}/{len(images)}: {img_path.name}")
            image = Image.open(img_path)
            raw_blocks = await self._provider.detect_text(image)
            normalized = self._normalize.process(raw_blocks, page=page, language=source_lang)

            # OCR 后处理（5 级修复）
            result = self._postprocess.process_page(normalized, page=page)
            processed_blocks = result["blocks"]
            page_entities = result["entities"]
            stats = result["stats"]

            for k in total_stats:
                total_stats[k] += stats.get(k, 0)

            if page_entities:
                all_entities.extend(page_entities)
                logger.debug(f"    Entities: {len(page_entities)}")

            pages.append(BatchPage(page=page, blocks=processed_blocks))

        # 输出统计
        logger.info(
            f"OCR postprocess: "
            f"raw={total_stats['raw']} "
            f"cleaned={total_stats['cleaned']} "
            f"repaired={total_stats['repaired']} "
            f"merged={total_stats['merged']} "
            f"entities={len(all_entities)}"
        )

        # Save cache
        if use_cache:
            data = [p.model_dump(mode="json") for p in pages]
            self._cache.set_ocr_cache(project_id, data)
            logger.info(f"OCR cache saved: {len(pages)} pages")

        return pages
