#!/usr/bin/env python
"""
Render Pipeline V2 Demo

流程:
  1. 读取 data/comics/ 下的图片
  2. OCR → OCR Repair → Batch Translation（可选）
  3. Render Pipeline → 输出最终图片
  4. QA 检查

用法:
  python demo/render_pipeline.py
  python demo/render_pipeline.py --page 1
  python demo/render_pipeline.py --skip-translate  (使用已有翻译缓存)
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import cv2
import numpy as np
from loguru import logger
from PIL import Image

logger.remove()
logger.add(sys.stderr, level="INFO")


async def run_render(
    project_name: str = "render_test",
    page_limit: int = 0,
    skip_translate: bool = False,
    output_dir: str = "data/output/rendered",
    db_path: str = "data/output/project.db",
):
    """Render Pipeline"""

    # === Init services ===
    from comic_translator.services.ocr_normalize_service import OCRNormalizeService
    from comic_translator.services.batch_cache import BatchCacheService
    from comic_translator.services.batch_ocr_service import BatchOCRService
    from comic_translator.services.ocr_postprocess_service import OCRPostprocessService
    from comic_translator.services.region_classifier import RegionClassifier
    from comic_translator.services.inpaint_service import InpaintService
    from comic_translator.services.background_cleaner import BackgroundCleaner
    from comic_translator.services.text_normalizer import TextNormalizer
    from comic_translator.services.typography_service import TypographyService
    from comic_translator.services.text_layout_service import TextLayoutService
    from comic_translator.services.render_qa_service import RenderQAService
    from comic_translator.services.render_service import RenderService

    # OCR
    cache = BatchCacheService()
    normalize = OCRNormalizeService()
    postprocess = OCRPostprocessService()

    # Render
    classifier = RegionClassifier()
    inpaint = InpaintService(padding=8)
    cleaner = BackgroundCleaner(inpaint)
    normalizer = TextNormalizer()
    typography = TypographyService()
    layout = TextLayoutService()
    qa = RenderQAService()

    render = RenderService(
        classifier=classifier,
        inpaint=inpaint,
        background_cleaner=cleaner,
        text_normalizer=normalizer,
        layout_service=layout,
        typography=typography,
        qa_service=qa,
    )

    # === OCR ===
    # 尝试从缓存读取
    ocr_pages = cache.get_ocr_cache(project_name)
    if ocr_pages is None:
        logger.error("No OCR cache found. Run demo/batch_pipeline.py first.")
        return

    images = sorted(
        [f for f in Path("data/comics").iterdir()
         if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")]
    )

    pages_to_render = ocr_pages
    if page_limit > 0:
        pages_to_render = pages_to_render[:page_limit]

    # === Load translations from DB ===
    from comic_translator.storage.database import get_engine, get_session
    from comic_translator.storage.models import TranslatedPageORM
    from comic_translator.models.translation_block import TranslationBlock
    from sqlalchemy import select

    translations_by_page: dict[int, list[TranslationBlock]] = {}
    session = next(get_session(db_path))
    try:
        rows = session.execute(
            select(TranslatedPageORM)
            .where(TranslatedPageORM.project_id == project_name)
        ).scalars().all()
        for row in rows:
            blocks_data = json.loads(row.page_data or "[]")
            blocks = [TranslationBlock(**b) for b in blocks_data]
            translations_by_page[row.page] = blocks
        logger.info(f"Loaded {len(rows)} translated pages from DB")
    finally:
        session.close()

    # === Render each page ===
    for page_data in pages_to_render:
        page = page_data.get("page", 0)
        if page not in translations_by_page:
            logger.warning(f"Page {page}: no translation found, skipping")
            continue

        blocks_data = page_data.get("blocks", [])
        from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock
        ocr_blocks = [NormalizedOCRBlock(**b) for b in blocks_data]
        translations = translations_by_page[page]

        # 找对应图片
        img_path = images[page - 1] if page <= len(images) else None
        if img_path is None or not img_path.exists():
            logger.warning(f"Page {page}: image not found, skipping")
            continue

        image = Image.open(img_path)
        logger.info(f"Rendering page {page}...")

        result = render.render_page(
            image=image,
            page=page,
            ocr_blocks=ocr_blocks,
            translations=translations,
            output_dir=output_dir,
            source_path=str(img_path),
            run_qa=True,
        )

        if result.qa and not result.qa.passed:
            for issue in result.qa.issues:
                logger.warning(f"  QA: {issue}")

    logger.info(f"\nRendered pages saved to: {output_dir}/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Render Pipeline Demo")
    parser.add_argument("--project", default="batch_project", help="项目名称")
    parser.add_argument("--page", type=int, default=0, help="只渲染第 N 页（0=全部）")
    args = parser.parse_args()

    asyncio.run(run_render(
        project_name=args.project,
        page_limit=args.page,
    ))