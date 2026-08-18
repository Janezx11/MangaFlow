#!/usr/bin/env python
"""
Text Detection Pipeline Demo

新 Pipeline: Text Detector → OCR → Bubble Matching(optional) → Translation → Render

用法:
  python demo/text_detection_pipeline.py
  python demo/text_detection_pipeline.py --page 1
  python demo/text_detection_pipeline.py --conf 0.3 --no-bubble
"""

import sys
import asyncio
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger

logger.remove()
logger.add(sys.stderr, level="INFO")


async def translate_dialogues(dialogues) -> None:
    """用 LLM 翻译对话块"""
    import yaml

    config_path = Path("config.yaml")
    if not config_path.exists():
        logger.warning("config.yaml not found, skipping translation")
        return

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    llm_cfg = config.get("llm", {})
    if llm_cfg.get("mode") != "api":
        logger.warning("config llm.mode != api, skipping translation")
        return

    api_cfg = llm_cfg.get("api", {})
    api_key = api_cfg.get("api_key", "")
    if not api_key or api_key == "your-api-key-here":
        logger.warning("No api_key set, skipping translation")
        return

    texts = [d.merged_text.strip() for d in dialogues if d.merged_text.strip()]
    if not texts:
        return

    import httpx

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    system_msg = (
        "You are a professional comic translator. "
        "Translate the following English comic text to Chinese. "
        "Keep the tone natural for comics. "
        "Return ONLY the translations, one per line, in the same order."
    )
    user_msg = "\n---\n".join(texts)

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{api_cfg['base_url'].rstrip('/')}/chat/completions",
                headers=headers,
                json={
                    "model": api_cfg["model"],
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                },
            )
            resp.raise_for_status()
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Translation failed: {e}")
        return

    lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
    idx = 0
    for d in dialogues:
        if d.merged_text.strip():
            if idx < len(lines):
                d.translated_text = lines[idx].lstrip("0123456789.- ")
                idx += 1
        else:
            d.translated_text = ""

    translated = sum(
        1 for d in dialogues
        if d.translated_text and d.translated_text != d.merged_text
    )
    logger.info(f"Translated {translated}/{len(dialogues)} dialogue blocks")


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Text Detection Pipeline Demo")
    parser.add_argument("--page", type=int, default=0, help="Page number (0=all)")
    parser.add_argument("--conf", type=float, default=0.3, help="CTBD confidence threshold")
    parser.add_argument("--no-bubble", action="store_true", help="Skip bubble detection (text-only mode)")
    parser.add_argument("--device", default="cpu", help="Device: cpu or cuda")
    args = parser.parse_args()

    from PIL import Image

    from comic_translator.providers.text_detection import CTBDTextDetectorProvider
    from comic_translator.providers.ocr.easyocr_provider import EasyOCRProvider
    from comic_translator.services.text_region_service import TextRegionService
    from comic_translator.services.text_region_adapter import TextRegionAdapter
    from comic_translator.services.text_normalizer import TextNormalizer
    from comic_translator.services.font_manager import FontManager
    from comic_translator.services.bubble_layout_service import BubbleLayoutService
    from comic_translator.services.shape_layout_solver import ShapeLayoutSolver
    from comic_translator.services.balanced_line_breaker import BalancedLineBreaker
    from comic_translator.services.padding_solver import PaddingSolver
    from comic_translator.services.font_size_solver import FontSizeSolver
    from comic_translator.services.optical_center_solver import OpticalCenterSolver
    from comic_translator.services.typography_service_v2 import TypographyServiceV2
    from comic_translator.services.typography_qa import TypographyQA
    from comic_translator.services.render_qa_service import RenderQAService
    from comic_translator.services.inpaint_service import InpaintService
    from comic_translator.services.background_cleaner import BackgroundCleaner
    from comic_translator.services.render_service import RenderService
    from comic_translator.services.bubble_interior_service import BubbleInteriorService
    from comic_translator.services.region_classifier import RegionClassifier

    # Text Detection Layer
    text_detector = CTBDTextDetectorProvider(
        confidence=args.conf,
        device=args.device,
    )
    text_region_service = TextRegionService()
    text_region_adapter = TextRegionAdapter()

    # OCR
    ocr_provider = EasyOCRProvider(lang="en", use_gpu=False)

    # Optional: Bubble Detection (auxiliary)
    bubble_provider = None
    if not args.no_bubble:
        from comic_translator.providers.bubble import YOLO11BubbleProvider
        bubble_provider = YOLO11BubbleProvider(
            model_path="models/bubble/yolo11n-seg-mixed-bubble.pt",
            device=args.device if args.device == "cuda" else "cpu",
        )

    # Render Pipeline
    normalizer = TextNormalizer()
    fm = FontManager()
    bl = BubbleLayoutService()
    sl = ShapeLayoutSolver(bl)
    brk = BalancedLineBreaker(fm)
    ps = PaddingSolver()
    fss = FontSizeSolver(fm, brk, sl, ps)
    oc = OpticalCenterSolver()
    typo_v2 = TypographyServiceV2(fm, bl, sl, brk, ps, fss, oc)
    typo_qa = TypographyQA()
    qa = RenderQAService()
    inpaint = InpaintService()
    cleaner = BackgroundCleaner(inpaint)
    render = RenderService(normalizer, fm, typo_v2, typo_qa, qa, inpaint, cleaner)
    interior = BubbleInteriorService()
    classifier = RegionClassifier()

    # Collect images
    images = sorted([
        f for f in Path("data/comics").iterdir()
        if f.suffix.lower() in (".png", ".jpg", ".jpeg")
    ])
    if not images:
        logger.error("No images found in data/comics/")
        return

    pages = [(i + 1, img) for i, img in enumerate(images)]
    if args.page > 0:
        pages = [p for p in pages if p[0] == args.page]

    for page, img_path in pages:
        logger.info(f"\n=== Page {page}: {img_path.name} ===")
        img = Image.open(img_path)

        # 1. Text Detection (PRIMARY)
        text_regions = await text_detector.detect(img)
        logger.info(f"Text regions: {len(text_regions)}")

        # 2. Optional: Bubble Detection (AUXILIARY)
        bubbles = None
        if bubble_provider:
            bubbles = await bubble_provider.detect(img)
            logger.info(f"Bubbles (aux): {len(bubbles)}")

        # 3. OCR within text regions
        ocr_blocks = await text_region_service.extract_text(img, text_regions, ocr_provider)
        logger.info(f"OCR blocks: {len(ocr_blocks)}")

        # 4. Text-Bubble Matching → DialogueBlock
        dialogues = text_region_adapter.to_dialogue_blocks(
            text_regions, ocr_blocks, bubbles=bubbles,
        )
        logger.info(f"Dialogues: {len(dialogues)}")

        # 5. Interior + Region Classification
        img_shape = (img.height, img.width)
        if bubbles:
            safe_areas = interior.compute_batch(bubbles, img_shape)
            for b in bubbles:
                b.safe_area = safe_areas.get(b.id)

        for d in dialogues:
            new_type = classifier.classify(d.bubble, img.height, img.width)
            if new_type != d.region_type:
                d.region_type = new_type
            logger.info(f"  [{d.region_type}] '{d.merged_text[:60]}'")

        # 6. Translation
        await translate_dialogues(dialogues)

        # 7. Render
        result = render.render_page(img, page, dialogues)
        logger.info(f"Rendered: {result.output_path}")

        # 8. Debug overlay
        import numpy as np
        import cv2

        img_np = np.array(img.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        render.draw_layout_debug(img_bgr, dialogues, Path("data/output/rendered"), page)
        logger.info(f"Layout debug: data/output/rendered/page_{page:04d}_layout.png")

    logger.info("\nDone! Check data/output/rendered/")


if __name__ == "__main__":
    asyncio.run(main())
