#!/usr/bin/env python3
"""Run one page through the pipeline and analyze the output"""
import sys, json, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from PIL import Image
import numpy as np
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="WARNING")

from comic_translator.providers.bubble import YOLO11BubbleProvider
from comic_translator.providers.ocr.easyocr_provider import EasyOCRProvider
from comic_translator.providers.translation.qwen_translator import QwenTranslator
from comic_translator.clients.openai_compatible_client import OpenAICompatibleClient
from comic_translator.models.translation_block import TranslationBlock
from comic_translator.models.context import TranslationContext
from comic_translator.services.bubble_matching_service import BubbleMatchingService
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
from comic_translator.services.bubble_postprocess_service import BubblePostProcessService
from comic_translator.services.bubble_interior_service import BubbleInteriorService
from comic_translator.services.region_classifier import RegionClassifier


async def main():
    page = 1
    img_path = Path("data/comics/0002_GREECE0001.png")
    if not img_path.exists():
        img_path = Path("data/comics/0003_GREECE0002.png")
        page = 2

    print(f"Processing page {page}: {img_path}")

    # Init services
    bubble_provider = YOLO11BubbleProvider(device="cpu")
    ocr_provider = EasyOCRProvider(lang="en", use_gpu=False)
    matcher = BubbleMatchingService()
    postproc = BubblePostProcessService()
    interior = BubbleInteriorService()
    classifier = RegionClassifier()
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

    # Load image
    img = Image.open(img_path)
    print(f"Image: {img.size}")

    # 1. YOLO detection
    bubbles = await bubble_provider.detect(img)
    print(f"Bubbles (raw): {len(bubbles)}")

    # 2. PostProcess
    bubbles = postproc.process(bubbles, img.size[::-1], ocr_blocks_map={})
    print(f"Bubbles (post): {len(bubbles)}")

    # 3. OCR
    raw_blocks = await ocr_provider.detect_text(img)
    print(f"OCR blocks: {len(raw_blocks)}")

    # 4. Match
    dialogues = matcher.match(bubbles, raw_blocks)
    print(f"Dialogues: {len(dialogues)}")

    # 5. Safe area + classify
    img_shape = (img.height, img.width)
    safe_areas = interior.compute_batch(bubbles, img_shape)
    for b in bubbles:
        b.safe_area = safe_areas.get(b.id)
    for d in dialogues:
        new_type = classifier.classify(d.bubble, img.height, img.width)
        d.region_type = new_type
        print(f"  [{d.region_type}] bbox={d.bubble.bbox} text='{d.merged_text[:50]}'")

    # 6. Translate English to Chinese
    print("\n--- Translation ---")
    # Read API config
    import yaml
    cfg_path = Path("config.yaml")
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
        api_cfg = cfg.get("llm", {}).get("api", {})
        api_key = api_cfg.get("api_key", "")
        base_url = api_cfg.get("base_url", "https://token.sensenova.cn/v1")
        model = api_cfg.get("model", "deepseek-v4-flash")
    else:
        api_key = ""
        base_url = "https://token.sensenova.cn/v1"
        model = "deepseek-v4-flash"

    if api_key and "..." not in api_key:
        translator_client = OpenAICompatibleClient(
            model=model,
            base_url=base_url,
            api_key=api_key,
            timeout=120,
        )
        translator = QwenTranslator(client=translator_client, model=model)

        # Build TranslationBlocks from dialogues
        blocks = [
            TranslationBlock(
                id=d.id,
                ocr_block_id=d.id,
                source_text=d.merged_text,
            ) for d in dialogues
        ]
        ctx = TranslationContext(
            source_language="en",
            target_language="zh",
        )
        translated = await translator.translate(blocks, ctx)

        # Copy results back
        for d in dialogues:
            for tb in translated:
                if tb.id == d.id and tb.translated_text:
                    d.translated_text = tb.translated_text
                    print(f"  {d.id}: '{d.merged_text[:40]}' → '{d.translated_text[:40]}'")
                    break
    else:
        print("  WARNING: No valid API key found in config.yaml")
        print("  Using placeholder Chinese text for testing")
        # Fallback: use hardcoded Chinese for testing
        fallback = {
            "bubble_yolo_0002": "哦宝贝！我真不敢相信这地方有多棒",
            "bubble_yolo_0001": "我开始相信你说得对，希腊确实看起来",
        }
        for d in dialogues:
            if d.id in fallback:
                d.translated_text = fallback[d.id]
                print(f"  {d.id}: '{d.merged_text[:40]}' → '{d.translated_text}'")

    # 7. Analyze safe area vs old boundingRect approach
    print("\n--- Safe Area Analysis ---")
    for d in dialogues:
        poly = d.bubble.polygon
        if poly and len(poly) >= 3:
            # Old approach: boundingRect
            cnt = np.array(poly, dtype=np.int32)
            ox, oy, ow, oh = cv2.boundingRect(cnt)
            old_rect = (ox, oy, ox + ow, oy + oh)

            # New approach: max inscribed rect
            new_safe = bl.safe_area(poly, img_shape, min_distance=0)
            new_rect = new_safe.max_inscribed_rect

            old_area = (old_rect[2] - old_rect[0]) * (old_rect[3] - old_rect[1])
            new_area = (new_rect[2] - new_rect[0]) * (new_rect[3] - new_rect[1])
            shrink = (1 - new_area / old_area) * 100 if old_area > 0 else 0
            print(f"  {d.bubble.id}: old={old_rect} new={new_rect} shrink={shrink:.1f}%")

    # 8. Render
    out_dir = "data/output/test_fix5"
    result = render.render_page(img, page, dialogues, output_dir=out_dir)
    print(f"\nRendered: {result.output_path}")

    # 9. Read debug JSON
    debug_path = Path(out_dir) / f"page_{page:04d}_render_debug.json"
    if debug_path.exists():
        data = json.loads(debug_path.read_text(encoding="utf-8"))
        print(f"\n--- Render Debug ---")
        for b in data["blocks"]:
            print(f"  {b['block_id']}: type={b['region_type']}, fs={b['font_size']}, lines={b['line_count']}, score={b['typography_score']}")
            print(f"    text: {b['translated_text'][:60]}")

    # 10. Compare with old output
    old_debug = Path("data/output/rendered") / f"page_{page:04d}_render_debug.json"
    if old_debug.exists():
        old_data = json.loads(old_debug.read_text(encoding="utf-8"))
        print(f"\n--- Old vs New Comparison ---")
        for b_new in data["blocks"]:
            b_old = next((o for o in old_data["blocks"] if o["block_id"] == b_new["block_id"]), None)
            if b_old:
                print(f"  {b_new['block_id']}:")
                print(f"    old: fs={b_old['font_size']}, lines={b_old['line_count']}, score={b_old['typography_score']}")
                print(f"    new: fs={b_new['font_size']}, lines={b_new['line_count']}, score={b_new['typography_score']}")

    print("\nDone!")


if __name__ == "__main__":
    import cv2  # needed for analysis
    asyncio.run(main())