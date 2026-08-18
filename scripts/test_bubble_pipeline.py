#!/usr/bin/env python
"""
YOLO Bubble Detection + OCR + Matching + Render 完整测试

用法:
  python scripts/test_bubble_pipeline.py
  python scripts/test_bubble_pipeline.py --page 1
"""

import sys, asyncio
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")


async def translate_dialogues(dialogues) -> None:
    """用 LLM 翻译对话块（JSON 结构化输出，支持 OCR 纠错）"""
    import yaml
    from pathlib import Path

    config = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8"))
    llm_cfg = config["llm"]
    if llm_cfg["mode"] != "api":
        logger.warning("config llm.mode != api, skipping translation")
        return

    api_cfg = llm_cfg["api"]
    api_key = api_cfg.get("api_key", "")
    if not api_key or api_key == "your-api-key-here":
        logger.warning("No api_key set in config.yaml, skipping translation")
        return

    # 收集所有需要翻译的文本（带 ID，用于 JSON 映射）
    blocks_to_translate = [(d.id, d.merged_text.strip()) for d in dialogues if d.merged_text.strip()]
    if not blocks_to_translate:
        return

    # 构建 JSON 结构化 prompt
    import json
    blocks_json = [{"id": bid, "text": text} for bid, text in blocks_to_translate]

    system_msg = (
        "You are a professional comic translator. Your tasks:\n"
        "1. Correct OCR errors in the source text (e.g. 'ANDPEw woodS' -> 'Andrew Woods', "
        "'Youll' -> 'You'll', 'tonicht' -> 'tonight')\n"
        "2. Translate the corrected English text to natural Simplified Chinese\n"
        "3. Preserve character names, speaking style, and emotional tone\n"
        "4. Keep the translation concise — fit for comic speech bubbles\n\n"
        "Output ONLY valid JSON. No explanations, no markdown fences."
    )

    user_msg = (
        "Translate the following comic dialogue blocks to Chinese.\n"
        "For each block, first correct any OCR errors, then translate.\n\n"
        "Input:\n" + json.dumps(blocks_json, ensure_ascii=False, indent=2) + "\n\n"
        "Output JSON format:\n"
        '{\n'
        '  "translations": [\n'
        '    {"id": "bubble_yolo_0001", "corrected": "corrected English text", "translated": "中文翻译"}\n'
        '  ]\n'
        '}'
    )

    import httpx
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            resp = await client.post(
                f"{api_cfg['base_url'].rstrip('/')}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                json={
                    "model": api_cfg["model"],
                    "messages": [
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": user_msg},
                    ],
                    "temperature": 0.1,
                },
            )
            resp.raise_for_status()
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Translation API call failed: {e}")
        return

    # 解析 JSON 响应
    content = content.strip()
    # 去除 markdown 代码块标记
    if content.startswith("```"):
        import re
        content = re.sub(r"^```\w*\n?", "", content)
        content = re.sub(r"\n```$", "", content)
        content = content.strip()

    # 找到 JSON 部分
    start = content.find("{")
    end = content.rfind("}") + 1
    if start < 0 or end <= start:
        logger.error(f"Could not find JSON in response: {content[:200]}")
        return

    try:
        parsed = json.loads(content[start:end])
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse failed: {e}")
        logger.debug(f"Raw response: {content[:500]}")
        return

    raw_translations = parsed.get("translations", [])
    if not raw_translations:
        logger.error("No translations in response")
        return

    # 按 ID 映射回 DialogueBlock
    trans_map = {t["id"]: t for t in raw_translations}
    mapped = 0
    for d in dialogues:
        if d.id in trans_map:
            entry = trans_map[d.id]
            d.translated_text = entry.get("translated", "") or entry.get("corrected", "")
            mapped += 1
        elif d.merged_text.strip():
            # fallback: 用原文（未翻译）
            d.translated_text = d.merged_text

    translated = sum(1 for d in dialogues if d.translated_text and d.translated_text != d.merged_text)
    logger.info(f"Translated {translated}/{len(dialogues)} dialogue blocks (mapped={mapped})")


async def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--page", type=int, default=0)
    parser.add_argument("--model", default="", help="Path to YOLO .pt model")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    from PIL import Image
    from comic_translator.providers.bubble import YOLO11BubbleProvider
    from comic_translator.providers.ocr.easyocr_provider import EasyOCRProvider
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

    # 1. Init services
    bubble_provider = YOLO11BubbleProvider(model_path=args.model, device=args.device)
    ocr_provider = EasyOCRProvider(lang="en", use_gpu=True)
    matcher = BubbleMatchingService()

    # Bubble Pipeline V2
    postproc = BubblePostProcessService()
    interior = BubbleInteriorService()
    classifier = RegionClassifier()

    normalizer = TextNormalizer()

    # V6 Typography Engine
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

    # 2. Collect images
    images = sorted([f for f in Path("data/comics").iterdir()
                     if f.suffix.lower() in (".png", ".jpg", ".jpeg")])
    if not images:
        logger.error("No images found in data/comics/")
        return

    pages_to_process = [(i + 1, img) for i, img in enumerate(images)]
    if args.page > 0:
        pages_to_process = [p for p in pages_to_process if p[0] == args.page]

    for page, img_path in pages_to_process:
        logger.info(f"\n=== Page {page}: {img_path.name} ===")
        img = Image.open(img_path)

        # 3. YOLO Bubble Detection
        bubbles = await bubble_provider.detect(img)
        logger.info(f"Bubbles (raw): {len(bubbles)}")

        # 3.5 Bubble PostProcess: filter, merge, smooth
        bubbles = postproc.process(bubbles, img.size[::-1], ocr_blocks_map={})
        logger.info(f"Bubbles (post-process): {len(bubbles)}")

        # 4. OCR
        raw_blocks = await ocr_provider.detect_text(img)
        logger.info(f"OCR blocks: {len(raw_blocks)}")

        # 5. Bubble ↔ OCR Matching
        dialogues = matcher.match(bubbles, raw_blocks)
        logger.info(f"Dialogues: {len(dialogues)}")

        # 5.2 Bubble Interior: 计算每个气泡的 safe area
        img_shape = (img.height, img.width)
        safe_areas = interior.compute_batch(bubbles, img_shape)
        for b in bubbles:
            b.safe_area = safe_areas.get(b.id)

        # 5.3 Region Classification: 重新分类 region type
        for d in dialogues:
            new_type = classifier.classify(d.bubble, img.height, img.width)
            if new_type != d.region_type:
                logger.debug(f"  Reclassify: {d.bubble.id} {d.region_type} → {new_type}")
                d.region_type = new_type

        for d in dialogues:
            logger.info(f"  [{d.region_type}] '{d.merged_text[:50]}...'")

        # 5.5 Translation: 把合并后的英文翻成中文
        await translate_dialogues(dialogues)

        # 6. Render
        result = render.render_page(img, page, dialogues)
        logger.info(f"Rendered: {result.output_path}")

        # 7. Layout debug overlay
        import numpy as np
        import cv2
        img_np = np.array(img.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        render.draw_layout_debug(img_bgr, dialogues, Path("data/output/rendered"), page)
        logger.info(f"Layout debug: data/output/rendered/page_{page:04d}_layout.png")

    logger.info("\nDone! Check data/output/rendered/")

if __name__ == "__main__":
    asyncio.run(main())