#!/usr/bin/env python
"""
Sprint 2 Demo: OCR → Normalize → Translate → JSON Output

This demo validates the core translation pipeline end-to-end.
It works in two modes:
  1. Full mode (requires PaddleOCR + Ollama running locally)
  2. Simulation mode (uses pre-defined OCR blocks, no external deps)

Usage:
  uv run python demo/translate_pipeline.py
"""

import json
import sys
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger
from comic_translator.models.enums import ComicStyle

# Configure logger
logger.remove()
logger.add(sys.stderr, level="INFO")


def simulate_ocr_blocks():
    """Generate simulated OCR blocks for demo purposes.

    Returns blocks that need normalization (unsorted, needs merging).
    """
    from comic_translator.models.ocr_block import OCRBlock

    return [
        OCRBlock(id="b1", page=1, bbox=(50, 120, 300, 145), text="Hello.", confidence=0.95),
        OCRBlock(id="b2", page=1, bbox=(50, 148, 300, 170), text="My name is", confidence=0.92),
        OCRBlock(id="b3", page=1, bbox=(50, 172, 300, 195), text="Tony Stark.", confidence=0.90),
        OCRBlock(id="b4", page=1, bbox=(50, 220, 350, 245), text="You may know me", confidence=0.88),
        OCRBlock(id="b5", page=1, bbox=(50, 248, 350, 270), text="as Iron Man .", confidence=0.85),
        OCRBlock(id="b6", page=1, bbox=(400, 50, 600, 80), text="Welcome to Stark Tower!", confidence=0.93),
    ]


def run_simulation_demo(output_dir: str = "data/output", cache_dir: str = "data/cache"):
    """Run pipeline with simulated OCR — no external dependencies."""
    from comic_translator.models.context import TranslationContext
    from comic_translator.models.glossary import GlossaryEntry
    from comic_translator.models.character_memory import CharacterMemory
    from comic_translator.services.ocr_normalize_service import OCRNormalizeService
    from comic_translator.services.glossary_service import GlossaryService
    from comic_translator.services.character_memory_service import CharacterMemoryService
    from comic_translator.services.translation_memory_service import TranslationMemoryService
    from comic_translator.services.prompt_cache_service import PromptCacheService
    from comic_translator.services.cache_service import CacheService
    from comic_translator.cache.file_cache_provider import FileCacheProvider

    project_id = "demo_project"
    page = 1
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Setup services ---
    logger.info("=== Setting up services ===")

    db_path = str(out_dir / "demo.db")
    glossary_service = GlossaryService(db_path=db_path)
    char_memory = CharacterMemoryService(db_path=db_path)
    tm_service = TranslationMemoryService(db_path=db_path)
    prompt_cache = PromptCacheService(cache_dir=cache_dir)
    cache_service = CacheService(provider=FileCacheProvider(cache_dir=cache_dir))

    # --- Seed glossary ---
    logger.info("=== Seeding glossary ===")
    glossary_entries = [
        GlossaryEntry(source="Tony Stark", target="托尼·斯塔克", category="character"),
        GlossaryEntry(source="Stark Tower", target="斯塔克大厦", category="location"),
        GlossaryEntry(source="Iron Man", target="钢铁侠", category="character"),
    ]
    for entry in glossary_entries:
        glossary_service.add_term(entry)
    logger.info(f"Added {len(glossary_entries)} glossary entries")

    # --- Seed character memory ---
    logger.info("=== Seeding character memory ===")
    char_memory.register_character(
        CharacterMemory(
            character_id="tony_stark",
            canonical_name="Tony Stark",
            aliases=["Tony", "Mr. Stark", "Iron Man"],
            description="Billionaire inventor of Stark Industries",
            speaking_style="sarcastic, confident, witty",
        )
    )
    logger.info("Registered character: Tony Stark")

    # Step 1: Simulate OCR blocks
    logger.info("=== Step 1: Simulated OCR ===")
    raw_blocks = simulate_ocr_blocks()
    logger.info(f"OCR returned {len(raw_blocks)} raw blocks")

    # Step 2: Normalize
    logger.info("=== Step 2: OCR Normalize ===")
    normalize_service = OCRNormalizeService()
    normalized = normalize_service.process(raw_blocks, page=page)
    logger.info(f"Normalized to {len(normalized)} blocks")
    for b in normalized:
        logger.info(f"  [{b.reading_order}] {b.text[:50]} ({'merged' if b.is_merged else 'single'})")

    # Step 3: Build Translation Context
    logger.info("=== Step 3: Build Translation Context ===")
    context = TranslationContext(
        glossary=glossary_service.get_all_entries(),
        comic_style=ComicStyle.WESTERN_COMIC,
        source_language="en",
        target_language="zh",
        metadata={
            "characters": [
                {
                    "canonical_name": c.canonical_name,
                    "aliases": c.aliases,
                    "speaking_style": c.speaking_style,
                }
                for c in char_memory.get_all_characters()
            ]
        },
    )

    # Step 4: Try translation via LLM (skip if Ollama not available)
    logger.info("=== Step 4: Translation ===")

    from comic_translator.providers.translation.qwen_translator import QwenTranslator
    from comic_translator.clients.ollama_client import OllamaClient

    try:
        client = OllamaClient(model="qwen3.5:9b", base_url="http://localhost:11434", timeout=10)
        # Try a quick health check
        import asyncio
        asyncio.run(client.generate("ping", max_tokens=1))
        ollama_available = True
    except Exception as e:
        logger.warning(f"Ollama not available ({e}) — using mock translation")
        ollama_available = False

    if ollama_available:
        from comic_translator.prompts.western_comic_prompt_builder import WesternComicPromptBuilder

        prompt_builder = WesternComicPromptBuilder()
        provider = QwenTranslator(client=client)

        from comic_translator.services.translation_service import TranslationService

        translation_service = TranslationService(
            provider=provider,
            prompt_builder=prompt_builder,
            glossary_service=glossary_service,
            character_memory_service=char_memory,
            translation_memory_service=tm_service,
            prompt_cache_service=prompt_cache,
            cache=cache_service,
        )

        import asyncio
        translations = asyncio.run(
            translation_service.translate_page(
                ocr_blocks=normalized,
                context=context,
                page=page,
                project_id=project_id,
                use_cache=True,
            )
        )
    else:
        # Mock translation — use direct OllamaClient prompt for demonstration
        logger.info("Using simplified translation (no Ollama)")
        from comic_translator.models.translation_block import TranslationBlock

        translations = []
        for b in normalized:
            block = TranslationBlock(id=b.id, ocr_block_id=b.id, source_text=b.text, provider="demo")
            block.translated_text = f"[translated] {b.text}"
            translations.append(block)

    # Step 5: Save outputs
    logger.info("=== Step 5: Save Outputs ===")

    # Save translation.json
    trans_output = out_dir / "page_001.translation.json"
    trans_data = [
        {
            "id": t.id,
            "source_text": t.source_text,
            "translated_text": t.translated_text,
            "provider": t.provider,
            "model": t.model,
        }
        for t in translations
    ]
    trans_output.write_text(json.dumps(trans_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved: {trans_output}")

    # Save normalized OCR for reference
    ocr_output = out_dir / "page_001.ocr.json"
    ocr_data = [
        {
            "id": b.id,
            "page": b.page,
            "bbox": b.bbox,
            "text": b.text,
            "reading_order": b.reading_order,
            "is_merged": b.is_merged,
            "merged_from_ids": b.merged_from_ids,
        }
        for b in normalized
    ]
    ocr_output.write_text(json.dumps(ocr_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"Saved: {ocr_output}")

    # Print summary
    logger.info("")
    logger.info("=== Pipeline Result ===")
    for t in translations:
        logger.info(f"  {t.source_text[:40]:40s} → {t.translated_text[:40]}")

    logger.info("")
    logger.info("=== Output Files ===")
    logger.info(f"  {ocr_output}")
    logger.info(f"  {trans_output}")
    logger.info(f"  DB: {db_path}")
    logger.info("")
    logger.info("Pipeline completed successfully!")


def run_ocr_pipeline_demo(output_dir: str = "data/output", cache_dir: str = "data/cache"):
    """Run pipeline with real PaddleOCR — requires paddlepaddle + paddleocr installed."""
    from comic_translator.providers.ocr.paddle_ocr_provider import PaddleOCRProvider
    from comic_translator.services.ocr_normalize_service import OCRNormalizeService
    from comic_translator.services.cache_service import CacheService
    from comic_translator.cache.file_cache_provider import FileCacheProvider
    from PIL import Image

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find sample image
    comic_dir = Path("data/comics")
    images = list(comic_dir.glob("*.png")) + list(comic_dir.glob("*.jpg")) + list(comic_dir.glob("*.jpeg"))
    if not images:
        logger.error("No sample images found in data/comics/. Place a comic page image there first.")
        return

    image_path = images[0]
    logger.info(f"Processing: {image_path}")

    image = Image.open(image_path)
    provider = PaddleOCRProvider(lang="en", use_gpu=False)

    import asyncio
    raw_blocks = asyncio.run(provider.detect_text(image))
    logger.info(f"OCR detected {len(raw_blocks)} blocks")

    normalize_service = OCRNormalizeService()
    normalized = normalize_service.process(raw_blocks, page=1)
    logger.info(f"Normalized to {len(normalized)} blocks")

    ocr_output = out_dir / "page_001.ocr.json"
    ocr_data = [
        {
            "id": b.id,
            "bbox": b.bbox,
            "text": b.text,
            "confidence": b.confidence,
            "reading_order": b.reading_order,
            "is_merged": b.is_merged,
        }
        for b in normalized
    ]
    ocr_output.write_text(json.dumps(ocr_data, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info(f"OCR results saved to: {ocr_output}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Comic Translation Pipeline Demo")
    parser.add_argument(
        "--mode",
        choices=["simulation", "ocr", "full"],
        default="simulation",
        help="Demo mode: simulation (no deps), ocr (requires PaddleOCR), full (requires PaddleOCR + Ollama)",
    )
    parser.add_argument("--output-dir", default="data/output", help="Output directory")
    parser.add_argument("--cache-dir", default="data/cache", help="Cache directory")
    args = parser.parse_args()

    # Import here since ComicStyle is not needed for argparse
    if args.mode == "simulation":
        run_simulation_demo(output_dir=args.output_dir, cache_dir=args.cache_dir)
    elif args.mode == "ocr":
        run_ocr_pipeline_demo(output_dir=args.output_dir, cache_dir=args.cache_dir)
    else:
        logger.error("Full mode not yet implemented")
        sys.exit(1)