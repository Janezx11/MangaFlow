#!/usr/bin/env python
"""
一键翻译：把漫画图片丢到 data/comics/，运行这个脚本。

用法:
  uv run python demo/batch_translate.py

流程:
  1. 读取 data/comics/ 下所有图片
  2. 按文件名排序逐页翻译
  3. 每页自动继承上一页的 Context + Summary
  4. CharacterIdentity 跨页统一角色名
  5. SemanticTM 跨页复用历史翻译
  6. 输出到 data/output/{project_name}/

首次运行需要:
  - Ollama 已启动 (ollama serve)
  - qwen3.5:9b 已拉取 (ollama pull qwen3.5:9b)
  - (可选) PaddleOCR: uv sync --extra paddle

不需要 PaddleOCR 也能跑 —— 脚本内置 OCR 模拟模式。
"""

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from loguru import logger
from comic_translator.models.enums import ComicStyle

logger.remove()
logger.add(sys.stderr, level="INFO")


async def translate_all(
    project_name: str = "my_comic",
    source_lang: str = "en",
    target_lang: str = "zh",
    comic_style: str = "western_comic",
    db_path: str = "data/output/project.db",
    cache_dir: str = "data/cache",
    output_dir: str = "data/output",
    use_ollama: bool = True,
):
    """批量翻译 data/comics/ 下的所有漫画图片"""

    comic_dir = Path("data/comics")
    images = sorted(
        [f for f in comic_dir.glob("*") if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")]
    )
    if not images:
        logger.error(f"没有找到图片！请把漫画图片放到 {comic_dir}/")
        logger.info(f"支持的格式: .png .jpg .jpeg .webp .bmp")
        return

    logger.info(f"找到 {len(images)} 页漫画: {project_name}")
    for img in images:
        logger.info(f"  {img.name}")

    # === 初始化服务 ===
    from comic_translator.models.glossary import GlossaryEntry
    from comic_translator.models.character_identity import CharacterIdentity
    from comic_translator.services.glossary_service import GlossaryService
    from comic_translator.services.character_identity_service import CharacterIdentityService
    from comic_translator.services.semantic_tm_service import SemanticTranslationMemoryService
    from comic_translator.services.page_summary_service import PageSummaryService
    from comic_translator.services.context_assembler import ContextAssembler
    from comic_translator.services.translation_service import TranslationService
    from comic_translator.services.cache_service import CacheService
    from comic_translator.cache.file_cache_provider import FileCacheProvider
    from comic_translator.services.ocr_normalize_service import OCRNormalizeService

    glossary = GlossaryService(db_path=db_path)
    char_id = CharacterIdentityService(db_path=db_path)
    tm = SemanticTranslationMemoryService(db_path=db_path)
    page_summary = PageSummaryService(cache_dir=cache_dir)
    cache = CacheService(provider=FileCacheProvider(cache_dir=cache_dir))
    normalize = OCRNormalizeService()

    # 按漫画风格选 PromptBuilder
    if comic_style == "manga":
        from comic_translator.prompts.manga_prompt_builder import MangaPromptBuilder
        prompt_builder = MangaPromptBuilder()
        logger.info("使用 Manga 风格 Prompt")
    elif comic_style == "webtoon":
        from comic_translator.prompts.webtoon_prompt_builder import WebtoonPromptBuilder
        prompt_builder = WebtoonPromptBuilder()
        logger.info("使用 Webtoon 风格 Prompt")
    else:
        from comic_translator.prompts.western_comic_prompt_builder import WesternComicPromptBuilder
        prompt_builder = WesternComicPromptBuilder()
        logger.info("使用 Western Comic 风格 Prompt")

    assembler = ContextAssembler(
        glossary_service=glossary,
        character_identity_service=char_id,
        semantic_tm_service=tm,
        page_summary_service=page_summary,
        prompt_builder=prompt_builder,
    )

    # === 选 OCR Provider（EasyOCR → PaddleOCR → 模拟）===
    ocr_provider = None

    # 优先 EasyOCR
    try:
        from comic_translator.providers.ocr.easyocr_provider import EasyOCRProvider

        easyocr_provider = EasyOCRProvider(lang=source_lang, use_gpu=False)
        # Quick health check: detect text on a small test region
        from PIL import Image, ImageDraw
        import numpy as np

        test_img = Image.new("RGB", (100, 30), color="white")
        draw = ImageDraw.Draw(test_img)
        draw.text((5, 5), "test", fill="black")
        _ = await easyocr_provider.detect_text(test_img)
        ocr_provider = easyocr_provider
        logger.info("使用 EasyOCR")
    except Exception:
        logger.warning("EasyOCR 不可用")

    # 备用 PaddleOCR
    if ocr_provider is None:
        try:
            from comic_translator.providers.ocr.paddle_ocr_provider import PaddleOCRProvider

            paddle_provider = PaddleOCRProvider(lang=source_lang, use_gpu=False)
            ocr_provider = paddle_provider
            logger.info("使用 PaddleOCR")
        except Exception:
            logger.warning("PaddleOCR 不可用")

    if ocr_provider is None:
        logger.info("使用模拟 OCR（仅提取文件名作为占位文本）")

    # === 选 LLM（Ollama / 模拟）===
    translation_svc = None
    if use_ollama:
        from comic_translator.clients.ollama_client import OllamaClient
        from comic_translator.providers.translation.qwen_translator import QwenTranslator

        try:
            client = OllamaClient(model="qwen3.5:9b", base_url="http://localhost:11434")
            # 使用 is_available() 做快速健康检查（仅检查 Ollama 是否运行，不加载模型）
            if not await client.is_available():
                raise ConnectionError("Ollama is not running")
            translator = QwenTranslator(client=client)
            translation_svc = TranslationService(
                provider=translator,
                prompt_builder=prompt_builder,
                cache=cache,
                context_assembler=assembler,
                semantic_tm_service=tm,
                page_summary_service=page_summary,
            )
            logger.info("Ollama 连接成功，使用 Qwen3.5:9b 翻译")
        except Exception as e:
            logger.warning(f"Ollama 不可用 ({e})，使用模拟翻译")
            use_ollama = False

    # === 可选：预填充术语表和角色 ===
    # 在这里添加你的自定义词条...
    # glossary.add_term(GlossaryEntry(source="...", target="...", category="character"))
    # char_id.create_identity(CharacterIdentity(character_id="...", canonical_name="...", ...))

    # === 逐页翻译 ===
    out_dir = Path(output_dir) / project_name
    out_dir.mkdir(parents=True, exist_ok=True)
    all_results = []

    for page_idx, img_path in enumerate(images):
        page = page_idx + 1
        logger.info(f"\n{'='*60}")
        logger.info(f"第 {page}/{len(images)} 页: {img_path.name}")
        logger.info(f"{'='*60}")

        # Step 1: OCR
        if ocr_provider:
            from PIL import Image
            image = Image.open(img_path)
            raw_blocks = await ocr_provider.detect_text(image)
        else:
            # 模拟 OCR：每张图片生成一个占位文本块
            from comic_translator.models.ocr_block import OCRBlock
            raw_blocks = [
                OCRBlock(
                    id=f"page_{page:04d}_sim",
                    page=page,
                    bbox=(50, 50, 300, 80),
                    text=f"[Page {page}] {img_path.stem}",
                    confidence=0.9,
                    language=source_lang,
                )
            ]

        # Step 2: Normalize
        blocks = normalize.process(raw_blocks, page=page)
        logger.info(f"  OCR: {len(raw_blocks)} raw → {len(blocks)} normalized blocks")

        # Step 3: ContextAssembler
        ctx = assembler.assemble_with_tm(
            project_id=project_name,
            page=page,
            blocks_source_texts=[b.text for b in blocks],
            source_language=source_lang,
            target_language=target_lang,
            comic_style=ComicStyle(comic_style),
        )
        prev_summary = page_summary.get_previous_summary(project_name, page)
        if prev_summary:
            logger.info(f"  加载上一页摘要: {prev_summary.summary[:60]}...")

        # Step 4: Translate
        if translation_svc:
            translations = await translation_svc.translate_page(
                ocr_blocks=blocks,
                context=ctx,
                page=page,
                project_id=project_name,
                use_cache=True,
            )
        else:
            from comic_translator.models.translation_block import TranslationBlock
            translations = [
                TranslationBlock(
                    id=b.id, ocr_block_id=b.id,
                    source_text=b.text,
                    translated_text=f"[待翻译] {b.text}",
                    provider="simulation",
                )
                for b in blocks
            ]

        # Step 5: Summary
        summary = await page_summary.generate_summary(
            blocks=translations,
            context=ctx,
            page=page,
            project_id=project_name,
        )

        # 保存本页结果
        page_result = {
            "page": page,
            "file": img_path.name,
            "translations": [
                {"source": t.source_text, "translated": t.translated_text}
                for t in translations
            ],
            "summary": summary.summary,
        }
        all_results.append(page_result)

        # 打印翻译结果
        for t in translations:
            logger.info(f"  {str(t.source_text)[:50]:50s} → {str(t.translated_text)[:40]}")

    # === 输出汇总 ===
    result_file = out_dir / "all_translations.json"
    result_file.write_text(
        json.dumps(all_results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"翻译完成！共 {len(images)} 页")
    logger.info(f"结果保存到: {result_file}")
    logger.info(f"数据库: {db_path}")
    logger.info(f"摘要缓存: {cache_dir}/{project_name}/summaries/")
    logger.info(f"翻译缓存: {cache_dir}/{project_name}/translation/")

    # 打印翻译记忆统计
    stats = tm.get_stats()
    logger.info(f"翻译记忆条目: {stats['total_entries']}")

    # 断点续跑提示
    logger.info(f"\n💡 断点续跑: 删除对应缓存文件即可重新翻译某页")
    logger.info(f"   rm {cache_dir}/{project_name}/translation/page_0000xx.json")
    logger.info(f"   rm {cache_dir}/{project_name}/summaries/page_0000xx_summary.json")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="一键批量翻译漫画")
    parser.add_argument("--project", default="my_comic", help="项目名称")
    parser.add_argument("--style", default="western_comic", choices=["western_comic", "manga", "webtoon"], help="漫画风格")
    parser.add_argument("--ollama", action="store_true", default=True, help="使用 Ollama 翻译")
    parser.add_argument("--no-ollama", action="store_false", dest="ollama", help="不使用 Ollama（模拟翻译）")
    parser.add_argument("--source-lang", default="en", help="源语言代码")
    parser.add_argument("--db", default="data/output/project.db", help="SQLite 数据库路径")
    args = parser.parse_args()

    asyncio.run(translate_all(
        project_name=args.project,
        source_lang=args.source_lang,
        comic_style=args.style,
        db_path=args.db,
        use_ollama=args.ollama,
    ))