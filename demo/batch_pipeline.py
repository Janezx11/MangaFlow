#!/usr/bin/env python
"""
Batch 翻译管线 —— 替代 demo/batch_translate.py

流程:
  1. OCR 全部页面 → 缓存
  2. 按 batch_size 切分
  3. 逐 Batch 调用 API（附 Glossary）
  4. 自动提取 Glossary → 写入 SQLite
  5. 结果写回 DB + 缓存
  6. 输出最终 JSON

用法:
  python demo/batch_pipeline.py
  python demo/batch_pipeline.py --batch-size 10
  python demo/batch_pipeline.py --style manga
  python demo/batch_pipeline.py --no-ollama  (模拟翻译)
"""

import asyncio
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import yaml
from loguru import logger


logger.remove()
logger.add(sys.stderr, level="INFO")


def load_config(path: str = "config.yaml") -> dict:
    """从 config.yaml 读取配置，文件不存在则返回空 dict"""
    p = Path(path)
    if not p.exists():
        logger.info(f"Config file {path} not found, using defaults")
        return {}
    with open(p, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
        logger.info(f"Loaded config from {path}")
        return cfg or {}


async def run_pipeline(
    project_name: str = "batch_project",
    batch_size: int = 20,
    comic_dir: str = "data/comics",
    source_lang: str = "en",
    target_lang: str = "zh",
    comic_style: str = "western_comic",
    db_path: str = "data/output/project.db",
    cache_dir: str = "data/cache",
    output_dir: str = "data/output",
    use_ollama: bool = True,
    api_base: str = "",
    api_key: str = "",
    api_model: str = "SenseNova-v6",
):
    """完整 Batch 翻译管线"""

    # === 1. Init services ===
    from comic_translator.services.ocr_normalize_service import OCRNormalizeService
    from comic_translator.services.glossary_service import GlossaryService
    from comic_translator.services.batch_cache import BatchCacheService
    from comic_translator.services.batch_ocr_service import BatchOCRService
    from comic_translator.services.batch_prompt_builder import BatchPromptBuilder
    from comic_translator.services.batch_translation_service import (
        BatchTranslationService,
    )
    from comic_translator.services.ocr_postprocess_service import OCRPostprocessService
    from comic_translator.services.entity_service import EntityService

    glossary = GlossaryService(db_path=db_path)
    normalize = OCRNormalizeService()
    cache = BatchCacheService(cache_dir=cache_dir)
    prompt_builder = BatchPromptBuilder()
    postprocess = OCRPostprocessService()
    entity_service = EntityService(db_path=db_path)

    # === 2. Select OCR Provider ===
    ocr_provider = None

    try:
        from comic_translator.providers.ocr.easyocr_provider import EasyOCRProvider

        ep = EasyOCRProvider(lang=source_lang, use_gpu=False)
        # Health check
        from PIL import Image, ImageDraw

        test_img = Image.new("RGB", (100, 30), color="white")
        ImageDraw.Draw(test_img).text((5, 5), "test", fill="black")
        _ = await ep.detect_text(test_img)
        ocr_provider = ep
        logger.info("OCR: EasyOCR")
    except Exception:
        logger.warning("EasyOCR unavailable")

    if ocr_provider is None:
        try:
            from comic_translator.providers.ocr.paddle_ocr_provider import (
                PaddleOCRProvider,
            )

            ocr_provider = PaddleOCRProvider(lang=source_lang, use_gpu=False)
            logger.info("OCR: PaddleOCR")
        except Exception:
            logger.warning("PaddleOCR unavailable")

    if ocr_provider is None:
        logger.warning("OCR: simulation mode (no real OCR)")

    ocr_service = BatchOCRService(
        provider=ocr_provider,
        normalize=normalize,
        cache=cache,
        postprocess=postprocess,
    )

    # === 3. Select LLM (Ollama / SenseNova / simulation) ===
    llm_client = None

    if use_ollama:
        from comic_translator.clients.ollama_client import OllamaClient

        try:
            client = OllamaClient(model="qwen3.5:9b", timeout=600)
            if await client.is_available():
                llm_client = client
                logger.info("LLM: Ollama (qwen3.5:9b)")
            else:
                logger.warning("Ollama not available")
        except Exception as e:
            logger.warning(f"Ollama: {e}")

    if llm_client is None and api_base:
        from comic_translator.clients.openai_compatible_client import (
            OpenAICompatibleClient,
        )

        try:
            client = OpenAICompatibleClient(
                model=api_model,
                base_url=api_base,
                api_key=api_key or "",
                timeout=300,
            )
            # Quick health check
            await client.generate("ping")
            llm_client = client
            logger.info(f"LLM: OpenAI-compatible ({api_base})")
        except Exception as e:
            logger.warning(f"API client failed: {e}")

    if llm_client is None:
        logger.warning("LLM: simulation mode (no translation)")

    # === 4. Run pipeline ===
    if llm_client:
        pipeline = BatchTranslationService(
            llm_client=llm_client,
            ocr_service=ocr_service,
            prompt_builder=prompt_builder,
            cache=cache,
            glossary_service=glossary,
            db_path=db_path,
        )

        batch_outputs = await pipeline.run(
            project_name=project_name,
            comic_dir=comic_dir,
            batch_size=batch_size,
            source_lang=source_lang,
            target_lang=target_lang,
            comic_style=comic_style,
            use_cache=True,
        )
    else:
        # Simulation: just run OCR
        all_pages = await ocr_service.process(
            project_id=project_name,
            comic_dir=comic_dir,
            source_lang=source_lang,
            use_cache=True,
        )
        logger.info(f"Simulation: OCR completed ({len(all_pages)} pages)")
        batch_outputs = []

    # === 5. Write output JSON ===
    out_dir = Path(output_dir) / project_name
    out_dir.mkdir(parents=True, exist_ok=True)

    all_translations = []
    if llm_client:
        for bo in batch_outputs:
            for pt in bo.pages:
                page_data = {
                    "page": pt.page,
                    "translations": [
                        {
                            "source": b.source_text,
                            "translated": b.translated_text,
                            "corrected": b.source_text,
                        }
                        for b in pt.blocks
                    ],
                }
                all_translations.append(page_data)

        # Also write glossary
        glossary_entries = glossary.get_all_entries()
        glossary_data = {g.source: g.target for g in glossary_entries}
        (out_dir / "glossary.json").write_text(
            json.dumps(glossary_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Glossary saved: {len(glossary_entries)} entries")

    result_file = out_dir / "all_translations.json"
    result_file.write_text(
        json.dumps(all_translations, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    logger.info(f"\nResults: {result_file}")
    logger.info(f"DB: {db_path}")
    logger.info(f"Cache: {cache_dir}/{project_name}/")

    # === 6. 可选：生成质检报告 ===
    try:
        from comic_translator.services.quality_orchestrator import QualityOrchestrator

        orchestrator = QualityOrchestrator(output_dir=output_dir)
        summary = orchestrator.aggregate_report(project_name=project_name)
        report_path = Path(output_dir) / "quality_report.json"
        orchestrator.save_report(summary, str(report_path))
        # 只打印摘要，不阻塞流程
        if summary.needs_manual_review > 0 or summary.overall_pass_rate < 90:
            logger.warning(
                f"Quality: {summary.overall_pass_rate}% pass rate, "
                f"{summary.needs_manual_review} pages need manual review. "
                f"See {report_path}"
            )
        else:
            logger.info(f"Quality: {summary.overall_pass_rate}% pass rate")
    except Exception as e:
        logger.debug(f"Quality report generation skipped: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Batch Comic Translation Pipeline")
    parser.add_argument("--project", default="batch_project", help="项目名称")
    parser.add_argument("--batch-size", type=int, default=20, help="每批页数")
    parser.add_argument("--style", default="western_comic", choices=["western_comic", "manga", "webtoon"])
    parser.add_argument("--ollama", action="store_true", default=True)
    parser.add_argument("--no-ollama", action="store_false", dest="ollama")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--db", default="data/output/project.db")
    parser.add_argument("--api-base", default="", help="SenseNova/OpenAI API 地址")
    parser.add_argument("--api-key", default="", help="API Key（也可设 SENSENOVA_API_KEY 环境变量）")
    parser.add_argument("--api-model", default="", help="API 模型名（默认读取 config.yaml）")
    args = parser.parse_args()

    # 读取 config.yaml，CLI 参数优先
    cfg = load_config()
    llm_cfg = cfg.get("llm", {})
    ocr_cfg = cfg.get("ocr", {})
    trans_cfg = cfg.get("translation", {})
    proj_cfg = cfg.get("project", {})

    # 合并参数: config.yaml → 环境变量 → CLI 参数
    project_name = args.project or proj_cfg.get("name", "batch_project")
    batch_size = args.batch_size or trans_cfg.get("batch_size", 20)
    style = args.style or trans_cfg.get("style", "western_comic")
    db_path = args.db or proj_cfg.get("db", "data/output/project.db")
    cache_dir = proj_cfg.get("cache_dir", "data/cache")
    output_dir = proj_cfg.get("output_dir", "data/output")

    # mode: config.yaml 优先，CLI --ollama/--no-ollama 可覆盖
    mode = llm_cfg.get("mode", "")
    if mode == "api":
        use_ollama = False
    elif mode == "ollama":
        use_ollama = True
    else:
        use_ollama = args.ollama  # 默认 True
    api_base = args.api_base or llm_cfg.get("api", {}).get("base_url", "")
    api_model = args.api_model or llm_cfg.get("api", {}).get("model", "SenseNova-v6")
    api_key = (args.api_key
               or os.environ.get("SENSENOVA_API_KEY", "")
               or llm_cfg.get("api", {}).get("api_key", ""))

    asyncio.run(run_pipeline(
        project_name=project_name,
        batch_size=batch_size,
        comic_style=style,
        use_ollama=use_ollama,
        api_base=api_base,
        api_key=api_key,
        api_model=api_model,
        source_lang=args.source_lang,
        db_path=db_path,
        cache_dir=cache_dir,
        output_dir=output_dir,
    ))