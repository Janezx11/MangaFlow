#!/usr/bin/env python
"""
翻译质检报告 + 自动重跑工具

用法:
  # 生成质检报告
  python scripts/quality_report.py --project my_comic --output report.json

  # 查看报告摘要
  python scripts/quality_report.py --project my_comic --show

  # 自动重跑不合格页面（需要完整 pipeline 环境）
  python scripts/quality_report.py --project my_comic --auto-retry --max-retries 3
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
    p = Path(path)
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def show_report(report_path: str) -> None:
    """显示已有的质检报告"""
    path = Path(report_path)
    if not path.exists():
        logger.error(f"Report not found: {path}")
        return

    data = json.loads(path.read_text(encoding="utf-8"))
    summary = data.get("summary", data)

    print(f"\n{'=' * 55}")
    print(f"  Quality Report: {summary.get('project_name', '(unnamed)')}")
    print(f"{'=' * 55}")
    print(f"  Total pages:        {summary.get('total_pages', 0)}")
    print(f"  Passed:             {summary.get('passed', 0)}")
    print(f"  Overall pass rate:  {summary.get('overall_pass_rate', 0)}%")
    print(f"  After retry:        {summary.get('retry_passed', 0)}")
    print(f"  Manual review:      {summary.get('needs_manual_review', 0)}")
    print(f"  Avg render quality: {summary.get('avg_render_quality', 0)}")
    print(f"  Avg translation:    {summary.get('avg_translation_status', 'N/A')}")
    print(f"{'=' * 55}")

    manual_pages = summary.get("manual_review_pages", [])
    if manual_pages:
        print(f"\n  Pages needing manual review:")
        for p in manual_pages:
            print(f"    #{p['page']:>4}  quality={p.get('render_quality', '?'):<8}  "
                  f"translation={p.get('translation_status', '?'):<8}  "
                  f"reason={p.get('retry_reason', '?')}")

    pages = data.get("pages", [])
    if pages:
        print(f"\n  All pages:")
        print(f"    {'Page':<6} {'Quality':<10} {'Trans':<10} {'Score':<8} {'Retry':<6} Notes")
        print(f"    {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*6} {'-'*20}")
        for p in pages:
            notes = p.get("retry_reason", "")[:20] if p.get("needs_retry") else ""
            print(f"    #{p['page']:<4} {p.get('render_quality', 0):<9} "
                  f"{p.get('translation_status', '?'):<10} "
                  f"{p.get('overall_score', 0):<7} "
                  f"{p.get('retry_count', 0):<5} {notes}")


async def run_auto_retry(
    project_name: str,
    db_path: str = "data/output/project.db",
    cache_dir: str = "data/cache",
    output_dir: str = "data/output",
    comic_dir: str = "data/comics",
    max_retries: int = 3,
    render: bool = True,
    source_lang: str = "en",
    target_lang: str = "zh",
    comic_style: str = "western_comic",
    use_ollama: bool = False,
    api_base: str = "",
    api_key: str = "",
    api_model: str = "SenseNova-v6",
) -> None:
    """自动重跑流程：读取报告 → 重跑不合格页面 → 生成新报告"""
    from comic_translator.services.quality_orchestrator import QualityOrchestrator

    # 1. 先检查已有报告
    report_path = Path(output_dir) / "quality_report.json"
    if not report_path.exists():
        # 尝试从渲染输出生成报告
        logger.info("No existing report found, generating from render output...")
        orchestrator = QualityOrchestrator(output_dir=output_dir)
        summary = orchestrator.aggregate_report(project_name=project_name)
        if summary.total_pages == 0:
            logger.error("No render output found. Run render pipeline first or use --report to specify a report file.")
            return
        orchestrator.save_report(summary, str(report_path))
        orchestrator.print_summary(summary)
    else:
        orchestrator = QualityOrchestrator(output_dir=output_dir)

    # 2. 初始化服务
    from comic_translator.services.ocr_normalize_service import OCRNormalizeService
    from comic_translator.services.glossary_service import GlossaryService
    from comic_translator.services.batch_cache import BatchCacheService
    from comic_translator.services.batch_ocr_service import BatchOCRService
    from comic_translator.services.batch_prompt_builder import BatchPromptBuilder

    glossary = GlossaryService(db_path=db_path)
    normalize = OCRNormalizeService()
    cache = BatchCacheService(cache_dir=cache_dir)
    prompt_builder = BatchPromptBuilder()

    # OCR Provider
    ocr_provider = None
    try:
        from comic_translator.providers.ocr.easyocr_provider import EasyOCRProvider
        from PIL import Image, ImageDraw

        ep = EasyOCRProvider(lang=source_lang, use_gpu=False)
        test_img = Image.new("RGB", (100, 30), color="white")
        ImageDraw.Draw(test_img).text((5, 5), "test", fill="black")
        _ = await ep.detect_text(test_img)
        ocr_provider = ep
        logger.info("OCR: EasyOCR")
    except Exception:
        logger.warning("EasyOCR unavailable")

    if ocr_provider is None:
        try:
            from comic_translator.providers.ocr.paddle_ocr_provider import PaddleOCRProvider

            ocr_provider = PaddleOCRProvider(lang=source_lang, use_gpu=False)
            logger.info("OCR: PaddleOCR")
        except Exception:
            logger.warning("PaddleOCR unavailable")

    if ocr_provider is None:
        logger.error("No OCR provider available")
        return

    ocr_service = BatchOCRService(
        provider=ocr_provider,
        normalize=normalize,
        cache=cache,
    )

    # LLM Client
    llm_client = None
    if use_ollama:
        from comic_translator.clients.ollama_client import OllamaClient

        try:
            client = OllamaClient(model="qwen3.5:9b", timeout=600)
            if await client.is_available():
                llm_client = client
                logger.info("LLM: Ollama (qwen3.5:9b)")
        except Exception as e:
            logger.warning(f"Ollama: {e}")

    if llm_client is None and api_base:
        from comic_translator.clients.openai_compatible_client import OpenAICompatibleClient

        try:
            client = OpenAICompatibleClient(
                model=api_model,
                base_url=api_base,
                api_key=api_key or "",
                timeout=300,
            )
            await client.generate("ping")
            llm_client = client
            logger.info(f"LLM: {api_base}")
        except Exception as e:
            logger.warning(f"API: {e}")

    if llm_client is None:
        logger.error("No LLM client available")
        return

    # Render Service
    render_service = None
    if render:
        try:
            from comic_translator.services.render_quality_service import RenderQualityService
            from comic_translator.services.text_normalizer import TextNormalizer
            from comic_translator.services.font_manager import FontManager
            from comic_translator.services.typography_service_v2 import TypographyServiceV2
            from comic_translator.services.typography_qa import TypographyQA
            from comic_translator.services.render_qa_service import RenderQAService

            with open("config.yaml", encoding="utf-8") as f:
                cfg = yaml.safe_load(f) or {}

            render_service = RenderQualityService(
                text_normalizer=TextNormalizer(),
                font_manager=FontManager(),
                typography_v2=TypographyServiceV2(),
                typography_qa=TypographyQA(),
                qa_service=RenderQAService(),
                render_config=cfg.get("render", {}),
            )
            logger.info("Render service ready")
        except Exception as e:
            logger.warning(f"Render service unavailable (translations only): {e}")

    # 3. 执行自动重跑
    logger.info(f"\nStarting auto-retry for {project_name}...")
    summary = await orchestrator.auto_retry_pages(
        summary=orchestrator.aggregate_report(project_name=project_name),
        project_name=project_name,
        comic_dir=comic_dir,
        ocr_service=ocr_service,
        llm_client=llm_client,
        prompt_builder=prompt_builder,
        glossary_service=glossary,
        cache=cache,
        render_service=render_service,
        source_lang=source_lang,
        target_lang=target_lang,
        comic_style=comic_style,
        render_output_dir=str(Path(output_dir) / "rendered"),
        max_retries=max_retries,
    )

    # 4. 保存报告
    orchestrator.save_report(summary, str(report_path))
    orchestrator.print_summary(summary)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Comic Translation Quality Report")
    parser.add_argument("--project", default="batch_project", help="项目名称")
    parser.add_argument("--output", default="data/output/quality_report.json", help="报告输出路径")
    parser.add_argument("--show", action="store_true", help="显示已有报告")
    parser.add_argument("--auto-retry", action="store_true", help="自动重跑不合格页面")
    parser.add_argument("--max-retries", type=int, default=3, help="每页最大重跑次数")
    parser.add_argument("--no-render", action="store_true", help="重跑时不执行渲染（只翻译）")
    parser.add_argument("--source-lang", default="en")
    parser.add_argument("--db", default="data/output/project.db")
    parser.add_argument("--api-base", default="", help="API 地址")
    parser.add_argument("--api-key", default="", help="API Key")
    parser.add_argument("--api-model", default="", help="API 模型名")
    args = parser.parse_args()

    cfg = load_config()
    llm_cfg = cfg.get("llm", {})
    proj_cfg = cfg.get("project", {})

    project_name = args.project or proj_cfg.get("name", "batch_project")
    db_path = args.db or proj_cfg.get("db", "data/output/project.db")
    cache_dir = proj_cfg.get("cache_dir", "data/cache")
    output_dir = proj_cfg.get("output_dir", "data/output")
    api_base = args.api_base or llm_cfg.get("api", {}).get("base_url", "")
    api_model = args.api_model or llm_cfg.get("api", {}).get("model", "SenseNova-v6")
    api_key = args.api_key or os.environ.get("SENSENOVA_API_KEY", "") or llm_cfg.get("api", {}).get("api_key", "")

    if args.show:
        show_report(args.output)
    elif args.auto_retry:
        asyncio.run(run_auto_retry(
            project_name=project_name,
            db_path=db_path,
            cache_dir=cache_dir,
            output_dir=output_dir,
            max_retries=args.max_retries,
            render=not args.no_render,
            source_lang=args.source_lang,
            api_base=api_base,
            api_key=api_key,
            api_model=api_model,
        ))
    else:
        # 默认：生成报告
        from comic_translator.services.quality_orchestrator import QualityOrchestrator

        orchestrator = QualityOrchestrator(output_dir=output_dir)
        summary = orchestrator.aggregate_report(project_name=project_name)
        orchestrator.save_report(summary, args.output)
        orchestrator.print_summary(summary)