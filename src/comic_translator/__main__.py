import asyncio
from pathlib import Path

import typer
from loguru import logger

from comic_translator.utils.di import get_container, create_llm_client_from_config, _load_config, _get_config_value

app = typer.Typer(name="comic-translator")


@app.command()
def run(
    project_id: str = typer.Option(..., "--project", "-p", help="Project ID"),
    start_page: int = typer.Option(1, "--start", "-s", help="Start page number"),
    end_page: int | None = typer.Option(None, "--end", "-e", help="End page number"),
    batch_size: int = typer.Option(20, "--batch-size", "-b", help="Pages per batch"),
    use_cache: bool = typer.Option(True, "--cache/--no-cache", help="Use cache"),
    batch_delay: float = typer.Option(0.0, "--batch-delay", help="Seconds between batch API calls (0 = no artificial wait)"),
) -> None:
    """Run translation workflow for a project"""
    container = get_container()
    project_service = container.get("project_service")
    llm_client = container.get_llm_client()

    project = asyncio.run(project_service.load_project(project_id))
    if project is None:
        logger.error(f"Project not found: {project_id}")
        raise typer.Exit(1)

    logger.info(f"Starting translation for project {project.name} ({project.metadata.title})")
    logger.info(f"  Pages: {start_page}-{end_page or project.metadata.total_pages}")
    logger.info(f"  Style: {project.metadata.comic_style}")

    # 使用 BatchTranslationService 运行翻译管线
    from comic_translator.services.batch_translation_service import BatchTranslationService
    from comic_translator.services.batch_ocr_service import BatchOCRService
    from comic_translator.services.batch_prompt_builder import BatchPromptBuilder
    from comic_translator.services.batch_cache import BatchCacheService
    from comic_translator.services.glossary_service import GlossaryService
    from comic_translator.services.ocr_normalize_service import OCRNormalizeService

    config = _load_config()
    cache_dir = _get_config_value(config, "project.cache_dir", "data/cache")
    batch_cache = BatchCacheService(cache_dir=cache_dir)
    ocr_normalize = OCRNormalizeService()
    ocr_service = BatchOCRService(
        provider=container.get("ocr_provider"),
        normalize=ocr_normalize,
        cache=batch_cache,
    )
    prompt_builder = BatchPromptBuilder()
    glossary_service = container.get("glossary_service")

    translation_service = BatchTranslationService(
        llm_client=llm_client,
        ocr_service=ocr_service,
        prompt_builder=prompt_builder,
        cache=batch_cache,
        glossary_service=glossary_service,
        db_path=container._db_path,
    )

    asyncio.run(translation_service.run(
        project_name=project.name,
        comic_dir=project.input_dir,
        batch_size=batch_size,
        source_lang=project.metadata.source_language,
        target_lang=project.metadata.target_language,
        comic_style=project.metadata.comic_style.value if hasattr(project.metadata.comic_style, 'value') else project.metadata.comic_style,
        use_cache=use_cache,
        batch_delay=batch_delay,
    ))

    logger.info("Translation pipeline completed!")


@app.command()
def init(
    name: str = typer.Argument(..., help="Project name"),
    input_dir: str = typer.Option("data/comics", "--input", "-i", help="Input directory with comic images"),
    title: str = typer.Option("", "--title", "-t", help="Comic title"),
    style: str = typer.Option("western_comic", "--style", help="Comic style (western_comic/manga/webtoon)"),
    source_lang: str = typer.Option("en", "--source-lang", "-sl", help="Source language"),
    target_lang: str = typer.Option("zh", "--target-lang", "-tl", help="Target language"),
    output_dir: str = typer.Option("data/output", "--output", "-o", help="Output directory"),
) -> None:
    """Initialize a new comic translation project"""
    container = get_container()
    project_service = container.get("project_service")

    project = asyncio.run(project_service.create_project(
        name=name,
        input_dir=input_dir,
        title=title or name,
        comic_style=style,
        source_language=source_lang,
        target_language=target_lang,
        output_dir=output_dir,
    ))

    logger.info(f"Project '{name}' created successfully!")
    logger.info(f"  ID: {project.id}")
    logger.info(f"  Input: {input_dir}")
    logger.info(f"  Pages: {project.metadata.total_pages}")
    logger.info(f"  Style: {style}")
    logger.info(f"  Translation: {source_lang} → {target_lang}")
    logger.info(f"")
    logger.info(f"To start translation, run:")
    logger.info(f"  python -m comic_translator run --project {project.id}")


@app.command()
def resume(
    project_id: str = typer.Argument(..., help="Project ID"),
    batch_size: int = typer.Option(20, "--batch-size", "-b", help="Pages per batch"),
) -> None:
    """Resume an interrupted translation project"""
    container = get_container()
    project_service = container.get("project_service")

    project = asyncio.run(project_service.load_project(project_id))
    if project is None:
        logger.error(f"Project not found: {project_id}")
        raise typer.Exit(1)

    next_page = project_service.resume(project_id)
    logger.info(f"Resuming project {project.name} from page {next_page}")

    if next_page > project.metadata.total_pages:
        logger.info("Project already completed!")
        return

    # 调用 run 从 next_page 开始
    run(project_id=project_id, start_page=next_page, batch_size=batch_size)


@app.command()
def list_projects() -> None:
    """List all projects"""
    container = get_container()
    project_service = container.get("project_service")

    projects = asyncio.run(project_service.list_projects())
    if not projects:
        logger.info("No projects found.")
        return

    logger.info(f"Found {len(projects)} project(s):")
    logger.info(f"{'ID':<40} {'Name':<20} {'Pages':<8} {'Status':<12} {'Style':<16}")
    logger.info("-" * 100)
    for p in projects:
        style = p.metadata.comic_style.value if hasattr(p.metadata.comic_style, 'value') else p.metadata.comic_style
        logger.info(f"{p.id:<40} {p.name:<20} {p.metadata.total_pages:<8} {p.status.value:<12} {style:<16}")


if __name__ == "__main__":
    app()