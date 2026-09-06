import asyncio

import typer
from loguru import logger

from comic_translator.utils.di import get_container

app = typer.Typer(name="comic-translator")


@app.callback()
def _configure_logging() -> None:
    """所有子命令共享的日志初始化（级别/文件可经 CT_LOG_LEVEL、CT_LOG_FILE 覆盖）"""
    from comic_translator.config.settings import Settings
    from comic_translator.utils.logging import setup_logging

    settings = Settings()
    setup_logging(level=settings.log_level, log_file=settings.log_file)


@app.command()
def run(
    project_id: str = typer.Option(..., "--project", "-p", help="Project ID"),
    start_page: int = typer.Option(1, "--start", "-s", help="Start page number (1-based)"),
    end_page: int | None = typer.Option(None, "--end", "-e", help="End page number (inclusive)"),
    output_dir: str = typer.Option("data/output/rendered", "--output", "-o", help="Rendered images output dir"),
    target_lang: str = typer.Option("zh", "--target-lang", "-tl", help="Target language"),
) -> None:
    """运行完整翻译流水线：检测 → OCR → 翻译 → 擦除 → 渲染，输出翻译后的图片"""
    _run_full_pipeline(project_id, start_page, end_page, output_dir, target_lang)


def _run_full_pipeline(
    project_id: str,
    start_page: int,
    end_page: int | None,
    output_dir: str,
    target_lang: str,
) -> None:
    """CLI 共享的完整管线执行器：消费进度生成器、打日志、持久化页码进度"""
    container = get_container()
    project_service = container.get("project_service")

    project = asyncio.run(project_service.load_project(project_id))
    if project is None:
        logger.error(f"Project not found: {project_id}")
        raise typer.Exit(1)

    total_images = len(project_service.get_page_paths(project))
    logger.info(f"Starting pipeline for project {project.name} ({project.metadata.title})")
    logger.info(f"  Pages: {start_page}-{end_page or 'end'} (total images: {total_images})")
    logger.info(f"  Style: {project.metadata.comic_style} | Output: {output_dir}")

    from comic_translator.core.pipeline import ComicPipelineService

    pipeline = ComicPipelineService(history_service=None)

    done_pages = 0
    failed_pages = 0
    for p in pipeline.run_pipeline(
        input_dir=project.input_dir,
        target_lang=target_lang,
        output_dir=output_dir,
        start_page=start_page,
        end_page=end_page,
    ):
        if p.message:
            logger.info(f"[{p.stage}] {p.message}")
        # 每页成功完成后持久化进度 —— 这是 resume 断点续跑的数据来源
        if p.stage == "done" and p.current_page > 0 and p.page_result is not None:
            if p.page_result.error:
                failed_pages += 1
            else:
                done_pages += 1
                asyncio.run(project_service.update_progress(project_id, p.page))

    logger.info(
        f"Pipeline completed: {done_pages} page(s) rendered, {failed_pages} failed. "
        f"Images → {output_dir}"
    )


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
    logger.info("")
    logger.info("To start translation, run:")
    logger.info(f"  comic-translator run --project {project.id}")


@app.command()
def resume(
    project_id: str = typer.Argument(..., help="Project ID"),
    output_dir: str = typer.Option("data/output/rendered", "--output", "-o", help="Rendered images output dir"),
    target_lang: str = typer.Option("zh", "--target-lang", "-tl", help="Target language"),
) -> None:
    """从上次完成的页码继续翻译（进度按页持久化于 run 过程中）"""
    container = get_container()
    project_service = container.get("project_service")

    project = asyncio.run(project_service.load_project(project_id))
    if project is None:
        logger.error(f"Project not found: {project_id}")
        raise typer.Exit(1)

    next_page = project_service.resume(project_id)
    total_images = len(project_service.get_page_paths(project))
    if next_page > total_images:
        logger.info(f"Project already completed ({total_images}/{total_images} pages)!")
        return

    logger.info(f"Resuming project {project.name} from page {next_page} ({total_images} total)")
    _run_full_pipeline(project_id, next_page, None, output_dir, target_lang)


@app.command()
def doctor() -> None:
    """环境体检：Python / 依赖 / 模型 / 配置 / 数据目录，冷启动自检"""
    from comic_translator.utils.environment_check import run_doctor

    report = run_doctor()
    for line in report.to_lines():
        logger.info(line)

    failed = len(report.failures)
    warned = len(report.warnings)
    if failed:
        logger.error(f"体检未通过: {failed} 项失败, {warned} 项警告")
        raise typer.Exit(1)
    logger.info(f"体检通过（{warned} 项警告）—— 可以开始翻译了")


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


