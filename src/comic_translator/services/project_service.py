from __future__ import annotations

import uuid
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger

from comic_translator.models.enums import ComicStyle, ProcessingStatus
from comic_translator.models.project import ComicMetadata, ComicProject

if TYPE_CHECKING:
    from comic_translator.storage.repositories import ProjectRepository


class ProjectService:
    """项目管理服务 —— 创建/加载/恢复项目"""

    def __init__(self, repository: ProjectRepository | None = None) -> None:
        self._repo = repository

    async def create_project(
        self,
        name: str,
        input_dir: str,
        title: str = "",
        comic_style: str = "western_comic",
        source_language: str = "en",
        target_language: str = "zh",
        output_dir: str = "data/output",
        cache_dir: str = "data/cache",
    ) -> ComicProject:
        """创建新项目"""
        # 计算图片数量
        input_path = Path(input_dir)
        total_pages = 0
        if input_path.exists():
            exts = ("*.png", "*.jpg", "*.jpeg", "*.webp", "*.bmp")
            for ext in exts:
                total_pages += len(list(input_path.glob(ext)))

        metadata = ComicMetadata(
            title=title or name,
            total_pages=total_pages,
            comic_style=ComicStyle(comic_style) if isinstance(comic_style, str) else comic_style,
            source_language=source_language,
            target_language=target_language,
        )

        project = ComicProject(
            id=str(uuid.uuid4()),
            name=name,
            metadata=metadata,
            input_dir=input_dir,
            output_dir=output_dir,
            cache_dir=cache_dir,
            status=ProcessingStatus.PENDING,
            pages_completed=0,
        )

        if self._repo:
            project = self._repo.save(project)
            logger.info(f"Created project '{name}' (id={project.id}, {total_pages} pages)")

        return project

    async def load_project(self, project_id: str) -> ComicProject | None:
        """加载项目"""
        if self._repo:
            return self._repo.find_by_id(project_id)
        raise NotImplementedError("Repository not configured")

    async def update_progress(self, project_id: str, page: int) -> None:
        """更新项目进度"""
        if self._repo:
            self._repo.update_progress(project_id, page)
        else:
            raise NotImplementedError("Repository not configured")

    def get_page_paths(self, project: ComicProject) -> list[str]:
        """获取项目所有图片路径（自然排序，页码与阅读顺序一致）"""
        input_dir = Path(project.input_dir)
        if not input_dir.exists():
            logger.warning(f"Input directory not found: {project.input_dir}")
            return []

        from comic_translator.utils.natural_sort import natural_key

        exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
        files = [p for p in input_dir.iterdir() if p.suffix.lower() in exts]
        return [str(p) for p in sorted(files, key=natural_key)]

    def resume(self, project_id: str) -> int:
        """查找需要继续的页码

        Returns:
            下一个未完成的页码（1-based），如果已完成则返回 total_pages + 1
        """
        if self._repo is None:
            raise NotImplementedError("Repository not configured")

        project = self._repo.find_by_id(project_id)
        if project is None:
            logger.error(f"Project not found: {project_id}")
            return 1

        return project.pages_completed + 1

    async def list_projects(self) -> list[ComicProject]:
        """列出所有项目"""
        if self._repo:
            return self._repo.find_all()
        return []

    async def delete_project(self, project_id: str) -> bool:
        """删除项目"""
        if self._repo:
            return self._repo.delete(project_id)
        return False
