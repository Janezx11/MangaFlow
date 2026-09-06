"""核心管线收敛后的行为测试

覆盖:
- ComicPipelineService.select_page_range 页码范围选择（resume 语义：页码保留全列表编号）
- ProjectService.update_progress / resume 进度持久化闭环（CLI run 每页持久化的数据来源）
"""

from __future__ import annotations

import functools
from pathlib import Path

import pytest

from comic_translator.core.pipeline import ComicPipelineService


def _images(n: int) -> list[Path]:
    return [Path(f"{i}.jpg") for i in range(1, n + 1)]


class TestSelectPageRange:
    def test_full_range_preserves_numbering(self):
        pages = ComicPipelineService.select_page_range(_images(5))
        assert [p for p, _ in pages] == [1, 2, 3, 4, 5]
        assert pages[0][1] == Path("1.jpg")

    def test_resume_from_middle_keeps_original_numbers(self):
        pages = ComicPipelineService.select_page_range(_images(5), start_page=3)
        assert [p for p, _ in pages] == [3, 4, 5]

    def test_end_page_inclusive(self):
        pages = ComicPipelineService.select_page_range(_images(10), start_page=2, end_page=4)
        assert [p for p, _ in pages] == [2, 3, 4]

    def test_start_below_one_clamped(self):
        pages = ComicPipelineService.select_page_range(_images(3), start_page=0)
        assert [p for p, _ in pages] == [1, 2, 3]

    def test_range_beyond_images_returns_empty(self):
        assert ComicPipelineService.select_page_range(_images(3), start_page=5) == []

    def test_end_none_means_to_the_end(self):
        pages = ComicPipelineService.select_page_range(_images(4), start_page=2)
        assert [p for p, _ in pages] == [2, 3, 4]


class TestProgressPersistence:
    """CLI run 每页调用 update_progress；resume 读取 pages_completed 续跑"""

    @pytest.fixture()
    def project_service(self, tmp_path, monkeypatch):
        import comic_translator.storage.repositories as repos
        from comic_translator.storage.database import get_session, init_db

        db = str(tmp_path / "proj.db")
        init_db(db)
        monkeypatch.setattr(repos, "get_session", functools.partial(get_session, db))
        return ProjectServiceWithTmp(tmp_path)

    def test_update_then_resume(self, project_service, tmp_path):
        import asyncio

        # 2 张假图片 → total_pages=2
        (tmp_path / "pages").mkdir()
        (tmp_path / "pages" / "1.jpg").write_bytes(b"x")
        (tmp_path / "pages" / "2.jpg").write_bytes(b"x")

        project = asyncio.run(project_service.create_project(
            name="p1", input_dir=str(tmp_path / "pages"),
        ))
        assert project.metadata.total_pages == 2
        assert project_service.resume(project.id) == 1  # 初始从第 1 页

        # 模拟 CLI run 完成第 1 页后的持久化
        asyncio.run(project_service.update_progress(project.id, 1))
        project = asyncio.run(project_service.load_project(project.id))
        assert project.pages_completed == 1
        assert project_service.resume(project.id) == 2

        # 完成第 2 页后 resume 应判定已完跑（CLI 侧 next_page > total_images）
        asyncio.run(project_service.update_progress(project.id, 2))
        assert project_service.resume(project.id) == 3


def ProjectServiceWithTmp(tmp_path):  # noqa: N802 — 工厂函数，保持 fixture 可读
    from comic_translator.services.project_service import ProjectService
    from comic_translator.storage.repositories import ProjectRepository

    return ProjectService(repository=ProjectRepository())
