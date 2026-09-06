from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from sqlalchemy import select

from comic_translator.models.enums import ComicStyle, ProcessingStatus
from comic_translator.models.project import ComicMetadata, ComicProject
from comic_translator.storage.database import get_session
from comic_translator.storage.models import ProjectORM

if TYPE_CHECKING:
    pass


class ProjectRepository:
    """Project 持久化仓库 —— 使用 SQLAlchemy ORM"""

    def _orm_to_domain(self, orm: ProjectORM) -> ComicProject:
        """ORM → Domain model"""
        metadata = ComicMetadata(
            title=orm.title or orm.name,
            total_pages=orm.total_pages,
            comic_style=ComicStyle(orm.comic_style) if orm.comic_style else ComicStyle.WESTERN_COMIC,
            source_language=orm.source_language or "en",
            target_language=orm.target_language or "zh",
        )
        return ComicProject(
            id=orm.id,
            name=orm.name,
            metadata=metadata,
            input_dir=orm.input_dir or "",
            output_dir=orm.output_dir or "",
            cache_dir=orm.cache_dir or "",
            status=ProcessingStatus(orm.status) if orm.status else ProcessingStatus.PENDING,
            pages_completed=orm.pages_completed or 0,
            created_at=orm.created_at or datetime.now(timezone.utc),
            updated_at=orm.updated_at or datetime.now(timezone.utc),
        )

    def _domain_to_orm(self, project: ComicProject, orm: ProjectORM | None = None) -> ProjectORM:
        """Domain model → ORM (upsert)"""
        if orm is None:
            orm = ProjectORM(id=project.id)

        orm.id = project.id
        orm.name = project.name
        orm.title = project.metadata.title
        orm.comic_style = project.metadata.comic_style.value if isinstance(project.metadata.comic_style, ComicStyle) else project.metadata.comic_style
        orm.source_language = project.metadata.source_language
        orm.target_language = project.metadata.target_language
        orm.total_pages = project.metadata.total_pages
        orm.pages_completed = project.pages_completed
        orm.status = project.status.value if isinstance(project.status, ProcessingStatus) else project.status
        orm.input_dir = project.input_dir
        orm.output_dir = project.output_dir
        orm.cache_dir = project.cache_dir
        orm.created_at = project.created_at
        orm.updated_at = datetime.now(timezone.utc)
        return orm

    def save(self, project: ComicProject) -> ComicProject:
        """保存项目（创建或更新）"""
        session = next(get_session())
        try:
            existing = session.execute(
                select(ProjectORM).where(ProjectORM.id == project.id)
            ).scalar_one_or_none()

            if existing:
                orm = self._domain_to_orm(project, existing)
            else:
                orm = self._domain_to_orm(project)

            session.add(orm)
            session.commit()
            return self._orm_to_domain(orm)
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def find_by_id(self, project_id: str) -> ComicProject | None:
        """按 ID 查找项目"""
        session = next(get_session())
        try:
            orm = session.execute(
                select(ProjectORM).where(ProjectORM.id == project_id)
            ).scalar_one_or_none()
            if orm is None:
                return None
            return self._orm_to_domain(orm)
        finally:
            session.close()

    def find_all(self) -> list[ComicProject]:
        """查找所有项目"""
        session = next(get_session())
        try:
            orms = session.execute(
                select(ProjectORM).order_by(ProjectORM.updated_at.desc())
            ).scalars().all()
            return [self._orm_to_domain(orm) for orm in orms]
        finally:
            session.close()

    def delete(self, project_id: str) -> bool:
        """删除项目"""
        session = next(get_session())
        try:
            orm = session.execute(
                select(ProjectORM).where(ProjectORM.id == project_id)
            ).scalar_one_or_none()
            if orm is None:
                return False
            session.delete(orm)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def update_progress(self, project_id: str, pages_completed: int) -> ComicProject | None:
        """更新项目进度"""
        session = next(get_session())
        try:
            orm = session.execute(
                select(ProjectORM).where(ProjectORM.id == project_id)
            ).scalar_one_or_none()
            if orm is None:
                return None
            orm.pages_completed = pages_completed
            orm.updated_at = datetime.now(timezone.utc)
            if pages_completed >= (orm.total_pages or 0):
                orm.status = "completed"
            elif pages_completed > 0:
                orm.status = "processing"
            session.commit()
            return self._orm_to_domain(orm)
        finally:
            session.close()


class TranslationMemoryRepository:
    """Translation Memory 持久化仓库"""

    def save(self, source: str, target: str, provider: str, model: str) -> None:
        from comic_translator.storage.models import TranslationMemoryORM
        session = next(get_session())
        try:
            orm = TranslationMemoryORM(
                source_text=source,
                translated_text=target,
                provider=provider,
                model=model,
            )
            session.add(orm)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def search(self, source_text: str, limit: int = 5) -> list:
        from comic_translator.storage.models import TranslationMemoryORM
        session = next(get_session())
        try:
            rows = session.execute(
                select(TranslationMemoryORM)
                .where(TranslationMemoryORM.source_text == source_text)
                .order_by(TranslationMemoryORM.usage_count.desc())
                .limit(limit)
            ).scalars().all()
            return rows
        finally:
            session.close()

    def find_similar(self, source_text: str, limit: int = 5) -> list:
        """模糊搜索翻译记忆（LIKE 匹配）"""
        from comic_translator.storage.models import TranslationMemoryORM
        session = next(get_session())
        try:
            rows = session.execute(
                select(TranslationMemoryORM)
                .where(TranslationMemoryORM.source_text.like(f"%{source_text}%"))
                .order_by(TranslationMemoryORM.usage_count.desc())
                .limit(limit)
            ).scalars().all()
            return rows
        finally:
            session.close()

    def update_usage(self, tm_id: int) -> None:
        """更新使用次数"""
        from datetime import datetime, timezone

        from comic_translator.storage.models import TranslationMemoryORM
        session = next(get_session())
        try:
            orm = session.get(TranslationMemoryORM, tm_id)
            if orm:
                orm.usage_count = (orm.usage_count or 0) + 1
                orm.last_used = datetime.now(timezone.utc)
                session.commit()
        finally:
            session.close()

    def count(self) -> int:
        from sqlalchemy import func

        from comic_translator.storage.models import TranslationMemoryORM
        session = next(get_session())
        try:
            return session.execute(
                select(func.count(TranslationMemoryORM.id))
            ).scalar() or 0
        finally:
            session.close()

    def delete(self, tm_id: int) -> bool:
        from comic_translator.storage.models import TranslationMemoryORM
        session = next(get_session())
        try:
            orm = session.get(TranslationMemoryORM, tm_id)
            if orm is None:
                return False
            session.delete(orm)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()


class GlossaryRepository:
    """术语表持久化仓库"""

    def save(self, source: str, target: str, category: str = "general", context: str = "") -> None:
        from comic_translator.storage.models import GlossaryEntryORM
        session = next(get_session())
        try:
            orm = GlossaryEntryORM(
                source=source,
                target=target,
                category=category,
                context=context,
            )
            session.add(orm)
            session.commit()
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def search(self, source: str, category: str | None = None) -> list:
        from comic_translator.storage.models import GlossaryEntryORM
        session = next(get_session())
        try:
            query = select(GlossaryEntryORM).where(GlossaryEntryORM.source == source)
            if category:
                query = query.where(GlossaryEntryORM.category == category)
            return session.execute(query).scalars().all()
        finally:
            session.close()

    def find_all(self, category: str | None = None) -> list:
        from comic_translator.storage.models import GlossaryEntryORM
        session = next(get_session())
        try:
            query = select(GlossaryEntryORM)
            if category:
                query = query.where(GlossaryEntryORM.category == category)
            return session.execute(
                query.order_by(GlossaryEntryORM.source)
            ).scalars().all()
        finally:
            session.close()

    def delete(self, entry_id: int) -> bool:
        from comic_translator.storage.models import GlossaryEntryORM
        session = next(get_session())
        try:
            orm = session.get(GlossaryEntryORM, entry_id)
            if orm is None:
                return False
            session.delete(orm)
            session.commit()
            return True
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def count(self) -> int:
        from sqlalchemy import func

        from comic_translator.storage.models import GlossaryEntryORM
        session = next(get_session())
        try:
            return session.execute(
                select(func.count(GlossaryEntryORM.id))
            ).scalar() or 0
        finally:
            session.close()
