from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from comic_translator.storage.database import get_session, get_session_local, init_db
from comic_translator.storage.models import GlossaryEntryORM

if TYPE_CHECKING:
    from comic_translator.models.glossary import GlossaryEntry


class GlossaryService:
    """术语表服务 —— SQLite 持久化

    支持 CRUD、搜索、批量导入。
    与 GlossaryMatcher 配合实现上下文匹配。
    """

    def __init__(self, db_path: str = "data/comic_translator.db") -> None:
        self._db_path = db_path
        init_db(db_path)

    # --- CRUD ---

    def add_term(self, entry: GlossaryEntry, project_id: str | None = None) -> int:
        """添加词条，返回 ID"""
        session = get_session_local(self._db_path)()
        try:
            orm = GlossaryEntryORM(
                source=entry.source,
                target=entry.target,
                context=entry.context or "",
                category=entry.category,
                priority=entry.priority,
                project_id=project_id or "",
            )
            session.add(orm)
            session.commit()
            return orm.id
        finally:
            session.close()

    def get_term(self, term_id: int) -> GlossaryEntry | None:
        session = get_session_local(self._db_path)()
        try:
            orm = session.query(GlossaryEntryORM).filter_by(id=term_id).first()
            if orm is None:
                return None
            from comic_translator.models.glossary import GlossaryEntry

            return GlossaryEntry(
                source=orm.source,
                target=orm.target,
                context=orm.context or None,
                category=orm.category,
                priority=orm.priority,
            )
        finally:
            session.close()

    def update_term(self, term_id: int, entry: GlossaryEntry) -> bool:
        session = get_session_local(self._db_path)()
        try:
            orm = session.query(GlossaryEntryORM).filter_by(id=term_id).first()
            if orm is None:
                return False
            orm.source = entry.source
            orm.target = entry.target
            orm.context = entry.context or ""
            orm.category = entry.category
            orm.priority = entry.priority
            session.commit()
            return True
        finally:
            session.close()

    def delete_term(self, term_id: int) -> bool:
        session = get_session_local(self._db_path)()
        try:
            orm = session.query(GlossaryEntryORM).filter_by(id=term_id).first()
            if orm is None:
                return False
            session.delete(orm)
            session.commit()
            return True
        finally:
            session.close()

    # --- Search ---

    def search_terms(self, query: str, category: str | None = None, project_id: str | None = None) -> list[GlossaryEntry]:
        """按源文本搜索，支持按分类和项目筛选"""
        session = get_session_local(self._db_path)()
        try:
            q = session.query(GlossaryEntryORM).filter(
                GlossaryEntryORM.source.ilike(f"%{query}%")
            )
            if category:
                q = q.filter(GlossaryEntryORM.category == category)
            if project_id:
                q = q.filter(GlossaryEntryORM.project_id == project_id)
            rows = q.order_by(GlossaryEntryORM.priority.desc()).all()
            from comic_translator.models.glossary import GlossaryEntry

            return [
                GlossaryEntry(
                    source=r.source,
                    target=r.target,
                    context=r.context or None,
                    category=r.category,
                    priority=r.priority,
                )
                for r in rows
            ]
        finally:
            session.close()

    def get_all_entries(self, category: str | None = None, project_id: str | None = None) -> list[GlossaryEntry]:
        """获取词条，支持按项目和分类筛选"""
        session = get_session_local(self._db_path)()
        try:
            q = session.query(GlossaryEntryORM)
            if category:
                q = q.filter(GlossaryEntryORM.category == category)
            if project_id:
                q = q.filter(GlossaryEntryORM.project_id == project_id)
            rows = q.order_by(GlossaryEntryORM.priority.desc()).all()
            from comic_translator.models.glossary import GlossaryEntry

            return [
                GlossaryEntry(
                    source=r.source,
                    target=r.target,
                    context=r.context or None,
                    category=r.category,
                    priority=r.priority,
                )
                for r in rows
            ]
        finally:
            session.close()

    def list_with_ids(
        self,
        category: str | None = None,
        project_id: str | None = None,
    ) -> list[tuple[int, GlossaryEntry]]:
        """获取词条及 ORM id，用于 GUI 列表展示与删除/更新定位"""
        session = get_session_local(self._db_path)()
        try:
            q = session.query(GlossaryEntryORM)
            if category:
                q = q.filter(GlossaryEntryORM.category == category)
            if project_id:
                q = q.filter(GlossaryEntryORM.project_id == project_id)
            rows = q.order_by(GlossaryEntryORM.priority.desc()).all()
            from comic_translator.models.glossary import GlossaryEntry

            return [
                (
                    r.id,
                    GlossaryEntry(
                        source=r.source,
                        target=r.target,
                        context=r.context or None,
                        category=r.category,
                        priority=r.priority,
                    ),
                )
                for r in rows
            ]
        finally:
            session.close()

    def count(self, project_id: str | None = None) -> int:
        session = get_session_local(self._db_path)()
        try:
            q = session.query(GlossaryEntryORM)
            if project_id:
                q = q.filter(GlossaryEntryORM.project_id == project_id)
            return q.count()
        finally:
            session.close()

    # --- Import ---

    def import_from_file(self, path: str) -> int:
        """从 glossary 文件导入，返回导入条目数"""
        import json

        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Glossary file not found: {path}")

        data = json.loads(p.read_text(encoding="utf-8"))
        count = 0
        for item in data:
            from comic_translator.models.glossary import GlossaryEntry

            entry = GlossaryEntry(
                source=item["source"],
                target=item["target"],
                context=item.get("context"),
                category=item.get("category", "general"),
                priority=item.get("priority", 0),
            )
            self.add_term(entry)
            count += 1
        return count