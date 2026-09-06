from __future__ import annotations

from typing import TYPE_CHECKING

from comic_translator.storage.database import get_session_local, init_db
from comic_translator.storage.models import TranslationMemoryORM

if TYPE_CHECKING:
    from comic_translator.models.glossary import TranslationMemory


class TranslationMemoryService:
    """翻译记忆服务 —— 存储和查询历史翻译

    用于跨页面一致性翻译。
    """

    def __init__(self, db_path: str = "data/comic_translator.db") -> None:
        self._db_path = db_path
        init_db(db_path)

    def store(
        self,
        source_text: str,
        translated_text: str,
        provider: str = "",
        model: str = "",
        context_hash: str = "",
        project_id: str | None = None,
    ) -> int:
        """存储一条翻译记忆，返回 ID"""
        session = get_session_local(self._db_path)()
        try:
            orm = TranslationMemoryORM(
                source_text=source_text,
                translated_text=translated_text,
                context_hash=context_hash,
                provider=provider,
                model=model,
                project_id=project_id or "",
                usage_count=0,
            )
            session.add(orm)
            session.commit()
            return orm.id
        finally:
            session.close()

    def find_similar(self, source_text: str, limit: int = 5) -> list[TranslationMemory]:
        """查询相似翻译

        优先级：精确匹配 → LIKE 模糊匹配
        """
        session = get_session_local(self._db_path)()
        try:
            from comic_translator.models.glossary import TranslationMemory

            results: list[TranslationMemory] = []

            # Phase 1: exact match
            exact = (
                session.query(TranslationMemoryORM)
                .filter(TranslationMemoryORM.source_text == source_text)
                .order_by(TranslationMemoryORM.usage_count.desc())
                .first()
            )
            if exact:
                tm = TranslationMemory(
                    source_text=exact.source_text,
                    translated_text=exact.translated_text,
                    context_hash=exact.context_hash,
                    similarity=1.0,
                    provider=exact.provider or "",
                    model=exact.model or "",
                    usage_count=exact.usage_count,
                )
                results.append(tm)
                # Bump usage count
                exact.usage_count = (exact.usage_count or 0) + 1
                session.commit()

            if len(results) >= limit:
                return results

            # Phase 2: LIKE match (contains)
            like_matches = (
                session.query(TranslationMemoryORM)
                .filter(
                    TranslationMemoryORM.source_text != source_text,
                    TranslationMemoryORM.source_text.like(f"%{source_text}%"),
                )
                .order_by(TranslationMemoryORM.usage_count.desc())
                .limit(limit - len(results))
                .all()
            )
            for orm in like_matches:
                sim = len(source_text) / max(len(orm.source_text), 1)
                tm = TranslationMemory(
                    source_text=orm.source_text,
                    translated_text=orm.translated_text,
                    context_hash=orm.context_hash,
                    similarity=sim,
                    provider=orm.provider or "",
                    model=orm.model or "",
                    usage_count=orm.usage_count,
                )
                results.append(tm)

            return results
        finally:
            session.close()

    def get_stats(self) -> dict:
        """获取 TM 统计信息"""
        session = get_session_local(self._db_path)()
        try:
            total = session.query(TranslationMemoryORM).count()
            return {"total_entries": total}
        finally:
            session.close()

    def list_entries(self, project_id: str | None = None, limit: int = 500) -> list[dict]:
        """获取翻译记忆列表（含 id），供 GUI 展示"""
        session = get_session_local(self._db_path)()
        try:
            q = session.query(TranslationMemoryORM)
            if project_id:
                q = q.filter(TranslationMemoryORM.project_id == project_id)
            rows = q.order_by(TranslationMemoryORM.id.desc()).limit(limit).all()
            return [
                {
                    "id": r.id,
                    "project_id": r.project_id or "",
                    "source_text": r.source_text,
                    "translated_text": r.translated_text,
                    "provider": r.provider or "",
                    "model": r.model or "",
                    "usage_count": r.usage_count or 0,
                    "created_at": r.created_at.isoformat() if r.created_at else "",
                }
                for r in rows
            ]
        finally:
            session.close()

    def delete_entry(self, entry_id: int) -> bool:
        """删除一条翻译记忆"""
        session = get_session_local(self._db_path)()
        try:
            orm = session.query(TranslationMemoryORM).filter_by(id=entry_id).first()
            if orm is None:
                return False
            session.delete(orm)
            session.commit()
            return True
        finally:
            session.close()
