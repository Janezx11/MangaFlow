"""实体服务 —— 管理 OCR 提取的命名实体"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from sqlalchemy.orm import Session


class EntityService:
    """实体管理：OCR 提取的命名实体存入 SQLite，翻译前自动注入"""

    def __init__(self, db_path: str = "data/output/project.db") -> None:
        self._db_path = db_path
        self._init_table()

    def _init_table(self) -> None:
        from comic_translator.storage.database import init_db

        init_db(self._db_path)

    def _get_session(self):
        from comic_translator.storage.database import get_session

        return get_session(self._db_path)

    def record_entities(
        self,
        entities: list[dict],
        page: int,
        project_id: str = "",
    ) -> int:
        """批量记录实体，已有实体增加频次"""
        from comic_translator.storage.database import get_engine
        from comic_translator.storage.models import EntityORM
        from sqlalchemy import select

        count = 0
        session_gen = self._get_session()
        session = next(session_gen)
        try:
            for ent in entities:
                name = ent.get("entity_name", "").strip()
                etype = ent.get("entity_type", "").strip()
                if not name:
                    continue
                existing = session.execute(
                    select(EntityORM).where(EntityORM.entity_name == name)
                ).scalar_one_or_none()
                if existing:
                    existing.frequency = EntityORM.frequency + 1
                    if page > 0:
                        existing.last_page = max(existing.last_page or 0, page)
                else:
                    session.add(EntityORM(
                        entity_name=name,
                        entity_type=etype,
                        frequency=1,
                        first_page=page,
                        last_page=page,
                    ))
                    count += 1
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"Entity record failed: {e}")
        finally:
            session.close()
        return count

    def get_all_entities(self) -> list[dict]:
        """获取所有实体（按频次降序）"""
        from comic_translator.storage.database import get_engine
        from comic_translator.storage.models import EntityORM
        from sqlalchemy import select

        session_gen = self._get_session()
        session = next(session_gen)
        try:
            rows = session.execute(
                select(EntityORM).order_by(EntityORM.frequency.desc())
            ).scalars().all()
            return [
                {
                    "entity_name": r.entity_name,
                    "entity_type": r.entity_type,
                    "frequency": r.frequency,
                    "first_page": r.first_page,
                    "last_page": r.last_page,
                }
                for r in rows
            ]
        finally:
            session.close()

    def get_entity_context(self) -> str:
        """获取实体上下文文本（用于注入翻译 Prompt）"""
        entities = self.get_all_entities()
        if not entities:
            return ""
        lines = ["Known entities:"]
        for e in entities:
            lines.append(f"  {e['entity_name']} ({e['entity_type']})")
        return "\n".join(lines)