from __future__ import annotations

import json
from typing import TYPE_CHECKING

from comic_translator.storage.database import get_session_local, init_db
from comic_translator.storage.models import CharacterMemoryORM

if TYPE_CHECKING:
    from comic_translator.models.character_memory import (
        CharacterMemory,
        CharacterMemoryEntry,
    )


class CharacterMemoryService:
    """角色记忆服务 —— 维护统一译名

    核心能力：
    - register_character: 注册新角色
    - resolve_name: 给定任意名称变体，返回规范名
    - 别名自动关联
    """

    def __init__(self, db_path: str = "data/comic_translator.db") -> None:
        self._db_path = db_path
        init_db(db_path)

    def register_character(
        self,
        character: CharacterMemory,
    ) -> int:
        """注册角色，返回 ID"""
        session = get_session_local(self._db_path)()
        try:
            # Check if character_id already exists
            existing = (
                session.query(CharacterMemoryORM)
                .filter_by(character_id=character.character_id)
                .first()
            )
            if existing:
                existing.canonical_name = character.canonical_name
                existing.aliases = json.dumps(character.aliases)
                existing.description = character.description
                existing.speaking_style = character.speaking_style
                existing.last_seen_page = max(existing.last_seen_page, character.last_seen_page)
                session.commit()
                return existing.id

            orm = CharacterMemoryORM(
                character_id=character.character_id,
                canonical_name=character.canonical_name,
                aliases=json.dumps(character.aliases),
                description=character.description,
                speaking_style=character.speaking_style,
                first_seen_page=character.first_seen_page,
                last_seen_page=character.last_seen_page,
            )
            session.add(orm)
            session.commit()
            return orm.id
        finally:
            session.close()

    def batch_register(self, characters: list[CharacterMemory]) -> int:
        """批量注册，返回注册数"""
        count = 0
        for c in characters:
            self.register_character(c)
            count += 1
        return count

    def get_by_name(self, name: str) -> CharacterMemory | None:
        """通过规范名查询"""
        session = get_session_local(self._db_path)()
        try:
            orm = (
                session.query(CharacterMemoryORM)
                .filter_by(canonical_name=name)
                .first()
            )
            if orm is None:
                return None
            return self._orm_to_model(orm)
        finally:
            session.close()

    def resolve_name(self, name: str) -> CharacterMemoryEntry | None:
        """给定任意名称变体，返回角色记忆条目

        匹配优先级：规范名 → 别名 → 子串匹配
        """
        name = name.strip()
        if not name:
            return None

        session = get_session_local(self._db_path)()
        try:
            rows = session.query(CharacterMemoryORM).all()
        finally:
            session.close()

        # Phase 1: exact canonical match
        for orm in rows:
            if orm.canonical_name.lower() == name.lower():
                return self._build_entry(orm, match_type="exact")

        # Phase 2: exact alias match
        for orm in rows:
            aliases = json.loads(orm.aliases or "[]")
            if any(a.lower() == name.lower() for a in aliases):
                return self._build_entry(orm, match_type="alias")

        # Phase 3: substring match (for partial names like "John" → "John Black")
        matches: list[tuple[CharacterMemoryORM, float]] = []
        for orm in rows:
            if orm.canonical_name.lower().startswith(name.lower()):
                ratio = len(name) / max(len(orm.canonical_name), 1)
                matches.append((orm, ratio))
            else:
                aliases = json.loads(orm.aliases or "[]")
                for a in aliases:
                    if a.lower().startswith(name.lower()):
                        ratio = len(name) / max(len(a), 1)
                        matches.append((orm, ratio))
                        break

        if matches:
            matches.sort(key=lambda x: x[1], reverse=True)
            return self._build_entry(matches[0][0], match_type="fuzzy")

        return None

    def update_last_seen(self, character_id: str, page: int) -> None:
        session = get_session_local(self._db_path)()
        try:
            orm = (
                session.query(CharacterMemoryORM)
                .filter_by(character_id=character_id)
                .first()
            )
            if orm:
                orm.last_seen_page = max(orm.last_seen_page, page)
                session.commit()
        finally:
            session.close()

    def get_all_characters(self) -> list[CharacterMemory]:
        session = get_session_local(self._db_path)()
        try:
            rows = session.query(CharacterMemoryORM).all()
            return [self._orm_to_model(r) for r in rows]
        finally:
            session.close()

    # --- Internal ---

    def _orm_to_model(self, orm: CharacterMemoryORM) -> CharacterMemory:
        from comic_translator.models.character_memory import CharacterMemory

        return CharacterMemory(
            character_id=orm.character_id,
            canonical_name=orm.canonical_name,
            aliases=json.loads(orm.aliases or "[]"),
            description=orm.description,
            speaking_style=orm.speaking_style,
            first_seen_page=orm.first_seen_page,
            last_seen_page=orm.last_seen_page,
        )

    def _build_entry(
        self, orm: CharacterMemoryORM, match_type: str
    ) -> CharacterMemoryEntry:
        from comic_translator.models.character_memory import (
            CharacterMemoryEntry,
        )

        char = self._orm_to_model(orm)
        return CharacterMemoryEntry(
            character=char,
            match_type=match_type,
            confidence=1.0 if match_type == "exact" else 0.8,
        )
