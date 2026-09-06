from __future__ import annotations

import json
from typing import TYPE_CHECKING

from comic_translator.storage.database import get_session_local, init_db
from comic_translator.storage.models import CharacterMemoryORM

if TYPE_CHECKING:
    from comic_translator.models.character_identity import (
        CharacterIdentity,
        IdentityMatchResult,
    )


class CharacterIdentityService:
    """角色身份服务 —— Sprint 3 升级版

    对比 CharacterMemoryService (Sprint 2):
    - 支持 titles (Iron Man, CEO)、nicknames (Shellhead)
    - 5 阶段匹配: 精确 → 别名 → 头衔 → 昵称 → 模糊
    - 支持关联角色
    """

    def __init__(self, db_path: str = "data/comic_translator.db") -> None:
        self._db_path = db_path
        init_db(db_path)

    # --- CRUD ---

    def create_identity(self, identity: CharacterIdentity, project_id: str | None = None) -> int:
        """创建角色身份，返回 ID"""
        session = get_session_local(self._db_path)()
        try:
            existing = (
                session.query(CharacterMemoryORM)
                .filter_by(character_id=identity.character_id)
                .first()
            )
            if existing:
                return self._update_orm(existing, identity, session)

            orm = CharacterMemoryORM(
                project_id=project_id or "",
                character_id=identity.character_id,
                canonical_name=identity.canonical_name,
                aliases=json.dumps(identity.aliases),
                titles=json.dumps(identity.titles),
                nicknames=json.dumps(identity.nicknames),
                description=identity.description,
                speaking_style=identity.speaking_style,
                related_character_ids=json.dumps(identity.related_character_ids),
                first_seen_page=identity.first_seen_page,
                last_seen_page=identity.last_seen_page,
            )
            session.add(orm)
            session.commit()
            return orm.id
        finally:
            session.close()

    def update_identity(self, character_id: str, identity: CharacterIdentity) -> bool:
        session = get_session_local(self._db_path)()
        try:
            orm = (
                session.query(CharacterMemoryORM)
                .filter_by(character_id=character_id)
                .first()
            )
            if orm is None:
                return False
            self._update_orm(orm, identity, session)
            return True
        finally:
            session.close()

    def link_alias(self, character_id: str, alias: str) -> bool:
        """为角色添加新别名"""
        session = get_session_local(self._db_path)()
        try:
            orm = (
                session.query(CharacterMemoryORM)
                .filter_by(character_id=character_id)
                .first()
            )
            if orm is None:
                return False
            aliases = json.loads(orm.aliases or "[]")
            if alias not in aliases:
                aliases.append(alias)
                orm.aliases = json.dumps(aliases)
                session.commit()
            return True
        finally:
            session.close()

    # --- Query ---

    def resolve_name(self, name: str) -> IdentityMatchResult | None:
        """给定任意名称变体，返回身份匹配结果

        5 阶段匹配:
        1. 规范名精确匹配
        2. 别名精确匹配
        3. 头衔匹配
        4. 昵称匹配
        5. 模糊匹配 (子串)
        """
        name = name.strip()
        if not name:
            return None

        session = get_session_local(self._db_path)()
        try:
            rows = session.query(CharacterMemoryORM).all()
        finally:
            session.close()

        name_lower = name.lower()

        # Phase 1: exact canonical
        for orm in rows:
            if orm.canonical_name.lower() == name_lower:
                return self._build_result(orm, "exact", 1.0)

        # Phase 2: exact alias
        for orm in rows:
            aliases = json.loads(orm.aliases or "[]")
            if any(a.lower() == name_lower for a in aliases):
                return self._build_result(orm, "alias", 0.95)

        # Phase 3: title match
        for orm in rows:
            titles = json.loads(orm.titles or "[]")
            if any(t.lower() == name_lower for t in titles):
                return self._build_result(orm, "title", 0.9)

        # Phase 4: nickname match
        for orm in rows:
            nicknames = json.loads(orm.nicknames or "[]")
            if any(n.lower() == name_lower for n in nicknames):
                return self._build_result(orm, "nickname", 0.85)

        # Phase 5: fuzzy match (substring)
        fuzzy_matches: list[tuple[CharacterMemoryORM, float]] = []
        for orm in rows:
            candidates = (
                [orm.canonical_name, *json.loads(orm.aliases or "[]"), *json.loads(orm.titles or "[]"), *json.loads(orm.nicknames or "[]")]
            )
            for candidate in candidates:
                if candidate.lower().startswith(name_lower):
                    ratio = len(name) / max(len(candidate), 1)
                    fuzzy_matches.append((orm, ratio))
                    break

        if fuzzy_matches:
            fuzzy_matches.sort(key=lambda x: x[1], reverse=True)
            return self._build_result(fuzzy_matches[0][0], "fuzzy", fuzzy_matches[0][1])

        return None

    def resolve_names(self, names: list[str]) -> list[IdentityMatchResult]:
        """批量解析名称"""
        return [r for name in names if (r := self.resolve_name(name)) is not None]

    def search_identity(self, query: str) -> list[CharacterIdentity]:
        """搜索角色身份（规范名 + 别名 + 头衔）"""
        session = get_session_local(self._db_path)()
        try:
            rows = session.query(CharacterMemoryORM).all()
        finally:
            session.close()

        q = query.lower()
        results: list[CharacterIdentity] = []
        for orm in rows:
            candidates = (
                [orm.canonical_name, *json.loads(orm.aliases or "[]"), *json.loads(orm.titles or "[]")]
            )
            if any(q in c.lower() for c in candidates):
                results.append(self._orm_to_model(orm))
        return results

    def get_all_identities(self, project_id: str | None = None) -> list[CharacterIdentity]:
        session = get_session_local(self._db_path)()
        try:
            q = session.query(CharacterMemoryORM)
            if project_id:
                q = q.filter(CharacterMemoryORM.project_id == project_id)
            rows = q.all()
            return [self._orm_to_model(r) for r in rows]
        finally:
            session.close()

    def get_identity(self, character_id: str) -> CharacterIdentity | None:
        session = get_session_local(self._db_path)()
        try:
            orm = (
                session.query(CharacterMemoryORM)
                .filter_by(character_id=character_id)
                .first()
            )
            if orm is None:
                return None
            return self._orm_to_model(orm)
        finally:
            session.close()

    def delete_identity(self, character_id: str) -> bool:
        """删除角色身份（按 character_id）"""
        session = get_session_local(self._db_path)()
        try:
            orm = (
                session.query(CharacterMemoryORM)
                .filter_by(character_id=character_id)
                .first()
            )
            if orm is None:
                return False
            session.delete(orm)
            session.commit()
            return True
        finally:
            session.close()

    # --- Internal ---

    def _orm_to_model(self, orm: CharacterMemoryORM) -> CharacterIdentity:
        from comic_translator.models.character_identity import CharacterIdentity

        return CharacterIdentity(
            character_id=orm.character_id,
            canonical_name=orm.canonical_name,
            aliases=json.loads(orm.aliases or "[]"),
            titles=json.loads(orm.titles or "[]"),
            nicknames=json.loads(orm.nicknames or "[]"),
            description=orm.description,
            speaking_style=orm.speaking_style,
            related_character_ids=json.loads(orm.related_character_ids or "[]"),
            first_seen_page=orm.first_seen_page,
            last_seen_page=orm.last_seen_page,
        )

    def _build_result(
        self, orm: CharacterMemoryORM, match_type: str, confidence: float
    ) -> IdentityMatchResult:
        from comic_translator.models.character_identity import IdentityMatchResult

        return IdentityMatchResult(
            identity=self._orm_to_model(orm),
            match_type=match_type,
            confidence=confidence,
        )

    def _update_orm(
        self, orm: CharacterMemoryORM, identity: CharacterIdentity, session
    ) -> int:
        orm.canonical_name = identity.canonical_name
        orm.aliases = json.dumps(identity.aliases)
        orm.titles = json.dumps(identity.titles)
        orm.nicknames = json.dumps(identity.nicknames)
        orm.description = identity.description
        orm.speaking_style = identity.speaking_style
        orm.related_character_ids = json.dumps(identity.related_character_ids)
        orm.last_seen_page = max(orm.last_seen_page, identity.last_seen_page)
        session.commit()
        return orm.id
