from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from loguru import logger

from comic_translator.clients.embedding_base import EmbeddingProvider
from comic_translator.storage.database import get_session_local, init_db
from comic_translator.storage.models import TranslationMemoryORM

if TYPE_CHECKING:
    from comic_translator.models.glossary import TranslationMemory


class SemanticTranslationMemoryService:
    """翻译记忆服务 —— Sprint 3 语义升级版

    在 Sprint 2 的字符串匹配基础上加入:
    - source_hash: 快速精确去重
    - last_used: LRU 淘汰策略预留
    - EmbeddingProvider: 语义搜索预留

    EmbeddingProvider 未来接入后启用 find_semantic()。
    """

    def __init__(
        self,
        db_path: str = "data/comic_translator.db",
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        self._db_path = db_path
        self._embedding_provider = embedding_provider
        init_db(db_path)

    def store(
        self,
        source_text: str,
        translated_text: str,
        provider: str = "",
        model: str = "",
        context_hash: str = "",
    ) -> int:
        """存储一条翻译记忆"""
        source_hash = hashlib.md5(source_text.encode("utf-8")).hexdigest()

        session = get_session_local(self._db_path)()
        try:
            # Dedup: if exact match exists, update instead of insert
            existing = (
                session.query(TranslationMemoryORM)
                .filter(
                    TranslationMemoryORM.source_hash == source_hash,
                    TranslationMemoryORM.source_text == source_text,
                )
                .first()
            )
            if existing:
                existing.usage_count = (existing.usage_count or 0) + 1
                existing.last_used = datetime.now(timezone.utc)
                session.commit()
                return existing.id

            orm = TranslationMemoryORM(
                source_text=source_text,
                translated_text=translated_text,
                source_hash=source_hash,
                context_hash=context_hash,
                provider=provider,
                model=model,
                usage_count=1,
                last_used=datetime.now(timezone.utc),
            )
            session.add(orm)
            session.commit()
            return orm.id
        finally:
            session.close()

    def find_similar(self, source_text: str, limit: int = 5) -> list[TranslationMemory]:
        """查询相似翻译

        优先级:
        1. 精确匹配 (source_hash)
        2. LIKE 模糊匹配
        3. 语义匹配 (future, when embedding_provider is set)
        """
        source_hash = hashlib.md5(source_text.encode("utf-8")).hexdigest()
        session = get_session_local(self._db_path)()
        try:
            from comic_translator.models.glossary import TranslationMemory

            results: list[TranslationMemory] = []

            # Phase 1: exact match by hash
            exact = (
                session.query(TranslationMemoryORM)
                .filter(
                    TranslationMemoryORM.source_hash == source_hash,
                    TranslationMemoryORM.source_text == source_text,
                )
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

            if len(results) >= limit:
                return results

            # Phase 2: LIKE match
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

            # Phase 3: semantic (future)
            if self._embedding_provider and not results:
                try:
                    semantic_results = self._find_semantic(source_text, session, limit)
                    results.extend(semantic_results)
                except Exception as e:
                    logger.debug(f"Semantic search failed (expected if no provider): {e}")

            return results
        finally:
            session.close()

    def _find_semantic(
        self, source_text: str, session, limit: int
    ) -> list[TranslationMemory]:
        """语义搜索 (预留) — 需要 embedding_provider"""
        # TODO: implement when EmbeddingProvider is available
        return []

    def get_stats(self) -> dict:
        session = get_session_local(self._db_path)()
        try:
            total = session.query(TranslationMemoryORM).count()
            return {"total_entries": total}
        finally:
            session.close()

    def get_recent(self, limit: int = 20) -> list[TranslationMemory]:
        from comic_translator.models.glossary import TranslationMemory

        session = get_session_local(self._db_path)()
        try:
            rows = (
                session.query(TranslationMemoryORM)
                .order_by(TranslationMemoryORM.last_used.desc())
                .limit(limit)
                .all()
            )
            return [
                TranslationMemory(
                    source_text=r.source_text,
                    translated_text=r.translated_text,
                    context_hash=r.context_hash,
                    similarity=1.0,
                    provider=r.provider or "",
                    model=r.model or "",
                    usage_count=r.usage_count,
                )
                for r in rows
            ]
        finally:
            session.close()