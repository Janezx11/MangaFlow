import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String, Text

from comic_translator.storage.database import Base


class ProjectORM(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    title = Column(String, default="")
    comic_style = Column(String, default="western_comic")
    source_language = Column(String, default="en")
    target_language = Column(String, default="zh")
    total_pages = Column(Integer, default=0)
    pages_completed = Column(Integer, default=0)
    status = Column(String, default="pending")
    input_dir = Column(Text, default="")
    output_dir = Column(Text, default="")
    cache_dir = Column(Text, default="")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class TranslationMemoryORM(Base):
    __tablename__ = "translation_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String, default="", index=True)  # 项目隔离
    source_text = Column(Text, nullable=False)
    translated_text = Column(Text, nullable=False)
    context_hash = Column(String, default="")
    source_hash = Column(String, default="", index=True)
    provider = Column(String, default="")
    model = Column(String, default="")
    usage_count = Column(Integer, default=0)
    last_used = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))


class GlossaryEntryORM(Base):
    __tablename__ = "glossary_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String, default="", index=True)  # 项目隔离
    source = Column(String, nullable=False, index=True)
    target = Column(String, nullable=False)
    context = Column(Text, default="")
    category = Column(String, default="general", index=True)
    priority = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class CharacterMemoryORM(Base):
    __tablename__ = "character_memory"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String, default="", index=True)  # 项目隔离
    character_id = Column(String, nullable=False, index=True)
    canonical_name = Column(String, nullable=False)
    aliases = Column(Text, default="")  # JSON list
    titles = Column(Text, default="")  # JSON list of titles
    nicknames = Column(Text, default="")  # JSON list of nicknames
    description = Column(Text, default="")
    speaking_style = Column(String, default="")
    related_character_ids = Column(Text, default="")  # JSON list
    first_seen_page = Column(Integer, default=0)
    last_seen_page = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class TranslatedPageORM(Base):
    """Batch 翻译结果持久化 —— 支持断点续跑"""

    __tablename__ = "translated_pages"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String, nullable=False, index=True)
    batch_index = Column(Integer, nullable=False)
    page = Column(Integer, nullable=False)
    page_data = Column(Text, default="")  # JSON — list[TranslationBlock]
    raw_source = Column(Text, default="")  # JSON — list[NormalizedOCRBlock]
    status = Column(String, default="pending")
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))


class EntityORM(Base):
    """OCR 提取的实体 —— 用于译名一致"""

    __tablename__ = "entities"

    id = Column(Integer, primary_key=True, autoincrement=True)
    project_id = Column(String, default="", index=True)  # 项目隔离
    entity_name = Column(String, nullable=False, index=True)
    entity_type = Column(String, default="")  # PERSON / LOCATION / ORGANIZATION
    translation = Column(String, default="")
    frequency = Column(Integer, default=1)
    first_page = Column(Integer, default=0)
    last_page = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
