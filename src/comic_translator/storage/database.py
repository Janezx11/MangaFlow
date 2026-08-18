"""Database engine and session management

Each call to get_engine creates or retrieves engine by db_path.
No global singleton — supports multiple databases for testing.
"""

from pathlib import Path
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker


class Base(DeclarativeBase):
    pass


_engines = {}
_session_makers = {}


def get_engine(db_path: str = "data/comic_translator.db"):
    """Get or create engine for the given db_path."""
    if db_path not in _engines:
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        _engines[db_path] = create_engine(f"sqlite:///{db_path}", echo=False)
    return _engines[db_path]


def get_session_local(db_path: str = "data/comic_translator.db"):
    """Get or create sessionmaker for the given db_path."""
    if db_path not in _session_makers:
        _session_makers[db_path] = sessionmaker(bind=get_engine(db_path))
    return _session_makers[db_path]


def get_session(db_path: str = "data/comic_translator.db"):
    session = get_session_local(db_path)()
    try:
        yield session
    finally:
        session.close()


def init_db(db_path: str = "data/comic_translator.db"):
    """Create all tables, then migrate existing tables to add missing columns."""
    engine = get_engine(db_path)
    Base.metadata.create_all(bind=engine)
    _migrate_schema(engine, db_path)


def _migrate_schema(engine, db_path: str):
    """轻量迁移：为已存在的旧表补充缺失列（如 project_id）。

    create_all 只会新建缺失的表，不会给旧表加列。
    旧版本 DB（无 project_id 列）会在这里被 ALTER TABLE ADD COLUMN 补齐。
    """
    try:
        insp = inspect(engine)
        existing_tables = set(insp.get_table_names())
    except Exception:
        return

    # 需要保证列与 ORM 定义一致（models.py 中带 default 的列）
    from comic_translator.storage import models as _models  # noqa: F401  (注册 Base.metadata)

    with engine.begin() as conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in existing_tables:
                continue
            try:
                existing_cols = {c["name"] for c in insp.get_columns(table.name)}
            except Exception:
                continue
            for col in table.columns:
                if col.name in existing_cols:
                    continue
                # 用 ORM 类型定义生成 ADD COLUMN 语句
                try:
                    coltype = col.type.compile(engine.dialect)
                except Exception:
                    coltype = "TEXT"
                stmt = f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" {coltype}'
                conn.execute(text(stmt))


def reset_db_for_testing():
    """Clear engine cache — for test isolation."""
    _engines.clear()
    _session_makers.clear()
