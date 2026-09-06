from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from comic_translator.cache.base import CacheProvider


class FileCacheProvider(CacheProvider):
    """文件系统缓存 —— JSON 序列化，按 {project_id}/{stage}/{page}.json 存储"""

    def __init__(self, cache_dir: str = "data/cache") -> None:
        self._cache_dir = Path(cache_dir)

    def _path(self, key: str) -> Path:
        return self._cache_dir / key.replace(":", "/")

    async def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        data = path.read_text(encoding="utf-8")
        return json.loads(data)

    async def set(self, key: str, value: Any, ttl: int | None = None) -> None:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, ensure_ascii=False, default=str), encoding="utf-8")

    async def exists(self, key: str) -> bool:
        return self._path(key).exists()

    async def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    async def clear(self) -> None:
        import shutil
        for child in self._cache_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
