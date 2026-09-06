from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class BatchCacheService:
    """Batch 级别缓存 —— 存放到 project 目录

    缓存文件:
      data/cache/{project_id}/ocr/all_pages_ocr.json   — 全部页面的 OCR 结果
      data/cache/{project_id}/batch/batch_{idx:03d}.json — Batch 翻译结果
    """

    def __init__(self, cache_dir: str = "data/cache") -> None:
        self._root = Path(cache_dir)

    # ---- OCR Cache ----

    def _ocr_path(self, project_id: str) -> Path:
        p = self._root / project_id / "ocr"
        p.mkdir(parents=True, exist_ok=True)
        return p / "all_pages_ocr.json"

    def get_ocr_cache(self, project_id: str) -> list | None:
        path = self._ocr_path(project_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else None

    def set_ocr_cache(self, project_id: str, pages_data: list[dict]) -> None:
        path = self._ocr_path(project_id)
        path.write_text(json.dumps(pages_data, ensure_ascii=False, indent=2), encoding="utf-8")

    def has_ocr_cache(self, project_id: str) -> bool:
        return self._ocr_path(project_id).exists()

    # ---- Batch Cache ----

    def _batch_path(self, project_id: str, batch_index: int) -> Path:
        p = self._root / project_id / "batch"
        p.mkdir(parents=True, exist_ok=True)
        return p / f"batch_{batch_index:03d}.json"

    def get_batch_output(self, project_id: str, batch_index: int) -> dict | None:
        path = self._batch_path(project_id, batch_index)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def set_batch_output(self, project_id: str, batch_index: int, data: dict) -> None:
        path = self._batch_path(project_id, batch_index)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def has_batch_cache(self, project_id: str, batch_index: int) -> bool:
        return self._batch_path(project_id, batch_index).exists()

    # ---- Cleanup ----

    def clear_project(self, project_id: str) -> None:
        import shutil
        path = self._root / project_id
        if path.exists():
            shutil.rmtree(path)
