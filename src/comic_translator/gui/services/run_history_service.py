"""Run History Service — 管理 runs/ 目录，保存与加载 Pipeline 运行记录

目录结构:
    runs/
      └── 20260811_001/
             run_config.json          # RunRecord
             pages/
                 page_0001.json       # PageSnapshot
                 page_0002.json
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from loguru import logger

from comic_translator.gui.models.gui_models import PageSnapshot, RunRecord


class RunHistoryService:
    """运行历史持久化服务"""

    def __init__(self, runs_root: str = "runs"):
        self.runs_root = Path(runs_root)
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def generate_run_id(self) -> str:
        """生成唯一 run_id，格式 YYYYMMDD_NNN"""
        today = datetime.now().strftime("%Y%m%d")
        existing = [
            d.name for d in self.runs_root.iterdir()
            if d.is_dir() and d.name.startswith(today)
        ]
        seq = len(existing) + 1
        # 防止冲突，向上找第一个空位
        while f"{today}_{seq:03d}" in existing:
            seq += 1
        return f"{today}_{seq:03d}"

    def get_run_dir(self, run_id: str) -> Path:
        return self.runs_root / run_id

    # ---------- 保存 ----------

    def save_run(
        self,
        run_record: RunRecord,
        page_snapshots: list[PageSnapshot],
    ) -> Path:
        """保存一次运行的完整记录"""
        run_dir = self.get_run_dir(run_record.id)
        run_record.save(run_dir)

        for snap in page_snapshots:
            try:
                snap.save(run_dir)
            except Exception as e:
                logger.error(f"Failed to save page snapshot {snap.page}: {e}")

        logger.info(f"Run saved: {run_dir} ({len(page_snapshots)} pages)")
        return run_dir

    # ---------- 列表 ----------

    def list_runs(self) -> list[RunRecord]:
        """按时间倒序返回所有运行记录"""
        if not self.runs_root.exists():
            return []

        records = []
        for run_dir in sorted(self.runs_root.iterdir(), reverse=True):
            if not run_dir.is_dir():
                continue
            rec = RunRecord.load(run_dir)
            if rec:
                records.append(rec)
        return records

    # ---------- 加载单个 ----------

    def load_run(self, run_id: str) -> RunRecord | None:
        return RunRecord.load(self.get_run_dir(run_id))

    def load_page(self, run_id: str, page: int) -> PageSnapshot | None:
        return PageSnapshot.load(self.get_run_dir(run_id), page)

    def load_all_pages(self, run_id: str) -> list[PageSnapshot]:
        """加载一次运行中的所有页面快照"""
        pages_dir = self.get_run_dir(run_id) / "pages"
        if not pages_dir.exists():
            return []

        snapshots = []
        for p in sorted(pages_dir.glob("page_*.json")):
            try:
                page_num = int(p.stem.split("_")[1])
                snap = PageSnapshot.load(self.get_run_dir(run_id), page_num)
                if snap:
                    snapshots.append(snap)
            except Exception as e:
                logger.error(f"Failed to load {p}: {e}")
        return snapshots

    # ---------- 删除 ----------

    def delete_run(self, run_id: str) -> bool:
        """删除一次运行记录"""
        import shutil
        run_dir = self.get_run_dir(run_id)
        if not run_dir.exists():
            return False
        try:
            shutil.rmtree(run_dir)
            logger.info(f"Run deleted: {run_dir}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete {run_dir}: {e}")
            return False
