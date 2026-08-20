"""GUI 数据模型 — RunRecord, PipelineSnapshot"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path


@dataclass
class RunRecord:
    """一次 Pipeline 运行记录"""
    id: str                          # 如 "20260811_001"
    project: str                     # 项目名称
    input_dir: str                   # 输入目录
    created_time: str                # ISO 格式时间
    total_pages: int = 0
    success_pages: int = 0
    failed_pages: int = 0
    total_dialogues: int = 0
    status: str = "running"          # running / success / partial / failed
    config: dict = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> RunRecord:
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, run_dir: Path):
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "run_config.json").write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, run_dir: Path) -> RunRecord | None:
        p = run_dir / "run_config.json"
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls.from_dict(data)


@dataclass
class PageSnapshot:
    """单页 Pipeline 数据快照"""
    page: int
    file_name: str
    detection: list[dict] = field(default_factory=list)
    ocr_blocks: list[dict] = field(default_factory=list)
    dialogues: list[dict] = field(default_factory=list)
    rendered_path: str = ""
    detection_time: float = 0.0
    ocr_time: float = 0.0
    translation_time: float = 0.0
    render_time: float = 0.0
    error: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, run_dir: Path):
        pages_dir = run_dir / "pages"
        pages_dir.mkdir(parents=True, exist_ok=True)
        page_file = pages_dir / f"page_{self.page:04d}.json"
        page_file.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, run_dir: Path, page: int) -> PageSnapshot | None:
        p = run_dir / "pages" / f"page_{page:04d}.json"
        if not p.exists():
            return None
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
