"""Benchmark Service — 读取模型 Benchmark JSON，为评估面板提供数据"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger


class BenchmarkService:
    """读取并聚合 benchmark_detection_models.py 输出的 JSON"""

    DEFAULT_PATH = "data/output/benchmark_models.json"

    def __init__(self, json_path: str | None = None):
        self.json_path = Path(json_path or self.DEFAULT_PATH)

    def exists(self) -> bool:
        return self.json_path.exists()

    def load(self) -> list[dict]:
        if not self.json_path.exists():
            return []
        try:
            return json.loads(self.json_path.read_text(encoding="utf-8"))
        except Exception as e:
            logger.error(f"Failed to load benchmark json: {e}")
            return []

    def available_models(self) -> list[str]:
        """返回所有页面中出现过的模型名"""
        models: set[str] = set()
        for page in self.load():
            for name in (page.get("models") or {}).keys():
                models.add(name)
        return sorted(models)

    def summary(self) -> dict[str, dict]:
        """按模型聚合总体统计：
        {model_name: {pages, total_regions, avg_confidence, avg_time_s, label_counts}}
        同时包含 'fused' 项。
        """
        agg: dict[str, dict] = {}
        for page in self.load():
            sources = dict(page.get("models") or {})
            if page.get("fused"):
                sources["fused"] = page["fused"]
            for name, m in sources.items():
                bucket = agg.setdefault(name, {
                    "pages": 0,
                    "total_regions": 0,
                    "conf_sum": 0.0,
                    "time_sum": 0.0,
                    "label_counts": {},
                })
                bucket["pages"] += 1
                bucket["total_regions"] += m.get("regions", 0)
                bucket["conf_sum"] += m.get("avg_confidence", 0.0)
                bucket["time_sum"] += m.get("time_s", 0.0)
                for label, count in (m.get("labels") or {}).items():
                    bucket["label_counts"][label] = (
                        bucket["label_counts"].get(label, 0) + count
                    )

        for _name, bucket in agg.items():
            pages = max(bucket["pages"], 1)
            bucket["avg_confidence"] = round(bucket["conf_sum"] / pages, 3)
            bucket["avg_time_s"] = round(bucket["time_sum"] / pages, 3)
            del bucket["conf_sum"]
            del bucket["time_sum"]
        return agg

    def per_page_table(self) -> list[dict]:
        """返回逐页明细，用于表格展示"""
        rows = []
        for page in self.load():
            row = {
                "page": page.get("page"),
                "file": page.get("file"),
            }
            for name, m in (page.get("models") or {}).items():
                row[f"{name}_regions"] = m.get("regions", 0)
                row[f"{name}_conf"] = round(m.get("avg_confidence", 0.0), 3)
                row[f"{name}_time"] = round(m.get("time_s", 0.0), 2)
            fused = page.get("fused") or {}
            row["fused_regions"] = fused.get("regions", 0)
            row["fused_conf"] = round(fused.get("avg_confidence", 0.0), 3)
            rows.append(row)
        return rows
