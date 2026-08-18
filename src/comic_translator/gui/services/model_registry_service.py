"""Model Registry Service — 模型注册表

每个 .pt 模型对应一个同名 .json 元数据文件:
    models/bubble/yolo11n-seg-custom.pt
    models/bubble/yolo11n-seg-custom.pt.json   <- metadata
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from loguru import logger


@dataclass
class ModelInfo:
    """模型注册信息"""
    name: str                       # 文件名（不含.pt）
    path: str                       # 完整路径
    type: str = "yolo_seg"          # yolo_seg / ctbd / other
    dataset: str = ""               # 训练用数据集
    metrics: dict = field(default_factory=dict)  # mAP50 / mAP50-95 / precision / recall
    created_time: str = ""
    enabled: bool = False           # 是否被 pipeline 使用
    notes: str = ""

    @property
    def size_mb(self) -> float:
        try:
            return round(Path(self.path).stat().st_size / (1024 * 1024), 2)
        except (OSError, FileNotFoundError):
            return 0.0


class ModelRegistryService:
    """扫描 models/ 目录，管理元数据"""

    def __init__(self, root: str = "models/bubble"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------- 扫描 ----------

    def list_models(self) -> list[ModelInfo]:
        """扫描所有 .pt 模型，合并元数据"""
        if not self.root.exists():
            return []

        models: list[ModelInfo] = []
        for pt in sorted(self.root.glob("*.pt")):
            info = self._load_or_create(pt)
            models.append(info)
        return models

    def get_model(self, name: str) -> ModelInfo | None:
        """按名称获取（name 不含 .pt）"""
        pt = self.root / f"{name}.pt"
        if not pt.exists():
            return None
        return self._load_or_create(pt)

    # ---------- 注册 / 更新 ----------

    def register(
        self,
        pt_path: str,
        dataset: str = "",
        metrics: dict | None = None,
        notes: str = "",
    ) -> ModelInfo:
        """注册新模型（或更新已有）"""
        pt = Path(pt_path)
        if not pt.exists():
            raise FileNotFoundError(f"模型文件不存在: {pt_path}")

        info = ModelInfo(
            name=pt.stem,
            path=str(pt),
            type="yolo_seg",
            dataset=dataset,
            metrics=metrics or {},
            created_time=datetime.now().isoformat(timespec="seconds"),
            enabled=False,
            notes=notes,
        )
        self._save_metadata(info)
        logger.info(f"Registered model: {info.name}")
        return info

    def update_metrics(self, name: str, metrics: dict) -> bool:
        info = self.get_model(name)
        if info is None:
            return False
        info.metrics.update(metrics)
        self._save_metadata(info)
        return True

    # ---------- 启用 / 禁用 ----------

    def enable(self, name: str) -> bool:
        """启用模型（同时禁用其他所有）"""
        target = self.get_model(name)
        if target is None:
            return False

        for m in self.list_models():
            m.enabled = (m.name == name)
            self._save_metadata(m)
        logger.info(f"Enabled model: {name}")
        return True

    def disable(self, name: str) -> bool:
        info = self.get_model(name)
        if info is None:
            return False
        info.enabled = False
        self._save_metadata(info)
        return True

    def get_active_model(self) -> ModelInfo | None:
        """返回当前启用的模型"""
        for m in self.list_models():
            if m.enabled:
                return m
        return None

    # ---------- 删除 ----------

    def delete(self, name: str, delete_file: bool = False) -> bool:
        """删除注册信息；delete_file=True 时同时删除 .pt 文件"""
        info = self.get_model(name)
        if info is None:
            return False
        meta = self._meta_path(Path(info.path))
        if meta.exists():
            meta.unlink()
        if delete_file:
            Path(info.path).unlink()
            logger.info(f"Deleted model file: {info.path}")
        return True

    # ---------- 私有 ----------

    def _load_or_create(self, pt: Path) -> ModelInfo:
        meta = self._meta_path(pt)
        if meta.exists():
            try:
                data = json.loads(meta.read_text(encoding="utf-8"))
                # path 字段以实际位置为准
                data["path"] = str(pt)
                return ModelInfo(**{k: v for k, v in data.items()
                                    if k in ModelInfo.__dataclass_fields__})
            except Exception as e:
                logger.warning(f"Bad metadata {meta}: {e}")

        # 没有元数据 → 创建基础记录
        return ModelInfo(
            name=pt.stem,
            path=str(pt),
            type="yolo_seg",
            created_time=datetime.fromtimestamp(pt.stat().st_mtime).isoformat(timespec="seconds"),
        )

    def _save_metadata(self, info: ModelInfo):
        meta = self._meta_path(Path(info.path))
        meta.write_text(
            json.dumps(asdict(info), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    @staticmethod
    def _meta_path(pt: Path) -> Path:
        return pt.with_suffix(pt.suffix + ".json")
