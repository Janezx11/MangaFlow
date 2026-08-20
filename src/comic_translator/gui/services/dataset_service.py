"""Dataset Service — 数据集管理

目录约定（统一用 `dataset/`，与 scripts/train_bubble_model.py 默认一致）:
    dataset/
      └── {name}/
             dataset.yaml
             images/{train,val}/*.png
             labels/{train,val}/*.txt
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from loguru import logger

from comic_translator.gui.services.annotation_service import CLASS_NAMES, CLASS_TO_ID


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass
class DatasetStats:
    """数据集统计"""
    name: str
    path: str
    train_images: int = 0
    val_images: int = 0
    train_labels: int = 0
    val_labels: int = 0
    class_counts: dict[str, int] = field(default_factory=dict)
    unlabeled_images: list[str] = field(default_factory=list)


class DatasetService:
    """数据集 CRUD + 统计"""

    def __init__(self, root: str = "dataset"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ---------- 列表 ----------

    def list_datasets(self) -> list[str]:
        """返回所有数据集名称"""
        if not self.root.exists():
            return []
        return sorted([
            d.name for d in self.root.iterdir()
            if d.is_dir() and (d / "dataset.yaml").exists()
        ])

    def get_dataset_dir(self, name: str) -> Path:
        return self.root / name

    # ---------- 创建 ----------

    def create_dataset(self, name: str) -> Path:
        """创建空数据集骨架"""
        if not name or "/" in name or "\\" in name:
            raise ValueError(f"非法数据集名: {name}")
        ds_dir = self.get_dataset_dir(name)
        if ds_dir.exists():
            raise FileExistsError(f"数据集已存在: {name}")

        for sub in ("images/train", "images/val", "labels/train", "labels/val"):
            (ds_dir / sub).mkdir(parents=True, exist_ok=True)
        self._write_dataset_yaml(ds_dir)
        logger.info(f"Created dataset: {ds_dir}")
        return ds_dir

    # ---------- 导入图片 ----------

    def import_images(
        self,
        name: str,
        source_dir: str,
        split: str = "train",
        max_count: int | None = None,
    ) -> int:
        """从 source_dir 拷贝图片到 datasets/{name}/images/{split}/"""
        if split not in ("train", "val"):
            raise ValueError(f"split 必须是 train/val: {split}")

        ds_dir = self.get_dataset_dir(name)
        if not ds_dir.exists():
            raise FileNotFoundError(f"数据集不存在: {name}")

        src = Path(source_dir)
        if not src.exists():
            raise FileNotFoundError(f"源目录不存在: {source_dir}")

        target = ds_dir / "images" / split
        target.mkdir(parents=True, exist_ok=True)

        count = 0
        for f in sorted(src.iterdir()):
            if not f.is_file() or f.suffix.lower() not in IMAGE_EXTS:
                continue
            if max_count is not None and count >= max_count:
                break
            dst = target / f.name
            if dst.exists():
                continue
            shutil.copy2(f, dst)
            count += 1

        logger.info(f"Imported {count} images into {name}/{split}")
        return count

    # ---------- 统计 ----------

    def get_stats(self, name: str) -> DatasetStats | None:
        ds_dir = self.get_dataset_dir(name)
        if not ds_dir.exists():
            return None

        stats = DatasetStats(name=name, path=str(ds_dir))

        for split in ("train", "val"):
            img_dir = ds_dir / "images" / split
            lbl_dir = ds_dir / "labels" / split
            images = [f for f in img_dir.iterdir()
                      if f.is_file() and f.suffix.lower() in IMAGE_EXTS] if img_dir.exists() else []
            labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

            if split == "train":
                stats.train_images = len(images)
                stats.train_labels = len(labels)
            else:
                stats.val_images = len(images)
                stats.val_labels = len(labels)

            # 找出未标注图片
            label_stems = {f.stem for f in labels}
            for img in images:
                if img.stem not in label_stems:
                    stats.unlabeled_images.append(f"{split}/{img.name}")

            # 类别分布统计
            for lbl in labels:
                self._count_classes(lbl, stats.class_counts)

        return stats

    @staticmethod
    def _count_classes(label_file: Path, bucket: dict[str, int]):
        try:
            for line in label_file.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                class_id = int(line.split()[0])
                name = CLASS_NAMES[class_id] if 0 <= class_id < len(CLASS_NAMES) else f"cls{class_id}"
                bucket[name] = bucket.get(name, 0) + 1
        except (ValueError, IndexError):
            pass

    # ---------- 数据访问 ----------

    def list_images(self, name: str, split: str = "train") -> list[str]:
        """返回 {split}/xxx.png 相对路径列表"""
        img_dir = self.get_dataset_dir(name) / "images" / split
        if not img_dir.exists():
            return []
        return sorted([
            f"{split}/{f.name}" for f in img_dir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTS
        ])

    def get_image_path(self, name: str, rel_path: str) -> Path:
        """rel_path 形如 'train/xxx.png'"""
        return self.get_dataset_dir(name) / "images" / rel_path

    def get_label_path(self, name: str, rel_path: str) -> Path:
        """rel_path 形如 'train/xxx.png'，返回对应 label 路径"""
        p = Path(rel_path)
        return self.get_dataset_dir(name) / "labels" / p.parent / f"{p.stem}.txt"

    # ---------- 删除 ----------

    def delete_image(self, name: str, rel_path: str) -> bool:
        """删除图片及其 label"""
        img = self.get_image_path(name, rel_path)
        lbl = self.get_label_path(name, rel_path)
        ok = False
        if img.exists():
            img.unlink()
            ok = True
        if lbl.exists():
            lbl.unlink()
        return ok

    def delete_dataset(self, name: str) -> bool:
        ds_dir = self.get_dataset_dir(name)
        if not ds_dir.exists():
            return False
        shutil.rmtree(ds_dir)
        logger.info(f"Deleted dataset: {name}")
        return True

    # ---------- YAML ----------

    def regenerate_yaml(self, name: str) -> Path:
        """重新生成 dataset.yaml"""
        ds_dir = self.get_dataset_dir(name)
        if not ds_dir.exists():
            raise FileNotFoundError(f"数据集不存在: {name}")
        return self._write_dataset_yaml(ds_dir)

    @staticmethod
    def _write_dataset_yaml(ds_dir: Path) -> Path:
        yaml_path = ds_dir / "dataset.yaml"
        config = {
            "path": ds_dir.as_posix(),
            "train": "images/train",
            "val": "images/val",
            "nc": len(CLASS_NAMES),
            "names": {i: n for i, n in enumerate(CLASS_NAMES)},
        }
        yaml_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        return yaml_path
