"""Annotation Service — 漫画检测数据标注业务逻辑

职责:
- 加载/保存 YOLO segmentation 格式标签
- 调用现有 DetectionFusionService 生成预标注
- 类别 id 映射

GUI 不应该直接处理 YOLO 格式细节。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from loguru import logger
from PIL import Image

from comic_translator.models.detection_region import DetectionRegion


# 类别映射 (id <-> name)
CLASS_NAMES = ["speech", "narration", "title", "sfx"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}
ID_TO_CLASS = {i: name for name, i in CLASS_TO_ID.items()}


@dataclass
class AnnotationRegion:
    """单条标注 (多边形 + 类别)"""
    class_name: str                 # speech / narration / title / sfx
    polygon: list[tuple[float, float]]  # 像素坐标
    source: str = "manual"          # manual / auto

    @property
    def class_id(self) -> int:
        return CLASS_TO_ID.get(self.class_name, 0)


@dataclass
class ImageAnnotation:
    """一张图片的全部标注"""
    image_path: str
    image_size: tuple[int, int]     # (width, height)
    regions: list[AnnotationRegion] = field(default_factory=list)


class AnnotationService:
    """标注业务服务"""

    # ---------- 加载 ----------

    def load_annotation(self, image_path: str, label_path: str | None = None) -> ImageAnnotation:
        """加载图片对应标注，如果没有 label 文件则返回空"""
        img_path = Path(image_path)
        if not img_path.exists():
            raise FileNotFoundError(f"图片不存在: {image_path}")

        with Image.open(img_path) as im:
            w, h = im.size

        regions: list[AnnotationRegion] = []
        label_file = Path(label_path) if label_path else self._default_label_path(img_path)
        if label_file.exists():
            regions = self._parse_yolo_seg(label_file, w, h)

        return ImageAnnotation(
            image_path=str(img_path),
            image_size=(w, h),
            regions=regions,
        )

    # ---------- 保存 ----------

    def save_annotation(
        self,
        annotation: ImageAnnotation,
        label_path: str | None = None,
    ) -> Path:
        """保存为 YOLO segmentation 格式"""
        img_path = Path(annotation.image_path)
        label_file = Path(label_path) if label_path else self._default_label_path(img_path)
        label_file.parent.mkdir(parents=True, exist_ok=True)

        w, h = annotation.image_size
        lines = []
        for r in annotation.regions:
            if len(r.polygon) < 3:
                continue
            coords = []
            for x, y in r.polygon:
                coords.append(f"{x / w:.6f}")
                coords.append(f"{y / h:.6f}")
            lines.append(f"{r.class_id} " + " ".join(coords))

        label_file.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        logger.info(f"Saved {len(lines)} labels to {label_file}")
        return label_file

    # ---------- AI 预标注 ----------

    def auto_annotate(
        self,
        image_path: str,
        detection_lists: list[list[DetectionRegion]],
        fusion_service,
    ) -> list[AnnotationRegion]:
        """用现有 Fusion 检测结果生成预标注

        Args:
            image_path: 图片路径
            detection_lists: 多源检测结果 (CTBD / YOLO 各自返回的 DetectionRegion)
            fusion_service: DetectionFusionService 实例
        """
        if not detection_lists:
            return []

        fused = fusion_service.fuse(detection_lists)
        regions: list[AnnotationRegion] = []
        for r in fused:
            label = r.label if r.label in CLASS_TO_ID else "speech"
            # 优先用 polygon，没有就用 bbox 构造矩形
            if r.polygon and len(r.polygon) >= 3:
                polygon = [(float(x), float(y)) for x, y in r.polygon]
            else:
                x1, y1, x2, y2 = r.bbox
                polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            regions.append(AnnotationRegion(
                class_name=label,
                polygon=polygon,
                source="auto",
            ))
        return regions

    # ---------- 私有 ----------

    @staticmethod
    def _default_label_path(image_path: Path) -> Path:
        """从图片路径推导 label 路径
        dataset/images/train/x.png -> dataset/labels/train/x.txt
        """
        parts = list(image_path.parts)
        if "images" in parts:
            idx = parts.index("images")
            parts[idx] = "labels"
            return Path(*parts).with_suffix(".txt")
        return image_path.with_suffix(".txt")

    @staticmethod
    def _parse_yolo_seg(label_file: Path, img_w: int, img_h: int) -> list[AnnotationRegion]:
        """解析 YOLO segmentation label 文件"""
        regions: list[AnnotationRegion] = []
        for line in label_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 7:  # class + 至少3个点
                continue
            try:
                class_id = int(parts[0])
                coords = [float(v) for v in parts[1:]]
                # 坐标是 (x, y) 对
                if len(coords) % 2 != 0:
                    continue
                polygon = []
                for i in range(0, len(coords), 2):
                    x = coords[i] * img_w
                    y = coords[i + 1] * img_h
                    polygon.append((x, y))
                regions.append(AnnotationRegion(
                    class_name=ID_TO_CLASS.get(class_id, "speech"),
                    polygon=polygon,
                    source="manual",
                ))
            except (ValueError, IndexError) as e:
                logger.warning(f"Bad label line in {label_file}: {line} ({e})")
        return regions
