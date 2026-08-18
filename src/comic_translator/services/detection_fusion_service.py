"""DetectionFusionService — 多源检测结果融合

将多个 Detection Provider 的结果融合为统一的 DetectionRegion 列表。

融合策略:
  1. IoU > threshold 的区域视为同一区域 → 合并
  2. 多源检测同一区域 → confidence 提升
  3. 有 polygon 的优先保留 polygon
  4. 单源独有区域 → 保留，不删除
"""

from __future__ import annotations

from loguru import logger

from comic_translator.models.detection_region import DetectionRegion

# 融合 IoU 阈值：超过此值视为同一区域
FUSION_IOU_THRESHOLD = 0.4

# 多源加成：每多一个源，confidence 提升系数
MULTI_SOURCE_BONUS = 0.1


class DetectionFusionService:
    """多源检测结果融合服务"""

    def fuse(
        self,
        detection_lists: list[list[DetectionRegion]],
        iou_threshold: float = FUSION_IOU_THRESHOLD,
    ) -> list[DetectionRegion]:
        """融合多个检测源的结果

        Args:
            detection_lists: 每个检测源的 DetectionRegion 列表
            iou_threshold: IoU 阈值，超过则视为同一区域

        Returns:
            融合后的 DetectionRegion 列表
        """
        all_regions = [r for regions in detection_lists for r in regions]
        if not all_regions:
            return []

        # 按 (y, x) 排序，贪心聚类
        all_regions.sort(key=lambda r: (r.bbox[1], r.bbox[0]))

        clusters: list[list[DetectionRegion]] = []
        for region in all_regions:
            best_cluster = None
            best_iou = 0.0

            for cluster in clusters:
                iou = _max_iou_with_cluster(region, cluster)
                if iou > best_iou:
                    best_iou = iou
                    best_cluster = cluster

            if best_cluster is not None and best_iou > iou_threshold:
                best_cluster.append(region)
            else:
                clusters.append([region])

        # 每个聚类合并为一个 DetectionRegion
        fused: list[DetectionRegion] = []
        for cluster in clusters:
            fused.append(self._merge_cluster(cluster))

        fused.sort(key=lambda r: (r.bbox[1], r.bbox[0]))

        # 重排 id
        for i, r in enumerate(fused):
            r.id = f"fused_{i:04d}"

        sources = set()
        for r in fused:
            sources.update(r.source.split("+"))

        logger.info(
            f"Fusion: {sum(len(l) for l in detection_lists)} raw → "
            f"{len(fused)} fused regions (sources={sources})"
        )
        return fused

    def _merge_cluster(self, cluster: list[DetectionRegion]) -> DetectionRegion:
        """将一个聚类合并为单个 DetectionRegion"""
        if len(cluster) == 1:
            return cluster[0]

        # 合并 bbox：取所有 box 的并集
        x1 = min(r.bbox[0] for r in cluster)
        y1 = min(r.bbox[1] for r in cluster)
        x2 = max(r.bbox[2] for r in cluster)
        y2 = max(r.bbox[3] for r in cluster)

        # 合并 label：投票，优先非 unknown
        labels = [r.label for r in cluster if r.label != "unknown"]
        if labels:
            label = max(set(labels), key=labels.count)
        else:
            label = "unknown"

        # 合并 confidence：max * (1 + bonus * (num_sources - 1))
        sources = list(dict.fromkeys(r.source for r in cluster))
        max_conf = max(r.confidence for r in cluster)
        confidence = min(1.0, max_conf * (1 + MULTI_SOURCE_BONUS * (len(sources) - 1)))

        # 合并 polygon：优先取有 polygon 的（YOLO 通常有精确 polygon）
        polygon = None
        for r in cluster:
            if r.polygon:
                polygon = r.polygon
                break

        return DetectionRegion(
            id=cluster[0].id,
            bbox=(x1, y1, x2, y2),
            polygon=polygon,
            label=label,
            confidence=round(confidence, 4),
            source="+".join(sources),
        )

    def to_text_regions(
        self, fused: list[DetectionRegion]
    ) -> list:
        """将融合结果转换为 TextRegion 列表（用于 TextRegionService）"""
        return [r.to_text_region() for r in fused]

    def to_bubbles(
        self, fused: list[DetectionRegion]
    ) -> list:
        """将融合结果转换为 Bubble 列表（用于 BubblePipeline）"""
        return [r.to_bubble() for r in fused]


def _max_iou_with_cluster(
    region: DetectionRegion, cluster: list[DetectionRegion]
) -> float:
    return max(_bbox_iou(region.bbox, c.bbox) for c in cluster)


def _bbox_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter

    return inter / union if union > 0 else 0.0
