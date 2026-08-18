"""BubblePostProcessService — 气泡后处理（合并、拆分、过滤、平滑、评分）

输入: list[Bubble] (from YOLO)
输出: list[Bubble] (经过合并、拆分、过滤、平滑后的高质量气泡)

处理流程:
  1. Score — 重新计算综合评分
  2. Filter — 删除低质量气泡
  3. Merge — 合并高 IoU 气泡
  4. Split — 超大气泡按 OCR 分布拆分
  5. Smooth — Polygon 边缘平滑
  6. Final Score — 更新最终评分
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np
from loguru import logger
from scipy.spatial import ConvexHull

from comic_translator.models.bubble import Bubble

if TYPE_CHECKING:
    from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock

# ---- 导入 scipy 可能失败，准备 fallback ----
try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class BubblePostProcessService:
    """气泡后处理"""

    # Filter 阈值
    MIN_CONFIDENCE = 0.3
    MIN_AREA = 800  # 像素
    MIN_POLYGON_POINTS = 3

    # Merge 阈值
    MERGE_IOU_THRESHOLD = 0.5
    MERGE_CENTER_DIST_RATIO = 0.3

    # Split 阈值
    SPLIT_MIN_AREA = 5000  # 超过此面积的气泡考虑拆分
    SPLIT_OCR_GAP_RATIO = 0.5  # OCR 行间距超过此比例时拆分

    # Smooth 参数
    APPROX_EPSILON_RATIO = 0.02  # approxPolyDP epsilon = 周长 * 此值

    def process(
        self,
        bubbles: list[Bubble],
        image_shape: tuple[int, int],
        ocr_blocks_map: dict[str, list] | None = None,
    ) -> list[Bubble]:
        """完整后处理流水线

        Args:
            bubbles: 原始 YOLO 输出
            image_shape: (h, w)
            ocr_blocks_map: 可选，{bubble_id: [OCR blocks]} 用于 split

        Returns:
            处理后的气泡列表
        """
        if not bubbles:
            return []

        # 1. 初始评分
        for b in bubbles:
            b.confidence = self._compute_score(b, ocr_blocks_map.get(b.id, []) if ocr_blocks_map else [])

        # 2. Filter: 删除低质量
        filtered = self._filter(bubbles, image_shape)
        if not filtered:
            return []

        # 3. Merge: 合并重叠气泡
        merged = self._merge_overlapping(filtered, image_shape)

        # 4. Split: 超大气泡按 OCR 拆分
        if ocr_blocks_map:
            merged = self._split_large(merged, ocr_blocks_map, image_shape)

        # 5. Smooth: 多边形平滑
        for b in merged:
            b.polygon = self._smooth_polygon(b.polygon)

        # 6. 最终评分
        for b in merged:
            blocks = ocr_blocks_map.get(b.id, []) if ocr_blocks_map else b.blocks
            b.confidence = self._compute_score(b, blocks)

        logger.info(
            f"PostProcess: {len(bubbles)} → {len(merged)} bubbles "
            f"(filtered={len(bubbles) - len(filtered)}, "
            f"merged={len(filtered) - len(merged) + max(0, len(merged) - len(filtered))}"
        )
        return merged

    # ---- Score ----

    def _compute_score(self, bubble: Bubble, ocr_blocks: list) -> float:
        """综合评分：YOLO confidence + OCR 文本数量 + polygon 面积

        Returns:
            0.0 ~ 1.0
        """
        # YOLO confidence
        conf_score = bubble.confidence

        # OCR 文本数量得分
        text_count = len([b for b in ocr_blocks if getattr(b, 'text', '')])
        ocr_score = min(text_count / 3, 1.0)  # 3+ OCR blocks → 满分

        # 面积得分（太大或太小都不好）
        x1, y1, x2, y2 = bubble.bbox
        area = (x2 - x1) * (y2 - y1)
        if area < self.MIN_AREA:
            area_score = 0.0
        elif area > 50000:
            area_score = 0.5  # 可能包含多个气泡
        else:
            area_score = 1.0

        # 加权
        return conf_score * 0.5 + ocr_score * 0.3 + area_score * 0.2

    # ---- Filter ----

    def _filter(self, bubbles: list[Bubble], image_shape: tuple[int, int]) -> list[Bubble]:
        """删除低质量气泡"""
        h, w = image_shape[:2]
        kept: list[Bubble] = []
        for b in bubbles:
            x1, y1, x2, y2 = b.bbox
            area = (x2 - x1) * (y2 - y1)

            # 面积过滤
            if area < self.MIN_AREA:
                continue

            # 置信度过滤
            if b.confidence < self.MIN_CONFIDENCE:
                continue

            # 多边形有效性
            if len(b.polygon) < self.MIN_POLYGON_POINTS:
                continue

            # 边界检查（不超出图片范围）
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                # 裁剪到图片范围内
                b.bbox = (max(0, x1), max(0, y1), min(w, x2), min(h, y2))

            kept.append(b)

        removed = len(bubbles) - len(kept)
        if removed > 0:
            logger.debug(f"Filter: removed {removed} low-quality bubbles")
        return kept

    # ---- Merge ----

    def _merge_overlapping(self, bubbles: list[Bubble], image_shape: tuple[int, int]) -> list[Bubble]:
        """合并高 IoU 的气泡"""
        if len(bubbles) <= 1:
            return bubbles

        merged = list(bubbles)
        changed = True
        while changed:
            changed = False
            new_merged = []
            skip = set()

            for i in range(len(merged)):
                if i in skip:
                    continue
                current = merged[i]
                for j in range(i + 1, len(merged)):
                    if j in skip:
                        continue
                    other = merged[j]
                    if self._should_merge(current, other):
                        current = self._do_merge(current, other)
                        skip.add(j)
                        changed = True
                new_merged.append(current)
            merged = new_merged

        return merged

    def _should_merge(self, a: Bubble, b: Bubble) -> bool:
        """判断两个气泡是否应该合并"""
        # IoU 计算
        ax1, ay1, ax2, ay2 = a.bbox
        bx1, by1, bx2, by2 = b.bbox

        # 重叠面积
        ox1 = max(ax1, bx1)
        oy1 = max(ay1, by1)
        ox2 = min(ax2, bx2)
        oy2 = min(ay2, by2)

        if ox2 <= ox1 or oy2 <= oy1:
            return False

        overlap = (ox2 - ox1) * (oy2 - oy1)
        a_area = (ax2 - ax1) * (ay2 - ay1)
        b_area = (bx2 - bx1) * (by2 - by1)
        iou = overlap / min(a_area, b_area)  # 用小面积做分母，更敏感

        if iou >= self.MERGE_IOU_THRESHOLD:
            return True

        # 中心距离
        a_cx = (ax1 + ax2) / 2
        a_cy = (ay1 + ay2) / 2
        b_cx = (bx1 + bx2) / 2
        b_cy = (by1 + by2) / 2
        dist = np.sqrt((a_cx - b_cx) ** 2 + (a_cy - b_cy) ** 2)
        avg_diag = (np.sqrt(a_area) + np.sqrt(b_area)) / 2
        if avg_diag <= 0:
            return False
        norm_dist = dist / avg_diag

        return norm_dist < self.MERGE_CENTER_DIST_RATIO

    def _do_merge(self, a: Bubble, b: Bubble) -> Bubble:
        """合并两个气泡"""
        # 合并 bbox
        x1 = min(a.bbox[0], b.bbox[0])
        y1 = min(a.bbox[1], b.bbox[1])
        x2 = max(a.bbox[2], b.bbox[2])
        y2 = max(a.bbox[3], b.bbox[3])

        # 合并 polygon: 取并集的凸包
        combined_poly = a.polygon + b.polygon
        if len(combined_poly) >= 3:
            pts = np.array(combined_poly, dtype=np.int32)
            if HAS_SCIPY and len(pts) >= 3:
                try:
                    hull = ConvexHull(pts)
                    combined_poly = [(int(pts[v, 0]), int(pts[v, 1])) for v in hull.vertices]
                except Exception:
                    combined_poly = self._fallback_merge_polygon(a, b)
            else:
                combined_poly = self._fallback_merge_polygon(a, b)
        else:
            combined_poly = a.polygon or b.polygon

        # 合并 mask：OR 操作
        a_mask = np.array(a.mask) if a.mask else np.zeros((1, 1), dtype=np.float32)
        b_mask = np.array(b.mask) if b.mask else np.zeros((1, 1), dtype=np.float32)
        # 简化：取较大的 mask
        if a_mask.size >= b_mask.size:
            merged_mask = a_mask.tolist() if a.mask else []
        else:
            merged_mask = b_mask.tolist() if b.mask else []

        # 合并 blocks
        merged_blocks = list(a.blocks) + list(b.blocks)

        # 标签：取置信度较高的
        best = a if a.confidence >= b.confidence else b

        return Bubble(
            id=f"merged_{a.id}_{b.id}",
            bbox=(x1, y1, x2, y2),
            polygon=combined_poly,
            mask=merged_mask,
            class_name=best.class_name,
            confidence=max(a.confidence, b.confidence),
            blocks=merged_blocks,
        )

    def _fallback_merge_polygon(self, a: Bubble, b: Bubble) -> list:
        """fallback: 用 bbox 合并"""
        x1, y1, x2, y2 = (
            min(a.bbox[0], b.bbox[0]),
            min(a.bbox[1], b.bbox[1]),
            max(a.bbox[2], b.bbox[2]),
            max(a.bbox[3], b.bbox[3]),
        )
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    # ---- Split ----

    def _split_large(
        self,
        bubbles: list[Bubble],
        ocr_blocks_map: dict[str, list],
        image_shape: tuple[int, int],
    ) -> list[Bubble]:
        """对超大气泡按 OCR 分布拆分"""
        result: list[Bubble] = []
        for b in bubbles:
            x1, y1, x2, y2 = b.bbox
            area = (x2 - x1) * (y2 - y1)
            if area < self.SPLIT_MIN_AREA:
                result.append(b)
                continue

            blocks = ocr_blocks_map.get(b.id, []) or b.blocks
            if len(blocks) < 2:
                result.append(b)
                continue

            # 按 Y 坐标聚类
            clusters = self._cluster_ocr_blocks(blocks, image_shape)
            if len(clusters) <= 1:
                result.append(b)
                continue

            # 每个 cluster 生成一个子气泡
            for i, cluster in enumerate(clusters):
                sub_bubble = self._make_sub_bubble(b, cluster, i)
                result.append(sub_bubble)

            logger.debug(f"Split: bubble {b.id} → {len(clusters)} sub-bubbles")

        return result

    def _cluster_ocr_blocks(self, blocks: list, image_shape: tuple[int, int]) -> list[list]:
        """按 Y 坐标聚类 OCR blocks"""
        if not blocks:
            return []

        # 按 Y 中心排序
        sorted_blocks = sorted(blocks, key=lambda b: (b.bbox[1] + b.bbox[3]) / 2)
        clusters: list[list] = [[sorted_blocks[0]]]
        _, _, _, prev_y2 = sorted_blocks[0].bbox

        for block in sorted_blocks[1:]:
            _, by1, _, by2 = block.bbox
            gap = by1 - prev_y2
            avg_h = (by2 - by1 + (clusters[-1][0].bbox[3] - clusters[-1][0].bbox[1])) / 2
            if avg_h > 0 and gap > avg_h * self.SPLIT_OCR_GAP_RATIO:
                clusters.append([block])
            else:
                clusters[-1].append(block)
            prev_y2 = by2

        return clusters

    def _make_sub_bubble(self, parent: Bubble, ocr_blocks: list, idx: int) -> Bubble:
        """从父气泡中根据 OCR blocks 创建子气泡"""
        # 计算子 bbox
        xs = [b.bbox[0] for b in ocr_blocks] + [b.bbox[2] for b in ocr_blocks]
        ys = [b.bbox[1] for b in ocr_blocks] + [b.bbox[3] for b in ocr_blocks]
        x1, y1 = min(xs), min(ys)
        x2, y2 = max(xs), max(ys)

        # 扩展 20% 但不超过父边界
        margin_x = int((x2 - x1) * 0.2)
        margin_y = int((y2 - y1) * 0.2)
        x1 = max(parent.bbox[0], x1 - margin_x)
        y1 = max(parent.bbox[1], y1 - margin_y)
        x2 = min(parent.bbox[2], x2 + margin_x)
        y2 = min(parent.bbox[3], y2 + margin_y)

        return Bubble(
            id=f"{parent.id}_split_{idx}",
            bbox=(x1, y1, x2, y2),
            polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
            mask=parent.mask,
            class_name=parent.class_name,
            confidence=parent.confidence,
            blocks=ocr_blocks,
        )

    # ---- Smooth ----

    def _smooth_polygon(self, polygon: list) -> list:
        """多边形平滑：approxPolyDP + Chaikin smooth"""
        if len(polygon) < 4:
            return polygon

        pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)

        # 1. approxPolyDP — 减少顶点数，去除锯齿
        perimeter = cv2.arcLength(pts, True)
        epsilon = perimeter * self.APPROX_EPSILON_RATIO
        approx = cv2.approxPolyDP(pts, epsilon, True)
        approx_pts = [(int(pt[0][0]), int(pt[0][1])) for pt in approx]

        # 2. Chaikin smoothing — 迭代 1 次
        if len(approx_pts) >= 4:
            smooth = self._chaikin_smooth(approx_pts, iterations=1)
            return smooth

        return approx_pts

    def _chaikin_smooth(self, pts: list, iterations: int = 1) -> list:
        """Chaikin 曲线平滑算法"""
        result = pts[:]
        for _ in range(iterations):
            if len(result) < 3:
                break
            new_pts = []
            for i in range(len(result)):
                curr = result[i]
                nxt = result[(i + 1) % len(result)]
                # 插入两个点：25% 和 75% 位置
                qx = int(0.75 * curr[0] + 0.25 * nxt[0])
                qy = int(0.75 * curr[1] + 0.25 * nxt[1])
                rx = int(0.25 * curr[0] + 0.75 * nxt[0])
                ry = int(0.25 * curr[1] + 0.75 * nxt[1])
                new_pts.append((qx, qy))
                new_pts.append((rx, ry))
            # 闭合
            if new_pts:
                new_pts.append(new_pts[0])
            result = new_pts
        return result