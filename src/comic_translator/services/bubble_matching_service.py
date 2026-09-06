"""BubbleMatchingService V2 — 综合评分匹配系统

不再依赖 cv2.pointPolygonTest() 单一规则。
使用四项综合评分 + mask dilation fallback。

评分公式:
  score = 0.45 * bbox_iou
        + 0.25 * polygon_overlap
        + 0.20 * center_distance_score
        + 0.10 * area_ratio_score

匹配策略:
  >= 0.65 → 直接匹配
  0.45 ~ 0.65 → 候选匹配
  < 0.45  → singleton

Bubble mask 扩展:
  原始 polygon → expanded mask (5px/10px/20px) → bbox overlap
"""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger

from comic_translator.models.bubble import Bubble, DialogueBlock
from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock

# 匹配阈值
DIRECT_MATCH_THRESHOLD = 0.50
CANDIDATE_THRESHOLD = 0.25

# 评分权重
# polygon_overlap 最重要（OCR 是否在 bubble mask 内）
# bbox_iou 对气泡内小文本块无效（OCR 远小于气泡），权重降低
W_POLYGON_OVERLAP = 0.50
W_CENTER_DIST = 0.25
W_AREA_RATIO = 0.15
W_BBOX_IOU = 0.10

# Mask dilation 扩展像素（更大 dilation 确保覆盖气泡内文本）
DILATION_SIZES = [10, 20, 40]


class BubbleMatchingService:
    """气泡匹配服务 V2 — 综合评分 + mask dilation"""

    def __init__(self) -> None:
        self._stats = {"total": 0, "direct": 0, "candidate": 0, "singleton": 0}

    def match(
        self,
        bubbles: list[Bubble],
        ocr_blocks: list[NormalizedOCRBlock],
    ) -> list[DialogueBlock]:
        """将 OCR blocks 匹配到 bubbles，输出 DialogueBlock 列表"""
        self._stats = {"total": len(ocr_blocks), "direct": 0, "candidate": 0, "singleton": 0}

        # 0. 预分割：把跨多个气泡的 OCR block 沿气泡边界切开
        ocr_blocks = self._split_cross_bubble_blocks(ocr_blocks, bubbles)

        # 预计算每个 bubble 的 mask 缓存
        mask_cache: dict[str, list[np.ndarray]] = {}
        for b in bubbles:
            mask_cache[b.id] = self._build_mask_stack(b)

        # 1. 计算所有 OCR block → Bubble 评分矩阵
        score_matrix: dict[str, list[tuple[str, float]]] = {b.id: [] for b in bubbles}
        assignments: dict[str, list[NormalizedOCRBlock]] = {b.id: [] for b in bubbles}
        unassigned: list[NormalizedOCRBlock] = []

        for block in ocr_blocks:
            best_bubble, best_score = None, CANDIDATE_THRESHOLD
            scores: list[tuple[str, float]] = []

            for bubble in bubbles:
                score = self._compute_score(block, bubble, mask_cache[bubble.id])
                scores.append((bubble.id, score))
                if score > best_score:
                    best_score = score
                    best_bubble = bubble.id

            if best_bubble and best_score >= DIRECT_MATCH_THRESHOLD:
                assignments[best_bubble].append(block)
                self._stats["direct"] += 1
            elif best_bubble and best_score >= CANDIDATE_THRESHOLD:
                # 候选：暂时分配，后续会检查冲突
                assignments[best_bubble].append(block)
                self._stats["candidate"] += 1
            else:
                unassigned.append(block)
                self._stats["singleton"] += 1

            score_matrix[best_bubble] = scores if best_bubble else []

        # 2. 对每个 bubble 的 blocks 按 Y→X 排序
        for bid in assignments:
            assignments[bid].sort(key=lambda b: (b.bbox[1], b.bbox[0]))
            bubble = next((b for b in bubbles if b.id == bid), None)
            if bubble:
                bubble.blocks = assignments[bid]

        # 3. 构建 DialogueBlocks（合并每个 bubble 的 blocks）
        blocks: list[DialogueBlock] = []
        for bubble in bubbles:
            assigned = assignments.get(bubble.id, [])
            if not assigned:
                continue
            merged = self._merge_ocr_texts(assigned)
            self._merge_bboxes(assigned)
            blocks.append(DialogueBlock(
                id=bubble.id,
                bubble=bubble,
                region_type=bubble.class_name,
                ocr_blocks=assigned,
                merged_text=merged,
            ))

        # 4. Unassigned → singleton DialogueBlocks
        for block in unassigned:
            x1, y1, x2, y2 = map(int, block.bbox)
            fake_bubble = Bubble(
                id=f"singleton_{block.id}",
                bbox=(x1, y1, x2, y2),
                polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                mask=[],
                class_name="speech",
            )
            blocks.append(DialogueBlock(
                id=fake_bubble.id,
                bubble=fake_bubble,
                region_type="speech",
                ocr_blocks=[block],
                merged_text=block.text.strip(),
            ))

        blocks.sort(key=lambda d: (d.bubble.bbox[1], d.bubble.bbox[0]))

        # 5. 输出统计
        len(blocks) - len(unassigned)
        rate = (self._stats["direct"] + self._stats["candidate"]) / max(len(ocr_blocks), 1) * 100
        logger.info(
            f"Matching V2: {len(blocks)} dialogues, "
            f"direct={self._stats['direct']}, "
            f"candidate={self._stats['candidate']}, "
            f"singleton={self._stats['singleton']}, "
            f"rate={rate:.1f}%"
        )
        return blocks

    # ---- 跨气泡 OCR 分割 ----

    def _split_cross_bubble_blocks(
        self,
        ocr_blocks: list[NormalizedOCRBlock],
        bubbles: list[Bubble],
    ) -> list[NormalizedOCRBlock]:
        """检测并分割跨多个气泡的 OCR block

        当两个气泡靠得很近时，OCR 可能把它们的文本识别成一个 block。
        此方法用气泡 bbox 做重叠检测——如果 OCR block 的 bbox 与 2+ 个气泡的 bbox
        都有交集，就在两个气泡质心之间垂直切开，生成两个独立 block。

        YOLO 能正确检出每个气泡，所以用 bbox 判交集就够了，不需要 polygon 测试。
        """
        if not bubbles or not ocr_blocks:
            return ocr_blocks

        # 预计算气泡质心（用 bbox 中心即可）
        bubble_centroids: dict[str, tuple[float, float]] = {}
        for b in bubbles:
            cx = (b.bbox[0] + b.bbox[2]) / 2
            cy = (b.bbox[1] + b.bbox[3]) / 2
            bubble_centroids[b.id] = (cx, cy)

        result: list[NormalizedOCRBlock] = []
        split_count = 0

        for block in ocr_blocks:
            bx1, by1, bx2, by2 = map(int, block.bbox)
            bw = bx2 - bx1
            bh = by2 - by1
            if bw <= 0 or bh <= 0:
                result.append(block)
                continue

            # 找到与此 block 的 bbox 有交集的所有气泡
            intersecting: list[tuple[str, float]] = []  # (bubble_id, 重叠面积比)
            for b in bubbles:
                ox1 = max(bx1, int(b.bbox[0]))
                oy1 = max(by1, int(b.bbox[1]))
                ox2 = min(bx2, int(b.bbox[2]))
                oy2 = min(by2, int(b.bbox[3]))
                if ox2 > ox1 and oy2 > oy1:
                    # 有声交集：计算重叠面积占 OCR block 的比例
                    overlap_area = (ox2 - ox1) * (oy2 - oy1)
                    ratio = overlap_area / (bw * bh)
                    if ratio > 0.05:  # 至少 5% 重叠才计
                        intersecting.append((b.id, ratio))

            # 如果只与 0 或 1 个气泡相交，无需分割
            if len(intersecting) < 2:
                result.append(block)
                continue

            # 与 2+ 个气泡相交 → 需要分割
            split_count += 1
            # 按重叠面积排序，取前两个气泡
            intersecting.sort(key=lambda x: -x[1])
            top2 = intersecting[:2]
            b1_id = top2[0][0]
            b2_id = top2[1][0]

            c1 = bubble_centroids.get(b1_id)
            c2 = bubble_centroids.get(b2_id)
            if c1 is None or c2 is None:
                result.append(block)
                continue

            # 判断两个气泡的相对位置，决定分割方向
            dx = abs(c1[0] - c2[0])
            dy = abs(c1[1] - c2[1])

            if dx >= dy:
                # 左右并排 → 垂直分割（在 x 方向切）
                split_x = int((c1[0] + c2[0]) / 2)
                split_x = max(bx1 + 4, min(bx2 - 4, split_x))
                split_a_bbox = (bx1, by1, split_x, by2)
                split_b_bbox = (split_x, by1, bx2, by2)
                id_a = block.id + "_splitL"
                id_b = block.id + "_splitR"
            else:
                # 上下排列 → 水平分割（在 y 方向切）
                split_y = int((c1[1] + c2[1]) / 2)
                split_y = max(by1 + 4, min(by2 - 4, split_y))
                split_a_bbox = (bx1, by1, bx2, split_y)
                split_b_bbox = (bx1, split_y, bx2, by2)
                id_a = block.id + "_splitT"
                id_b = block.id + "_splitB"

            # 创建两个子 block，各自带完整文本
            # 不切分文本——几何比例切文本不可靠，且分裂后两个气泡都有内容
            # 比"一个气泡空着、另一个气泡揉着两段话"好得多
            split_kwargs = {
                "id": id_a,
                "page": block.page,
                "bbox": split_a_bbox,
                "text": block.text,
                "confidence": block.confidence,
                "language": block.language,
                "metadata": block.metadata,
                "reading_order": block.reading_order,
                "is_merged": False,
                "merged_from_ids": [],
            }
            block_a = NormalizedOCRBlock(**split_kwargs)
            split_kwargs["id"] = id_b
            split_kwargs["bbox"] = split_b_bbox
            block_b = NormalizedOCRBlock(**split_kwargs)

            result.append(block_a)
            result.append(block_b)

            split_pos = split_x if dx >= dy else split_y
            logger.debug(
                f"Split OCR block {block.id} at {'x' if dx >= dy else 'y'}={split_pos}: "
                f"'{block.text[:40]}...' overlaps {b1_id} + {b2_id}"
            )

        if split_count > 0:
            logger.info(
                f"Split {split_count} OCR blocks that crossed bubble boundaries "
                f"({len(ocr_blocks)} -> {len(result)} blocks)"
            )
        return result

    # ---- 评分系统 ----

    def _compute_score(
        self,
        block: NormalizedOCRBlock,
        bubble: Bubble,
        mask_stack: list[np.ndarray],
    ) -> float:
        """计算 OCR block 与 bubble 的综合评分"""
        bx1, by1, bx2, by2 = map(int, block.bbox)
        bb_x1, bb_y1, bb_x2, bb_y2 = bubble.bbox

        # 0. 快速排除：完全不重叠
        if bx2 < bb_x1 or bx1 > bb_x2 or by2 < bb_y1 or by1 > bb_y2:
            return 0.0

        # 1. bbox_iou
        iou = self._bbox_iou((bx1, by1, bx2, by2), (bb_x1, bb_y1, bb_x2, bb_y2))

        # 2. polygon_overlap (使用 mask_stack 逐级尝试)
        overlap = self._polygon_overlap(block, bubble, mask_stack)

        # 3. center_distance_score
        center_dist = self._center_distance_score(block, bubble)

        # 4. area_ratio_score
        area_ratio = self._area_ratio_score(block, bubble)

        # 综合评分
        score = (
            W_BBOX_IOU * iou +
            W_POLYGON_OVERLAP * overlap +
            W_CENTER_DIST * center_dist +
            W_AREA_RATIO * area_ratio
        )
        return score

    def _bbox_iou(
        self,
        ocr_bbox: tuple[int, int, int, int],
        bubble_bbox: tuple[int, int, int, int],
    ) -> float:
        """计算 OCR bbox 和 Bubble bbox 的 IoU"""
        ox1, oy1, ox2, oy2 = ocr_bbox
        bx1, by1, bx2, by2 = bubble_bbox

        # 交集
        ix1 = max(ox1, bx1)
        iy1 = max(oy1, by1)
        ix2 = min(ox2, bx2)
        iy2 = min(oy2, by2)

        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0

        intersection = (ix2 - ix1) * (iy2 - iy1)
        ocr_area = (ox2 - ox1) * (oy2 - oy1)
        bubble_area = (bx2 - bx1) * (by2 - by1)
        union = ocr_area + bubble_area - intersection

        if union <= 0:
            return 0.0
        return intersection / union

    def _polygon_overlap(
        self,
        block: NormalizedOCRBlock,
        bubble: Bubble,
        mask_stack: list[np.ndarray],
    ) -> float:
        """计算 OCR bbox 与 Bubble mask 的重叠比例

        逐级尝试:
          1. 原始 polygon (mask)
          2. expanded mask (5px)
          3. expanded mask (10px)
          4. expanded mask (20px)
          5. bbox overlap (fallback)
        """
        if not mask_stack:
            return self._bbox_overlap(block.bbox, bubble.bbox)

        bx1, by1, bx2, by2 = map(int, block.bbox)
        bw = bx2 - bx1
        bh = by2 - by1
        if bw <= 0 or bh <= 0:
            return 0.0

        # 对每级 mask 计算重叠
        for mask in mask_stack:
            if mask is None or mask.size == 0:
                continue
            mh, mw = mask.shape[:2]
            # 裁剪 OCR bbox 到 mask 范围内
            rx1 = max(0, bx1)
            ry1 = max(0, by1)
            rx2 = min(mw, bx2)
            ry2 = min(mh, by2)
            if rx2 <= rx1 or ry2 <= ry1:
                continue
            # OCR bbox 在 mask 内的有效像素
            ocr_region = mask[ry1:ry2, rx1:rx2]
            overlap_px = int(ocr_region.sum())
            ocr_px = bw * bh
            if ocr_px > 0:
                ratio = overlap_px / ocr_px
                if ratio > 0.02:  # 至少 2% 重叠才有效
                    return min(ratio, 1.0)

        # fallback: bbox overlap
        return self._bbox_overlap(block.bbox, bubble.bbox)

    def _bbox_overlap(
        self,
        ocr_bbox: tuple,
        bubble_bbox: tuple,
    ) -> float:
        """bbox 重叠比例（OCR 面积在 Bubble bbox 内的比例）"""
        ox1, oy1, ox2, oy2 = ocr_bbox
        bx1, by1, bx2, by2 = bubble_bbox
        overlap = max(0, min(ox2, bx2) - max(ox1, bx1)) * \
                  max(0, min(oy2, by2) - max(oy1, by1))
        ocr_area = (ox2 - ox1) * (oy2 - oy1)
        if ocr_area <= 0:
            return 0.0
        return overlap / ocr_area

    def _center_distance_score(self, block: NormalizedOCRBlock, bubble: Bubble) -> float:
        """OCR 中心点到 Bubble polygon 的距离评分

        距离越近分数越高。
        在 polygon 内 = 1.0
        """
        bx1, by1, bx2, by2 = map(int, block.bbox)
        ocr_cx = (bx1 + bx2) / 2
        ocr_cy = (by1 + by2) / 2

        poly = np.array(bubble.polygon, dtype=np.int32) if len(bubble.polygon) >= 3 else None
        if poly is None or len(poly) < 3:
            # 无 polygon，用 bbox 中心距离
            bb_cx = (bubble.bbox[0] + bubble.bbox[2]) / 2
            bb_cy = (bubble.bbox[1] + bubble.bbox[3]) / 2
            dist = np.sqrt((ocr_cx - bb_cx) ** 2 + (ocr_cy - bb_cy) ** 2)
            bb_diag = np.sqrt((bubble.bbox[2] - bubble.bbox[0]) ** 2 +
                              (bubble.bbox[3] - bubble.bbox[1]) ** 2)
            if bb_diag <= 0:
                return 0.0
            return max(0, 1.0 - dist / bb_diag)

        # 点在 polygon 内 → 满分
        inside = cv2.pointPolygonTest(poly, (float(ocr_cx), float(ocr_cy)), False)
        if inside >= 0:
            return 1.0

        # 点到 polygon 的有符号距离
        dist = cv2.pointPolygonTest(poly, (float(ocr_cx), float(ocr_cy)), True)
        if dist < 0:
            # 在外面：用气泡对角线的 1/4 作为归一化基准
            bb_diag = np.sqrt((bubble.bbox[2] - bubble.bbox[0]) ** 2 +
                              (bubble.bbox[3] - bubble.bbox[1]) ** 2)
            norm_dist = min(abs(dist) / max(bb_diag * 0.25, 1), 1.0)
            return max(0, 1.0 - norm_dist)
        return 0.5

    def _area_ratio_score(self, block: NormalizedOCRBlock, bubble: Bubble) -> float:
        """面积比例评分：防止小文字块匹配到巨大气泡或反之"""
        bx1, by1, bx2, by2 = map(int, block.bbox)
        bb_x1, bb_y1, bb_x2, bb_y2 = bubble.bbox

        ocr_area = (bx2 - bx1) * (by2 - by1)
        bubble_area = (bb_x2 - bb_x1) * (bb_y2 - bb_y1)

        if bubble_area <= 0 or ocr_area <= 0:
            return 0.0

        ratio = ocr_area / bubble_area

        # 理想范围：0.05 ~ 0.8
        if 0.05 <= ratio <= 0.8:
            return 1.0
        elif ratio < 0.05:
            return ratio / 0.05  # 0~1 线性
        else:
            # 太大：OCR 几乎等于气泡
            return max(0, 1.0 - (ratio - 0.8) / 0.2)

    # ---- Mask 构建 ----

    def _build_mask_stack(self, bubble: Bubble) -> list[np.ndarray]:
        """构建多级 mask（原始 + 扩展）"""
        masks: list[np.ndarray] = []

        # 1. 原始 mask（从 YOLO mask 或 polygon 构建）
        base_mask = self._get_base_mask(bubble)
        if base_mask is not None:
            masks.append(base_mask)

        # 2. 扩展 mask（dilation）
        if base_mask is not None:
            for d in DILATION_SIZES:
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (d * 2 + 1, d * 2 + 1))
                dilated = cv2.dilate(base_mask, kernel, iterations=1)
                masks.append(dilated)

        return masks

    def _get_base_mask(self, bubble: Bubble) -> np.ndarray | None:
        """从 Bubble 获取原始 mask（优先使用 YOLO mask，fallback polygon）"""
        # 尝试 YOLO mask（已 resize 到原图尺寸）
        if bubble.mask and len(bubble.mask) > 0:
            try:
                mask = np.array(bubble.mask, dtype=np.float32)
                if mask.ndim == 3:
                    mask = mask.squeeze()
                # 检查 mask 尺寸是否合理（至少和 bbox 匹配）
                if mask.ndim == 2 and mask.size > 0:
                    h, w = mask.shape
                    _, _, x2, y2 = bubble.bbox
                    if h >= y2 and w >= x2:
                        return (mask > 0.5).astype(np.uint8) * 255
                    else:
                        # mask 太小，需要 resize
                        target_h = max(y2 + 10, h)
                        target_w = max(x2 + 10, w)
                        import cv2
                        mask_resized = cv2.resize(mask, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                        return (mask_resized > 0.5).astype(np.uint8) * 255
            except Exception:
                pass

        # fallback: 从 polygon 构建 mask
        if len(bubble.polygon) >= 3:
            _x1, _y1, x2, y2 = bubble.bbox
            # 确保 mask 足够大
            h = max(y2 + 10, 100)
            w = max(x2 + 10, 100)
            mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(bubble.polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(mask, [pts], 255)
            return mask

        return None

    # ---- 辅助 ----

    def _merge_ocr_texts(self, blocks: list[NormalizedOCRBlock]) -> str:
        """合并多个 OCR block 的文本"""
        texts = [b.text.strip() for b in blocks if b.text.strip()]
        return " ".join(texts)

    def _merge_bboxes(
        self,
        blocks: list[NormalizedOCRBlock],
    ) -> tuple[float, float, float, float]:
        """合并多个 OCR block 的 bbox"""
        xs = [b.bbox[0] for b in blocks] + [b.bbox[2] for b in blocks]
        ys = [b.bbox[1] for b in blocks] + [b.bbox[3] for b in blocks]
        return (min(xs), min(ys), max(xs), max(ys))

    def get_stats(self) -> dict:
        return dict(self._stats)
