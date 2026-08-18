"""RegionClassifier — 基于规则的 Region Type 分类

不依赖 YOLO class_name，根据气泡几何和文本特征重新分类。

规则:
  - 页面顶部 10% + 短文本 → TITLE
  - 近似矩形（polygon 与 bbox 面积比 > 0.85）→ NARRATION
  - 极小区域 + 高密度文字 → SFX
  - 其余 → DIALOGUE
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from comic_translator.models.bubble import Bubble
    from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock


class RegionClassifier:
    """基于规则的 Region Type 分类器"""

    # 页面顶部多少比例视为 TITLE 区域
    TITLE_TOP_RATIO = 0.1
    # 矩形判定：polygon 面积 / bbox 面积
    RECT_AREA_RATIO = 0.85
    # SFX: bbox 面积阈值
    SFX_MAX_AREA = 5000

    def classify(
        self,
        bubble: Bubble,
        image_height: int,
        image_width: int,
    ) -> str:
        """对气泡进行 region type 分类

        优先使用 YOLO class_name（微调模型），仅 speech 走几何分类做二次确认。

        Args:
            bubble: 气泡
            image_height: 图片高度
            image_width: 图片宽度

        Returns:
            speech / narration / title / sfx
        """
        # 优先使用 YOLO class_name（微调模型已分类，直接采用）
        if bubble.class_name:
            return bubble.class_name

        x1, y1, x2, y2 = bubble.bbox
        bw = x2 - x1
        bh = y2 - y1
        area = bw * bh
        if area <= 0:
            return "speech"

        # 1. SFX: 极小区域
        if area < self.SFX_MAX_AREA:
            if bubble.blocks:
                text_len = sum(len(b.text or "") for b in bubble.blocks)
                if text_len > 0 and text_len / area > 0.01:
                    return "sfx"
            return "sfx"

        # 2. TITLE: 页面顶部 + 短文本
        if y1 < image_height * self.TITLE_TOP_RATIO and y2 < image_height * 0.15:
            if bubble.blocks:
                total_text = " ".join(b.text or "" for b in bubble.blocks)
                if len(total_text) < 30:
                    return "title"
            return "title"

        # 3. 判断是否为矩形旁白框 vs 椭圆/带尾巴的对话气泡
        # 使用 convex hull 面积比 + 凸性缺陷检测
        if self._is_narration_box(bubble):
            return "narration"

        # 4. 默认：speech
        return "speech"

    def _is_narration_box(self, bubble: Bubble) -> bool:
        """判断是否为矩形旁白框（非对话气泡）"""
        poly = np.array(bubble.polygon, dtype=np.int32) if len(bubble.polygon) >= 3 else None
        if poly is None:
            return True  # 无 polygon 视为矩形

        x1, y1, x2, y2 = bubble.bbox
        bw = x2 - x1
        bh = y2 - y1
        bbox_area = bw * bh
        if bbox_area <= 0:
            return True

        # 面积比：接近 1 表示矩形
        hull = cv2.convexHull(poly)
        hull_area = abs(cv2.contourArea(hull))
        area_ratio = hull_area / bbox_area

        # 凸性缺陷：有尾巴的气泡会有明显缺陷
        hull_rev = cv2.convexHull(poly, returnPoints=False)
        defects = cv2.convexityDefects(poly, hull_rev) if len(poly) >= 3 and hull_rev.shape[0] >= 3 else None

        # 有显著凸性缺陷 → 有尾巴 → 对话气泡
        has_tail = False
        if defects is not None:
            for i in range(defects.shape[0]):
                _, _, depth, _ = defects[i, 0]
                # 深度 > 平均边长的 10% 视为显著缺陷
                avg_edge = (bw + bh) / 2
                if depth > avg_edge * 0.08:
                    has_tail = True
                    break

        # 面积比 > 0.90 且无尾巴 → 矩形旁白框
        return area_ratio > 0.90 and not has_tail


# Lazy import to avoid circular deps at module level
import cv2  # noqa: E402