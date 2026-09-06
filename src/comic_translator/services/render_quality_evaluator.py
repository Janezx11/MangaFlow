"""Render Quality Evaluator — 评估最终渲染质量

指标:
1. Text Residual Score — 检测是否残留原文字 (mask 区域内外文字对比)
2. Blur Score — 检测 inpaint 区域是否模糊 (拉普拉斯方差)
3. Edge Preservation — 检测气泡边缘是否损坏 (mask 边缘梯度保留率)
4. Overall Score — 加权综合 0-100
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class RenderQualityScores:
    text_residual: float
    blur: float
    edge_preservation: float
    overall: float


class RenderQualityEvaluator:
    """渲染质量评估器"""

    def __init__(
        self,
        text_residual_weight: float = 0.35,
        blur_weight: float = 0.30,
        edge_weight: float = 0.35,
    ):
        self.text_residual_weight = text_residual_weight
        self.blur_weight = blur_weight
        self.edge_weight = edge_weight

    def evaluate(
        self,
        original: np.ndarray,
        rendered: np.ndarray,
        mask: np.ndarray,
    ) -> RenderQualityScores:
        """评估渲染质量

        Args:
            original: 原图 (BGR/RGB numpy)
            rendered: 最终渲染图 (BGR/RGB numpy)
            mask: uint8 mask,255 为被处理区域

        Returns:
            RenderQualityScores
        """
        mask_bin = (mask > 0).astype(np.uint8) * 255

        # 1. 文字残留分数 —— mask 内原图 vs 渲染图的文字对比度差异
        text_residual = self._text_residual_score(original, rendered, mask_bin)

        # 2. 模糊分数 —— mask 区域内拉普拉斯方差 (越高越清晰)
        blur = self._blur_score(rendered, mask_bin)

        # 3. 边缘保留 —— mask 边界附近梯度保留率
        edge = self._edge_preservation_score(original, rendered, mask_bin)

        overall = (
            self.text_residual_weight * text_residual +
            self.blur_weight * blur +
            self.edge_weight * edge
        ) * 100

        return RenderQualityScores(
            text_residual=round(text_residual * 100, 2),
            blur=round(blur * 100, 2),
            edge_preservation=round(edge * 100, 2),
            overall=round(overall, 2),
        )

    def translation_integrity_check(self, dialogue_blocks: list, qa=None) -> dict:
        """翻译完整性检查：空译文、未变化、渲染不一致、英文残留。

        Returns:
            {"status": "PASS" | "WARNING" | "FAIL", "issues": [...], "checks": {...}}
        """
        issues: list[str] = []
        checks = {
            "empty_translation": [],
            "unchanged": [],
            "mismatch": [],
            "english_residue": getattr(qa, "english_residue_blocks", []) if qa else [],
        }

        for db in dialogue_blocks:
            bid = getattr(db, "id", "?")
            source = (getattr(db, "merged_text", "") or "").strip()
            translated = (getattr(db, "translated_text", "") or "").strip()
            render_text = (getattr(db, "render_text", "") or "").strip()
            trans_status = getattr(db, "translation_status", "")
            render_status = getattr(db, "render_status", "")

            # 1. 空译文 / 翻译失败
            if trans_status == "failed" or render_text == "TRANSLATION FAILED":
                issues.append(f"[{bid}] Translation failed: source='{source[:40]}'")
                checks["empty_translation"].append(bid)

            # 2. 译文与原文相同
            if trans_status == "unchanged" or (source and translated and source == translated):
                issues.append(f"[{bid}] Translation unchanged: '{source[:40]}'")
                checks["unchanged"].append(bid)

            # 3. 渲染文本与译文不一致（且不是因为显式 fallback）
            norm_render = "".join(render_text.split())
            norm_translated = "".join(translated.split())
            if translated and render_text and norm_render != norm_translated:
                if render_status != "fallback":
                    issues.append(
                        f"[{bid}] Render text mismatch: expected='{translated[:40]}' "
                        f"got='{render_text[:40]}'"
                    )
                    checks["mismatch"].append(bid)

        # 英文残留由外部 RenderQAService 提供
        for bid in checks["english_residue"]:
            if bid not in {i.split("]", 1)[0].lstrip("[") for i in issues}:
                issues.append(f"[{bid}] English residue detected in rendered text")

        if any(checks["empty_translation"]) or any(checks["mismatch"]):
            status = "FAIL"
        elif any(checks["unchanged"]) or any(checks["english_residue"]):
            status = "WARNING"
        else:
            status = "PASS"

        return {"status": status, "issues": issues, "checks": checks}

    def _text_residual_score(
        self, original: np.ndarray, rendered: np.ndarray, mask: np.ndarray,
    ) -> float:
        """文字残留: mask 内原图如果有高对比度边缘，而渲染图没有，说明残留"""
        if mask.sum() == 0:
            return 1.0

        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY) if len(original.shape) == 3 else original
        rend_gray = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY) if len(rendered.shape) == 3 else rendered

        # 计算原图 mask 区域内的高频能量 (文字通常有强边缘)
        sobel_orig = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_orig = cv2.convertScaleAbs(sobel_orig)
        sobel_rend = cv2.Sobel(rend_gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_rend = cv2.convertScaleAbs(sobel_rend)

        mask_bool = mask > 0
        energy_orig = sobel_orig[mask_bool].mean()
        energy_rend = sobel_rend[mask_bool].mean()

        # 如果渲染后高频能量大幅下降，说明文字残留少
        if energy_orig <= 1:
            return 1.0
        ratio = energy_rend / energy_orig
        # ratio 越小越好，但完全模糊也不好，所以用 sigmoid 映射
        score = 1.0 - min(ratio, 1.0)
        return float(score)

    def _blur_score(self, rendered: np.ndarray, mask: np.ndarray) -> float:
        """inpaint 区域清晰度: 拉普拉斯方差"""
        if mask.sum() == 0:
            return 1.0

        gray = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY) if len(rendered.shape) == 3 else rendered
        lap = cv2.Laplacian(gray, cv2.CV_64F)
        mask_bool = mask > 0
        variance = lap[mask_bool].var()

        # 经验阈值: < 50 模糊，> 300 清晰
        score = min(max((variance - 50) / 250, 0.0), 1.0)
        return float(score)

    def _edge_preservation_score(
        self, original: np.ndarray, rendered: np.ndarray, mask: np.ndarray,
    ) -> float:
        """mask 边界附近原图边缘是否保留"""
        if mask.sum() == 0:
            return 1.0

        # 膨胀 mask 得到边界区域
        kernel = np.ones((5, 5), np.uint8)
        dilated = cv2.dilate(mask, kernel, iterations=1)
        eroded = cv2.erode(mask, kernel, iterations=1)
        boundary = cv2.bitwise_xor(dilated, eroded)

        orig_gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY) if len(original.shape) == 3 else original
        rend_gray = cv2.cvtColor(rendered, cv2.COLOR_RGB2GRAY) if len(rendered.shape) == 3 else rendered

        grad_orig = cv2.Sobel(orig_gray, cv2.CV_64F, 1, 1, ksize=3)
        grad_rend = cv2.Sobel(rend_gray, cv2.CV_64F, 1, 1, ksize=3)

        grad_orig = cv2.convertScaleAbs(grad_orig)
        grad_rend = cv2.convertScaleAbs(grad_rend)

        boundary_bool = boundary > 0
        if boundary_bool.sum() == 0:
            return 1.0

        go = grad_orig[boundary_bool].mean()
        gr = grad_rend[boundary_bool].mean()
        if go <= 1:
            return 1.0
        ratio = gr / go
        # 渲染后边界梯度应接近原图 (说明没有破坏气泡边缘)
        score = 1.0 - abs(ratio - 1.0)
        return float(max(score, 0.0))
