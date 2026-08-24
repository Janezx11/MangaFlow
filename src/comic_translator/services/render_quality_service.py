"""Render Quality Service — 渲染质量 Sprint 编排器

新流程:
  OCR → MaskProvider → MaskRefinement → InpaintProvider → Typography Render

v0.4 增强:
  - 支持动态切换 Mask/Inpaint Provider、字体、背景清理
  - 逐块记录 source/translated/render 文本与状态
  - 禁止 Render 默认回退到 source_text（可配置 fallback）
  - 输出 translation_debug.json + 完整中间产物
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image, ImageDraw

from comic_translator.models.bubble import DialogueBlock
from comic_translator.models.render_models import QAResult, RenderPageResult
from comic_translator.providers.inpaint import create_inpaint_provider
from comic_translator.providers.inpaint.base_inpaint_provider import InpaintProvider
from comic_translator.services.font_manager import FontManager
from comic_translator.services.mask import create_mask_provider
from comic_translator.services.mask.base_mask_provider import MaskProvider
from comic_translator.services.mask_refinement_service import MaskRefinementService
from comic_translator.services.render_qa_service import RenderQAService
from comic_translator.services.render_quality_evaluator import RenderQualityEvaluator
from comic_translator.services.text_normalizer import TextNormalizer
from comic_translator.services.typography_qa import TypographyQA
from comic_translator.services.typography_service_v2 import TypographyServiceV2


class RenderQualityService:
    """高质量渲染编排服务"""

    def __init__(
        self,
        text_normalizer: TextNormalizer,
        font_manager: FontManager,
        typography_v2: TypographyServiceV2,
        typography_qa: TypographyQA,
        qa_service: RenderQAService,
        render_config: dict | None = None,
    ):
        self._normalizer = text_normalizer
        self._fm = font_manager
        self._typo_v2 = typography_v2
        self._typo_qa = typography_qa
        self._qa = qa_service

        cfg = render_config or {}
        self._render_cfg = cfg
        mask_cfg = cfg.get("mask", {})
        inpaint_cfg = cfg.get("inpaint", {})

        self.mask_provider = create_mask_provider(mask_cfg)
        self.mask_refinement = MaskRefinementService(
            enabled=mask_cfg.get("refinement", {}).get("enabled", True),
        )
        self.inpaint_provider = create_inpaint_provider(inpaint_cfg)
        self.evaluator = RenderQualityEvaluator()

    def get_provider_summary(self) -> dict:
        return {
            "mask_provider": self.mask_provider.get_name(),
            "inpaint_provider": self.inpaint_provider.get_name(),
            "inpaint_available": self.inpaint_provider.is_available(),
        }

    def _reconfigure_providers(self, options: dict) -> None:
        """根据 GUI 选项重新配置 Provider（只在变化时重建）"""
        mask_cfg = dict(self._render_cfg.get("mask", {}))
        mask_cfg["provider"] = options.get("mask_provider", "ocr")
        new_mask = create_mask_provider(mask_cfg)

        inpaint_cfg = dict(self._render_cfg.get("inpaint", {}))
        inpaint_cfg["provider"] = options.get("inpaint_provider", "opencv")
        new_inpaint = create_inpaint_provider(inpaint_cfg)

        if new_mask.get_name() != self.mask_provider.get_name():
            self.mask_provider = new_mask
            logger.info(f"Switched mask provider -> {self.mask_provider.get_name()}")

        if new_inpaint.get_name() != self.inpaint_provider.get_name():
            self.inpaint_provider = new_inpaint
            logger.info(f"Switched inpaint provider -> {self.inpaint_provider.get_name()}")

        font_name = options.get("typography_font", "comic")
        try:
            self._fm.set_font_override(font_name)
        except Exception as e:
            logger.warning(f"Font override failed ({font_name}): {e}")

    def _prepare_render_text(self, dialogue_blocks: list[DialogueBlock], fallback_to_source: bool) -> None:
        """确定每块的最终渲染文本，并记录翻译/渲染状态。"""
        for db in dialogue_blocks:
            source = (db.merged_text or "").strip()
            translated = (db.translated_text or "").strip()

            # 没有 OCR 文本的块直接跳过，不渲染任何内容
            if not source:
                db.translation_status = "skipped"
                db.render_text = ""
                db.render_status = "skipped"
                continue

            # 规范已有的 translation_status（如果上游没设置）
            if not db.translation_status or db.translation_status == "pending":
                if not translated:
                    db.translation_status = "failed"
                elif translated == source:
                    db.translation_status = "unchanged"
                else:
                    db.translation_status = "success"

            # 决定渲染文本：禁止默认回退到原文，除非显式允许
            if translated:
                db.render_text = translated
                db.render_status = "pending"
            elif fallback_to_source:
                db.render_text = source
                db.render_status = "fallback"
            else:
                db.render_text = "TRANSLATION FAILED"
                db.render_status = "failed"

            # 中文规范化（对失败标记跳过，避免破坏提示信息）
            if db.render_text != "TRANSLATION FAILED":
                db.render_text = self._normalizer.normalize(db.render_text)

    def _expand_mask_for_background_cleaning(
        self,
        mask: np.ndarray,
        dialogue_blocks: list[DialogueBlock],
    ) -> np.ndarray:
        """背景清理：把 OCR block / bubble 的完整区域加入 mask"""
        h, w = mask.shape[:2]
        expanded = mask.copy()
        for db in dialogue_blocks:
            blocks = getattr(db, "ocr_blocks", []) or []
            if blocks:
                for block in blocks:
                    x1, y1, x2, y2 = map(int, block.bbox)
                    x1 = max(0, x1)
                    y1 = max(0, y1)
                    x2 = min(w, x2)
                    y2 = min(h, y2)
                    if x2 > x1 and y2 > y1:
                        expanded[y1:y2, x1:x2] = 255
            else:
                x1, y1, x2, y2 = map(int, db.bubble.bbox)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                if x2 > x1 and y2 > y1:
                    expanded[y1:y2, x1:x2] = 255
        return expanded

    @staticmethod
    def _merge_ocr_rect(dialogue_block) -> tuple[int, int, int, int] | None:
        """合并该块所有 OCR block 的 bbox，得到原文占位矩形。

        用于排版时把译文字号/换行限制在原文占位内（贴合原文）。
        无 OCR 块时返回 None（退回气泡安全区排版）。
        """
        rects = [
            b.bbox for b in (getattr(dialogue_block, "ocr_blocks", None) or [])
            if getattr(b, "bbox", None)
        ]
        if not rects:
            return None
        x1 = min(int(r[0]) for r in rects)
        y1 = min(int(r[1]) for r in rects)
        x2 = max(int(r[2]) for r in rects)
        y2 = max(int(r[3]) for r in rects)
        if x2 <= x1 or y2 <= y1:
            return None
        return (x1, y1, x2, y2)

    def _build_translation_debug(self, dialogue_blocks: list[DialogueBlock]) -> dict:
        blocks = []
        for db in dialogue_blocks:
            blocks.append({
                "block_id": db.id,
                "source_text": db.merged_text or "",
                "translated_text": db.translated_text or "",
                "render_text": db.render_text or "",
                "translation_status": db.translation_status or "pending",
                "render_status": db.render_status or "pending",
            })
        return {"blocks": blocks}

    def render_page(
        self,
        image: Image.Image,
        page: int,
        dialogue_blocks: list[DialogueBlock],
        output_dir: str = "data/output/rendered",
        source_path: str = "",
        run_qa: bool = True,
        render_options: dict | None = None,
    ) -> RenderPageResult:
        """渲染单页（带 mask/inpaint provider 的高质量版本）"""
        options = render_options or {}
        self._reconfigure_providers(options)

        fallback_to_source = options.get("fallback_to_source", False)
        background_cleaning = options.get("background_cleaning", False)

        img_np = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. 确定最终渲染文本 + 状态
        self._prepare_render_text(dialogue_blocks, fallback_to_source)

        # 2. Mask Provider
        mask = self.mask_provider.generate(image, dialogue_blocks)

        # 2b. 背景清理：扩展 mask 覆盖完整 OCR/bubble 区域
        if background_cleaning:
            mask = self._expand_mask_for_background_cleaning(mask, dialogue_blocks)

        # 3. Mask Refinement
        bubble_polygons = [
            db.bubble.polygon for db in dialogue_blocks
            if db.bubble.polygon and len(db.bubble.polygon) >= 3
        ]
        refined_mask = self.mask_refinement.refine(mask, (h, w), bubble_polygons)

        # 4. Inpaint Provider（传入原始 PIL Image，provider 内部处理颜色转换）
        if refined_mask.sum() > 0:
            try:
                cleaned_bgr = self.inpaint_provider.inpaint(image, refined_mask)
            except Exception as e:
                logger.error(f"Inpaint provider failed ({self.inpaint_provider.get_name()}): {e}")
                # fallback OpenCV
                cleaned_bgr = cv2.inpaint(img_bgr, refined_mask, inpaintRadius=5,
                                          flags=cv2.INPAINT_TELEA)
        else:
            cleaned_bgr = img_bgr.copy()

        # 5. Typography Render
        draw_img = Image.fromarray(cv2.cvtColor(cleaned_bgr, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(draw_img)
        debug_blocks: list[dict] = []
        typography_scores: list[dict] = []

        for db in dialogue_blocks:
            polygon = db.bubble.polygon if db.bubble.polygon and len(db.bubble.polygon) >= 3 else []
            bbox = db.bubble.bbox

            render_text = db.render_text or ""
            if not render_text.strip():
                db.render_status = "failed"
                continue

            layout = self._typo_v2.layout(
                polygon=polygon,
                bbox=bbox,
                text=render_text,
                region_type=db.region_type,
                image_shape=(h, w),
                mask=getattr(db.bubble, 'mask', None),
                ocr_rect=self._merge_ocr_rect(db),
            )
            if layout is None:
                db.render_status = "failed"
                continue

            db.font_size = layout.font_size
            db.line_count = layout.line_count
            db.lines = [li.text for li in layout.lines]

            profile = self._fm.get_profile_by_name(db.region_type)
            font = self._fm.load_font(profile.font_path, layout.font_size)

            for li in layout.lines:
                self._fm.draw_text(draw, li.x, li.y, li.text, font, profile)

            typo_score = self._typo_qa.check(layout, polygon, (h, w))
            typography_scores.append({
                "block_id": db.id,
                "region_type": db.region_type,
                "typography_score": typo_score.overall,
                "balance": typo_score.balance,
                "font_size_score": typo_score.font_size_score,
                "overflow_score": typo_score.overflow_score,
                "center_score": typo_score.center_score,
                "padding_score": typo_score.padding_score,
                "issues": typo_score.issues,
            })

            # 渲染后检查：实际绘制文本与期望渲染文本是否一致（忽略空白差异）
            rendered_text = "".join("\n".join(db.lines).split())
            expected_text = "".join(render_text.split())
            mismatch = rendered_text != expected_text
            if db.render_status == "pending":
                db.render_status = "failed" if mismatch else "success"

            debug_blocks.append({
                "block_id": db.id,
                "region_type": db.region_type,
                "bubble_bbox": list(db.bubble.bbox),
                "source_text": db.merged_text or "",
                "translated_text": db.translated_text or "",
                "render_text": render_text,
                "translation_status": db.translation_status or "pending",
                "render_status": db.render_status,
                "font_size": layout.font_size,
                "line_count": layout.line_count,
                "typography_score": typo_score.overall,
                "font_name": layout.font_name,
                "mask_provider": self.mask_provider.get_name(),
                "inpaint_provider": self.inpaint_provider.get_name(),
            })

        # 6. QA + Quality Eval
        qa = self._qa.check(dialogue_blocks, w, h) if run_qa else None
        quality = self.evaluator.evaluate(img_np, np.array(draw_img), refined_mask)
        translation_integrity = self.evaluator.translation_integrity_check(
            dialogue_blocks, qa
        )

        # 7. Save rendered image to flat output directory（优先继承原文件名）
        if source_path:
            out_path = out_dir / f"{Path(source_path).stem}_translated.png"
        else:
            out_path = out_dir / f"page_{page:04d}_rendered.png"
        draw_img.save(str(out_path))

        # 8. Debug outputs → 放入 _debug/ 子目录，不污染用户输出目录
        debug_dir = out_dir / "_debug"
        self._save_debug(
            debug_dir, page, img_bgr, mask, refined_mask, cleaned_bgr, draw_img,
            debug_blocks, typography_scores, qa, quality, translation_integrity,
        )

        # 9. Translation debug JSON → 也放 _debug/
        translation_debug = self._build_translation_debug(dialogue_blocks)
        translation_debug["page"] = page
        translation_debug["mask_provider"] = self.mask_provider.get_name()
        translation_debug["inpaint_provider"] = self.inpaint_provider.get_name()
        (debug_dir / f"page_{page:04d}_translation_debug.json").write_text(
            json.dumps(translation_debug, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        avg_typo = 0.0
        if typography_scores:
            avg_typo = sum(s["typography_score"] for s in typography_scores) / len(typography_scores)

        logger.info(
            f"RenderQuality page {page}: {len(dialogue_blocks)} blocks, "
            f"typo={avg_typo:.1f}, quality={quality.overall}, "
            f"mask={self.mask_provider.get_name()}, inpaint={self.inpaint_provider.get_name()}, "
            f"translation_integrity={translation_integrity['status']}"
        )

        metadata = {
            "mask_provider": self.mask_provider.get_name(),
            "inpaint_provider": self.inpaint_provider.get_name(),
            "quality": quality.__dict__,
            "translation_integrity": translation_integrity,
        }
        result = RenderPageResult(
            page=page, source_path=source_path, output_path=str(out_path),
            blocks_rendered=len(dialogue_blocks), qa=qa,
        )
        result.metadata = metadata
        return result

    def _save_debug(
        self,
        out_dir: Path,
        page: int,
        img_bgr: np.ndarray,
        raw_mask: np.ndarray,
        refined_mask: np.ndarray,
        cleaned_bgr: np.ndarray,
        final_img: Image.Image,
        debug_blocks: list[dict],
        typography_scores: list[dict],
        qa: QAResult | None,
        quality: "RenderQualityScores",
        translation_integrity: dict,
    ) -> None:
        page_dir = out_dir / f"page_{page:04d}"
        page_dir.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(page_dir / "original.png"), img_bgr)
        cv2.imwrite(str(page_dir / "text_mask.png"), raw_mask)
        cv2.imwrite(str(page_dir / "refined_mask.png"), refined_mask)
        cv2.imwrite(str(page_dir / "inpaint_result.png"), cleaned_bgr)
        final_img.save(str(page_dir / "final_render.png"))

        json_path = page_dir / "render_debug.json"
        json_data = {
            "page": page,
            "mask_provider": self.mask_provider.get_name(),
            "inpaint_provider": self.inpaint_provider.get_name(),
            "quality": quality.__dict__,
            "translation_integrity": translation_integrity,
            "blocks": debug_blocks,
            "typography_scores": typography_scores,
            "qa": {"passed": qa.passed, "issues": qa.issues} if qa else None,
        }
        json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")
