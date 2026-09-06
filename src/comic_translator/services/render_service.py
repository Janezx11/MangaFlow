"""Render Engine V7 —— Comic Typography Engine 渲染管线

输入: Comic Image → YOLO Bubble Detection → OCR → BubbleMatching → DialogueBlock
输出: Rendered comic page with comic-quality typography

全部排版委托给 TypographyServiceV2。
RenderService 仅负责:
  - Text Normalization
  - Inpaint (OCR polygon)
  - Typography Engine 编排
  - 文本绘制
  - QA
  - Debug 输出
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import ClassVar


def merge_ocr_rect(dialogue_block) -> tuple[int, int, int, int] | None:
    """合并一个 DialogueBlock 内所有 OCR block 的 bbox，得到原文占位矩形。

    用于排版时把字号/换行限制在原文占位内（贴合原文）。无 OCR 块时返回 None。
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
from typing import TYPE_CHECKING

import cv2
import numpy as np
from loguru import logger
from PIL import Image, ImageDraw

from comic_translator.models.render_models import RenderPageResult

if TYPE_CHECKING:
    from comic_translator.models.bubble import DialogueBlock
    from comic_translator.services.background_cleaner import BackgroundCleaner
    from comic_translator.services.font_manager import FontManager
    from comic_translator.services.inpaint_service import InpaintService
    from comic_translator.services.render_qa_service import RenderQAService
    from comic_translator.services.text_normalizer import TextNormalizer
    from comic_translator.services.typography_qa import TypographyQA
    from comic_translator.services.typography_service_v2 import TypographyServiceV2


class RenderService:
    """渲染服务 V7 —— Comic Typography Engine

    流程:
      DialogueBlock (from YOLO + OCR Matching)
      ↓
      Text Normalization
      ↓
      OCR Polygon Text Mask → Inpaint
      ↓
      TypographyServiceV2 (Bubble-aware layout)
      ↓
      FontManager → Draw
      ↓
      TypographyQA → RenderQA
      ↓
      Debug Output (incl. layout overlay)
    """

    REGION_COLORS: ClassVar[dict[str, tuple[int, int, int]]] = {
        "speech": (0, 255, 0),
        "narration": (0, 128, 255),
        "title": (255, 0, 0),
        "sfx": (255, 255, 0),
    }

    def __init__(
        self,
        text_normalizer: TextNormalizer,
        font_manager: FontManager,
        typography_v2: TypographyServiceV2,
        typography_qa: TypographyQA,
        qa_service: RenderQAService,
        inpaint_service: InpaintService,
        background_cleaner: BackgroundCleaner,
    ) -> None:
        self._normalizer = text_normalizer
        self._fm = font_manager
        self._typo_v2 = typography_v2
        self._typo_qa = typography_qa
        self._qa = qa_service
        self._inpaint = inpaint_service
        self._cleaner = background_cleaner

    # ---- OCR polygon text mask ----

    def _create_ocr_polygon_mask(
        self,
        dialogue_blocks: list[DialogueBlock],
        image_shape: tuple[int, int],
    ) -> np.ndarray:
        """生成覆盖气泡内部区域的 mask（保留气泡边框）

        不再只擦 OCR 文本区域，改为擦除气泡内部安全区域，
        确保渲染文字位置和擦除位置一致。
        """
        h, w = image_shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        for db in dialogue_blocks:
            # 使用气泡 polygon 或 bbox 擦除内部区域
            polygon = db.bubble.polygon
            if polygon and len(polygon) >= 3:
                pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
                # 向内缩进 3px 保留气泡边框
                mask_temp = np.zeros((h, w), dtype=np.uint8)
                cv2.fillPoly(mask_temp, [pts], 255)
                dist = cv2.distanceTransform(mask_temp, cv2.DIST_L2, 3)
                erode = (dist >= 3).astype(np.uint8) * 255
                mask = cv2.bitwise_or(mask, erode)
            else:
                # fallback: 用 bbox，向内缩进 2px
                x1, y1, x2, y2 = map(int, db.bubble.bbox)
                x1 = max(0, x1 + 2)
                y1 = max(0, y1 + 2)
                x2 = min(w - 1, x2 - 2)
                y2 = min(h - 1, y2 - 2)
                if x2 > x1 and y2 > y1:
                    mask[y1:y2, x1:x2] = 255
        return mask

    # ---- Main render ----

    def _prepare_render_text(self, dialogue_blocks, fallback_to_source: bool):
        for db in dialogue_blocks:
            source = (db.merged_text or "").strip()
            translated = (db.translated_text or "").strip()

            if not source:
                db.translation_status = "skipped"
                db.render_text = ""
                db.render_status = "skipped"
                continue

            if not db.translation_status or db.translation_status == "pending":
                if not translated:
                    db.translation_status = "failed"
                elif translated == source:
                    db.translation_status = "unchanged"
                else:
                    db.translation_status = "success"

            if translated:
                db.render_text = translated
                db.render_status = "pending"
            elif fallback_to_source:
                db.render_text = source
                db.render_status = "fallback"
            else:
                db.render_text = "TRANSLATION FAILED"
                db.render_status = "failed"

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
        """渲染单页（Comic Typography Engine）"""
        options = render_options or {}
        fallback_to_source = options.get("fallback_to_source", False)
        font_name = options.get("typography_font", "comic")
        try:
            self._fm.set_font_override(font_name)
        except Exception as e:
            logger.warning(f"Font override failed ({font_name}): {e}")

        img_np = np.array(image.convert("RGB"))
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Text Normalization + 状态
        self._prepare_render_text(dialogue_blocks, fallback_to_source)
        for db in dialogue_blocks:
            if db.render_text != "TRANSLATION FAILED":
                db.render_text = self._normalizer.normalize(db.render_text)

        # 2. Inpaint: 只擦 OCR polygon（保留气泡边框）
        ocr_mask = self._create_ocr_polygon_mask(dialogue_blocks, (h, w))
        if ocr_mask.sum() > 0:
            try:
                cleaned = cv2.inpaint(img_bgr, ocr_mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
            except Exception:
                cleaned = cv2.inpaint(img_bgr, ocr_mask, inpaintRadius=5, flags=cv2.INPAINT_NS)
        else:
            cleaned = img_bgr.copy()

        # Save inpaint mask
        cv2.imwrite(str(out_dir / f"page_{page:04d}_mask.png"), ocr_mask)

        # 3. Typography Engine V2 → Layout + Draw
        draw_img = Image.fromarray(cv2.cvtColor(cleaned, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(draw_img)
        debug_blocks: list[dict] = []
        typography_scores: list[dict] = []

        for db in dialogue_blocks:
            polygon = db.bubble.polygon if db.bubble.polygon and len(db.bubble.polygon) >= 3 else []
            bbox = db.bubble.bbox
            render_text = db.render_text or ""
            if not render_text.strip():
                if db.render_status != "skipped":
                    db.render_status = "failed"
                continue

            bubble_mask = getattr(db.bubble, 'mask', None)
            layout = self._typo_v2.layout(
                polygon=polygon,
                bbox=bbox,
                text=render_text,
                region_type=db.region_type,
                image_shape=(h, w),
                mask=bubble_mask,
                ocr_rect=merge_ocr_rect(db),
            )

            if layout is None:
                db.render_status = "failed"
                continue

            # 更新 DialogueBlock
            db.font_size = layout.font_size
            db.line_count = layout.line_count
            db.lines = [li.text for li in layout.lines]

            # 绘制
            profile = self._fm.get_profile_by_name(db.region_type)
            font = self._fm.load_font(profile.font_path, layout.font_size)

            for li in layout.lines:
                self._fm.draw_text(draw, li.x, li.y, li.text, font, profile)

            # Typography QA
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
                "mask_provider": "ocr_polygon",
                "inpaint_provider": "opencv",
            })

        # 4. Render QA
        qa = self._qa.check(dialogue_blocks, w, h) if run_qa else None

        # 5. Save rendered image（优先继承原文件名；无 source_path 时退回页码命名）
        if source_path:
            out_path = out_dir / f"{Path(source_path).stem}_translated.png"
        else:
            out_path = out_dir / f"page_{page:04d}_rendered.png"
        draw_img.save(str(out_path))

        # 6. Debug JSON (with typography scores)
        json_path = out_dir / f"page_{page:04d}_render_debug.json"
        json_data = {
            "page": page,
            "image_size": [w, h],
            "mask_provider": "ocr_polygon",
            "inpaint_provider": "opencv",
            "blocks": debug_blocks,
            "typography_scores": typography_scores,
            "qa": {"passed": qa.passed, "issues": qa.issues} if qa else None,
        }
        json_path.write_text(json.dumps(json_data, ensure_ascii=False, indent=2), encoding="utf-8")

        translation_debug = {
            "page": page,
            "mask_provider": "ocr_polygon",
            "inpaint_provider": "opencv",
            "blocks": [
                {
                    "block_id": db.id,
                    "source_text": db.merged_text or "",
                    "translated_text": db.translated_text or "",
                    "render_text": db.render_text or "",
                    "translation_status": db.translation_status or "pending",
                    "render_status": db.render_status or "pending",
                }
                for db in dialogue_blocks
            ],
        }
        (out_dir / f"page_{page:04d}_translation_debug.json").write_text(
            json.dumps(translation_debug, ensure_ascii=False, indent=2), encoding="utf-8"
        )

        # 7. Debug overlays
        self._draw_debug_overlay(img_bgr.copy(), dialogue_blocks, out_dir, page)

        # Compute avg typography score
        avg_typo_score = 0.0
        if typography_scores:
            avg_typo_score = sum(s["typography_score"] for s in typography_scores) / len(typography_scores)

        logger.info(
            f"Page {page}: {len(dialogue_blocks)} blocks, "
            f"typo_score={avg_typo_score:.1f}, -> {out_path}"
        )
        return RenderPageResult(
            page=page, source_path=source_path, output_path=str(out_path),
            blocks_rendered=len(dialogue_blocks), qa=qa,
        )

    # ---- Debug overlays ----

    def _draw_debug_overlay(
        self,
        image: np.ndarray,
        blocks: list[DialogueBlock],
        out_dir: Path,
        page: int,
    ) -> None:
        """绘制调试叠加：bubble bbox + polygon + region_type + OCR 映射"""
        for db in blocks:
            color = self.REGION_COLORS.get(db.region_type, (128, 128, 128))
            x1, y1, x2, y2 = db.bubble.bbox
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            if db.bubble.polygon and len(db.bubble.polygon) >= 3:
                pts = np.array(db.bubble.polygon, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(image, [pts], True, color, 1)
            label = f"{db.id} [{db.region_type}] sz={db.font_size} ln={db.line_count}"
            cv2.putText(image, label, (x1 + 2, y1 + 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
            for ocr in db.ocr_blocks:
                ox1, oy1, ox2, oy2 = map(int, ocr.bbox)
                cv2.rectangle(image, (ox1, oy1), (ox2, oy2), (255, 255, 255), 1)

        cv2.imwrite(str(out_dir / f"page_{page:04d}_debug_overlay.png"), image)

    def draw_layout_debug(
        self,
        image: np.ndarray,
        dialogue_blocks: list[DialogueBlock],
        out_dir: Path,
        page: int,
    ) -> None:
        """绘制排版调试图：安全区域、段落实框、基线、视觉中心"""
        debug = image.copy()
        for db in dialogue_blocks:
            color = self.REGION_COLORS.get(db.region_type, (128, 128, 128))

            # Bubble polygon
            if db.bubble.polygon and len(db.bubble.polygon) >= 3:
                pts = np.array(db.bubble.polygon, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(debug, [pts], True, (200, 200, 200), 1)

            # 重新排版获取详细布局信息
            bubble_mask = getattr(db.bubble, 'mask', None)
            layout = self._typo_v2.layout(
                polygon=db.bubble.polygon if db.bubble.polygon else [],
                bbox=db.bubble.bbox,
                text=db.translated_text,
                region_type=db.region_type,
                image_shape=debug.shape[:2],
                mask=bubble_mask,
            )
            if layout is None:
                continue

            # Safe area
            sa = layout.safe_area
            if len(sa.polygon) >= 3:
                sa_pts = np.array(sa.polygon, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(debug, [sa_pts], True, (0, 200, 255), 1)

            # Paragraph box (dashed)
            x1, y1, x2, y2 = sa.bbox
            for dash_x in range(x1, x2, 8):
                cv2.rectangle(debug, (dash_x, y1), (min(dash_x + 4, x2), y2), (255, 200, 0), 1)

            # Optical center
            oc = (int((x1 + x2) / 2), int(layout.optical_center_y + layout.total_height / 2))
            cv2.circle(debug, oc, 4, (0, 0, 255), -1)
            cv2.putText(debug, "OC", (oc[0] + 6, oc[1] + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

            # 每行基线
            for li in layout.lines:
                cv2.line(debug, (li.x, li.y), (li.x + li.width, li.y), (0, 255, 255), 1)

            # Label
            x1, y1, _, _ = db.bubble.bbox
            cv2.putText(debug, f"score={layout.score:.2f}", (x1 + 2, y1 + 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

        cv2.imwrite(str(out_dir / f"page_{page:04d}_layout.png"), debug)
