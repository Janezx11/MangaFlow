"""Render Lab 页面 — 交互式测试 Mask + Inpaint Provider"""

from __future__ import annotations

import asyncio
import concurrent.futures
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

from comic_translator.gui.services.gui_pipeline_service import GuiPipelineService
from comic_translator.models.detection_region import DetectionRegion
from comic_translator.providers.inpaint import create_inpaint_provider
from comic_translator.services.mask import create_mask_provider
from comic_translator.services.mask_refinement_service import MaskRefinementService
from comic_translator.services.render_quality_evaluator import RenderQualityEvaluator


def _run_coro(coro):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def _pil_to_bgr(pil: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(bgr: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def create_render_lab_page(pipeline_service: GuiPipelineService):
    """创建 Render Lab 标签页"""

    with gr.Column(elem_classes="mf-page"):
        gr.Markdown("Render Lab", elem_classes="mf-page-title")

        with gr.Row():
            image_input = gr.Image(label="选择图片", type="pil", elem_classes="mf-img-stable")
            with gr.Column():
                mask_provider = gr.Dropdown(
                    label="Mask Provider",
                    choices=["OCR Mask"],
                    value="OCR Mask",
                )
                inpaint_provider = gr.Dropdown(
                    label="Inpaint Provider",
                    choices=["OpenCV", "LaMa", "SD"],
                    value="OpenCV",
                    allow_custom_value=True,
                )
                auto_detect = gr.Checkbox(label="AI 检测文字区域", value=True)
                run_btn = gr.Button("Run", variant="primary")

        status_md = gr.Markdown("就绪", elem_classes="mf-info-line")

        with gr.Row():
            with gr.Column():
                gr.Markdown("Original", elem_classes="mf-desc")
                orig_img = gr.Image(label="原图", elem_classes="mf-img-stable")
            with gr.Column():
                gr.Markdown("Mask", elem_classes="mf-desc")
                mask_img = gr.Image(label="Mask", elem_classes="mf-img-stable")

        with gr.Row():
            with gr.Column():
                gr.Markdown("Inpaint Result", elem_classes="mf-desc")
                inpaint_img = gr.Image(label="修复结果", elem_classes="mf-img-stable")
            with gr.Column():
                gr.Markdown("Final Render", elem_classes="mf-desc")
                final_img = gr.Image(label="最终渲染", elem_classes="mf-img-stable")

        with gr.Row():
            scores_md = gr.Markdown("**质量评分:** 未运行", elem_classes="mf-info-line")

        with gr.Row():
            ab_btn = gr.Button("A/B 对比: OpenCV vs LaMa", variant="secondary")

        with gr.Row():
            with gr.Column():
                gr.Markdown("A/B · OpenCV", elem_classes="mf-desc")
                ab_opencv_img = gr.Image(label="OpenCV Inpaint", elem_classes="mf-img-stable")
            with gr.Column():
                gr.Markdown("A/B · LaMa", elem_classes="mf-desc")
                ab_lama_img = gr.Image(label="LaMa Inpaint", elem_classes="mf-img-stable")

        ab_scores_md = gr.Markdown("**A/B 评分:** 未运行", elem_classes="mf-info-line")

    # ---------- 逻辑 ----------

    def run_lab(image, mask_name, inpaint_name, auto_detect):
        if image is None:
            return (None, None, None, None,
                    "请先上传图片")

        # 1. 检测文字区域 (可选)
        regions = []
        if auto_detect:
            try:
                pipeline_service._ensure_initialized()
                text_regions = _run_coro(pipeline_service.text_detector.detect(image))
                regions = [DetectionRegion.from_text_region(r) for r in text_regions]
            except Exception as e:
                return (image, None, None, None,
                        f"检测失败: {e}")

        # 2. 生成 mask
        img_bgr = _pil_to_bgr(image)
        h, w = img_bgr.shape[:2]

        # 伪造 DialogueBlock 兼容 OCRMaskProvider
        from comic_translator.models.bubble import DialogueBlock, Bubble
        if regions:
            dbs = []
            for i, r in enumerate(regions):
                from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock
                bubble = Bubble(
                    id=f"b{i}",
                    bbox=r.bbox,
                    polygon=r.polygon or [],
                    mask=[],
                    class_name=r.label,
                )
                block = NormalizedOCRBlock(
                    id=f"o{i}", page=1, bbox=r.bbox, text="x", confidence=1.0,
                )
                dbs.append(DialogueBlock(id=f"d{i}", bubble=bubble, ocr_blocks=[block]))
        else:
            bubble = Bubble(id="b0", bbox=(0, 0, w, h), polygon=[], mask=[], class_name="speech")
            dbs = [DialogueBlock(id="d0", bubble=bubble)]

        mask_provider = create_mask_provider({
            "provider": "ocr",
            "padding": 4,
            "text_ratio": 0.5,
            "smooth_kernel": 5,
        })
        raw_mask = mask_provider.generate(image, dbs)
        refined_mask = MaskRefinementService().refine(raw_mask, (h, w),
                                                      [db.bubble.polygon for db in dbs])

        # 3. Inpaint
        provider = create_inpaint_provider({"provider": inpaint_name.lower()})

        if not provider.is_available():
            return (image, _bgr_to_pil((refined_mask[..., None] * np.ones(3, dtype=np.uint8)).astype(np.uint8)),
                    None, None,
                    f"{inpaint_name} 不可用")

        try:
            inpainted_bgr = provider.inpaint(img_bgr, refined_mask)
        except Exception as e:
            return (image, _bgr_to_pil(raw_mask), None, None,
                    f"Inpaint 失败: {e}")

        # 4. 质量评分
        evaluator = RenderQualityEvaluator()
        scores = evaluator.evaluate(
            np.array(image.convert("RGB")),
            cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB),
            refined_mask,
        )

        scores_text = (
            f"**质量评分**: Overall {scores.overall} | "
            f"Text Residual {scores.text_residual} | "
            f"Blur {scores.blur} | Edge {scores.edge_preservation}"
        )

        # 5. 最终渲染 (简单画框，演示 Typography 占位)
        final_bgr = inpainted_bgr.copy()
        for db in dbs:
            x1, y1, x2, y2 = db.bubble.bbox
            cv2.rectangle(final_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(final_bgr, "Typography", (x1 + 5, y1 + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # mask 可视化
        mask_color = np.zeros((h, w, 3), dtype=np.uint8)
        mask_color[refined_mask > 0] = (255, 255, 255)

        return (
            image,
            _bgr_to_pil(mask_color),
            _bgr_to_pil(inpainted_bgr),
            _bgr_to_pil(final_bgr),
            scores_text,
        )

    def run_ab(image, auto_detect):
        """使用相同的 mask 对 OpenCV 和 LaMa 进行 A/B 对比"""
        if image is None:
            return None, None, "请先上传图片"

        regions = []
        if auto_detect:
            try:
                pipeline_service._ensure_initialized()
                text_regions = _run_coro(pipeline_service.text_detector.detect(image))
                regions = [DetectionRegion.from_text_region(r) for r in text_regions]
            except Exception as e:
                return None, None, f"检测失败: {e}"

        img_bgr = _pil_to_bgr(image)
        h, w = img_bgr.shape[:2]

        from comic_translator.models.bubble import DialogueBlock, Bubble
        if regions:
            dbs = []
            for i, r in enumerate(regions):
                from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock
                bubble = Bubble(
                    id=f"b{i}",
                    bbox=r.bbox,
                    polygon=r.polygon or [],
                    mask=[],
                    class_name=r.label,
                )
                block = NormalizedOCRBlock(
                    id=f"o{i}", page=1, bbox=r.bbox, text="x", confidence=1.0,
                )
                dbs.append(DialogueBlock(id=f"d{i}", bubble=bubble, ocr_blocks=[block]))
        else:
            bubble = Bubble(id="b0", bbox=(0, 0, w, h), polygon=[], mask=[], class_name="speech")
            dbs = [DialogueBlock(id="d0", bubble=bubble)]

        mask_provider = create_mask_provider({
            "provider": "ocr",
            "padding": 4,
            "text_ratio": 0.5,
            "smooth_kernel": 5,
        })
        raw_mask = mask_provider.generate(image, dbs)
        refined_mask = MaskRefinementService().refine(raw_mask, (h, w),
                                                      [db.bubble.polygon for db in dbs])

        evaluator = RenderQualityEvaluator()
        results = {}
        for name in ("opencv", "lama"):
            provider = create_inpaint_provider({"provider": name})
            if not provider.is_available():
                results[name] = None
                continue
            try:
                inpainted_bgr = provider.inpaint(img_bgr, refined_mask)
            except Exception as e:
                results[name] = (None, str(e))
                continue
            scores = evaluator.evaluate(
                np.array(image.convert("RGB")),
                cv2.cvtColor(inpainted_bgr, cv2.COLOR_BGR2RGB),
                refined_mask,
            )
            results[name] = (_bgr_to_pil(inpainted_bgr), scores)

        opencv_result = results.get("opencv")
        lama_result = results.get("lama")

        def _fmt(res, label):
            if res is None:
                return f"**{label}**: 不可用"
            img, scores = res
            if img is None:
                return f"**{label}**: 失败 ({scores})"
            return (
                f"**{label}**: Overall {scores.overall} | "
                f"Text Residual {scores.text_residual} | "
                f"Blur {scores.blur} | Edge {scores.edge_preservation}"
            )

        scores_text = f"{_fmt(opencv_result, 'OpenCV')}  \n{_fmt(lama_result, 'LaMa')}"

        opencv_img = opencv_result[0] if opencv_result and opencv_result[0] else None
        lama_img = lama_result[0] if lama_result and lama_result[0] else None
        return opencv_img, lama_img, scores_text

    # ---------- 绑定 ----------

    run_btn.click(
        fn=run_lab,
        inputs=[image_input, mask_provider, inpaint_provider, auto_detect],
        outputs=[orig_img, mask_img, inpaint_img, final_img, scores_md],
    )
    ab_btn.click(
        fn=run_ab,
        inputs=[image_input, auto_detect],
        outputs=[ab_opencv_img, ab_lama_img, ab_scores_md],
    )
