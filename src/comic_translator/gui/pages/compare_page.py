"""模型对比页面 — 同一张图，多模型并行检测，并排展示"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time
from pathlib import Path

import gradio as gr
from loguru import logger
from PIL import Image

from comic_translator.gui.components.detection_overlay import (
    bgr_to_pil,
    draw_detection_overlay,
)
from comic_translator.gui.services.gui_pipeline_service import GuiPipelineService
from comic_translator.models.detection_region import DetectionRegion


def _run_coro(coro):
    """在独立线程中跑 coroutine，避免 asyncio.run 在已有 event loop 时失败"""
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def create_compare_page(pipeline_service: GuiPipelineService):
    """创建模型对比标签页"""

    with gr.Column(elem_classes="mf-page"):
        gr.Markdown("检测模型对比", elem_classes="mf-page-title")

        with gr.Row():
            image_input = gr.Image(label="上传图片", type="pil", elem_classes="mf-img-stable")
            with gr.Column():
                use_ctbd = gr.Checkbox(label="CTBD (文本)", value=True)
                use_yolo = gr.Checkbox(label="YOLO11 (气泡)", value=True)
                use_locate = gr.Checkbox(label="LocateAnything-3B", value=False)
                conf_slider = gr.Slider(
                    minimum=0.1, maximum=0.9, value=0.3, step=0.05,
                    label="置信度阈值",
                )
                run_btn = gr.Button("开始对比", variant="primary")

        status_md = gr.Markdown("", elem_classes="mf-info-line")

        with gr.Row():
            with gr.Column():
                gr.Markdown("CTBD", elem_classes="mf-desc")
                ctbd_img = gr.Image(label="CTBD 检测", elem_classes="mf-img-stable")
                ctbd_info = gr.Markdown("", elem_classes="mf-info-line")
            with gr.Column():
                gr.Markdown("YOLO11", elem_classes="mf-desc")
                yolo_img = gr.Image(label="YOLO 检测", elem_classes="mf-img-stable")
                yolo_info = gr.Markdown("", elem_classes="mf-info-line")

        with gr.Row():
            with gr.Column():
                gr.Markdown("LocateAnything-3B", elem_classes="mf-desc")
                locate_img = gr.Image(label="LocateAnything 检测", elem_classes="mf-img-stable")
                locate_info = gr.Markdown("", elem_classes="mf-info-line")
            with gr.Column():
                gr.Markdown("融合结果", elem_classes="mf-desc")
                fused_img = gr.Image(label="融合检测", elem_classes="mf-img-stable")
                fused_info = gr.Markdown("", elem_classes="mf-info-line")

    # ---------- 逻辑 ----------

    def run_compare(image, use_ctbd, use_yolo, use_locate, conf):
        if image is None:
            empty = (None, "", None, "", None, "", None, "")
            return (*empty, "请先上传图片")

        pipeline_service._ensure_initialized(conf)

        results = {
            "ctbd": (None, ""),
            "yolo": (None, ""),
            "locate": (None, ""),
            "fused": (None, ""),
        }

        det_lists = []

        # CTBD
        if use_ctbd:
            t0 = time.time()
            try:
                regions = _run_coro(pipeline_service.text_detector.detect(image))
                det_regions = [DetectionRegion.from_text_region(r) for r in regions]
                det_lists.append(det_regions)
                elapsed = time.time() - t0
                overlay = draw_detection_overlay(image, det_regions, color_by="source")
                results["ctbd"] = (
                    bgr_to_pil(overlay),
                    f"**{len(det_regions)}** 个区域 | 耗时 {elapsed:.2f}s",
                )
            except Exception as e:
                results["ctbd"] = (None, f"错误: {e}")

        # YOLO
        if use_yolo:
            t0 = time.time()
            try:
                bubbles = _run_coro(pipeline_service.bubble_provider.detect(image))
                det_regions = [DetectionRegion.from_bubble(b) for b in bubbles]
                det_lists.append(det_regions)
                elapsed = time.time() - t0
                overlay = draw_detection_overlay(image, det_regions, color_by="source")
                results["yolo"] = (
                    bgr_to_pil(overlay),
                    f"**{len(det_regions)}** 个气泡 | 耗时 {elapsed:.2f}s",
                )
            except Exception as e:
                results["yolo"] = (None, f"错误: {e}")

        # LocateAnything-3B
        if use_locate:
            t0 = time.time()
            try:
                from comic_translator.providers.vision import LocateAnythingProvider
                provider = LocateAnythingProvider()
                if not provider.is_available():
                    results["locate"] = (None, "GPU 不可用或 VRAM 不足")
                else:
                    det_regions = _run_coro(provider.detect(image))
                    det_lists.append(det_regions)
                    elapsed = time.time() - t0
                    overlay = draw_detection_overlay(image, det_regions, color_by="source")
                    results["locate"] = (
                        bgr_to_pil(overlay),
                        f"**{len(det_regions)}** 个区域 | 耗时 {elapsed:.2f}s",
                    )
            except Exception as e:
                logger.exception("LocateAnything failed")
                results["locate"] = (None, f"错误: {e}")

        # Fusion
        if len(det_lists) >= 1:
            try:
                fused = pipeline_service.fusion.fuse(det_lists)
                overlay = draw_detection_overlay(image, fused, color_by="source")
                results["fused"] = (
                    bgr_to_pil(overlay),
                    f"**{len(fused)}** 个融合区域（来自 {len(det_lists)} 个模型）",
                )
            except Exception as e:
                results["fused"] = (None, f"错误: {e}")

        return (
            results["ctbd"][0], results["ctbd"][1],
            results["yolo"][0], results["yolo"][1],
            results["locate"][0], results["locate"][1],
            results["fused"][0], results["fused"][1],
            "对比完成",
        )

    run_btn.click(
        fn=run_compare,
        inputs=[image_input, use_ctbd, use_yolo, use_locate, conf_slider],
        outputs=[
            ctbd_img, ctbd_info,
            yolo_img, yolo_info,
            locate_img, locate_info,
            fused_img, fused_info,
            status_md,
        ],
    )
