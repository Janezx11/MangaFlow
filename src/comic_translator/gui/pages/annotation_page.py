"""标注中心页面 — 漫画检测数据标注工具

工作流:
1. 选数据集 + 图片
2. 点击画布添加多边形顶点 → "完成区域" 闭合
3. Auto Annotate 调 DetectionFusionService 预标注
4. 保存为 YOLO segmentation 格式

GUI 不写业务逻辑，所有数据操作走 AnnotationService / DatasetService。
"""

from __future__ import annotations

import asyncio
import concurrent.futures
from dataclasses import dataclass, field

import cv2
import gradio as gr
import numpy as np
from loguru import logger
from PIL import Image

from comic_translator.gui.services.annotation_service import (
    CLASS_NAMES,
    AnnotationRegion,
    AnnotationService,
    ImageAnnotation,
)
from comic_translator.gui.services.dataset_service import DatasetService
from comic_translator.gui.services.gui_pipeline_service import GuiPipelineService
from comic_translator.models.detection_region import DetectionRegion

CLASS_COLORS = {
    "speech": (0, 255, 0),
    "narration": (0, 128, 255),
    "title": (255, 0, 255),
    "sfx": (255, 255, 0),
}


@dataclass
class AnnotationState:
    """标注页面会话状态"""
    dataset: str = ""
    split: str = "train"
    image_rel: str = ""
    annotation: ImageAnnotation | None = None
    pending_points: list[tuple[float, float]] = field(default_factory=list)


def _run_coro(coro):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
        return ex.submit(asyncio.run, coro).result()


def _render_overlay(
    image_path: str,
    annotation: ImageAnnotation | None,
    pending: list[tuple[float, float]],
) -> Image.Image | None:
    """把已保存的多边形 + 待完成的点绘制到图片上"""
    if not image_path:
        return None
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return None

    arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    if annotation:
        for r in annotation.regions:
            color = CLASS_COLORS.get(r.class_name, (128, 128, 128))
            pts = np.array(r.polygon, dtype=np.int32).reshape((-1, 1, 2))
            cv2.polylines(arr, [pts], True, color, 2)
            x, y = int(r.polygon[0][0]), int(r.polygon[0][1])
            cv2.putText(arr, r.class_name, (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # 待完成的点
    for i, (x, y) in enumerate(pending):
        cv2.circle(arr, (int(x), int(y)), 5, (0, 255, 255), -1)
        cv2.putText(arr, str(i + 1), (int(x) + 6, int(y) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    if len(pending) >= 2:
        pts = np.array(pending, dtype=np.int32).reshape((-1, 1, 2))
        cv2.polylines(arr, [pts], False, (0, 255, 255), 1)

    return Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def create_annotation_page(
    annotation_service: AnnotationService,
    dataset_service: DatasetService,
    pipeline_service: GuiPipelineService,
):
    """创建标注中心标签页"""

    state = gr.State(AnnotationState())

    with gr.Column():
        gr.Markdown("## 标注中心")

        with gr.Row():
            dataset_dd = gr.Dropdown(
                label="数据集",
                choices=dataset_service.list_datasets(),
                value=None,
                interactive=True,
                allow_custom_value=True,
                scale=2,
            )
            split_dd = gr.Dropdown(
                label="划分",
                choices=["train", "val"],
                value="train",
                scale=1,
            )
            image_dd = gr.Dropdown(
                label="图片",
                choices=[],
                value=None,
                interactive=True,
                allow_custom_value=True,
                scale=3,
            )
            refresh_btn = gr.Button("刷新", size="sm", scale=1, min_width=80)

        with gr.Row():
            # 左侧: 画布
            with gr.Column(scale=3):
                canvas = gr.Image(
                    label="点击画布添加多边形顶点",
                    type="pil",
                    interactive=True,
                    height=600,
                )
                with gr.Row():
                    class_dd = gr.Dropdown(
                        label="类别",
                        choices=CLASS_NAMES,
                        value="speech",
                        scale=2,
                    )
                    finish_btn = gr.Button("完成区域", variant="primary", scale=1)
                    undo_btn = gr.Button("撤销点", scale=1)
                    clear_btn = gr.Button("清空待定点", scale=1)

            # 右侧: 区域列表 + 操作
            with gr.Column(scale=2):
                gr.Markdown("### 已标注区域")
                region_table = gr.Dataframe(
                    headers=["#", "类别", "来源", "顶点数"],
                    value=[],
                    interactive=False,
                )
                with gr.Row():
                    delete_btn = gr.Button("删除选中行", variant="stop", scale=1)
                    delete_index = gr.Number(label="行号", value=0, precision=0, visible=False)

                gr.Markdown("### AI 辅助")
                auto_btn = gr.Button("Auto Annotate (CTBD+YOLO+Fusion)", variant="secondary")
                save_btn = gr.Button("保存 YOLO 标签", variant="primary")

                status_md = gr.Markdown("就绪")

    # ---------- 状态渲染辅助 ----------

    def _render(state: AnnotationState):
        if state.annotation is None:
            return None
        return _render_overlay(
            state.annotation.image_path, state.annotation, state.pending_points,
        )

    def _regions_table(state: AnnotationState):
        if state.annotation is None:
            return []
        return [
            [i, r.class_name, r.source, len(r.polygon)]
            for i, r in enumerate(state.annotation.regions)
        ]

    # ---------- 事件 ----------

    def on_dataset_or_split_change(dataset, split, state):
        if not dataset:
            return gr.update(choices=[], value=None), None, [], state, "请先选数据集"
        images = dataset_service.list_images(dataset, split)
        choices = images
        state.dataset = dataset
        state.split = split
        state.image_rel = ""
        state.annotation = None
        state.pending_points = []
        return (
            gr.update(choices=choices, value=choices[0] if choices else None),
            None, [], state,
            f"数据集 {dataset}/{split} 共 {len(choices)} 张",
        )

    def on_image_change(image_rel, state: AnnotationState):
        if not state.dataset or not image_rel:
            return None, [], state, ""
        img_path = dataset_service.get_image_path(state.dataset, image_rel)
        label_path = dataset_service.get_label_path(state.dataset, image_rel)
        try:
            ann = annotation_service.load_annotation(str(img_path), str(label_path))
        except Exception as e:
            return None, [], state, f"加载失败: {e}"

        state.image_rel = image_rel
        state.annotation = ann
        state.pending_points = []
        return _render(state), _regions_table(state), state, \
            f"已加载 {image_rel} ({len(ann.regions)} 个已有区域)"

    def on_canvas_click(state: AnnotationState, evt: gr.SelectData):
        if state.annotation is None:
            return None, state, "请先选择图片"
        x, y = float(evt.index[0]), float(evt.index[1])
        state.pending_points.append((x, y))
        return _render(state), state, f"待定点 {len(state.pending_points)} 个"

    def on_finish(class_name, state: AnnotationState):
        if state.annotation is None:
            return None, [], state, "请先选择图片"
        if len(state.pending_points) < 3:
            return _render(state), _regions_table(state), state, \
                "至少需要 3 个点才能闭合多边形"

        state.annotation.regions.append(AnnotationRegion(
            class_name=class_name,
            polygon=list(state.pending_points),
            source="manual",
        ))
        state.pending_points = []
        return _render(state), _regions_table(state), state, \
            f"已添加 {class_name} 区域 (共 {len(state.annotation.regions)} 个)"

    def on_undo(state: AnnotationState):
        if state.pending_points:
            state.pending_points.pop()
        return _render(state), state, f"待定点 {len(state.pending_points)} 个"

    def on_clear_pending(state: AnnotationState):
        state.pending_points = []
        return _render(state), state, "已清空待定点"

    def on_delete_row(row_idx, state: AnnotationState):
        if state.annotation is None:
            return None, [], state, "无标注"
        try:
            idx = int(row_idx)
        except (TypeError, ValueError):
            return _render(state), _regions_table(state), state, "行号无效"
        if not (0 <= idx < len(state.annotation.regions)):
            return _render(state), _regions_table(state), state, f"行号 {idx} 超出范围"
        removed = state.annotation.regions.pop(idx)
        return _render(state), _regions_table(state), state, \
            f"已删除区域 {idx} ({removed.class_name})"

    def on_table_select(evt: gr.SelectData):
        return evt.index[0] if evt.index else 0

    def on_auto_annotate(state: AnnotationState):
        if state.annotation is None:
            return None, [], state, "请先选择图片"

        try:
            pipeline_service._ensure_initialized()
        except Exception as e:
            return _render(state), _regions_table(state), state, f"模型初始化失败: {e}"

        try:
            img = Image.open(state.annotation.image_path).convert("RGB")
            text_regions = _run_coro(pipeline_service.text_detector.detect(img))
            bubbles = _run_coro(pipeline_service.bubble_provider.detect(img))

            det_lists = []
            if text_regions:
                det_lists.append([DetectionRegion.from_text_region(tr) for tr in text_regions])
            if bubbles:
                det_lists.append([DetectionRegion.from_bubble(b) for b in bubbles])

            auto_regions = annotation_service.auto_annotate(
                state.annotation.image_path, det_lists, pipeline_service.fusion,
            )
            # 替换所有 source=auto 的旧区域，保留 manual
            state.annotation.regions = [
                r for r in state.annotation.regions if r.source == "manual"
            ] + auto_regions
            return _render(state), _regions_table(state), state, \
                f"AI 预标注 {len(auto_regions)} 个区域"
        except Exception as e:
            logger.exception("Auto annotate failed")
            return _render(state), _regions_table(state), state, f"AI 标注失败: {e}"

    def on_save(state: AnnotationState):
        if state.annotation is None or not state.dataset:
            return state, "无标注可保存"
        label_path = dataset_service.get_label_path(state.dataset, state.image_rel)
        try:
            annotation_service.save_annotation(state.annotation, str(label_path))
        except Exception as e:
            return state, f"保存失败: {e}"
        # 重新生成 yaml
        try:
            dataset_service.regenerate_yaml(state.dataset)
        except Exception:
            pass
        return state, f"已保存到 {label_path.name}（{len(state.annotation.regions)} 个区域，无效区域已跳过，详见日志）"

    # ---------- 绑定 ----------

    refresh_btn.click(
        fn=lambda: gr.update(choices=dataset_service.list_datasets()),
        outputs=[dataset_dd],
    )
    dataset_dd.change(
        fn=on_dataset_or_split_change,
        inputs=[dataset_dd, split_dd, state],
        outputs=[image_dd, canvas, region_table, state, status_md],
    )
    split_dd.change(
        fn=on_dataset_or_split_change,
        inputs=[dataset_dd, split_dd, state],
        outputs=[image_dd, canvas, region_table, state, status_md],
    )
    image_dd.change(
        fn=on_image_change,
        inputs=[image_dd, state],
        outputs=[canvas, region_table, state, status_md],
    )
    canvas.select(
        fn=on_canvas_click,
        inputs=[state],
        outputs=[canvas, state, status_md],
    )
    finish_btn.click(
        fn=on_finish,
        inputs=[class_dd, state],
        outputs=[canvas, region_table, state, status_md],
    )
    undo_btn.click(
        fn=on_undo,
        inputs=[state],
        outputs=[canvas, state, status_md],
    )
    clear_btn.click(
        fn=on_clear_pending,
        inputs=[state],
        outputs=[canvas, state, status_md],
    )
    region_table.select(
        fn=on_table_select,
        outputs=[delete_index],
    )
    delete_btn.click(
        fn=on_delete_row,
        inputs=[delete_index, state],
        outputs=[canvas, region_table, state, status_md],
    )
    auto_btn.click(
        fn=on_auto_annotate,
        inputs=[state],
        outputs=[canvas, region_table, state, status_md],
    )
    save_btn.click(
        fn=on_save,
        inputs=[state],
        outputs=[state, status_md],
    )
