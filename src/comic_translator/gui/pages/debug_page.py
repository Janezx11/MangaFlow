"""调试查看器页面 v0.2 — 查看流水线每步结果 + 右侧数据详情 Inspector

增强:
- 数据源切换: 当前 Pipeline 内存 / 历史运行记录
- 右侧 Inspector 面板显示结构化数据
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr
from loguru import logger
from PIL import Image

from comic_translator.gui.components.detection_overlay import (
    bgr_to_pil,
    draw_detection_overlay,
    draw_ocr_overlay,
)
from comic_translator.gui.services.gui_pipeline_service import GuiPipelineService
from comic_translator.gui.services.run_history_service import RunHistoryService
from comic_translator.models.detection_region import DetectionRegion
from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock

MEMORY_SOURCE = "当前内存"
HISTORY_PREFIX = "历史:"


def _detect_mismatch(d: dict) -> str:
    """检测翻译/渲染不一致"""
    (d.get("merged_text") or "").strip()
    translated = (d.get("translated_text") or "").strip()
    render_text = (d.get("render_text") or "").strip()
    trans_status = d.get("translation_status", "")
    render_status = d.get("render_status", "")

    if render_status == "fallback":
        return ""
    if trans_status == "failed" and render_text != "TRANSLATION FAILED":
        return "ERROR: 翻译失败但渲染未显示失败标记"
    if trans_status == "success" and translated and render_text != translated:
        return "ERROR: 渲染文本与译文不一致"
    if render_status == "failed":
        return "ERROR: 渲染失败"
    return ""


def _build_dialogue_rows(dialogues: list[dict]) -> list[list[str]]:
    rows = []
    for d in dialogues:
        mismatch = _detect_mismatch(d)
        rows.append([
            d.get("id", ""),
            (d.get("merged_text") or "")[:80],
            (d.get("translated_text") or "")[:80],
            (d.get("render_text") or "")[:80],
            d.get("translation_status", ""),
            d.get("render_status", ""),
            mismatch,
        ])
    return rows


def _build_integrity(dialogues: list[dict]) -> str:
    issues = []
    status_counts = {"success": 0, "failed": 0, "unchanged": 0, "skipped": 0, "pending": 0}
    for d in dialogues:
        status_counts[d.get("translation_status", "pending")] = status_counts.get(d.get("translation_status", "pending"), 0) + 1
        mismatch = _detect_mismatch(d)
        if mismatch:
            issues.append(mismatch)

    summary = (
        f"**翻译**: success={status_counts['success']}, failed={status_counts['failed']}, "
        f"unchanged={status_counts['unchanged']}, skipped={status_counts['skipped']}  \n"
        f"**不一致**: {len(issues)} 项"
    )
    if issues:
        summary += "\n\n" + "\n".join(f"- {i}" for i in issues[:10])
    return summary


def create_debug_page(
    pipeline_service: GuiPipelineService,
    history_service: RunHistoryService | None = None,
):
    """创建调试查看器标签页"""

    # ---------- 数据源辅助（在组件创建前定义，供初始 choices 使用） ----------

    def list_source_choices() -> list[str]:
        if history_service is None:
            return [MEMORY_SOURCE]
        runs = history_service.list_runs()
        return [MEMORY_SOURCE] + [f"{HISTORY_PREFIX}{r.id}" for r in runs]

    def get_run_input_dir(run_id: str) -> str | None:
        """从历史记录中读取 input_dir"""
        if history_service is None:
            return None
        rec = history_service.load_run(run_id)
        return rec.input_dir if rec else None

    with gr.Column(elem_classes="mf-page"):
        gr.Markdown("调试", elem_classes="mf-page-title")
        gr.Markdown(
            "Debug translation pipeline results and inspect intermediate outputs.",
            elem_classes="mf-desc",
        )

        # ---- Context Toolbar（紧凑，一行内） ----
        with gr.Row():
            source_dropdown = gr.Dropdown(
                label="Source",
                choices=list_source_choices(),
                value=MEMORY_SOURCE,
                interactive=True,
                allow_custom_value=True,
                scale=2,
            )
            page_selector = gr.Dropdown(
                label="Page",
                choices=[],
                value=None,
                interactive=True,
                allow_custom_value=True,
                scale=3,
            )
            refresh_btn = gr.Button("刷新", size="sm", scale=0, min_width=80)

        # ---- Visual Debug Workspace（2×2 四图，等权） ----
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("Original", elem_classes="mf-desc")
                orig_img = gr.Image(label="原图", show_label=False, elem_classes="mf-debug-img")
            with gr.Column(scale=1):
                gr.Markdown("Detection", elem_classes="mf-desc")
                det_img = gr.Image(label="检测区域覆盖图", show_label=False, elem_classes="mf-debug-img")

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("OCR", elem_classes="mf-desc")
                ocr_img = gr.Image(label="OCR 覆盖图", show_label=False, elem_classes="mf-debug-img")
            with gr.Column(scale=1):
                gr.Markdown("Render", elem_classes="mf-desc")
                render_img = gr.Image(label="翻译渲染图", show_label=False, elem_classes="mf-debug-img")

        # ---- Inspector（下方，紧凑，默认只开最常用项） ----
        gr.Markdown("Inspector", elem_classes="mf-section-title")

        with gr.Accordion("页面元数据", open=True):
            meta_md = gr.Markdown("未选择页面")

        with gr.Accordion("检测区域", open=False):
            det_table = gr.Dataframe(
                headers=["ID", "标签", "置信度", "来源", "BBox"],
                value=[],
                interactive=False,
                wrap=False,
                elem_classes="mf-table",
            )

        with gr.Accordion("OCR 文本块", open=False):
            ocr_table = gr.Dataframe(
                headers=["文本", "置信度"],
                value=[],
                interactive=False,
                wrap=False,
                elem_classes="mf-table",
            )

        with gr.Accordion("对话 / 翻译 / 渲染", open=True):
            dlg_table = gr.Dataframe(
                headers=[
                    "ID", "原文", "译文", "渲染文本",
                    "翻译状态", "渲染状态", "不一致",
                ],
                value=[],
                interactive=False,
                wrap=False,
                elem_classes="mf-table",
            )

        with gr.Accordion("翻译完整性检查", open=True):
            integrity_md = gr.Markdown("未选择页面")

        with gr.Row():
            retry_failed_btn = gr.Button(
                "重试本页失败气泡",
                variant="secondary",
                size="sm",
                scale=2,
            )
            retry_status_md = gr.Markdown("", scale=3)

        with gr.Accordion("原始 JSON（高级诊断）", open=False):
            raw_json = gr.Code(label="PageSnapshot JSON", language="json")

    # ---------- 加载页面列表 ----------

    def refresh_all(source):
        """刷新数据源下拉 + 页面列表"""
        sources = list_source_choices()
        # 当前 source 若不在最新列表中（被删除），回退到内存
        if source not in sources:
            source = MEMORY_SOURCE

        if source == MEMORY_SOURCE:
            results = pipeline_service.get_all_results()
            choices = [f"第 {p} 页: {r.file_name}" for p, r in sorted(results.items())]
        else:
            run_id = source.replace(HISTORY_PREFIX, "")
            snapshots = history_service.load_all_pages(run_id) if history_service else []
            choices = [f"第 {s.page} 页: {s.file_name}" for s in snapshots]

        return (
            gr.update(choices=sources, value=source),
            gr.update(choices=choices, value=choices[0] if choices else None),
        )

    # ---------- 显示单页 ----------

    empty_page = (None, None, None, None, "未选择页面", [], [], [], "", "未选择页面")

    def _empty_full():
        """返回 10 个空输出（与 page_outputs 对应）"""
        return empty_page

    def show_page(source, page_str):
        if not page_str:
            return empty_page

        try:
            page_num = int(page_str.split(":")[0].replace("第 ", "").replace(" 页", ""))
        except (ValueError, IndexError):
            return (*empty_page[:4], "页面编号无效", [], [], [], "", "页面编号无效")

        if source == MEMORY_SOURCE:
            return _show_from_memory(page_num)
        run_id = source.replace(HISTORY_PREFIX, "")
        return _show_from_history(run_id, page_num)

    # ---------- 图片查找 ----------

    def _find_image(file_name: str, extra_dir: str | None = None) -> Image.Image | None:
        candidates = []
        if extra_dir:
            candidates.append(Path(extra_dir))
        candidates.extend([Path("data/comics"), Path("data/input")])
        # 也搜索 data/ 下的子目录
        data_dir = Path("data")
        if data_dir.exists():
            for sub in data_dir.iterdir():
                if sub.is_dir() and sub.name not in ("output", "cache"):
                    candidates.append(sub)
        for d in candidates:
            p = d / file_name
            if p.exists():
                return Image.open(p).convert("RGB")
        return None

    def _find_rendered(page: int, rendered_path: str = "") -> Image.Image | None:
        """查找渲染结果图片，支持多位置 fallback"""
        # 1. 优先使用 recorded path
        if rendered_path:
            rp = Path(rendered_path)
            if rp.exists():
                return Image.open(rp)

        # 2. 在标准输出目录中查找
        candidates = [
            Path(f"data/output/rendered/page_{page:04d}_rendered.png"),
            Path(f"data/output/rendered/page_{page:04d}/final_render.png"),
        ]
        for c in candidates:
            if c.exists():
                return Image.open(c)
        return None

    # ---------- 内存数据源 ----------

    def _show_from_memory(page_num: int):
        result = pipeline_service.get_result(page_num)
        if not result:
            return (None, None, None, None, f"第 {page_num} 页无结果", [], [], [], "", "")

        img = _find_image(result.file_name)
        if img is None:
            return (None, None, None, None, f"找不到图片: {result.file_name}", [], [], [], "", "")

        orig = img.copy()

        det_overlay = None
        if result.fused_regions:
            det_bgr = draw_detection_overlay(img, result.fused_regions, color_by="source")
            det_overlay = bgr_to_pil(det_bgr)
        elif result.text_regions:
            regions = [DetectionRegion.from_text_region(tr) for tr in result.text_regions]
            det_bgr = draw_detection_overlay(img, regions, color_by="source")
            det_overlay = bgr_to_pil(det_bgr)

        ocr_overlay = None
        if result.ocr_blocks:
            ocr_bgr = draw_ocr_overlay(img, result.ocr_blocks)
            ocr_overlay = bgr_to_pil(ocr_bgr)

        rendered = _find_rendered(result.page, result.rendered_path)

        meta = (
            f"**页码**: {result.page}\n\n"
            f"**文件**: {result.file_name}\n\n"
            f"**检测耗时**: {result.detection_time:.2f}s\n\n"
            f"**OCR 耗时**: {result.ocr_time:.2f}s\n\n"
            f"**渲染耗时**: {result.render_time:.2f}s\n\n"
            f"**区域数**: {len(result.fused_regions)}\n\n"
            f"**OCR 块数**: {len(result.ocr_blocks)}\n\n"
            f"**对话数**: {len(result.dialogues)}"
            + (f"\n\n**错误**: {result.error}" if result.error else "")
        )

        det_rows = [
            [r.id, r.label, f"{r.confidence:.2f}", r.source, str(r.bbox)]
            for r in result.fused_regions
        ]
        ocr_rows = [
            [(b.text or "")[:50], f"{getattr(b, 'confidence', 0.0):.2f}"]
            for b in result.ocr_blocks
        ]
        dialogue_dicts = [
            {
                "id": d.id,
                "merged_text": d.merged_text,
                "translated_text": d.translated_text,
                "render_text": d.render_text,
                "translation_status": d.translation_status,
                "render_status": d.render_status,
            }
            for d in result.dialogues
        ]
        dlg_rows = _build_dialogue_rows(dialogue_dicts)
        integrity = _build_integrity(dialogue_dicts)

        snapshot = pipeline_service._to_snapshot(result)
        raw = json.dumps(snapshot.to_dict(), indent=2, ensure_ascii=False)

        return orig, det_overlay, ocr_overlay, rendered, meta, det_rows, ocr_rows, dlg_rows, raw, integrity

    # ---------- 历史数据源 ----------

    def _show_from_history(run_id: str, page_num: int):
        if history_service is None:
            return (None, None, None, None, "历史服务未启用", [], [], [], "", "")

        snap = history_service.load_page(run_id, page_num)
        if snap is None:
            return (None, None, None, None, f"运行 {run_id} 第 {page_num} 页不存在", [], [], [], "", "")

        # 从 RunRecord 读取原始 input_dir，提高找图命中率
        input_dir = get_run_input_dir(run_id)
        img = _find_image(snap.file_name, extra_dir=input_dir)
        if img is None:
            return (None, None, None, None,
                    f"找不到图片: {snap.file_name} (搜索: {input_dir}, data/comics, data/input)",
                    [], [], [], "", "")

        orig = img.copy()

        det_overlay = None
        if snap.detection:
            try:
                regions = [DetectionRegion(**r) for r in snap.detection]
                det_bgr = draw_detection_overlay(img, regions, color_by="source")
                det_overlay = bgr_to_pil(det_bgr)
            except Exception as e:
                meta_extra = f"\n\n**检测覆盖图错误**: {e}"
            else:
                meta_extra = ""
        else:
            meta_extra = ""

        ocr_overlay = None
        if snap.ocr_blocks:
            try:
                blocks = [NormalizedOCRBlock(**b) for b in snap.ocr_blocks]
                ocr_bgr = draw_ocr_overlay(img, blocks)
                ocr_overlay = bgr_to_pil(ocr_bgr)
            except Exception as e:
                # 与检测分支一致：把失败原因写进元信息，而不是静默丢弃覆盖图
                meta_extra += f"\n\n**OCR 覆盖图错误**: {e}"

        rendered = _find_rendered(snap.page, snap.rendered_path)

        meta = (
            f"**运行**: {run_id}\n\n"
            f"**页码**: {snap.page}\n\n"
            f"**文件**: {snap.file_name}\n\n"
            f"**检测耗时**: {snap.detection_time:.2f}s\n\n"
            f"**OCR 耗时**: {snap.ocr_time:.2f}s\n\n"
            f"**渲染耗时**: {snap.render_time:.2f}s\n\n"
            f"**区域数**: {len(snap.detection)}\n\n"
            f"**OCR 块数**: {len(snap.ocr_blocks)}\n\n"
            f"**对话数**: {len(snap.dialogues)}"
            + (f"\n\n**错误**: {snap.error}" if snap.error else "")
            + meta_extra
        )

        det_rows = [
            [r.get("id", ""), r.get("label", ""),
             f"{r.get('confidence', 0.0):.2f}",
             r.get("source", ""), str(r.get("bbox", ""))]
            for r in snap.detection
        ]
        ocr_rows = [
            [(b.get("text") or "")[:50], f"{b.get('confidence', 0.0):.2f}"]
            for b in snap.ocr_blocks
        ]
        dialogue_dicts = [
            {
                "id": d.get("id", ""),
                "merged_text": d.get("merged_text", ""),
                "translated_text": d.get("translated_text", ""),
                "render_text": d.get("render_text", ""),
                "translation_status": d.get("translation_status", ""),
                "render_status": d.get("render_status", ""),
            }
            for d in snap.dialogues
        ]
        dlg_rows = _build_dialogue_rows(dialogue_dicts)
        integrity = _build_integrity(dialogue_dicts)

        raw = json.dumps(snap.to_dict(), indent=2, ensure_ascii=False)

        return orig, det_overlay, ocr_overlay, rendered, meta, det_rows, ocr_rows, dlg_rows, raw, integrity

    # ---------- 事件绑定 ----------

    page_outputs = [orig_img, det_img, ocr_img, render_img, meta_md,
                    det_table, ocr_table, dlg_table, raw_json, integrity_md]

    # 重试本页失败气泡
    def retry_failed_bubbles(source, page_str):
        """重试当前页所有翻译失败的气泡，重新渲染并刷新显示"""
        if source != MEMORY_SOURCE:
            return ("仅支持对「当前内存」数据源重试失败气泡。\n\n" "历史运行记录请回到流水线页重新运行。", *_empty_full())

        try:
            page_num = int(page_str.split(":")[0].replace("第 ", "").replace(" 页", ""))
        except (ValueError, IndexError, AttributeError):
            return ("页面编号无效。", *_empty_full())

        result = pipeline_service.get_result(page_num)
        if result is None:
            return (f"第 {page_num} 页无结果。", *_empty_full())

        dialogues = result.dialogues
        failed = [d for d in dialogues if d.merged_text.strip() and d.translation_status == "failed"]
        if not failed:
            return ("本页没有失败气泡，无需重试。", *_show_from_memory(page_num))

        # 在线程中跑 async 重试 + 渲染
        import asyncio

        from comic_translator.gui.services.gui_pipeline_service import GuiPipelineService

        def _do_retry():
            loop = asyncio.new_event_loop()
            try:
                loop.run_until_complete(
                    GuiPipelineService.retry_page_failed_bubbles(
                        dialogues, target_lang="zh", project_id=Path(result.file_name).stem,
                    )
                )
                # 重新渲染该页（render_service 从 pipeline_service 获取）
                if pipeline_service.render_service is not None:
                    # 找图优先用当前 run 记录的真实 input_dir，再退回历史默认目录
                    img_path = _find_image(
                        result.file_name,
                        extra_dir=get_run_input_dir(pipeline_service.current_run_id)
                        if pipeline_service.current_run_id else None,
                    )
                    if img_path is not None:
                        img = Image.open(img_path)
                        render_opts = {"fallback_to_source": True}
                        render_res = pipeline_service.render_service.render_page(
                            img, page_num, dialogues, render_options=render_opts,
                        )
                        result.rendered_path = render_res.output_path
            finally:
                loop.close()

        try:
            _do_retry()
        except Exception as e:
            logger.error(f"Retry failed bubbles error: {e}")
            return (f"重试出错: {e}", *_show_from_memory(page_num))

        # 重试后刷新显示
        remaining_failed = sum(
            1 for d in dialogues if d.merged_text.strip() and d.translation_status == "failed"
        )
        msg = (
            f"已重试 {len(failed)} 个失败气泡，"
            f"仍有 {remaining_failed} 个失败。"
            if remaining_failed
            else f"已重试 {len(failed)} 个失败气泡，全部成功 ✅"
        )
        return (msg, *_show_from_memory(page_num))

    retry_failed_btn.click(
        fn=retry_failed_bubbles,
        inputs=[source_dropdown, page_selector],
        outputs=[retry_status_md, *page_outputs],
    )

    refresh_btn.click(
        fn=refresh_all,
        inputs=[source_dropdown],
        outputs=[source_dropdown, page_selector],
    ).then(
        fn=show_page,
        inputs=[source_dropdown, page_selector],
        outputs=page_outputs,
    )
    source_dropdown.change(
        fn=refresh_all,
        inputs=[source_dropdown],
        outputs=[source_dropdown, page_selector],
    ).then(
        fn=show_page,
        inputs=[source_dropdown, page_selector],
        outputs=page_outputs,
    )
    page_selector.change(
        fn=show_page,
        inputs=[source_dropdown, page_selector],
        outputs=page_outputs,
    )
