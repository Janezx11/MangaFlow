"""性能监控页面 — GPU 状态 + 流水线阶段耗时 + 历史运行汇总

运行中/运行后可随时刷新查看：
- 每页各阶段平均耗时（检测 / OCR / 翻译 / 渲染）
- 限流 429 与翻译重试次数
- GPU 设备与显存
- 最近 N 次运行记录
"""

from __future__ import annotations

import time

import gradio as gr

from comic_translator.gui.components.perf_format import (
    format_perf_md,
    format_summary_md,
    get_gpu_info,
)
from comic_translator.gui.services.gui_pipeline_service import GuiPipelineService
from comic_translator.gui.services.run_history_service import RunHistoryService

HISTORY_HEADERS = ["运行ID", "项目", "创建时间", "总页数", "成功", "失败", "对话数", "状态", "耗时(s)"]


def create_perf_monitor_page(
    pipeline_service: GuiPipelineService,
    history_service: RunHistoryService | None = None,
):
    """创建性能监控标签页"""

    with gr.Column():
        gr.Markdown("## 性能监控")

        with gr.Row():
            refresh_btn = gr.Button("刷新", variant="primary", scale=1, min_width=100)
            status_md = gr.Markdown("点击刷新获取最新数据", scale=4)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 当前 / 最近一次运行")
                live_md = gr.Markdown("_无数据_")
            with gr.Column():
                gr.Markdown("### 硬件环境")
                gr.Markdown(get_gpu_info())

        with gr.Accordion("本次运行汇总", open=True):
            summary_md = gr.Markdown("_无数据_")

        gr.Markdown("### 历史运行记录")
        history_table = gr.Dataframe(
            headers=HISTORY_HEADERS,
            value=[],
            interactive=False,
            wrap=True,
            row_count=15,
        )

    # ---------- 逻辑 ----------

    def refresh():
        snap = pipeline_service.get_perf_snapshot()
        live = format_perf_md(snap)

        # 汇总：如果 run_started 且 pages_done>0，显示汇总
        if snap and snap.get("pages_done", 0) > 0:
            elapsed = None
            if snap.get("run_started"):
                elapsed = time.time() - snap["run_started"]
            summary = format_summary_md(snap, elapsed)
        else:
            summary = "_尚未运行或运行中（等完成后再看汇总）_"

        # 历史记录
        rows = []
        if history_service is not None:
            for r in history_service.list_runs():
                rows.append([
                    r.id, r.project, r.created_time,
                    r.total_pages, r.success_pages, r.failed_pages,
                    r.total_dialogues, r.status, r.elapsed_s,
                ])

        msg = f"刷新完成 — 历史 {len(rows)} 条"
        return live, summary, rows, msg

    refresh_btn.click(
        fn=refresh,
        outputs=[live_md, summary_md, history_table, status_md],
    )

    # 页面加载时自动刷新一次
    try:
        live, summary, rows, msg = refresh()
        live_md.value = live
        summary_md.value = summary
        history_table.value = rows
        status_md.value = msg
    except Exception:
        pass
