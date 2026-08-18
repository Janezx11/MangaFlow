"""历史任务页面 — 列出所有 Pipeline 运行记录"""

from __future__ import annotations

import gradio as gr

from comic_translator.gui.services.run_history_service import RunHistoryService


def create_history_page(history_service: RunHistoryService):
    """创建历史任务标签页"""

    with gr.Column():
        gr.Markdown("## 历史任务")

        with gr.Row():
            refresh_btn = gr.Button("刷新列表", variant="primary", scale=1)
            delete_btn = gr.Button("删除选中", variant="stop", scale=1)

        status_md = gr.Markdown("")

        runs_table = gr.Dataframe(
            headers=["运行ID", "项目", "输入目录", "创建时间", "总页数", "成功", "失败", "对话数", "状态", "耗时(s)"],
            value=[],
            interactive=False,
            wrap=True,
        )

        selected_run = gr.Textbox(label="选中的运行ID", interactive=False, visible=False)

    # ---------- 逻辑 ----------

    def refresh_table():
        runs = history_service.list_runs()
        rows = []
        for r in runs:
            rows.append([
                r.id, r.project, r.input_dir, r.created_time,
                r.total_pages, r.success_pages, r.failed_pages,
                r.total_dialogues, r.status, r.elapsed_s,
            ])
        return rows, f"共 {len(rows)} 条记录"

    def on_select(evt: gr.SelectData, table):
        if evt.index is None:
            return ""
        row_idx = evt.index[0]
        if 0 <= row_idx < len(table):
            return str(table[row_idx][0])
        return ""

    def delete_selected(run_id):
        if not run_id:
            return [], "未选中任何运行"
        ok = history_service.delete_run(run_id)
        rows, msg = refresh_table()
        prefix = "已删除 " if ok else "删除失败 "
        return rows, prefix + run_id

    refresh_btn.click(fn=refresh_table, outputs=[runs_table, status_md])
    runs_table.select(fn=on_select, inputs=[runs_table], outputs=[selected_run])
    delete_btn.click(fn=delete_selected, inputs=[selected_run], outputs=[runs_table, status_md])

    # 页面加载时自动刷新一次
    refresh_table()
