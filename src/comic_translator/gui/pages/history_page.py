"""历史任务页面 — 列出所有 Pipeline 运行记录（Phase 3B-1 迁移）

对齐 Translation Workspace 视觉基准（Zinc + Teal / 紧凑专业工具）：
- 页面标题统一 mf-page-title
- Toolbar（刷新 secondary / 删除选中 danger）
- DataFrame 紧凑表 + 状态文本
- 空状态引导

业务逻辑（list_runs / delete_run / on_select / refresh_table）完全保留，零改动。
"""

from __future__ import annotations

import gradio as gr

from comic_translator.gui.services.run_history_service import RunHistoryService


def create_history_page(history_service: RunHistoryService):
    """创建历史任务页面"""

    # ---------- 业务逻辑 ----------

    def refresh_table():
        runs = history_service.list_runs()
        rows = []
        for r in runs:
            rows.append([
                r.id, r.project, r.input_dir, r.created_time,
                r.total_pages, r.success_pages, r.failed_pages,
                r.total_dialogues, r.status, r.elapsed_s,
            ])
        if not rows:
            return rows, "共 0 条记录", gr.update(visible=True)
        return rows, f"共 {len(rows)} 条记录", gr.update(visible=False)

    # Gradio 6 下「构建后、启动前」对组件 .value/.visible 赋值不会进入客户端
    # 配置 —— 首屏数据必须在构造时传入。
    _initial_rows, _initial_msg, _initial_empty = refresh_table()

    with gr.Column(elem_classes="mf-page"):
        gr.Markdown("历史", elem_classes="mf-page-title")
        gr.Markdown(
            "所有翻译任务的运行记录。选择一行可删除该次运行。",
            elem_classes="mf-desc",
        )

        # ---- Toolbar ----
        with gr.Row():
            refresh_btn = gr.Button("刷新列表", size="sm", scale=1)
            delete_btn = gr.Button("删除选中", variant="stop", size="sm", scale=1)
            status_md = gr.Markdown(_initial_msg, scale=3)

        # ---- 表格（紧凑、专业工具密度） ----
        runs_table = gr.Dataframe(
            headers=["运行ID", "项目", "输入目录", "创建时间", "总页数", "成功", "失败", "对话数", "状态", "耗时(s)"],
            value=_initial_rows,
            interactive=False,
            wrap=False,
            elem_classes="mf-table",
        )

        # 空状态提示（无记录时）
        empty_md = gr.Markdown(
            "**暂无运行记录**\n\n在「翻译」工作区执行一次翻译后，这里会列出每次运行的详情。",
            visible=bool(_initial_empty.get("visible", False)),
        )

        # 选中运行 ID（隐藏，业务保留）
        selected_run = gr.Textbox(label="选中的运行ID", interactive=False, visible=False)

    def on_select(evt: gr.SelectData, table):
        if evt.index is None:
            return ""
        row_idx = evt.index[0]
        # Gradio 6 的 Dataframe 默认 preprocess 成 pandas DataFrame，
        # table[int] 是按列名取列（KeyError），必须用 iloc 按位置取行
        try:
            import pandas as pd
            if isinstance(table, pd.DataFrame):
                if 0 <= row_idx < len(table):
                    return str(table.iloc[row_idx, 0])
                return ""
        except Exception:
            pass
        if 0 <= row_idx < len(table):
            return str(table[row_idx][0])
        return ""

    def delete_selected(run_id):
        if not run_id:
            rows, _msg, empty_vis = refresh_table()
            return rows, "未选中任何运行", empty_vis
        ok = history_service.delete_run(run_id)
        rows, _msg, empty_vis = refresh_table()
        prefix = "已删除 " if ok else "删除失败 "
        return rows, prefix + run_id, empty_vis

    refresh_btn.click(
        fn=refresh_table,
        outputs=[runs_table, status_md, empty_md],
    )
    runs_table.select(fn=on_select, inputs=[runs_table], outputs=[selected_run])
    delete_btn.click(
        fn=delete_selected,
        inputs=[selected_run],
        outputs=[runs_table, status_md, empty_md],
    )
