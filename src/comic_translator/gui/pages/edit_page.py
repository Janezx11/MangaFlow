"""精修工坊页 —— 交互式编辑闭环：历史机翻 → 表格里改译文 → 一键重渲染"""

from __future__ import annotations

import gradio as gr
from loguru import logger

from comic_translator.gui.services import edit_service


def create_edit_page(pipeline_service, history_service):
    """创建精修工坊标签页"""

    def list_run_ids() -> list[str]:
        if history_service is None:
            return []
        return [r.id for r in history_service.list_runs()]

    _run_ids = list_run_ids()

    with gr.Column(elem_classes="mf-page"):
        gr.Markdown("精修工坊", elem_classes="mf-page-title")
        gr.Markdown(
            "机翻打底、人工精修：选择运行与页码，在表格里直接修改译文，一键重渲染该页。",
            elem_classes="mf-desc",
        )

        with gr.Row():
            run_dropdown = gr.Dropdown(
                label="选择运行",
                choices=_run_ids,
                value=_run_ids[0] if _run_ids else None,
                scale=3,
            )
            refresh_runs_btn = gr.Button("刷新列表", scale=1)
            page_num = gr.Number(label="页码", value=1, precision=0, scale=1)
            load_btn = gr.Button("加载对白", variant="primary", scale=1)

        status_md = gr.Markdown("就绪 — 选择运行后点「加载对白」")

        dlg_table = gr.Dataframe(
            headers=["块ID", "原文", "译文（可直接编辑）", "字号(0=原值)", "X偏移", "Y偏移"],
            datatype=["str", "str", "str", "number", "number", "number"],
            value=[],
            interactive=True,
            wrap=True,
            elem_classes="mf-table",
        )

        with gr.Row():
            canvas = gr.Image(label="页面画布（点击气泡选中）", interactive=False, scale=3)

        with gr.Row():
            sel_id = gr.Textbox(label="选中块", interactive=False, scale=1)
            sel_source = gr.Textbox(label="原文", interactive=False, scale=2)
            sel_text = gr.Textbox(label="译文", scale=3)

        with gr.Row():
            sel_font = gr.Number(label="字号", value=0, precision=0, scale=1)
            sel_dx = gr.Number(label="X偏移", value=0, precision=0, scale=1)
            sel_dy = gr.Number(label="Y偏移", value=0, precision=0, scale=1)
            apply_sel_btn = gr.Button("写入表格行", scale=1)

        with gr.Row():
            rerender_btn = gr.Button("应用修改并重渲染", variant="primary", scale=2)
            preview = gr.Image(label="重渲染结果", interactive=False, scale=3)

    # ---------- 逻辑 ----------

    def load_dialogues(run_id, page):
        if history_service is None or not run_id:
            return [], None, "请先选择运行"
        snapshot = history_service.load_page(run_id, int(page or 1))
        if snapshot is None:
            return [], None, f"运行 {run_id} 第 {page} 页快照不存在"
        typo = edit_service.load_typography_params(snapshot)
        rows = [
            [
                d.get("id", ""),
                d.get("merged_text", ""),
                d.get("translated_text", ""),
                int(typo.get(d.get("id", ""), {}).get("font_size") or 0),
                0,
                0,
            ]
            for d in snapshot.dialogues
        ]
        img_path = snapshot.rendered_path or None
        if not img_path:
            src = edit_service.find_source_image(history_service, run_id, snapshot)
            img_path = str(src) if src else None
        typo_all = edit_service.load_typography_params(snapshot)
        canvas_img = edit_service.draw_selection_overlay(img_path, snapshot, typo_all)
        return rows, canvas_img, f"共 {len(rows)} 个对白块 —— 可点画布选中气泡，或直接编辑表格"

    def apply_and_rerender(run_id, page, table):
        if history_service is None or not run_id:
            return "请先选择运行", None
        # Dataframe preprocess: pandas DataFrame（列 = 表头）→ 逐行取值
        # 注意：DataFrame 不能用布尔判断（ambiguous truth value），先转行列表再判空
        try:
            import pandas as pd
            if isinstance(table, pd.DataFrame):
                rows = table.values.tolist()
            else:
                rows = [list(r) for r in table]
        except Exception:
            rows = [list(r) for r in table]
        if not rows:
            return "对白表为空，请先加载对白", None

        edits: dict[str, dict] = {}
        for row in rows:
            if len(row) < 3:
                continue
            block_id = str(row[0] or "").strip()
            if not block_id:
                continue
            edit = {
                "text": str(row[2] or "").strip(),
                "font_size": int(row[3] or 0) if len(row) > 3 else 0,
                "dx": int(row[4] or 0) if len(row) > 4 else 0,
                "dy": int(row[5] or 0) if len(row) > 5 else 0,
            }
            if edit["text"] or edit["font_size"] or edit["dx"] or edit["dy"]:
                edits[block_id] = edit

        try:
            msg, rendered = edit_service.rerender_with_edits(
                run_id=run_id,
                page=int(page or 1),
                edits=edits,
                pipeline_service=pipeline_service,
                history_service=history_service,
            )
            logger.info(msg)
            return msg, rendered or None
        except Exception as e:
            logger.error(f"Re-render failed: {e}")
            return f"重渲染失败: {e}", None


    # ---------- 绑定 ----------

    def refresh_runs_now():
        ids = list_run_ids()
        msg = f"共 {len(ids)} 次运行（刚完成的批次会出现在最上方）"
        return gr.update(choices=ids, value=ids[0] if ids else None), msg

    refresh_runs_btn.click(
        fn=refresh_runs_now,
        inputs=[],
        outputs=[run_dropdown, status_md],
    )

    load_btn.click(
        fn=load_dialogues,
        inputs=[run_dropdown, page_num],
        outputs=[dlg_table, canvas, status_md],
    )

    def select_block(evt: gr.SelectData, run_id, page):
        try:
            x, y = int(evt.index[0]), int(evt.index[1])
        except (TypeError, ValueError, IndexError):
            return "", "", 0, 0, 0, None
        if history_service is None or not run_id:
            return "", "", 0, 0, 0, None
        snapshot = history_service.load_page(run_id, int(page or 1))
        if snapshot is None:
            return "", "", 0, 0, 0, None
        typo = edit_service.load_typography_params(snapshot)
        bid = edit_service.find_block_at(snapshot, typo, x, y)
        if not bid:
            return "", "", 0, 0, 0, None
        src = next((d.get("merged_text", "") for d in snapshot.dialogues if d.get("id") == bid), "")
        text = next((d.get("translated_text", "") for d in snapshot.dialogues if d.get("id") == bid), "")
        font_size = int(typo.get(bid, {}).get("font_size") or 0)
        overlay = edit_service.draw_selection_overlay(
            snapshot.rendered_path or None, snapshot, typo, selected_id=bid)
        return bid, src, text, font_size, 0, 0, overlay

    canvas.select(
        fn=select_block,
        inputs=[run_dropdown, page_num],
        outputs=[sel_id, sel_source, sel_text, sel_font, sel_dx, sel_dy, canvas],
    )

    def apply_selected_to_table(sel_id_val, sel_text_val, sel_font, sel_dx, sel_dy, table):
        try:
            import pandas as pd
            if isinstance(table, pd.DataFrame):
                rows = table.values.tolist()
            else:
                rows = [list(r) for r in table]
        except Exception:
            rows = [list(r) for r in table]
        if not sel_id_val:
            return table, "未选中气泡"
        hit = False
        for r in rows:
            if str(r[0] or "").strip() == sel_id_val.strip():
                r[2] = sel_text_val
                if len(r) > 3:
                    r[3] = int(sel_font or 0)
                if len(r) > 4:
                    r[4] = int(sel_dx or 0)
                if len(r) > 5:
                    r[5] = int(sel_dy or 0)
                hit = True
                break
        if not hit:
            return table, f"表格中未找到块 {sel_id_val}（请先加载对白）"
        return rows, f"块 {sel_id_val} 已写入表格 —— 点「应用修改并重渲染」生效"

    apply_sel_btn.click(
        fn=apply_selected_to_table,
        inputs=[sel_id, sel_text, sel_font, sel_dx, sel_dy, dlg_table],
        outputs=[dlg_table, status_md],
    )
    rerender_btn.click(
        fn=apply_and_rerender,
        inputs=[run_dropdown, page_num, dlg_table],
        outputs=[status_md, preview],
    )
