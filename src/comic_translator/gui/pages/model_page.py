"""模型注册表页面 — 列出/启用/禁用/删除模型"""

from __future__ import annotations

from pathlib import Path

import gradio as gr
from loguru import logger

from comic_translator.gui.services.gui_config_service import GuiConfigService
from comic_translator.gui.services.model_registry_service import ModelRegistryService


def create_model_page(registry_service: ModelRegistryService, config_service: GuiConfigService, pipeline_service):
    """创建模型注册表标签页"""

    # ---------- 逻辑 ----------

    def refresh():
        models = registry_service.list_models()
        rows = []
        active = None
        for m in models:
            rows.append([
                m.name, m.type, m.dataset or "-",
                f"{m.metrics.get('mAP50', 0):.4f}" if m.metrics else "-",
                f"{m.metrics.get('mAP50-95', 0):.4f}" if m.metrics else "-",
                m.size_mb,
                m.created_time[:19],
                "✓" if m.enabled else "",
            ])
            if m.enabled:
                active = m
        active_text = f"**当前启用:** {active.name}" if active else "**当前启用:** 无"
        return rows, active_text, f"共 {len(models)} 个模型"

    # Gradio 6 下「构建后、启动前」对组件 .value 赋值不会进入客户端配置 ——
    # 首屏数据必须在构造时传入。
    _initial_rows, _initial_active, _initial_status = refresh()

    with gr.Column():
        gr.Markdown("## 模型注册表")

        with gr.Row():
            refresh_btn = gr.Button("刷新", size="sm", scale=1, min_width=80)
            active_md = gr.Markdown(_initial_active)

        model_table = gr.Dataframe(
            headers=["名称", "类型", "数据集", "mAP50", "mAP50-95",
                     "大小(MB)", "创建时间", "启用"],
            value=_initial_rows,
            interactive=False,
            wrap=True,
        )

        with gr.Row():
            selected_name = gr.Textbox(label="选中模型", interactive=False, scale=2)
            enable_btn = gr.Button("启用", variant="primary", scale=1)
            disable_btn = gr.Button("禁用", scale=1)
            delete_btn = gr.Button("删除注册", variant="stop", scale=1)

        status_md = gr.Markdown(_initial_status)

        with gr.Accordion("模型详情", open=False):
            detail_md = gr.Markdown("未选中")

    # ---------- 绑定 ----------
    def on_select(evt: gr.SelectData, table):
        if evt.index is None:
            return "", "未选中"
        row_idx = evt.index[0]
        # Gradio 6 的 Dataframe 默认 preprocess 成 pandas DataFrame，
        # table[int] 是按列名取列（KeyError），必须用 iloc 按位置取行
        try:
            import pandas as pd
            if isinstance(table, pd.DataFrame):
                if not (0 <= row_idx < len(table)):
                    return "", "未选中"
                row = table.iloc[row_idx].tolist()
            else:
                row = list(table[row_idx])
        except Exception:
            return "", "未选中"
        name = row[0]
        info = registry_service.get_model(name)
        if info is None:
            return name, "模型不存在"
        detail = (
            f"**路径**: `{info.path}`\n\n"
            f"**类型**: {info.type}\n\n"
            f"**数据集**: {info.dataset or '-'}\n\n"
            f"**指标**: {info.metrics}\n\n"
            f"**创建**: {info.created_time}\n\n"
            f"**启用**: {'是' if info.enabled else '否'}\n\n"
            f"**备注**: {info.notes or '-'}"
        )
        return name, detail

    def enable(name):
        if not name:
            return gr.update(), gr.update(), "未选中模型"
        ok = registry_service.enable(name)
        if ok:
            # 同步 config.yaml 的 bubble.model，让启用真正生效
            info = registry_service.get_model(name)
            if info:
                config_service.set("bubble.model", Path(info.path).as_posix())
                config_service.save()
                logger.info(f"Enabled model -> config.yaml bubble.model: {Path(info.path).as_posix()}")
            try:
                pipeline_service.refresh_bubble_provider()
            except Exception as e:
                logger.warning(f"Bubble provider refresh failed: {e}")
        rows, active, _ = refresh()
        return rows, active, (
            f"已启用 {name}（已同步 config.yaml，立即生效）" if ok else f"启用失败 {name}"
        )

    def disable(name):
        if not name:
            return gr.update(), gr.update(), "未选中模型"
        ok = registry_service.disable(name)
        rows, active, _ = refresh()
        return rows, active, (f"已禁用 {name}" if ok else f"禁用失败 {name}")

    def delete(name):
        if not name:
            return gr.update(), gr.update(), "未选中模型"
        ok = registry_service.delete(name, delete_file=False)
        rows, active, _ = refresh()
        return rows, active, (f"已删除注册 {name}(.pt 文件保留)" if ok else f"删除失败 {name}")

    # ---------- 绑定 ----------

    refresh_btn.click(fn=refresh, outputs=[model_table, active_md, status_md])
    model_table.select(fn=on_select, inputs=[model_table],
                       outputs=[selected_name, detail_md])
    enable_btn.click(fn=enable, inputs=[selected_name],
                     outputs=[model_table, active_md, status_md])
    disable_btn.click(fn=disable, inputs=[selected_name],
                      outputs=[model_table, active_md, status_md])
    delete_btn.click(fn=delete, inputs=[selected_name],
                     outputs=[model_table, active_md, status_md])

