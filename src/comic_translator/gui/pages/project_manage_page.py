"""项目管理页面 — 术语表 / 角色记忆 / 翻译记忆 的项目级管理

每个漫画目录（data/comics/<名字>）对应一个 project_id（目录名）。
此页面按选中项目过滤并编辑三类数据，与翻译流水线共享同一套服务与隔离逻辑。
"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from comic_translator.gui.components.dir_scanner import scan_image_dirs
from comic_translator.services.character_identity_service import CharacterIdentityService
from comic_translator.services.glossary_service import GlossaryService
from comic_translator.services.translation_memory_service import TranslationMemoryService
from comic_translator.models.character_identity import CharacterIdentity
from comic_translator.models.glossary import GlossaryEntry

GLOSSARY_HEADERS = ["ID", "源文", "译文", "上下文", "分类", "优先级"]
CHAR_HEADERS = ["角色ID", "规范名", "别名", "头衔", "昵称", "描述", "口癖"]
TM_HEADERS = ["ID", "源文", "译文", "provider", "model", "使用次数"]


def _project_choices() -> list[str]:
    """返回可选项目（与流水线页一致），project_id = 目录名"""
    dirs = scan_image_dirs()
    # 去重：同名目录只保留一个代表路径
    seen: dict[str, str] = {}
    for d in dirs:
        name = Path(d).name
        if name not in seen:
            seen[name] = d
    return sorted(seen.values())


def _pid_from_dir(dir_path: str) -> str:
    return Path(dir_path).name


def _glossary_rows(entries: list[tuple[int, GlossaryEntry]]) -> list[list]:
    rows = []
    for term_id, e in entries:
        rows.append([term_id, e.source, e.target, e.context or "", e.category, e.priority])
    return rows


def _char_rows(identities: list[CharacterIdentity]) -> list[list]:
    rows = []
    for c in identities:
        rows.append([
            c.character_id,
            c.canonical_name,
            ", ".join(c.aliases),
            ", ".join(c.titles),
            ", ".join(c.nicknames),
            c.description,
            c.speaking_style,
        ])
    return rows


def _tm_rows(entries: list[dict]) -> list[list]:
    rows = []
    for e in entries:
        rows.append([
            e["id"],
            (e["source_text"] or "")[:60],
            (e["translated_text"] or "")[:60],
            e["provider"],
            e["model"],
            e["usage_count"],
        ])
    return rows


def create_project_page():
    """创建项目管理标签页"""
    glossary_service = GlossaryService()
    char_service = CharacterIdentityService()
    tm_service = TranslationMemoryService()

    with gr.Column():
        gr.Markdown("## 项目管理")
        gr.Markdown(
            "每个漫画目录对应一个项目，术语表 / 角色记忆 / 翻译记忆均按项目隔离。"
            "翻译时自动抽取的术语会写回当前项目。"
        )

        with gr.Row():
            project_dd = gr.Dropdown(
                label="选择项目（漫画目录）",
                choices=_project_choices(),
                value=None,
                interactive=True,
                scale=3,
            )
            refresh_proj_btn = gr.Button("刷新项目", size="sm", scale=1, min_width=100)

        status_md = gr.Markdown("请选择项目")

        with gr.Tabs():
            # ================= 术语表 =================
            with gr.Tab("术语表"):
                glossary_table = gr.Dataframe(
                    headers=GLOSSARY_HEADERS,
                    value=[],
                    interactive=False,
                    wrap=True,
                    label="术语表（点击行加载到下方表单）",
                    row_count=12,
                )
                with gr.Row():
                    g_source = gr.Textbox(label="源文", scale=1)
                    g_target = gr.Textbox(label="译文", scale=1)
                with gr.Row():
                    g_context = gr.Textbox(label="上下文（可选）", scale=2)
                    g_category = gr.Dropdown(
                        label="分类",
                        choices=["general", "person", "location", "organization", "skill", "item"],
                        value="general",
                        scale=1,
                    )
                    g_priority = gr.Slider(
                        minimum=0, maximum=10, value=5, step=1,
                        label="优先级", scale=1,
                    )
                with gr.Row():
                    g_add_btn = gr.Button("新增术语", variant="primary", scale=1)
                    g_update_btn = gr.Button("更新选中", scale=1)
                    g_delete_btn = gr.Button("删除选中", variant="stop", scale=1)
                    g_refresh_btn = gr.Button("刷新", scale=1)
                g_msg = gr.Markdown("")
                g_selected_id = gr.State(None)

            # ================= 角色记忆 =================
            with gr.Tab("角色记忆"):
                char_table = gr.Dataframe(
                    headers=CHAR_HEADERS,
                    value=[],
                    interactive=False,
                    wrap=True,
                    label="角色记忆（点击行加载到下方表单）",
                    row_count=12,
                )
                with gr.Row():
                    c_id = gr.Textbox(label="角色ID", placeholder="如: ironman", scale=1)
                    c_name = gr.Textbox(label="规范名", placeholder="如: Iron Man", scale=1)
                with gr.Row():
                    c_aliases = gr.Textbox(label="别名（逗号分隔）", scale=1)
                    c_titles = gr.Textbox(label="头衔（逗号分隔）", scale=1)
                    c_nicknames = gr.Textbox(label="昵称（逗号分隔）", scale=1)
                with gr.Row():
                    c_desc = gr.Textbox(label="描述", scale=2)
                    c_style = gr.Textbox(label="口癖/说话风格", scale=1)
                with gr.Row():
                    c_add_btn = gr.Button("新增角色", variant="primary", scale=1)
                    c_update_btn = gr.Button("更新选中", scale=1)
                    c_delete_btn = gr.Button("删除选中", variant="stop", scale=1)
                    c_refresh_btn = gr.Button("刷新", scale=1)
                c_msg = gr.Markdown("")
                c_selected_id = gr.State(None)

            # ================= 翻译记忆 =================
            with gr.Tab("翻译记忆"):
                tm_table = gr.Dataframe(
                    headers=TM_HEADERS,
                    value=[],
                    interactive=False,
                    wrap=True,
                    label="翻译记忆（只读）",
                    row_count=12,
                )
                with gr.Row():
                    tm_delete_btn = gr.Button("删除选中", variant="stop", scale=1)
                    tm_refresh_btn = gr.Button("刷新", scale=1)
                tm_msg = gr.Markdown("")
                tm_selected_id = gr.State(None)

    # ---------- 逻辑 ----------

    def refresh_projects():
        choices = _project_choices()
        return gr.update(choices=choices, value=choices[0] if choices else None)

    def load_glossary(project_dir):
        if not project_dir:
            return [], "", None
        pid = _pid_from_dir(project_dir)
        entries = glossary_service.list_with_ids(project_id=pid)
        return _glossary_rows(entries), f"共 {len(entries)} 条术语（项目: {pid}）", None

    def on_select_glossary(evt: gr.SelectData, table):
        if evt.index is None or not table:
            return "", "", "", "general", 5, "", None
        row = table[evt.index[0]]
        try:
            return (
                str(row[1]), str(row[2]), str(row[3] or ""),
                str(row[4] or "general"), int(row[5] or 0),
                "已加载术语 #" + str(row[0]), int(row[0]),
            )
        except Exception:
            return "", "", "", "general", 5, "", None

    def add_glossary(project_dir, source, target, context, category, priority):
        if not project_dir:
            return "请先选择项目", None
        if not source.strip() or not target.strip():
            return "源文和译文不能为空", None
        pid = _pid_from_dir(project_dir)
        entry = GlossaryEntry(
            source=source.strip(),
            target=target.strip(),
            context=context.strip() or None,
            category=category,
            priority=int(priority),
        )
        term_id = glossary_service.add_term(entry, project_id=pid)
        return f"已新增术语 #{term_id}", None

    def update_glossary(project_dir, selected_id, source, target, context, category, priority):
        if not project_dir or selected_id is None:
            return "请先在表格中选择一行", None
        entry = GlossaryEntry(
            source=source.strip(),
            target=target.strip(),
            context=context.strip() or None,
            category=category,
            priority=int(priority),
        )
        ok = glossary_service.update_term(int(selected_id), entry)
        return ("已更新" if ok else "更新失败") , None

    def delete_glossary(project_dir, selected_id):
        if selected_id is None:
            return "请先在表格中选择一行", None
        ok = glossary_service.delete_term(int(selected_id))
        return ("已删除" if ok else "删除失败"), None

    def refresh_glossary(project_dir):
        return load_glossary(project_dir)[0], load_glossary(project_dir)[1]

    def load_chars(project_dir):
        if not project_dir:
            return [], "", None
        pid = _pid_from_dir(project_dir)
        identities = char_service.get_all_identities(project_id=pid)
        return _char_rows(identities), f"共 {len(identities)} 个角色（项目: {pid}）", None

    def on_select_char(evt: gr.SelectData, table):
        if evt.index is None or not table:
            return ("", "", "", "", "", "", "", None)
        row = table[evt.index[0]]
        try:
            return (
                str(row[0]), str(row[1]), str(row[2] or ""),
                str(row[3] or ""), str(row[4] or ""),
                str(row[5] or ""), str(row[6] or ""),
                str(row[0]),
            )
        except Exception:
            return ("", "", "", "", "", "", "", None)

    def _parse_csv(s: str) -> list[str]:
        return [x.strip() for x in s.split(",") if x.strip()]

    def add_char(project_dir, cid, name, aliases, titles, nicknames, desc, style):
        if not project_dir:
            return "请先选择项目", None
        if not cid.strip() or not name.strip():
            return "角色ID和规范名不能为空", None
        pid = _pid_from_dir(project_dir)
        identity = CharacterIdentity(
            character_id=cid.strip(),
            canonical_name=name.strip(),
            aliases=_parse_csv(aliases),
            titles=_parse_csv(titles),
            nicknames=_parse_csv(nicknames),
            description=desc.strip(),
            speaking_style=style.strip(),
        )
        char_id = char_service.create_identity(identity, project_id=pid)
        return f"已新增角色 {name.strip()}", None

    def update_char(project_dir, selected_id, cid, name, aliases, titles, nicknames, desc, style):
        if not project_dir or not selected_id:
            return "请先在表格中选择一行", None
        identity = CharacterIdentity(
            character_id=selected_id,
            canonical_name=name.strip() or selected_id,
            aliases=_parse_csv(aliases),
            titles=_parse_csv(titles),
            nicknames=_parse_csv(nicknames),
            description=desc.strip(),
            speaking_style=style.strip(),
        )
        ok = char_service.update_identity(selected_id, identity)
        return ("已更新" if ok else "更新失败"), None

    def delete_char(project_dir, selected_id):
        if not selected_id:
            return "请先在表格中选择一行", None
        ok = char_service.delete_identity(selected_id)
        return ("已删除" if ok else "删除失败"), None

    def refresh_chars(project_dir):
        return load_chars(project_dir)[0], load_chars(project_dir)[1]

    def load_tm(project_dir):
        if not project_dir:
            return [], "", None
        pid = _pid_from_dir(project_dir)
        entries = tm_service.list_entries(project_id=pid)
        return _tm_rows(entries), f"共 {len(entries)} 条翻译记忆（项目: {pid}）", None

    def on_select_tm(evt: gr.SelectData, table):
        if evt.index is None or not table:
            return "", None
        row = table[evt.index[0]]
        try:
            return "已选中记忆 #" + str(row[0]), int(row[0])
        except Exception:
            return "", None

    def delete_tm(project_dir, selected_id):
        if selected_id is None:
            return "请先在表格中选择一行", None
        ok = tm_service.delete_entry(int(selected_id))
        return ("已删除" if ok else "删除失败"), None

    # ---------- 绑定 ----------

    refresh_proj_btn.click(fn=refresh_projects, outputs=[project_dd])
    project_dd.change(fn=load_glossary, inputs=[project_dd], outputs=[glossary_table, g_msg, g_selected_id])
    project_dd.change(fn=load_chars, inputs=[project_dd], outputs=[char_table, c_msg, c_selected_id])
    project_dd.change(fn=load_tm, inputs=[project_dd], outputs=[tm_table, tm_msg, tm_selected_id])

    glossary_table.select(
        fn=on_select_glossary,
        inputs=[glossary_table],
        outputs=[g_source, g_target, g_context, g_category, g_priority, g_msg, g_selected_id],
    )
    g_add_btn.click(
        fn=add_glossary,
        inputs=[project_dd, g_source, g_target, g_context, g_category, g_priority],
        outputs=[g_msg, g_selected_id],
    ).then(fn=load_glossary, inputs=[project_dd], outputs=[glossary_table, g_msg, g_selected_id])
    g_update_btn.click(
        fn=update_glossary,
        inputs=[project_dd, g_selected_id, g_source, g_target, g_context, g_category, g_priority],
        outputs=[g_msg, g_selected_id],
    ).then(fn=load_glossary, inputs=[project_dd], outputs=[glossary_table, g_msg, g_selected_id])
    g_delete_btn.click(
        fn=delete_glossary,
        inputs=[project_dd, g_selected_id],
        outputs=[g_msg, g_selected_id],
    ).then(fn=load_glossary, inputs=[project_dd], outputs=[glossary_table, g_msg, g_selected_id])
    g_refresh_btn.click(fn=load_glossary, inputs=[project_dd], outputs=[glossary_table, g_msg, g_selected_id])

    char_table.select(
        fn=on_select_char,
        inputs=[char_table],
        outputs=[c_id, c_name, c_aliases, c_titles, c_nicknames, c_desc, c_style, c_selected_id],
    )
    c_add_btn.click(
        fn=add_char,
        inputs=[project_dd, c_id, c_name, c_aliases, c_titles, c_nicknames, c_desc, c_style],
        outputs=[c_msg, c_selected_id],
    ).then(fn=load_chars, inputs=[project_dd], outputs=[char_table, c_msg, c_selected_id])
    c_update_btn.click(
        fn=update_char,
        inputs=[project_dd, c_selected_id, c_id, c_name, c_aliases, c_titles, c_nicknames, c_desc, c_style],
        outputs=[c_msg, c_selected_id],
    ).then(fn=load_chars, inputs=[project_dd], outputs=[char_table, c_msg, c_selected_id])
    c_delete_btn.click(
        fn=delete_char,
        inputs=[project_dd, c_selected_id],
        outputs=[c_msg, c_selected_id],
    ).then(fn=load_chars, inputs=[project_dd], outputs=[char_table, c_msg, c_selected_id])
    c_refresh_btn.click(fn=load_chars, inputs=[project_dd], outputs=[char_table, c_msg, c_selected_id])

    tm_table.select(
        fn=on_select_tm,
        inputs=[tm_table],
        outputs=[tm_msg, tm_selected_id],
    )
    tm_delete_btn.click(
        fn=delete_tm,
        inputs=[project_dd, tm_selected_id],
        outputs=[tm_msg, tm_selected_id],
    ).then(fn=load_tm, inputs=[project_dd], outputs=[tm_table, tm_msg, tm_selected_id])
    tm_refresh_btn.click(fn=load_tm, inputs=[project_dd], outputs=[tm_table, tm_msg, tm_selected_id])
