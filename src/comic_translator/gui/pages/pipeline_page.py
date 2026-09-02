"""翻译流水线页面 — Translation Workspace（MangaFlow UI 标杆）

工作台布局：
- 控制栏（左）：输入/语言/模型/Provider + 高级选项折叠
- 预览区（右）：核心区，渲染结果大图 + 页导航
- Pipeline 步骤条：检测 → OCR → 翻译 → 渲染 → 完成
- 底部日志：mono 等宽、可分级过滤

约束：本文件只做 UI / 布局 / 设计系统改造，业务逻辑、事件绑定、输入输出全部保留。
仅新增一个 preview 预览输出（读取 page_result.rendered_path，不改 Pipeline 本身）。
"""

from __future__ import annotations

import gradio as gr
import yaml
from pathlib import Path

from comic_translator.gui.components.dir_scanner import scan_image_dirs
from comic_translator.gui.components.perf_format import format_perf_md
from comic_translator.gui.native_picker import pick_directory
from comic_translator.gui.services.gui_pipeline_service import GuiPipelineService


def _load_render_defaults() -> dict:
    """从 config.yaml 读取渲染默认值"""
    defaults = {
        "target_lang": "zh",
        "mask_provider": "OCR Mask",
        "inpaint_provider": "OpenCV",
        "typography_font": "Comic Font",
        "background_cleaning": False,
        "fallback_to_source": False,
        "output_dir": "data/output/rendered",
    }
    try:
        cfg = yaml.safe_load(Path("config.yaml").read_text(encoding="utf-8")) or {}
        trans = cfg.get("translation", {})
        render = cfg.get("render", {})
        defaults["target_lang"] = trans.get("target_lang", defaults["target_lang"])
        defaults["inpaint_provider"] = _canonical_inpaint(render.get("inpaint", {}).get("provider", "opencv"))
        defaults["typography_font"] = "System Font" if render.get("typography_font") in ("system", "fallback") else "Comic Font"
        defaults["background_cleaning"] = bool(render.get("background_cleaning", False))
        defaults["fallback_to_source"] = bool(render.get("fallback_to_source", False))
        proj = cfg.get("project", {})
        if proj.get("output_dir"):
            defaults["output_dir"] = proj["output_dir"]
    except Exception:
        pass
    return defaults


def _canonical_inpaint(name: str) -> str:
    mapping = {"opencv": "OpenCV", "lama": "LaMa", "sd": "SD"}
    return mapping.get(name.lower(), name.capitalize())


def _map_render_options(mask_provider, inpaint_provider, typography_font, background_cleaning, fallback_to_source) -> dict:
    return {
        "mask_provider": "ocr" if mask_provider == "OCR Mask" else mask_provider.lower(),
        "inpaint_provider": inpaint_provider.lower(),
        "typography_font": "system" if typography_font == "System Font" else "comic",
        "background_cleaning": background_cleaning,
        "fallback_to_source": fallback_to_source,
    }


LEVEL_ICONS = {"INFO": "[INFO]", "WARNING": "[WARN]", "ERROR": "[ERR ]"}

# Pipeline 步骤（与 pipeline_service 上报的 stage 对齐）
_PIPELINE_STEPS = [
    ("detection", "检测"),
    ("ocr", "OCR"),
    ("translation", "翻译"),
    ("render", "渲染"),
    ("done", "完成"),
]


def _pipeline_steps_md(stage: str) -> str:
    """渲染 Pipeline 步骤条：当前步高亮，已完成打勾（文本符号，非 emoji）"""
    order = [s for s, _ in _PIPELINE_STEPS]
    idx = order.index(stage) if stage in order else -1
    cells = []
    for i, (key, label) in enumerate(_PIPELINE_STEPS):
        if idx == -1:
            cls = ""
        elif i < idx:
            cls = " done"      # 已完成
        elif i == idx:
            cls = " active"    # 当前
        else:
            cls = ""
        mark = "✓ " if (idx != -1 and i < idx) else ""
        cells.append(f"<span class='mf-step{cls}'>{mark}{label}</span>")
    return (
        "<div class='mf-pipeline-steps'>" + "<span class='mf-step-sep'></span>".join(cells) + "</div>"
    )


def _classify(message: str) -> str:
    """根据消息内容判断日志级别"""
    m = message.lower()
    if "error" in m or "失败" in m or "错误" in m:
        return "ERROR"
    if "warn" in m or "警告" in m or "no images" in m:
        return "WARNING"
    return "INFO"


def _format(level: str, msg: str) -> str:
    return f"{LEVEL_ICONS.get(level, '[INFO]')} {msg}"


def _filter(lines: list[tuple[str, str]], allowed: list[str]) -> str:
    """按级别过滤日志"""
    if not allowed:
        allowed = list(LEVEL_ICONS.keys())
    visible = [_format(lvl, msg) for lvl, msg in lines if lvl in allowed]
    return "\n".join(visible[-80:])


def create_pipeline_page(pipeline_service: GuiPipelineService):
    """创建翻译工作台（Translation Workspace）"""

    with gr.Column(elem_classes="mf-page"):
        gr.Markdown("翻译", elem_classes="mf-page-title")

        # ====== Pipeline 步骤条 ======
        steps_md = gr.HTML(_pipeline_steps_md(""))

        with gr.Row():
            # ====== 左：控制栏 ======
            with gr.Column(scale=1, min_width=320, elem_classes="mf-controls"):
                gr.Markdown("输入", elem_classes="mf-section-title")

                with gr.Group(elem_classes="mf-card"):
                    dir_input = gr.Dropdown(
                        label="输入目录",
                        choices=scan_image_dirs(),
                        value="data/comics",
                        allow_custom_value=True,
                        interactive=True,
                    )
                    with gr.Row():
                        refresh_dirs_btn = gr.Button("刷新", size="sm", scale=1)
                        browse_in_btn = gr.Button("选择文件夹", size="sm", scale=1)

                # 输入目录的「唯一事实来源」：浏览按钮写入、手动输入同步、开始翻译只读它
                input_dir_state = gr.State("data/comics")

                defaults = _load_render_defaults()
                with gr.Group(elem_classes="mf-card"):
                    with gr.Row():
                        target_lang = gr.Dropdown(
                            label="目标语言",
                            choices=[("中文", "zh"), ("英语", "en"), ("日语", "ja"), ("韩语", "ko")],
                            value=defaults["target_lang"],
                        )
                    output_dir = gr.Textbox(
                        label="输出目录",
                        value=defaults["output_dir"],
                        placeholder="例如: data/output/rendered",
                    )
                    browse_out_btn = gr.Button("选择输出文件夹", size="sm")

                gr.Markdown("模型 / Provider", elem_classes="mf-section-title")
                with gr.Group(elem_classes="mf-card"):
                    with gr.Row():
                        mask_provider = gr.Dropdown(
                            label="Mask Provider", choices=["OCR Mask"], value=defaults["mask_provider"],
                        )
                        inpaint_provider = gr.Dropdown(
                            label="Inpaint Provider", choices=["OpenCV", "LaMa", "SD"],
                            value=defaults["inpaint_provider"],
                        )
                    with gr.Row():
                        typography_font = gr.Dropdown(
                            label="字体", choices=["Comic Font", "System Font"],
                            value=defaults["typography_font"],
                        )
                        translate = gr.Checkbox(label="翻译（LLM）", value=True)

                # 高级选项（折叠，默认隐藏）
                with gr.Accordion("高级选项", open=False):
                    with gr.Row():
                        with gr.Column(scale=1):
                            gr.Markdown("检测模型", elem_classes="mf-section-title")
                            use_ctbd = gr.Checkbox(label="CTBD 文本检测", value=True)
                            use_yolo = gr.Checkbox(label="YOLO 气泡检测", value=True)
                            use_fusion = gr.Checkbox(label="检测融合", value=True)
                            bubble_aware = gr.Checkbox(label="Bubble-aware OCR", value=True)
                            split_strategy = gr.Dropdown(
                                label="拆分策略",
                                choices=[("自动", "auto"), ("细粒度 Line", "line_split"), ("气泡裁剪 OCR", "bubble_crop_ocr")],
                                value="auto",
                            )
                            cross_threshold = gr.Slider(
                                minimum=0.05, maximum=0.5, value=0.15, step=0.01, label="跨气泡阈值",
                            )
                        with gr.Column(scale=1):
                            gr.Markdown("渲染", elem_classes="mf-section-title")
                            conf_slider = gr.Slider(
                                minimum=0.1, maximum=0.9, value=0.3, step=0.05, label="检测置信度",
                            )
                            background_cleaning = gr.Checkbox(label="背景清理", value=defaults["background_cleaning"])
                            fallback_to_source = gr.Checkbox(
                                label="失败回退原文（否则显示 TRANSLATION FAILED）",
                                value=defaults["fallback_to_source"],
                            )

                # ====== 主操作 ======
                with gr.Row():
                    start_btn = gr.Button(
                        "开始翻译", variant="primary", size="lg",
                        elem_id="mf-start-btn", scale=3,
                    )
                    cancel_btn = gr.Button(
                        "取消", size="lg", elem_id="mf-cancel-btn",
                        visible=False, scale=1,
                    )

                progress_bar = gr.Slider(
                    minimum=0, maximum=100, value=0, step=1,
                    label="完成度", interactive=False,
                )
                status_md = gr.Markdown("就绪 — 选择目录后点击「开始翻译」", elem_id="mf-status")
                perf_md = gr.Markdown("运行后自动显示各阶段耗时 / GPU / 限流统计", elem_id="mf-perf")

            # ====== 右：预览区（核心区） ======
            with gr.Column(scale=2, elem_classes="mf-preview"):
                gr.Markdown("预览", elem_classes="mf-section-title")
                preview_image = gr.Image(
                    label="渲染结果", interactive=False,
                    elem_id="mf-preview", type="filepath",
                )
                gr.Markdown(
                    "运行翻译后，这里会显示当前页的渲染结果（{原文件名}_translated.png）。",
                    elem_classes="mf-desc",
                )

        # ====== 底部：日志 ======
        with gr.Group(elem_classes="mf-card"):
            with gr.Row():
                gr.Markdown("运行日志", elem_classes="mf-section-title", scale=2)
                log_filter = gr.CheckboxGroup(
                    choices=["INFO", "WARNING", "ERROR"],
                    value=["INFO", "WARNING", "ERROR"],
                    label="日志级别", scale=2,
                )
                clear_log_btn = gr.Button("清空日志", size="sm", scale=1)

            log_output = gr.Textbox(
                label="", lines=10, max_lines=20,
                interactive=False, autoscroll=True, elem_classes="mono",
            )

        # 内部日志缓冲
        log_state = gr.State([])

    # ---------- 事件逻辑（保持原业务绑定） ----------

    def refresh_dirs():
        dirs = scan_image_dirs()
        return gr.update(choices=dirs, value=dirs[0] if dirs else None)

    def clear_log():
        return [], ""

    def apply_filter(lines, allowed):
        return _filter(lines, allowed)

    def run(input_dir_value, use_ctbd, use_yolo, use_fusion, translate, conf,
            target_lang, mask_provider, inpaint_provider, typography_font,
            background_cleaning, fallback_to_source, out_dir,
            bubble_aware, split_strategy, cross_threshold,
            allowed_levels, progress=gr.Progress()):
        """执行流水线并流式更新日志 + 预览"""
        input_dir = (input_dir_value or "").strip()
        if not input_dir:
            yield _pack("请先选择或输入漫画目录", 0, "错误：未选择目录", [], "_未运行_", running=False, preview=None, stage="")
            return

        output_dir = out_dir.strip() if out_dir.strip() else "data/output/rendered"

        render_options = _map_render_options(
            mask_provider, inpaint_provider, typography_font,
            background_cleaning, fallback_to_source,
        )

        log_lines: list[tuple[str, str]] = []

        def log(msg: str, level: str | None = None):
            lvl = level or _classify(msg)
            log_lines.append((lvl, msg))
            return _filter(log_lines, allowed_levels)

        stage_names = {
            "init": "初始化模型",
            "detection": "检测",
            "ocr": "文字识别",
            "translation": "翻译",
            "render": "渲染",
            "done": "完成",
        }

        yield _pack(
            log(f"开始处理: {input_dir}"), 0, "运行中...", log_lines,
            format_perf_md(pipeline_service.get_perf_snapshot()), running=True,
            preview=None, stage="init",
        )

        try:
            for p in pipeline_service.run_pipeline(
                input_dir=input_dir,
                use_ctbd=use_ctbd,
                use_yolo=use_yolo,
                use_fusion=use_fusion,
                translate=translate,
                conf=conf,
                target_lang=target_lang,
                render_options=render_options,
                output_dir=output_dir,
                bubble_aware=bubble_aware,
                split_strategy=split_strategy,
                cross_threshold=cross_threshold,
            ):
                pct = int(p.current_page / max(p.total_pages, 1) * 100)
                stage_cn = stage_names.get(p.stage, p.stage)
                progress(pct / 100, desc=p.message)

                # 预览：每页渲染完成后显示结果图
                preview_path = None
                if getattr(p, "page_result", None) is not None and p.page_result.rendered_path:
                    preview_path = p.page_result.rendered_path

                if p.stage == "init":
                    status_text = f"初始化中 — {p.message}"
                elif p.stage == "done":
                    status_text = p.message
                elif p.current_page >= 1:
                    status_text = f"{stage_cn} — 第 {p.current_page}/{p.total_pages} 页"
                else:
                    status_text = f"{stage_cn} — 准备中…"

                yield _pack(
                    log(p.message),
                    pct,
                    status_text,
                    log_lines,
                    format_perf_md(pipeline_service.get_perf_snapshot()),
                    running=True,
                    preview=preview_path,
                    stage=p.stage,
                )

            results = pipeline_service.get_all_results()
            total_dialogues = sum(len(r.dialogues) for r in results.values())
            errors = sum(1 for r in results.values() if r.error)
            summary = f"完成！共 {len(results)} 页，{total_dialogues} 个对话块"
            perf_summary = format_perf_md(pipeline_service.get_perf_snapshot())
            if errors:
                summary += f"，{errors} 页出错"
                yield _pack(log(summary, "WARNING"), 100, f"{summary}", log_lines, perf_summary, running=False, preview=None, stage="done")
            else:
                yield _pack(log(summary), 100, f"{summary}", log_lines, perf_summary, running=False, preview=None, stage="done")

            if pipeline_service.current_run_id:
                yield _pack(
                    log(f"运行记录已保存: runs/{pipeline_service.current_run_id}"),
                    100, f"{summary}", log_lines,
                    format_perf_md(pipeline_service.get_perf_snapshot()),
                    running=False, preview=None, stage="done",
                )

        except Exception as e:
            yield _pack(
                log(f"错误: {e}", "ERROR"), 0, f"错误: {e}", log_lines,
                format_perf_md(pipeline_service.get_perf_snapshot()),
                running=False, preview=None, stage="",
            )

    refresh_dirs_btn.click(fn=refresh_dirs, outputs=[dir_input])
    clear_log_btn.click(fn=clear_log, outputs=[log_state, log_output])
    log_filter.change(fn=apply_filter, inputs=[log_state, log_filter], outputs=[log_output])

    # Windows 原生文件夹选择对话框 → 同时更新下拉框（显示）与 State（权威值）
    def _browse_dir(current: str) -> tuple[str, str]:
        picked = pick_directory(initial_dir=current.strip() if current and current.strip() else "")
        value = picked or current
        return value, value

    browse_in_btn.click(fn=_browse_dir, inputs=[dir_input], outputs=[dir_input, input_dir_state])
    browse_out_btn.click(fn=_browse_dir, inputs=[output_dir], outputs=[output_dir])

    # 手动输入 / 选择预设 → 同步到 State（唯一事实来源）
    dir_input.change(fn=lambda v: (v or "").strip(), inputs=[dir_input], outputs=[input_dir_state])

    start_event = start_btn.click(
        fn=run,
        inputs=[
            input_dir_state,
            use_ctbd, use_yolo, use_fusion, translate, conf_slider,
            target_lang, mask_provider, inpaint_provider, typography_font,
            background_cleaning, fallback_to_source,
            output_dir,
            bubble_aware, split_strategy, cross_threshold,
            log_filter,
        ],
        outputs=[log_output, progress_bar, status_md, log_state, perf_md, start_btn, cancel_btn, preview_image, steps_md],
    )

    # 取消：终止进行中的翻译生成器，并复位按钮状态
    def _on_cancel():
        return gr.update(interactive=True), gr.update(visible=False, interactive=False)

    cancel_btn.click(
        fn=_on_cancel,
        outputs=[start_btn, cancel_btn],
        cancels=[start_event],
    )


def _pack(log_text: str, pct: int, status: str, lines: list, perf: str, running: bool,
          preview: str | None, stage: str):
    """统一打包生成器输出：日志 + 进度 + 状态 + 性能 + 按钮 + 预览 + 步骤条"""
    return (
        log_text,
        pct,
        status,
        lines,
        perf,
        gr.update(interactive=not running),
        gr.update(visible=running, interactive=running),
        gr.update(value=preview) if preview else gr.update(value=None),
        gr.HTML(_pipeline_steps_md(stage)),
    )
