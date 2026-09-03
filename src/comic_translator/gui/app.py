"""MangaFlow / AI 漫画翻译工作台（Gradio GUI 入口）

Shell:
- Header（应用名 + 当前页）
- Sidebar（分组导航，按钮切换视图可见性，无嵌套 Tabs 主导航）
- Main（14 个视图，默认显示 Translation Workspace）

设计系统：Light-first / Zinc + Teal（见 docs/UI_DESIGN_SYSTEM.md）。

用法:
  python -m comic_translator.gui.app
  python -m comic_translator.gui.app --port 7860
  python -m comic_translator.gui.app --share
"""

import os
import sys
from pathlib import Path

# 强制 localhost 绕过系统代理 / 防止 Gradio 自检和事件流被 Clash 等代理拦截
for var in ("NO_PROXY", "no_proxy"):
    existing = os.environ.get(var, "")
    additions = "localhost,127.0.0.1,::1"
    os.environ[var] = f"{existing},{additions}" if existing else additions

# 确保项目 src 在 path 中
_src = str(Path(__file__).resolve().parent.parent.parent)
if _src not in sys.path:
    sys.path.insert(0, _src)

import gradio as gr
from loguru import logger

from comic_translator.gui.services.gui_config_service import GuiConfigService
from comic_translator.gui.services.gui_pipeline_service import GuiPipelineService
from comic_translator.gui.services.run_history_service import RunHistoryService
from comic_translator.gui.services.benchmark_service import BenchmarkService
from comic_translator.gui.services.annotation_service import AnnotationService
from comic_translator.gui.services.dataset_service import DatasetService
from comic_translator.gui.services.training_service import TrainingService
from comic_translator.gui.services.model_registry_service import ModelRegistryService
from comic_translator.gui.pages.pipeline_page import create_pipeline_page
from comic_translator.gui.pages.debug_page import create_debug_page
from comic_translator.gui.pages.history_page import create_history_page
from comic_translator.gui.pages.compare_page import create_compare_page
from comic_translator.gui.pages.benchmark_page import create_benchmark_page
from comic_translator.gui.pages.settings_page import create_settings_page
from comic_translator.gui.pages.annotation_page import create_annotation_page
from comic_translator.gui.pages.dataset_page import create_dataset_page
from comic_translator.gui.pages.training_page import create_training_page
from comic_translator.gui.pages.model_page import create_model_page
from comic_translator.gui.pages.render_lab_page import create_render_lab_page
from comic_translator.gui.pages.quality_report_page import create_quality_report_page
from comic_translator.gui.pages.project_manage_page import create_project_page
from comic_translator.gui.pages.perf_monitor_page import create_perf_monitor_page

# ---------------------------------------------------------------------------
# Design Tokens（Light / Zinc + Teal）— 见 docs/UI_DESIGN_SYSTEM.md
# ---------------------------------------------------------------------------
_THEME = None


def _build_theme() -> gr.themes.Soft:
    """Light-first / Zinc + Teal。Teal 仅作 Primary/Accent，不铺满全屏。"""
    global _THEME
    if _THEME is None:
        _THEME = gr.themes.Soft(
            primary_hue="teal",
            secondary_hue="zinc",
            neutral_hue="zinc",
        )
    return _THEME


_CSS = """
/* ============================================================
   MangaFlow Design Tokens — Light / Zinc + Teal
   ============================================================ */
:root {
  --mf-bg: #f4f4f5;                 /* zinc-100 全局背景 */
  --mf-surface: #ffffff;            /* 卡片/面板/输入 */
  --mf-surface-elev: #fafafa;
  --mf-border: #e4e4e7;             /* zinc-200 */
  --mf-border-strong: #d4d4d8;      /* zinc-300 */
  --mf-primary: #0d9488;            /* teal-600 */
  --mf-primary-hover: #0f766e;      /* teal-700 */
  --mf-primary-soft: rgba(13,148,136,.10);
  --mf-text: #18181b;               /* zinc-900 */
  --mf-text-2: #52525b;             /* zinc-600 */
  --mf-text-3: #a1a1aa;             /* zinc-400 */
  --mf-success: #16a34a;
  --mf-warning: #d97706;
  --mf-error: #dc2626;
  --mf-info: #2563eb;
  --mf-font-sans: 'Geist', 'Geist Fallback', -apple-system, BlinkMacSystemFont,
                  'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei',
                  system-ui, sans-serif;
  --mf-font-mono: 'Geist Mono', 'JetBrains Mono', ui-monospace, 'SFMono-Regular',
                  'Cascadia Code', Consolas, monospace;
  --mf-r-sm: 4px;
  --mf-r-md: 6px;
  --mf-space-1: 4px; --mf-space-2: 8px; --mf-space-3: 12px;
  --mf-space-4: 16px; --mf-space-5: 20px; --mf-space-6: 24px; --mf-space-8: 32px;
}

/* ============================================================
   Global layout
   ============================================================ */
body, .gradio-container, .gr-block { font-family: var(--mf-font-sans); color: var(--mf-text); }
.gradio-container {
  max-width: 1500px !important;
  margin: 0 auto !important;
  background: var(--mf-bg) !important;
  padding: 12px 16px 24px !important;
}
code, pre, .mono, kbd, samp { font-family: var(--mf-font-mono); }

/* ---- Header ---- */
.mf-header {
  display: flex; align-items: baseline; justify-content: space-between;
  padding: 10px 2px 10px 2px; margin-bottom: 8px;
  border-bottom: 1px solid var(--mf-border);
}
.mf-header .mf-brand { font-size: 20px; font-weight: 700; color: var(--mf-text); letter-spacing: .3px; }
.mf-header .mf-brand .mf-brand-tag { color: var(--mf-primary); }
.mf-header .mf-current { font-size: 13px; color: var(--mf-text-2); }

/* ============================================================
   Sidebar
   ============================================================ */
.mf-sidebar { border-right: 1px solid var(--mf-border); padding-right: 12px; }
.mf-nav-group { font-size: 11px; font-weight: 600; text-transform: uppercase;
  letter-spacing: .06em; color: var(--mf-text-3); margin: 14px 0 4px 4px; }
.mf-nav-btn {
  display: block; width: 100%; text-align: left;
  padding: 6px 10px !important; margin: 1px 0 !important;
  font-size: 13px !important; font-weight: 500 !important;
  color: var(--mf-text-2) !important;
  background: transparent !important; border: none !important;
  border-left: 3px solid transparent !important;
  border-radius: var(--mf-r-sm) !important;
  transition: background .12s ease, color .12s ease, border-color .12s ease;
}
.mf-nav-btn:hover { background: var(--mf-surface) !important; color: var(--mf-text) !important; }
.mf-nav-btn.mf-nav-active {
  background: var(--mf-primary-soft) !important;
  color: var(--mf-primary) !important;
  border-left-color: var(--mf-primary) !important;
  font-weight: 600 !important;
}

/* ============================================================
   Main content / density
   ============================================================ */
.mf-main { padding-left: 16px; }
.mf-main > .block, .mf-main > .gr-block { background: transparent !important; }
.mf-page-title { font-size: 18px; font-weight: 600; color: var(--mf-text); margin: 0 0 12px 0; }
.mf-section-title { font-size: 14px; font-weight: 600; color: var(--mf-text); margin: 4px 0 8px 0; }
.mf-desc { color: var(--mf-text-2); font-size: 12px; }

/* 卡片：仅表达层级时使用（Surface + 边框 + 4px 圆角，不用大阴影） */
.mf-card {
  background: var(--mf-surface);
  border: 1px solid var(--mf-border);
  border-radius: var(--mf-r-md);
  padding: 12px;
  margin-bottom: 12px;
}

/* ============================================================
   Components
   ============================================================ */

/* ============================================================
   Checkbox（基于 Gradio 6.22 真实 DOM）
   真实结构: label.checkbox-container > input[type=checkbox][data-testid=checkbox] + span.label-text
   checked 仅改变 input.checked；用 CSS 伪元素自绘勾选，不依赖 Gradio Svelte 内部。
   ============================================================ */
.gradio-container label.checkbox-container {
  position: relative;
  display: inline-flex;
  align-items: center;
  gap: 8px;
  cursor: pointer;
  user-select: none;
}
/* 原生 input 视觉隐藏（保留交互），只占位不显示 */
.gradio-container label.checkbox-container input[type="checkbox"] {
  position: absolute;
  opacity: 0;
  width: 16px;
  height: 16px;
  margin: 0;
  cursor: pointer;
  accent-color: unset; /* 不依赖 accent-color（appearance:none 时无效） */
}
/* 自定义方框（::before） */
.gradio-container label.checkbox-container::before {
  content: "";
  width: 15px;
  height: 15px;
  flex: 0 0 15px;
  box-sizing: border-box;
  border: 1.5px solid var(--mf-border-strong);
  border-radius: 3px;
  background: var(--mf-surface);
  transition: background-color 120ms ease, border-color 120ms ease, box-shadow 120ms ease;
}
/* checked: teal 底 + 轻微 teal 光晕 */
.gradio-container label.checkbox-container:has(input[type="checkbox"]:checked)::before {
  background-color: var(--mf-primary);
  border-color: var(--mf-primary);
  box-shadow: 0 0 0 2px var(--mf-primary-soft);
}
/* 勾（::after，用 CSS 画，不依赖 Unicode 字符） */
.gradio-container label.checkbox-container:has(input[type="checkbox"]:checked)::after {
  content: "";
  position: absolute;
  left: 4px;
  top: 50%;
  width: 6px;
  height: 10px;
  border: solid #fff;
  border-width: 0 2px 2px 0;
  transform: translateY(-60%) rotate(45deg);
}
/* 文字 checked 时轻微 teal emphasis */
.gradio-container label.checkbox-container:has(input[type="checkbox"]:checked) .label-text {
  color: var(--mf-primary);
  font-weight: 600;
}
/* hover */
.gradio-container label.checkbox-container:hover::before {
  border-color: var(--mf-primary);
}
/* focus-visible: 轻微 teal 环（非强烈 glow） */
.gradio-container label.checkbox-container:has(input[type="checkbox"]:focus-visible)::before {
  outline: 2px solid var(--mf-primary-soft);
  outline-offset: 2px;
}

/* ---- Radio：原生 appearance 保留，仅 teal 染色（不作为主要方案，纯保险） ---- */
.gradio-container input[type="radio"] {
  accent-color: var(--mf-primary) !important;
  cursor: pointer;
}

/* ---- Primary CTA（开始翻译）——独立 ID 强约束，不被下面规则影响 ---- */
#mf-start-btn {
  font-family: var(--mf-font-sans) !important;
  background: var(--mf-primary) !important; color: #fff !important;
  border: 1px solid var(--mf-primary) !important; font-weight: 600;
  border-radius: var(--mf-r-sm) !important; font-size: 14px !important;
  min-height: 36px !important;
}
#mf-start-btn:hover { background: var(--mf-primary-hover) !important; }
#mf-start-btn:active { transform: scale(.98); }

/* ---- Cancel（危险操作，红色 ghost） ---- */
#mf-cancel-btn {
  font-family: var(--mf-font-sans) !important;
  background: transparent !important; color: var(--mf-error) !important;
  border: 1px solid var(--mf-error) !important; font-weight: 500;
  border-radius: var(--mf-r-sm) !important; font-size: 14px !important;
  min-height: 36px !important;
}

/* ---- Sidebar nav 按钮（独立 elem_classes .mf-nav-btn） ---- */
.mf-nav-btn {
  display: block; width: 100%; text-align: left;
  padding: 6px 10px !important; margin: 1px 0 !important;
  font-size: 13px !important; font-weight: 500 !important;
  color: var(--mf-text-2) !important; font-family: var(--mf-font-sans) !important;
  background: transparent !important; border: none !important;
  border-left: 3px solid transparent !important;
  border-radius: var(--mf-r-sm) !important;
  min-height: 0 !important;
  transition: background .12s ease, color .12s ease, border-color .12s ease;
}
.mf-nav-btn:hover { background: var(--mf-surface) !important; color: var(--mf-text) !important; }
.mf-nav-btn.mf-nav-active {
  background: var(--mf-primary-soft) !important;
  color: var(--mf-primary) !important;
  border-left-color: var(--mf-primary) !important;
  font-weight: 600 !important;
}

/* ---- 取消按钮：保持原有缩放反馈，**不缩放 tab 按钮**（防抖动） ---- */
.mf-nav-btn:active, #mf-start-btn:active { transform: scale(.98); }
.gradio-container .tab-nav button:active { transform: none !important; }

/* ---- 次要按钮（scoped，不影响 checkbox/radio/原生 form 控件） ---- */
/* 覆盖普通按钮（nav/CTA 已有独立样式，不受影响） */
.gradio-container button:not(.tab-nav button):not([type="checkbox"]):not([type="radio"]) {
  font-family: var(--mf-font-sans) !important;
  border-radius: var(--mf-r-sm) !important;
  font-size: 13px !important;
  min-height: 32px !important;
}
/* 默认/secondary 按钮（白底灰字） */
.gradio-container button:not(.tab-nav button):not([variant=primary]):not([variant=stop]):not(#mf-start-btn):not(#mf-cancel-btn):not(.mf-nav-btn) {
  background: var(--mf-surface) !important;
  color: var(--mf-text) !important;
  border: 1px solid var(--mf-border-strong) !important;
}
.gradio-container button:not(.tab-nav button):not([variant=primary]):not([variant=stop]):not(#mf-start-btn):not(#mf-cancel-btn):not(.mf-nav-btn):hover {
  background: var(--mf-surface-elev) !important;
  border-color: var(--mf-primary) !important;
}

/* ---- 输入/表单 ---- */
.gradio-container input, .gradio-container textarea, .gradio-container select {
  background: var(--mf-surface) !important; color: var(--mf-text) !important;
  border: 1px solid var(--mf-border) !important; border-radius: var(--mf-r-sm) !important;
  font-size: 13px !important;
  font-family: var(--mf-font-sans) !important;
}
.gradio-container input:focus, .gradio-container textarea:focus, .gradio-container select:focus {
  border-color: var(--mf-primary) !important;
  box-shadow: 0 0 0 2px var(--mf-primary-soft) !important; outline: none !important;
}
/* Dropdown 当前选中值：focus / 有值时更明显的左边 teal 条 */
.gradio-container .gr-dropdown:focus-within {
  border-color: var(--mf-primary) !important;
  box-shadow: 0 0 0 2px var(--mf-primary-soft) !important;
}
/* 输入 group 整体在 focused 时高亮 */
.gradio-container .gr-input:focus-within,
.gradio-container .gr-text-input:focus-within {
  border-color: var(--mf-primary) !important;
  box-shadow: 0 0 0 2px var(--mf-primary-soft) !important;
}
.gradio-container label { color: var(--mf-text-2) !important; font-size: 12px !important; }

/* ---- Disabled 状态（按钮被禁用 / 处理中） ---- */
.gradio-container button[disabled], .gradio-container button:disabled,
.gradio-container input:disabled, .gradio-container textarea:disabled,
.gradio-container select:disabled {
  opacity: 0.5 !important;
  cursor: not-allowed !important;
}

/* 分组/面板：紧凑密度 */
.gradio-container .gr-group, .gradio-container .gr-form,
.gradio-container .gr-row, .gradio-container .gr-column { gap: 8px !important; }
.gradio-container .gr-block, .gradio-container .gr-panel { padding: 8px !important; }

/* 表格（Dataframe）：紧凑 */
.gradio-container table, .gradio-container .gr-dataframe { font-size: 12px !important; }
.gradio-container table th { font-size: 11px !important; color: var(--mf-text-3) !important; font-weight: 600; }
.gradio-container table td { padding: 4px 8px !important; }

/* 日志/等宽文本 */
.mono, .gr-textbox textarea { font-family: var(--mf-font-mono) !important; font-size: 12.5px !important; line-height: 1.45 !important; }

/* 进度条 */
.gradio-container .progress-bar, .gradio-container .gr-progress {
  background: var(--mf-primary) !important; border-radius: 999px !important; height: 4px !important;
}

/* Accordion */
.gradio-container .gr-accordion .label-wrap { font-size: 13px !important; font-weight: 600; color: var(--mf-text); }
.gradio-container .gr-accordion { border: 1px solid var(--mf-border); border-radius: var(--mf-r-md); }

/* 滚动条（暗色表面下更顺眼） */
::-webkit-scrollbar { width: 10px; height: 10px; }
::-webkit-scrollbar-thumb { background: #d4d4d8; border-radius: 5px; }

/* a11y */
:focus-visible { outline: 2px solid var(--mf-primary) !important; outline-offset: 2px !important; }
::selection { background: var(--mf-primary-soft); }

/* ---- 状态徽标（文本为主，非 emoji） ---- */
.mf-status-ok { color: var(--mf-success); font-weight: 600; }
.mf-status-run { color: var(--mf-primary); font-weight: 600; }
.mf-status-warn { color: var(--mf-warning); font-weight: 600; }
.mf-status-err { color: var(--mf-error); font-weight: 600; }

/* ============================================================
   Translation Workspace
   ============================================================ */
.mf-controls { border-right: 1px solid var(--mf-border); padding-right: 12px; }
.mf-preview .gr-image, .mf-preview img {
  background: var(--mf-surface);
  border: 1px solid var(--mf-border);
  border-radius: var(--mf-r-md);
}
#mf-preview { min-height: 420px; }
/* Pipeline 步骤条 */
.mf-pipeline-steps {
  display: flex; align-items: center; gap: 6px; flex-wrap: wrap;
  padding: 10px 12px; margin-bottom: 12px;
  background: var(--mf-surface); border: 1px solid var(--mf-border);
  border-radius: var(--mf-r-md);
}
.mf-step {
  font-size: 12px; font-weight: 500; color: var(--mf-text-3);
  padding: 4px 10px; border-radius: 999px; border: 1px solid var(--mf-border);
}
.mf-step.active { color: var(--mf-primary); border-color: var(--mf-primary); background: var(--mf-primary-soft); font-weight: 600; }
.mf-step.done { color: var(--mf-success); border-color: rgba(22,163,74,.4); background: rgba(22,163,74,.06); }
.mf-step-sep { width: 14px; height: 1px; background: var(--mf-border); display: inline-block; }

/* ============================================================
   Layout Stability（Phase 3B Stability Fix）
   统一稳定容器模式：图片固定高度 + 等比 contain，动态文本预留空间，表格限高滚动
   ============================================================ */
/* 稳定图片容器：高度固定，图片等比 contain，不因加载改变父容器尺寸 */
.mf-img-stable {
  height: 360px !important;
  overflow: hidden;
  background: var(--mf-surface);
  border: 1px solid var(--mf-border);
  border-radius: var(--mf-r-md);
}
.mf-img-stable img {
  object-fit: contain !important;
  width: 100% !important;
  height: 100% !important;
}
/* 动态信息行：预留一行高度，内容增多不撑高 */
.mf-info-line { min-height: 22px !important; }
/* 固定表格区域：表头固定，数据区限高滚动，长文本不撑高页面 */
.mf-table-fixed {
  max-height: 420px !important;
  overflow-y: auto !important;
}
"""


# ---------------------------------------------------------------------------
# Navigation definition（Sidebar 分组 + 页面工厂）
# ---------------------------------------------------------------------------
# 每个条目: (group_label, [(key, 显示名, 工厂函数), ...])
_NAV_GROUPS = [
    ("Workspace", [
        ("translate", "翻译", lambda s: create_pipeline_page(s["pipeline"])),
    ]),
    ("Project", [
        ("projects", "项目", lambda s: create_project_page()),
        ("history", "历史", lambda s: create_history_page(s["history"])),
    ]),
    ("Tools", [
        ("debug", "调试", lambda s: create_debug_page(s["pipeline"], s["history"])),
        ("perf", "性能监控", lambda s: create_perf_monitor_page(s["pipeline"], s["history"])),
        ("quality", "质检报告", lambda s: create_quality_report_page()),
        ("compare", "模型对比", lambda s: create_compare_page(s["pipeline"])),
        ("render_lab", "Render Lab", lambda s: create_render_lab_page(s["pipeline"])),
    ]),
    ("ML Lab", [
        ("annotation", "标注中心", lambda s: create_annotation_page(s["annotation"], s["dataset"], s["pipeline"])),
        ("dataset", "数据集", lambda s: create_dataset_page(s["dataset"])),
        ("training", "训练中心", lambda s: create_training_page(s["training"], s["dataset"], s["registry"])),
        ("models", "模型库", lambda s: create_model_page(s["registry"], s["config"], s["pipeline"])),
        ("evaluation", "评估面板", lambda s: create_benchmark_page(s["benchmark"])),
    ]),
    ("System", [
        ("settings", "设置", lambda s: create_settings_page(s["config"])),
    ]),
]

# key -> 显示名
_PAGE_TITLES = {key: label for _, items in _NAV_GROUPS for key, label, _ in items}
_DEFAULT_PAGE = "translate"


def create_app() -> gr.Blocks:
    svc = {
        "config": GuiConfigService(),
        "history": RunHistoryService(),
        "benchmark": BenchmarkService(),
        "annotation": AnnotationService(),
        "dataset": DatasetService(),
        "training": TrainingService(),
        "registry": ModelRegistryService(),
    }
    svc["pipeline"] = GuiPipelineService(history_service=svc["history"])

    # 后台线程预加载模型，避免首次运行时长时间无反馈
    svc["pipeline"].warmup()

    with gr.Blocks(title="MangaFlow") as app:
        # ---- Header ----
        current_md = gr.HTML(
            f'<div class="mf-header">'
            f'<div class="mf-brand">MangaFlow<span class="mf-brand-tag"> / AI 漫画翻译工作台</span></div>'
            f'<div class="mf-current" id="mf-current">{_PAGE_TITLES[_DEFAULT_PAGE]}</div>'
            f'</div>'
        )

        with gr.Row():
            # ---- Sidebar ----
            with gr.Column(scale=0, min_width=190, elem_classes="mf-sidebar"):
                nav_buttons: dict[str, gr.Button] = {}
                for group, items in _NAV_GROUPS:
                    gr.Markdown(group, elem_classes="mf-nav-group")
                    for key, label, _ in items:
                        cls = ["mf-nav-btn"] + (["mf-nav-active"] if key == _DEFAULT_PAGE else [])
                        btn = gr.Button(
                            label,
                            elem_id=f"nav-{key}",
                            elem_classes=cls,
                            size="sm",
                        )
                        nav_buttons[key] = btn

            # ---- Main：14 个隐藏视图 ----
            with gr.Column(scale=1, elem_classes="mf-main"):
                views: dict[str, gr.Column] = {}
                for group, items in _NAV_GROUPS:
                    for key, label, factory in items:
                        with gr.Column(visible=(key == _DEFAULT_PAGE), elem_id=f"view-{key}") as col:
                            factory(svc)
                            views[key] = col

        # ---- 导航切换（闭包绑定 key，避免创建多余 State） ----
        nav_outputs = list(views.values()) + list(nav_buttons.values()) + [current_md]

        def _go(key: str):
            out = [gr.update(visible=(k == key)) for k in views]
            out += [gr.update(elem_classes=["mf-nav-btn"] + (["mf-nav-active"] if k == key else [])) for k in nav_buttons]
            out.append(
                gr.HTML(
                    f'<div class="mf-header">'
                    f'<div class="mf-brand">MangaFlow<span class="mf-brand-tag"> / AI 漫画翻译工作台</span></div>'
                    f'<div class="mf-current" id="mf-current">{_PAGE_TITLES.get(key, key)}</div>'
                    f'</div>'
                )
            )
            return out

        for key, btn in nav_buttons.items():
            def _handler(key=key):
                return _go(key)
            btn.click(fn=_handler, outputs=nav_outputs)

        gr.Markdown("MangaFlow / 本地运行的 AI 漫画翻译工具", elem_id="mf-footer")

    # queue：翻译任务为长耗时生成器，启用队列后进度流式、可取消、防并发卡顿
    app.queue(default_concurrency_limit=4)
    return app


def main():
    import argparse

    parser = argparse.ArgumentParser(description="MangaFlow / AI 漫画翻译工作台")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    app = create_app()
    logger.info(f"Starting MangaFlow on {args.host}:{args.port}")
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=_build_theme(),
        css=_CSS,
    )


if __name__ == "__main__":
    main()
