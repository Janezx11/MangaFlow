"""MangaFlow / AI 漫画翻译工作台（Gradio GUI 入口）

布局:
- 主 Tab: 翻译 / 项目数据 / 历史 / 设置
- 高级工具: 调试/性能/质检/模型对比/Render Lab + ML 开发
  (标注/数据集/训练/模型库/评估)

用法:
  python -m comic_translator.gui.app
  python -m comic_translator.gui.app --port 7860
  python -m comic_translator.gui.app --share
"""

import os
import sys
from pathlib import Path

# ---- 界面主题与样式 ----
_THEME = None


def _build_theme():
    """MangaFlow 主题（Soft 变体 + 暖中性强调色）

    配色一致性锁：全站单一强调色 orange，深底文字 slate，避免紫/蓝 AI 默认。
    形状一致性锁：全站 8px 圆角（按钮/输入/卡片/tab 选中条统一）。
    """
    global _THEME
    if _THEME is None:
        _THEME = gr.themes.Soft(
            primary_hue=gr.themes.Color("orange", hue_500="#c2410c", hue_600="#c2410c", hue_700="#9a3412"),
            secondary_hue="stone",
            neutral_hue="stone",
        )
    return _THEME


_CSS = """
/* ---- 字体（首选 Geist；无网时优雅回退到系统字体，含中文回退） ---- */
:root {
  --mf-font-sans: 'Geist', 'Geist Fallback', -apple-system, BlinkMacSystemFont,
                   'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei',
                   system-ui, sans-serif;
  --mf-font-mono: 'Geist Mono', 'JetBrains Mono', ui-monospace, 'SFMono-Regular',
                   'Cascadia Code', Consolas, monospace;
  --mf-accent: #c2410c;
  --mf-accent-700: #9a3412;
  --mf-accent-50: #fff7ed;
  --mf-text: #1c1917;
  --mf-text-sub: #57534e;
  --mf-bg: #fafaf9;
  --mf-bg-elev: #ffffff;
  --mf-border: #e7e5e4;
  --mf-radius: 8px;
}
@media (prefers-color-scheme: dark) {
  :root {
    --mf-accent: #fb923c;
    --mf-accent-700: #fdba74;
    --mf-accent-50: #431407;
    --mf-text: #f5f5f4;
    --mf-text-sub: #a8a29e;
    --mf-bg: #0c0a09;
    --mf-bg-elev: #1c1917;
    --mf-border: #292524;
  }
}
body, .gradio-container, .gr-block { font-family: var(--mf-font-sans); }
code, pre, .mono, .gr-code, kbd, samp { font-family: var(--mf-font-mono); }

/* ---- 品牌头部（暖中性强调，亮/暗双模式锁） ---- */
#mf-title { border-bottom: 2px solid var(--mf-accent); padding-bottom: 0.4rem; margin-bottom: 0.6rem; }
#mf-title h1 { font-size: 2.2rem; margin: 0.35rem 0 0.1rem; letter-spacing: 0.5px; color: var(--mf-text); font-weight: 700; }
#mf-tagline { color: var(--mf-text-sub); margin-top: -0.15rem; margin-bottom: 0.4rem; font-size: 1rem; }
#mf-footer { color: var(--mf-text-sub); font-size: 0.82rem; text-align: center; margin-top: 2.4rem; }
.gradio-container { max-width: 1320px !important; margin: 0 auto !important; padding: 0.6rem 1.2rem 1.6rem; background: var(--mf-bg); color: var(--mf-text); }

/* ---- 形状一致性锁：全站 8px 圆角 ---- */
button, .gr-button, .gr-box, .gr-input, .gr-text-input, .gr-dropdown, .gr-tabs,
.block, .gr-block, .gr-panel, .gr-form, .gr-group, .gr-accordion,
.gap, .min, .max, .wrap, .form, .gallery, .panel { border-radius: var(--mf-radius) !important; }

/* ---- 卡片化分组（仅用于真正需要层级的地方；不滥用） ---- */
.mf-card { border: 1px solid var(--mf-border); border-radius: var(--mf-radius);
  padding: 0.85rem 1rem 0.6rem; background: var(--mf-bg-elev);
  margin-bottom: 0.5rem; }
.mf-card h3 { margin: 0.05rem 0 0.55rem; font-size: 1.02rem; color: var(--mf-text); }
.mf-card .mf-desc { color: var(--mf-text-sub); font-size: 0.9rem; margin-bottom: 0.5rem; }
.mf-section-title h2 { margin: 0.2rem 0 0.4rem; color: var(--mf-text); }

/* ---- 紧凑化：表单/输入行/分组间距收紧（Daily App 风格，py-4 ~ py-6） ---- */
.gr-form, .gr-group, .gr-input, .gr-dropdown, .gr-box,
.gr-row, .gr-column { gap: 0.5rem !important; }
.gr-block, .gr-panel { padding: 0.6rem 0.8rem !important; }
.gr-block-label, label.block-label { color: var(--mf-text) !important; font-size: 0.9rem; }
.gr-input, .gr-text-input, .gr-dropdown { background: var(--mf-bg-elev) !important; color: var(--mf-text) !important; border-color: var(--mf-border) !important; }
.tabs { margin-top: 0.2rem; }

/* ---- 强调按钮（CTA 统一强调色 + contrast 必过 WCAG AA） ---- */
#mf-start-btn { height: 46px !important; font-size: 1.05rem !important; font-weight: 600 !important; border-radius: var(--mf-radius) !important; background: var(--mf-accent) !important; border-color: var(--mf-accent) !important; color: #ffffff !important; }
#mf-cancel-btn { height: 46px !important; border-radius: var(--mf-radius) !important; }

/* ---- 进度与状态 ---- */
#mf-status { font-size: 1rem; color: var(--mf-text); }
#mf-perf { font-size: 0.92rem; color: var(--mf-text-sub); }

/* ---- Tab 栏（强调条用暖中性） ---- */
.tab-nav button.selected, .tabitem[aria-selected="true"] { border-color: var(--mf-accent) !important; color: var(--mf-accent-700) !important; }
"""


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


def create_app() -> gr.Blocks:
    config_service = GuiConfigService()
    history_service = RunHistoryService()
    benchmark_service = BenchmarkService()
    pipeline_service = GuiPipelineService(history_service=history_service)

    # v0.3 AI Data Lab 服务
    annotation_service = AnnotationService()
    dataset_service = DatasetService()
    training_service = TrainingService()
    registry_service = ModelRegistryService()

    # 后台线程预加载模型，避免首次运行时长时间无反馈
    pipeline_service.warmup()

    # Gradio 6：theme / css 改由 launch() 传入（create_app 不挂 Blocks 参数）
    with gr.Blocks(title="MangaFlow") as app:
        gr.Markdown("# 📖 MangaFlow", elem_id="mf-title")
        gr.Markdown(
            "_AI 漫画翻译工作台 / 文字检测 / OCR / LLM 翻译 / 渲染回填_",
            elem_id="mf-tagline",
        )

        with gr.Tabs():            # ====== 主功能 Tab（普通用户常用） ======
            with gr.Tab("翻译"):
                create_pipeline_page(pipeline_service)

            with gr.Tab("项目数据"):
                create_project_page()

            with gr.Tab("历史"):
                create_history_page(history_service)

            with gr.Tab("设置"):
                create_settings_page(config_service)

            # ====== 高级工具（开发者 / 调试用） ======
            with gr.Tab("高级工具"):
                gr.Markdown(
                    "### 调试与分析\n"
                    "_运行中查看单页详情、性能指标、渲染质量、模型对比等_"
                )
                with gr.Tabs():
                    with gr.Tab("调试查看器"):
                        create_debug_page(pipeline_service, history_service)
                    with gr.Tab("性能监控"):
                        create_perf_monitor_page(pipeline_service, history_service)
                    with gr.Tab("质检报告"):
                        create_quality_report_page()
                    with gr.Tab("模型对比"):
                        create_compare_page(pipeline_service)
                    with gr.Tab("Render Lab"):
                        create_render_lab_page(pipeline_service)

                gr.Markdown(
                    "---\n"
                    "### ML 模型开发\n"
                    "_标注数据 -> 训练模型 -> 注册到模型库 -> 评估_"
                )
                with gr.Tabs():
                    with gr.Tab("标注中心"):
                        create_annotation_page(annotation_service, dataset_service, pipeline_service)
                    with gr.Tab("数据集"):
                        create_dataset_page(dataset_service)
                    with gr.Tab("训练中心"):
                        create_training_page(training_service, dataset_service, registry_service)
                    with gr.Tab("模型库"):
                        create_model_page(registry_service, config_service, pipeline_service)
                    with gr.Tab("评估面板"):
                        create_benchmark_page(benchmark_service)

        gr.Markdown(
            "MangaFlow / 本地运行的 AI 漫画翻译工具 / 数据留在本机，译文记忆可复用",
            elem_id="mf-footer",
        )

    # queue()：翻译任务为长耗时生成器，启用队列后进度流式、可取消、防并发卡顿
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
