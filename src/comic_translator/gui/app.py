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
    """MangaFlow 主题（Soft 变体 + 青绿强调，科技感）

    配色一致性锁：全站单一强调色 cyan，深色工作台基调（由 CSS 强压）。
    形状一致性锁：全站 8px 圆角。
    深色为默认主题（科技感工作台），亮色模式保留作为可选项。
    """
    global _THEME
    if _THEME is None:
        _THEME = gr.themes.Soft(
            primary_hue="cyan",
            secondary_hue="slate",
            neutral_hue="slate",
        )
    return _THEME


_CSS = """
/* ============================================================
   MangaFlow 深色科技工作台皮肤
   设计读：本地专业工具 / 冷静高信息密度 / 青绿强调
   VARIANCE 4 / MOTION 2 / DENSITY 6
   ============================================================ */
:root {
  --mf-font-sans: 'Geist', 'Geist Fallback', -apple-system, BlinkMacSystemFont,
                   'Segoe UI', 'PingFang SC', 'Hiragino Sans GB', 'Microsoft YaHei',
                   system-ui, sans-serif;
  --mf-font-mono: 'Geist Mono', 'JetBrains Mono', ui-monospace, 'SFMono-Regular',
                   'Cascadia Code', Consolas, monospace;
  --mf-accent: #22d3ee;          /* 青绿强调 */
  --mf-accent-strong: #06b6d4;
  --mf-accent-dim: rgba(34, 211, 238, 0.14);
  --mf-accent-text: #0c0a09;     /* 青绿按钮上的深色文字（对比度佳） */
  --mf-text: #e7e5e4;            /* 深色下主文字 */
  --mf-text-sub: #a8a29e;        /* 次级文字 */
  --mf-bg: #0c0a09;              /* 深黑底 */
  --mf-bg-elev: #171412;         /* 提升层（卡片/输入） */
  --mf-bg-hover: #1f1b19;        /* hover */
  --mf-border: #2b2624;          /* 细边框 */
  --mf-border-accent: rgba(34, 211, 238, 0.5);
  --mf-radius: 8px;
  --mf-glow: 0 0 0 1px rgba(34,211,238,.12), 0 8px 30px rgba(0,0,0,.45);
}
body, .gradio-container, .gr-block { font-family: var(--mf-font-sans); color: var(--mf-text); }
code, pre, .mono, .gr-code, kbd, samp { font-family: var(--mf-font-mono); }

/* ---- 根容器：深色工作台底板 ---- */
.gradio-container {
  max-width: 1440px !important;
  margin: 0 auto !important;
  padding: 1rem 1.2rem 1.8rem !important;
  background: var(--mf-bg) !important;
  border: 1px solid var(--mf-border);
  border-radius: 16px;
  box-shadow: var(--mf-glow);
}

/* ---- 自定义头部横幅（深色 + 青绿边条 + 微光） ---- */
#mf-hero {
  background: linear-gradient(135deg, #0c0a09 0%, #131010 55%, rgba(6,182,212,.10) 120%) !important;
  border-radius: 12px !important;
  border: 1px solid var(--mf-border);
  border-left: 4px solid var(--mf-accent) !important;
  box-shadow: inset 0 1px 0 rgba(255,255,255,.04);
}
#mf-hero .mf-hero-title { font-size: 1.8rem; font-weight: 700; letter-spacing: .5px; color: #fff; }
#mf-hero .mf-hero-sub { color: var(--mf-text-sub); font-size: 1rem; margin-top: .15rem; }
#mf-footer { color: var(--mf-text-sub); font-size: .82rem; text-align: center; margin-top: 2.4rem; }

/* ---- 形状一致性锁：全站 8px 圆角 ---- */
button, .gr-button, .gr-box, .gr-input, .gr-text-input, .gr-dropdown, .gr-tabs,
.block, .gr-block, .gr-panel, .gr-form, .gr-group, .gr-accordion,
.gap, .min, .max, .wrap, .form, .gallery, .panel, .gr-dataframe, .gr-slider {
  border-radius: var(--mf-radius) !important;
}

/* ---- 分组/卡片：半透明深色卡片 + 细边框 ---- */
.mf-card {
  border: 1px solid var(--mf-border);
  border-radius: var(--mf-radius);
  padding: 0.85rem 1rem 0.6rem;
  background: var(--mf-bg-elev);
  margin-bottom: 0.5rem;
  transition: border-color .18s ease, box-shadow .18s ease;
}
.mf-card:hover { border-color: var(--mf-border-accent); }
.mf-card h3 { margin: .05rem 0 .55rem; font-size: 1.02rem; color: var(--mf-text); }
.mf-card .mf-desc { color: var(--mf-text-sub); font-size: .9rem; margin-bottom: .5rem; }
.mf-section-title h2 { margin: .2rem 0 .4rem; color: var(--mf-text); }

/* ---- 输入/下拉/表单：深色底 + 细边框 + focus 光晕 ---- */
.gradio-container input, .gradio-container textarea, .gradio-container select,
.gradio-container .gr-input, .gradio-container .gr-text-input,
.gradio-container .gr-dropdown, .gradio-container .gr-number,
.gradio-container .gr-slider input[type=range] {
  background: var(--mf-bg-elev) !important;
  color: var(--mf-text) !important;
  border: 1px solid var(--mf-border) !important;
  border-radius: var(--mf-radius) !important;
}
.gradio-container input:focus, .gradio-container textarea:focus,
.gradio-container select:focus, .gradio-container .gr-input:focus {
  border-color: var(--mf-accent) !important;
  box-shadow: 0 0 0 3px var(--mf-accent-dim) !important;
  outline: none !important;
}
.gradio-container label, .gradio-container .block-label { color: var(--mf-text-sub) !important; font-size: .9rem; }

/* ---- 按钮：深底 + 强调 hover 发光 + 按压反馈 ---- */
.gradio-container button {
  background: var(--mf-bg-elev) !important;
  color: var(--mf-text) !important;
  border: 1px solid var(--mf-border) !important;
  border-radius: var(--mf-radius) !important;
  transition: border-color .15s ease, box-shadow .15s ease, transform .08s ease;
}
.gradio-container button:hover { border-color: var(--mf-accent) !important; box-shadow: 0 0 12px var(--mf-accent-dim) !important; }
.gradio-container button:active { transform: scale(.98); }
.gradio-container button.primary, .gradio-container button[variant=primary],
#mf-start-btn {
  background: var(--mf-accent) !important;
  color: var(--mf-accent-text) !important;
  border: none !important;
  font-weight: 600;
}
#mf-start-btn { height: 46px !important; font-size: 1.05rem !important; }
#mf-cancel-btn { height: 46px !important; }

/* ---- 紧凑化：表单/输入/分组间距（D6） ---- */
.gradio-container .gr-form, .gradio-container .gr-group,
.gradio-container .gr-row, .gradio-container .gr-column { gap: 0.5rem !important; }
.gradio-container .gr-block, .gradio-container .gr-panel { padding: 0.6rem 0.8rem !important; }

/* ---- 进度条：细线 + 青绿 ---- */
.gradio-container .progress-bar, .gradio-container .gr-progress {
  background: var(--mf-accent-strong) !important; border-radius: 999px !important;
}

/* ---- Tab 栏：选中态青绿下划线 ---- */
.gradio-container .tab-nav button { color: var(--mf-text-sub) !important; }
.gradio-container .tab-nav button.selected { color: var(--mf-accent) !important; border-color: var(--mf-accent) !important; }

/* ---- 状态/性能文本 ---- */
#mf-status { font-size: 1rem; color: var(--mf-text); }
#mf-perf { font-size: .92rem; color: var(--mf-text-sub); font-family: var(--mf-font-mono); }

/* ---- a11y：焦点环 + 选中文本 ---- */
:focus-visible { outline: 2px solid var(--mf-accent) !important; outline-offset: 2px !important; }
::selection { background: var(--mf-accent-dim); color: var(--mf-text); }
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
        # 自定义头部横幅（深色科技感，内联样式保证渲染）
        gr.HTML(
            """
            <div id="mf-hero">
              <div class="mf-hero-title">MangaFlow</div>
              <div class="mf-hero-sub">AI 漫画翻译工作台 / 文字检测 / OCR / LLM 翻译 / 渲染回填</div>
            </div>
            """
        )

        with gr.Tabs():            # ====== 信息架构：翻译 / 记录 / 设置 / 模型 / 高级 ======
            # ① 翻译：核心工作流，唯一主 CTA
            with gr.Tab("翻译"):
                create_pipeline_page(pipeline_service)

            # ② 记录：低频但重要的项目与历史
            with gr.Tab("记录"):
                gr.Markdown("### 项目数据 / 历史")
                with gr.Tabs():
                    with gr.Tab("项目数据"):
                        create_project_page()
                    with gr.Tab("历史"):
                        create_history_page(history_service)

            # ③ 设置
            with gr.Tab("设置"):
                create_settings_page(config_service)

            # ④ 模型：ML 开发（有真实产出价值的训练/标注/数据集从"高级"提上来）
            with gr.Tab("模型"):
                gr.Markdown("### ML 模型开发\n_标注数据 -> 训练模型 -> 注册到模型库 -> 评估_")
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

            # ⑤ 高级：纯调试 / 开发者工具
            with gr.Tab("高级"):
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
