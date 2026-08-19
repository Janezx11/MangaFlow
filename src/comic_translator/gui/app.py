"""Comic AI Studio v0.4 — Gradio GUI 入口

人体工学重构:
- 14 个平铺 Tab → 5 个主 Tab + 🛠 高级工具折叠
- 普通用户只需看前 3 个 Tab：翻译 → 项目数据 → 历史
- 开发者工具归入「高级工具」

用法:
  python -m comic_translator.gui.app
  python -m comic_translator.gui.app --port 7860
  python -m comic_translator.gui.app --share
"""

import os
import sys
from pathlib import Path

# 强制 localhost 绕过系统代理 — 防止 Gradio 自检和事件流被 Clash 等代理拦截
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

    with gr.Blocks(title="Comic AI Studio v0.4") as app:
        gr.Markdown("# Comic AI Studio v0.4 · 人体工学重构")

        with gr.Tabs():
            # ====== 主功能 Tab（普通用户常用） ======
            with gr.Tab("📖 翻译"):
                create_pipeline_page(pipeline_service)

            with gr.Tab("📁 项目数据"):
                create_project_page()

            with gr.Tab("📜 历史"):
                create_history_page(history_service)

            with gr.Tab("⚙️ 设置"):
                create_settings_page(config_service)

            # ====== 🛠 高级工具（开发者 / 调试用） ======
            with gr.Tab("🛠 高级工具"):
                gr.Markdown(
                    "### 🔍 调试与分析\n"
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
                    "### 🤖 ML 模型开发\n"
                    "_标注数据 → 训练模型 → 注册到模型库 → 评估_"
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

    return app


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Comic AI Studio")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    app = create_app()
    logger.info(f"Starting Comic AI Studio v0.4 on {args.host}:{args.port}")
    app.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        inbrowser=True,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
