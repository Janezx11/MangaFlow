"""设置页面 — 显示和修改配置"""

from __future__ import annotations

import gradio as gr

from comic_translator.gui.services.gui_config_service import GuiConfigService


def create_settings_page(config_service: GuiConfigService):
    """创建设置标签页"""

    cfg = config_service.summary()

    with gr.Column():
        gr.Markdown("## 设置")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 大语言模型")
                llm_mode = gr.Textbox(label="运行模式", value=cfg["llm_mode"], interactive=False)
                llm_model = gr.Textbox(label="模型", value=cfg["llm_model"], interactive=False)
                llm_url = gr.Textbox(label="API 地址", value=cfg["llm_base_url"], interactive=False)

            with gr.Column():
                gr.Markdown("### 检测模型")
                bubble_model = gr.Textbox(label="气泡检测模型路径", value=cfg["bubble_model"])
                bubble_conf = gr.Slider(
                    minimum=0.1, maximum=0.9, value=cfg["bubble_conf"], step=0.05,
                    label="气泡检测置信度",
                )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 翻译")
                target_lang = gr.Dropdown(
                    label="目标语言",
                    choices=[("中文", "zh"), ("日语", "ja"), ("韩语", "ko"), ("英语", "en")],
                    value=cfg["target_lang"],
                )
                style = gr.Dropdown(
                    label="漫画风格",
                    choices=[("欧美漫画", "western_comic"), ("日本漫画", "manga"), ("条漫", "webtoon")],
                    value=cfg["style"],
                )

            with gr.Column():
                gr.Markdown("### 输出")
                output_dir = gr.Textbox(label="输出目录", value=cfg["output_dir"])

        with gr.Row():
            save_btn = gr.Button("保存配置", variant="primary")
            reload_btn = gr.Button("重新加载")

        status_md = gr.Markdown("")

        gr.Markdown("### GPU 状态")
        gpu_md = gr.Markdown(_get_gpu_status())

    def save_config(bubble_model, bubble_conf, target_lang, style, output_dir):
        config_service.set("bubble.model", bubble_model)
        config_service.set("bubble.conf_threshold", bubble_conf)
        config_service.set("translation.target_lang", target_lang)
        config_service.set("translation.style", style)
        config_service.set("project.output_dir", output_dir)
        config_service.save()
        return "配置已保存到 config.yaml"

    def reload_config():
        config_service.reload()
        cfg = config_service.summary()
        return (
            cfg["bubble_model"], cfg["bubble_conf"],
            cfg["target_lang"], cfg["style"], cfg["output_dir"],
            "配置已重新加载",
        )

    save_btn.click(
        fn=save_config,
        inputs=[bubble_model, bubble_conf, target_lang, style, output_dir],
        outputs=[status_md],
    )
    reload_btn.click(
        fn=reload_config,
        outputs=[bubble_model, bubble_conf, target_lang, style, output_dir, status_md],
    )


def _get_gpu_status() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = getattr(props, 'total_memory', getattr(props, 'total_mem', 0)) / (1024**3)
            return f"**CUDA 可用**: {name}（{vram:.1f} GB 显存）"
        return "**CUDA 不可用** — 当前使用 CPU 运行"
    except ImportError:
        return "**PyTorch 未安装** — 无法检测 GPU"
