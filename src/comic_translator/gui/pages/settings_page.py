"""设置页面 — 显示和修改配置"""

from __future__ import annotations

import gradio as gr

from comic_translator.gui.services.gui_config_service import GuiConfigService, mask_api_key


def create_settings_page(config_service: GuiConfigService):
    """创建设置标签页"""

    cfg = config_service.summary()
    key_state = config_service.get_api_key_state()

    with gr.Column():
        gr.Markdown("## 设置")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 大语言模型")
                llm_mode = gr.Dropdown(
                    label="运行模式",
                    choices=[("API（OpenAI 兼容）", "api"), ("Ollama 本地", "ollama")],
                    value=cfg["llm_mode"],
                )
                with gr.Row():
                    llm_model = gr.Dropdown(
                        label="模型",
                        choices=[cfg["llm_model"]] if cfg["llm_model"] else [],
                        value=cfg["llm_model"] or None,
                        allow_custom_value=True,
                        scale=3,
                    )
                    fetch_btn = gr.Button("获取可用模型", size="sm", scale=1)
                fetch_status_md = gr.Markdown("")
                llm_url = gr.Textbox(label="API 地址", value=cfg["llm_base_url"])
                api_key = gr.Textbox(
                    label="API Key",
                    type="password",
                    placeholder=_key_placeholder(key_state),
                    value="",
                )
                api_key_status = gr.Markdown(_key_status_md(key_state))

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

    def save_config(
        bubble_model, bubble_conf, target_lang, style, output_dir,
        llm_mode, llm_model, llm_url, api_key,
    ):
        # 记录 LLM 字段是否变化（决定是否需要热更新客户端）
        old_llm = (
            config_service.get("llm.mode"),
            config_service.get("llm.api.model"),
            config_service.get("llm.api.base_url"),
        )
        llm_changed = (llm_mode, llm_model, llm_url) != old_llm

        config_service.set("llm.mode", llm_mode)
        config_service.set("llm.api.model", llm_model)
        config_service.set("llm.api.base_url", llm_url)
        config_service.set("bubble.model", bubble_model)
        config_service.set("bubble.conf_threshold", bubble_conf)
        config_service.set("translation.target_lang", target_lang)
        config_service.set("translation.style", style)
        config_service.set("project.output_dir", output_dir)

        key_changed = bool(api_key and api_key.strip())
        if key_changed:
            config_service.set_api_key(api_key)
        config_service.save()

        notes = []
        if key_changed or llm_changed:
            note = _refresh_llm_client()
            if note:
                notes.append(note)
        else:
            notes.append("API key 留空，未改动")

        new_state = config_service.get_api_key_state()
        msg = "配置已保存到 config.yaml"
        if notes:
            msg += "（" + "；".join(notes) + "）"
        # 返回清空的 key 输入框 + 最新状态
        return msg, _key_status_md(new_state), ""

    def reload_config():
        config_service.reload()
        cfg = config_service.summary()
        new_state = config_service.get_api_key_state()
        return (
            cfg["llm_mode"], cfg["llm_model"], cfg["llm_base_url"],
            cfg["bubble_model"], cfg["bubble_conf"],
            cfg["target_lang"], cfg["style"], cfg["output_dir"],
            "配置已重新加载",
            _key_status_md(new_state),
            "",
        )

    def fetch_models(api_key_typed, llm_url, llm_mode, llm_model):
        """根据 key + 地址自动拉取可用模型，填充下拉框。"""
        try:
            if llm_mode == "ollama":
                base = config_service.get("llm.ollama.base_url", "http://localhost:11434")
                models = config_service.fetch_ollama_models(base)
                if not models:
                    return gr.update(), "⚠️ 本机没有已安装的 Ollama 模型"
                new_value = llm_model if llm_model in models else models[0]
                return (
                    gr.update(choices=models, value=new_value),
                    f"✅ 获取到 {len(models)} 个本地模型",
                )
            # API 模式：优先用刚填的 key，否则用配置里已生效的 key
            key = (api_key_typed or "").strip() or config_service.effective_api_key()
            if not key:
                return gr.update(), "❌ 请先填写 API Key（或已在 config.yaml 中配置）"
            models = config_service.fetch_models(llm_url, key)
            if not models:
                return gr.update(), "⚠️ 接口返回了空模型列表"
            new_value = llm_model if llm_model in models else models[0]
            return (
                gr.update(choices=models, value=new_value),
                f"✅ 获取到 {len(models)} 个可用模型",
            )
        except Exception as e:  # noqa: BLE001
            return gr.update(), f"❌ 获取模型失败：{e}"

    save_btn.click(
        fn=save_config,
        inputs=[
            bubble_model, bubble_conf, target_lang, style, output_dir,
            llm_mode, llm_model, llm_url, api_key,
        ],
        outputs=[status_md, api_key_status, api_key],
    )
    reload_btn.click(
        fn=reload_config,
        outputs=[
            llm_mode, llm_model, llm_url,
            bubble_model, bubble_conf, target_lang, style, output_dir,
            status_md, api_key_status, api_key,
        ],
    )
    fetch_btn.click(
        fn=fetch_models,
        inputs=[api_key, llm_url, llm_mode, llm_model],
        outputs=[llm_model, fetch_status_md],
    )


def _key_placeholder(state: dict) -> str:
    if state["configured"]:
        return f"已设置（{mask_api_key(state['preview'])}），留空保持不变"
    return "未设置 — 在此粘贴 API key"


def _key_status_md(state: dict) -> str:
    if not state["configured"]:
        return "🟡 **API key：未设置**（翻译调用将失败）"
    return f"🟢 **API key：已设置**（来源 {state['source']}，`{mask_api_key(state['preview'])}`）"


def _refresh_llm_client() -> str:
    """热刷新 DI 容器中的 LLM 客户端；返回警示信息或空串。"""
    try:
        from comic_translator.utils.di import refresh_llm_client
        refreshed = refresh_llm_client()
        if refreshed is None:
            return "容器未初始化，key 将在下次运行时生效"
        return ""
    except Exception as e:  # noqa: BLE001
        return f"LLM 客户端刷新失败，重启后生效（{e}）"


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
