"""训练中心页面 — 包装 scripts/train_bubble_model.py，流式显示日志和指标"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from comic_translator.gui.services.dataset_service import DatasetService
from comic_translator.gui.services.model_registry_service import ModelRegistryService
from comic_translator.gui.services.training_service import TrainingService


BASE_MODELS = [
    ("YOLO11n-seg (nano, 快)", "models/bubble/yolo11n-seg.pt"),
    ("YOLO11m-seg (medium, 准)", "models/bubble/yolo11m-seg.pt"),
    ("YOLO11n-seg-bubble (已微调)", "models/bubble/yolo11n-seg-bubble.pt"),
]


def create_training_page(
    training_service: TrainingService,
    dataset_service: DatasetService,
    registry_service: ModelRegistryService,
):
    """创建训练中心标签页"""

    with gr.Column():
        gr.Markdown("## 训练中心")

        with gr.Row():
            dataset_dd = gr.Dropdown(
                label="数据集",
                choices=dataset_service.list_datasets(),
                value=None,
                interactive=True,
                allow_custom_value=True,
                scale=2,
            )
            model_dd = gr.Dropdown(
                label="基础模型",
                choices=[(label, path) for label, path in BASE_MODELS],
                value=BASE_MODELS[0][1],
                scale=2,
            )

        with gr.Row():
            epochs = gr.Slider(minimum=1, maximum=300, value=50, step=1, label="Epochs")
            batch = gr.Slider(minimum=1, maximum=32, value=4, step=1, label="Batch Size")
            imgsz = gr.Dropdown(
                choices=[640, 960, 1280, 1600], value=1600, label="Image Size",
            )
            run_name = gr.Textbox(label="实验名", value="bubble")

        with gr.Row():
            start_btn = gr.Button("开始训练", variant="primary", scale=2)
            cancel_btn = gr.Button("取消", variant="stop", scale=1)
            register_btn = gr.Button("注册到模型库", scale=1)

        with gr.Row():
            epoch_md = gr.Markdown("**Epoch:** -")
            metrics_md = gr.Markdown("**指标:** -")

        log_output = gr.Textbox(
            label="训练日志", lines=20, max_lines=30,
            interactive=False, autoscroll=True,
        )
        status_md = gr.Markdown("就绪")

    # ---------- 逻辑 ----------

    def start_training(dataset, model, epochs, batch, imgsz, run_name):
        if not dataset:
            yield "请先选择数据集", "**Epoch:** -", "**指标:** -", "错误"
            return
        if training_service.is_running():
            yield "已有训练在进行", "**Epoch:** -", "**指标:** -", "错误"
            return

        yaml_path = dataset_service.get_dataset_dir(dataset) / "dataset.yaml"
        if not yaml_path.exists():
            yield f"数据集缺少 dataset.yaml: {yaml_path}", "**Epoch:** -", "**指标:** -", "错误"
            return

        try:
            training_service.start_training(
                dataset_yaml=str(yaml_path),
                base_model=model,
                epochs=int(epochs),
                batch=int(batch),
                imgsz=int(imgsz),
                name=run_name or "bubble",
            )
        except Exception as e:
            yield f"启动失败: {e}", "**Epoch:** -", "**指标:** -", "错误"
            return

        # 流式读取日志
        log_lines = []
        for new_lines, status in training_service.stream_training(poll_interval=1.0):
            log_lines.extend(new_lines)
            log_text = "\n".join(log_lines[-80:])

            epoch_text = f"**Epoch:** {status.current_epoch}/{status.total_epochs}"
            if status.latest_metrics:
                m = status.latest_metrics
                metrics_text = (
                    f"**mAP50:** {m.get('mAP50', 0):.4f} | "
                    f"**mAP50-95:** {m.get('mAP50-95', 0):.4f} | "
                    f"**P:** {m.get('precision', 0):.4f} | "
                    f"**R:** {m.get('recall', 0):.4f}"
                )
            else:
                metrics_text = "**指标:** 等待首轮验证..."

            if status.running:
                phase = f"训练中 — epoch {status.current_epoch}/{status.total_epochs}"
            elif status.error:
                phase = f"失败: {status.error}"
            elif status.return_code == 0:
                phase = f"完成 → {status.output_path}"
            else:
                phase = f"退出码 {status.return_code}"

            yield log_text, epoch_text, metrics_text, phase

    def cancel_training():
        ok = training_service.cancel()
        return "已取消" if ok else "无运行中的训练"

    def register_model(dataset):
        status = training_service.status
        if not status.output_path:
            return "尚无训练产出"
        pt = Path(status.output_path)
        if not pt.exists():
            return f"模型文件不存在: {pt}"
        try:
            registry_service.register(
                pt_path=str(pt),
                dataset=dataset or status.dataset,
                metrics=status.latest_metrics,
                notes=f"epochs={status.total_epochs}",
            )
        except Exception as e:
            return f"注册失败: {e}"
        return f"已注册模型 {pt.name}"

    # ---------- 绑定 ----------

    start_btn.click(
        fn=start_training,
        inputs=[dataset_dd, model_dd, epochs, batch, imgsz, run_name],
        outputs=[log_output, epoch_md, metrics_md, status_md],
    )
    cancel_btn.click(fn=cancel_training, outputs=[status_md])
    register_btn.click(fn=register_model, inputs=[dataset_dd], outputs=[status_md])
