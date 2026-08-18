"""数据集管理页面 — 创建/导入/统计/清理"""

from __future__ import annotations

import gradio as gr

from comic_translator.gui.services.dataset_service import DatasetService


def create_dataset_page(dataset_service: DatasetService):
    """创建数据集管理标签页"""

    with gr.Column():
        gr.Markdown("## 数据集管理")

        with gr.Row():
            dataset_list = gr.Dropdown(
                label="已有数据集",
                choices=dataset_service.list_datasets(),
                value=None,
                interactive=True,
                allow_custom_value=True,
                scale=2,
            )
            refresh_btn = gr.Button("刷新", size="sm", scale=1, min_width=80)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 创建数据集")
                new_name = gr.Textbox(label="名称", placeholder="例如: bubble_v1")
                create_btn = gr.Button("创建", variant="primary")

            with gr.Column():
                gr.Markdown("### 导入图片")
                src_dir = gr.Textbox(label="源目录", placeholder="例如: data/comics")
                with gr.Row():
                    split_radio = gr.Radio(
                        choices=["train", "val"], value="train", label="划分", scale=1,
                    )
                    max_count = gr.Number(label="最多导入", value=0, precision=0, scale=1)
                import_btn = gr.Button("导入", variant="primary")

        with gr.Row():
            delete_dataset_btn = gr.Button("删除当前数据集", variant="stop")

        status_md = gr.Markdown("就绪")

        gr.Markdown("### 数据集统计")
        stats_table = gr.Dataframe(
            headers=["指标", "值"],
            value=[],
            interactive=False,
        )

        gr.Markdown("### 类别分布")
        class_table = gr.Dataframe(
            headers=["类别", "数量"],
            value=[],
            interactive=False,
        )

        gr.Markdown("### 未标注图片")
        unlabeled_table = gr.Dataframe(
            headers=["图片"],
            value=[],
            interactive=False,
        )

        with gr.Row():
            unlabeled_select = gr.Dropdown(
                label="选择要删除的图片", choices=[],
                interactive=True, allow_custom_value=True, scale=3,
            )
            delete_img_btn = gr.Button("删除图片+标签", variant="stop", scale=1)

    # ---------- 逻辑 ----------

    def refresh_list():
        choices = dataset_service.list_datasets()
        return gr.update(choices=choices, value=choices[0] if choices else None)

    def show_stats(name):
        if not name:
            return [], [], [], gr.update(choices=[]), "未选择数据集"
        stats = dataset_service.get_stats(name)
        if stats is None:
            return [], [], [], gr.update(choices=[]), f"数据集 {name} 不存在"

        stats_rows = [
            ["路径", stats.path],
            ["训练图片", stats.train_images],
            ["训练标签", stats.train_labels],
            ["验证图片", stats.val_images],
            ["验证标签", stats.val_labels],
            ["未标注", len(stats.unlabeled_images)],
        ]
        class_rows = [[k, v] for k, v in sorted(stats.class_counts.items())]
        unlabeled_rows = [[p] for p in stats.unlabeled_images]
        unlabeled_choices = stats.unlabeled_images

        return (
            stats_rows, class_rows, unlabeled_rows,
            gr.update(choices=unlabeled_choices, value=None),
            f"已加载 {name}",
        )

    def create_dataset(name):
        try:
            dataset_service.create_dataset(name.strip())
        except Exception as e:
            return gr.update(), f"创建失败: {e}"
        return refresh_list(), f"已创建数据集 {name}"

    def import_images(name, src_dir, split, max_count):
        if not name:
            return "请先选择数据集"
        if not src_dir:
            return "请输入源目录"
        try:
            mc = int(max_count) if max_count else None
            if mc == 0:
                mc = None
            count = dataset_service.import_images(name, src_dir, split, mc)
        except Exception as e:
            return f"导入失败: {e}"
        return f"已导入 {count} 张图片到 {name}/{split}"

    def delete_dataset(name):
        if not name:
            return gr.update(), [], [], [], "未选择数据集"
        ok = dataset_service.delete_dataset(name)
        if not ok:
            return gr.update(), [], [], [], f"数据集 {name} 不存在"
        return refresh_list(), [], [], [], f"已删除数据集 {name}"

    def delete_image(name, rel_path):
        if not name or not rel_path:
            return gr.update(), [], [], [], gr.update(), "未选择图片"
        ok = dataset_service.delete_image(name, rel_path)
        stats_rows, class_rows, unlabeled_rows, unlabeled_update, _ = show_stats(name)
        msg = f"已删除 {rel_path}" if ok else f"删除失败 {rel_path}"
        return (
            gr.update(), stats_rows, class_rows, unlabeled_rows, unlabeled_update, msg,
        )

    # ---------- 绑定 ----------

    refresh_btn.click(fn=refresh_list, outputs=[dataset_list])
    dataset_list.change(
        fn=show_stats,
        inputs=[dataset_list],
        outputs=[stats_table, class_table, unlabeled_table, unlabeled_select, status_md],
    )
    create_btn.click(
        fn=create_dataset,
        inputs=[new_name],
        outputs=[dataset_list, status_md],
    )
    import_btn.click(
        fn=import_images,
        inputs=[dataset_list, src_dir, split_radio, max_count],
        outputs=[status_md],
    ).then(
        fn=show_stats,
        inputs=[dataset_list],
        outputs=[stats_table, class_table, unlabeled_table, unlabeled_select, status_md],
    )
    delete_dataset_btn.click(
        fn=delete_dataset,
        inputs=[dataset_list],
        outputs=[dataset_list, stats_table, class_table, unlabeled_table, status_md],
    )
    delete_img_btn.click(
        fn=delete_image,
        inputs=[dataset_list, unlabeled_select],
        outputs=[dataset_list, stats_table, class_table, unlabeled_table,
                 unlabeled_select, status_md],
    )
