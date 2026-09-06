"""评估面板页面 — 展示模型 Benchmark 结果"""

from __future__ import annotations

from pathlib import Path

import gradio as gr

from comic_translator.gui.services.benchmark_service import BenchmarkService


def create_benchmark_page(benchmark_service: BenchmarkService):
    """创建评估面板标签页"""

    # ---------- 逻辑 ----------

    def reload(path: str):
        p = Path(path) if path else Path(BenchmarkService.DEFAULT_PATH)
        if not p.exists():
            return [], [], f"文件不存在: {p}"

        service = BenchmarkService(str(p))
        summary = service.summary()
        if not summary:
            return [], [], f"JSON 为空或格式错误: {p}"

        # 汇总表
        summary_rows = []
        for name, s in sorted(summary.items()):
            labels = ", ".join(f"{k}×{v}" for k, v in s["label_counts"].items())
            summary_rows.append([
                name,
                s["pages"],
                s["total_regions"],
                s["avg_confidence"],
                s["avg_time_s"],
                labels or "-",
            ])

        # 逐页明细（动态列，把所有指标合成字符串）
        detail_rows = []
        for row in service.per_page_table():
            metrics = []
            for k, v in row.items():
                if k in ("page", "file"):
                    continue
                metrics.append(f"{k}={v}")
            detail_rows.append([
                row.get("page"),
                row.get("file"),
                "; ".join(metrics),
            ])

        msg = f"加载 {len(summary)} 个模型，共 {len(detail_rows)} 页数据"
        return summary_rows, detail_rows, msg

    # Gradio 6 下「构建后、启动前」对组件 .value 赋值不会进入客户端配置 ——
    # 首屏数据必须在构造时传入。
    if benchmark_service.exists():
        _rows0, _detail0, _msg0 = reload(str(benchmark_service.json_path))
    else:
        _rows0, _detail0, _msg0 = [], [], f"未找到 benchmark JSON: {benchmark_service.json_path}"

    with gr.Column(elem_classes="mf-page"):
        gr.Markdown("评估面板", elem_classes="mf-page-title")

        with gr.Row():
            path_input = gr.Textbox(
                label="Benchmark JSON 路径",
                value=str(benchmark_service.json_path),
                scale=4,
            )
            reload_btn = gr.Button("重新加载", scale=1, min_width=120)

        status_md = gr.Markdown(_msg0, elem_classes="mf-info-line")

        gr.Markdown("模型汇总", elem_classes="mf-section-title")
        summary_table = gr.Dataframe(
            headers=["模型", "页数", "总区域数", "平均置信度", "平均耗时(s)", "标签分布"],
            value=_rows0,
            interactive=False,
            wrap=False,
            elem_classes="mf-table-fixed",
        )

        gr.Markdown("逐页明细", elem_classes="mf-section-title")
        detail_table = gr.Dataframe(
            headers=["页码", "文件", "指标"],
            value=_detail0,
            interactive=False,
            wrap=False,
            elem_classes="mf-table-fixed",
        )

    reload_btn.click(
        fn=reload,
        inputs=[path_input],
        outputs=[summary_table, detail_table, status_md],
    )
