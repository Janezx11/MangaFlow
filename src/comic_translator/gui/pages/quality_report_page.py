"""质检报告页面 — 翻译后质量评估 + 问题页面高亮"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr


def _load_report(report_path: str = "data/output/quality_report.json") -> dict | None:
    """加载质检报告 JSON"""
    p = Path(report_path)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _build_summary_md(report: dict) -> str:
    """构建报告摘要 Markdown"""
    s = report
    lines = [
        "## 质检报告",
        "",
        f"| 指标 | 数值 |",
        f"|------|------|",
        f"| 总页数 | {s.get('total_pages', 0)} |",
        f"| 通过 | {s.get('passed', 0)} ({s.get('overall_pass_rate', 0)}%) |",
        f"| 重跑后通过 | {s.get('retry_passed', 0)} |",
        f"| 需人工审核 | {s.get('needs_manual_review', 0)} |",
        f"| 平均渲染质量 | {s.get('avg_render_quality', 0)} / 100 |",
        f"| 翻译状态 | {s.get('avg_translation_status', 'N/A')} |",
    ]
    return "\n".join(lines)


def _build_page_table(report: dict) -> tuple[list[list], list[str]]:
    """构建页面表格数据"""
    pages = report.get("pages", [])
    if not pages:
        return [], ["Page", "Quality", "Translation", "Score", "Retry", "Issues"]

    headers = ["Page", "Quality", "Trans Status", "Score", "Retry", "Issues"]
    rows = []
    for p in pages:
        issues = p.get("retry_reason", "")[:30] if p.get("needs_retry") else ""
        rows.append([
            p.get("page", 0),
            f"{p.get('render_quality', 0):.1f}",
            p.get("translation_status", ""),
            f"{p.get('overall_score', 0):.1f}",
            p.get("retry_count", 0),
            issues,
        ])
    return rows, headers


def _build_manual_review_md(report: dict) -> str:
    """构建人工审核列表 Markdown"""
    manual = report.get("manual_review_pages", [])
    if not manual:
        return "无需人工审核的页面"

    lines = ["## 需人工审核", "", "| Page | Quality | Trans | 原因 |", "|------|---------|-------|------|"]
    for p in manual:
        lines.append(f"| #{p['page']} | {p.get('render_quality', 0)} | "
                     f"{p.get('translation_status', '')} | {p.get('retry_reason', '')} |")
    return "\n".join(lines)


def _load_page_detail(report: dict, page_num: int) -> str:
    """加载单页详细质检数据"""
    pages = report.get("pages", [])
    for p in pages:
        if p.get("page") == page_num:
            return (
                f"```json\n{json.dumps(p, ensure_ascii=False, indent=2)}\n```"
            )
    return f"Page {page_num} not found"


def create_quality_report_page():
    """创建质检报告标签页"""

    with gr.Column():
        gr.Markdown("## 质检报告")

        with gr.Row():
            report_path_input = gr.Textbox(
                label="报告路径",
                value="data/output/quality_report.json",
                scale=3,
            )
            refresh_btn = gr.Button("刷新报告", variant="primary", scale=1)
            open_folder_btn = gr.Button("打开输出目录", scale=1)

        summary_md = gr.Markdown("点击「刷新报告」加载质检数据")

        with gr.Row():
            with gr.Column(scale=2):
                page_table = gr.Dataframe(
                    headers=["Page", "Quality", "Trans Status", "Score", "Retry", "Issues"],
                    value=[],
                    interactive=False,
                    wrap=True,
                    label="各页面评分",
                    row_count=20,
                )

            with gr.Column(scale=1):
                manual_md = gr.Markdown("")

        with gr.Row():
            page_detail_input = gr.Number(label="查看页面详情 (页码)", value=1, precision=0)
            load_detail_btn = gr.Button("加载详情")

        page_detail = gr.Markdown("")

        # 状态提示
        status_md = gr.Markdown("就绪")

    # 事件绑定
    def refresh(report_path: str):
        report = _load_report(report_path)
        if report is None:
            return (
                "报告文件不存在，请先运行翻译流水线",
                [],
                "无数据",
                "无数据",
            )
        summary = _build_summary_md(report)
        rows, headers = _build_page_table(report)
        manual = _build_manual_review_md(report)
        return summary, rows, manual, f"已加载: {len(report.get('pages', []))} 页"

    def load_detail(report_path: str, page_num: int):
        report = _load_report(report_path)
        if report is None:
            return "报告文件不存在"
        return _load_page_detail(report, int(page_num))

    def open_folder():
        import subprocess
        import sys
        p = Path("data/output")
        p.mkdir(parents=True, exist_ok=True)
        if sys.platform == "win32":
            subprocess.Popen(["explorer", str(p.resolve())])
        return f"已打开: {p.resolve()}"

    refresh_btn.click(
        fn=refresh,
        inputs=[report_path_input],
        outputs=[summary_md, page_table, manual_md, status_md],
    )

    load_detail_btn.click(
        fn=load_detail,
        inputs=[report_path_input, page_detail_input],
        outputs=[page_detail],
    )

    open_folder_btn.click(
        fn=open_folder,
        inputs=[],
        outputs=[status_md],
    )

    # 页面加载时自动刷新一次
    refresh("data/output/quality_report.json")