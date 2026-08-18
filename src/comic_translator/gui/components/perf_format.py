"""性能监控格式化工具 — GPU 状态 + 流水线性能快照 → Markdown

供流水线页（实时面板）和「性能监控」Tab 共用。
"""

from __future__ import annotations


def get_gpu_info() -> str:
    """返回 GPU/设备状态 Markdown"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            props = torch.cuda.get_device_properties(0)
            vram = getattr(props, "total_memory", getattr(props, "total_mem", 0)) / (1024**3)
            try:
                used = torch.cuda.memory_allocated(0) / (1024**3)
                free = vram - used
                usage = f"已用 {used:.1f} GB / 空闲 {free:.1f} GB"
            except Exception:
                usage = f"共 {vram:.1f} GB"
            return f"**GPU**: {name}（{usage}）"
        return "**GPU**: 未检测到 CUDA — 当前使用 CPU 运行"
    except ImportError:
        return "**GPU**: PyTorch 未安装"
    except Exception as e:
        return f"**GPU**: 检测失败 ({e})"


def format_perf_md(snap: dict | None) -> str:
    """把 get_perf_snapshot() 的结果格式化为 Markdown 面板"""
    if not snap:
        return "_尚未运行流水线_"

    total = snap.get("total_pages", 0)
    done = snap.get("pages_done", 0)
    running = total > 0 and done < total

    lines = [
        "### ⚡ 性能面板",
        "",
        f"- **进度**: {done}/{total} 页"
        + ("（运行中...）" if running else ""),
        f"- **平均每页检测**: {_fmt_avg(snap, 'detect_total')}（最近 {snap.get('detect_last', 0):.2f}s）",
        f"- **平均每页 OCR**: {_fmt_avg(snap, 'ocr_total')}（最近 {snap.get('ocr_last', 0):.2f}s）",
        f"- **平均每页翻译**: {_fmt_avg(snap, 'trans_total')}（最近 {snap.get('trans_last', 0):.2f}s）",
        f"- **平均每页渲染**: {_fmt_avg(snap, 'render_total')}（最近 {snap.get('render_last', 0):.2f}s）",
        "",
    ]

    rate = snap.get("rate_limited", 0)
    retries = snap.get("trans_retries", 0)
    if rate:
        lines.append(f"- ⚠️ **限流 (429)**: 已发生 {rate} 次 — 已自动等待重试")
    else:
        lines.append("- **限流 (429)**: 无")
    lines.append(f"- **翻译重试**: {retries} 次")

    lines.append("")
    lines.append(get_gpu_info())
    return "\n".join(lines)


def _fmt_avg(snap: dict, key: str) -> str:
    total = snap.get(key, 0.0)
    done = snap.get("pages_done", 0)
    if done <= 0:
        return "-"
    return f"{total / done:.2f}s"


def format_summary_md(snap: dict | None, elapsed_s: float | None = None) -> str:
    """运行结束后的汇总面板（含总耗时）"""
    if not snap:
        return "_无数据_"
    done = snap.get("pages_done", 0)
    total = snap.get("total_pages", 0)
    lines = [
        "### 📊 本次运行汇总",
        "",
        f"- **完成**: {done}/{total} 页",
        f"- **检测累计**: {snap.get('detect_total', 0):.2f}s",
        f"- **OCR 累计**: {snap.get('ocr_total', 0):.2f}s",
        f"- **翻译累计**: {snap.get('trans_total', 0):.2f}s",
        f"- **渲染累计**: {snap.get('render_total', 0):.2f}s",
    ]
    if elapsed_s:
        lines.append(f"- **总耗时**: {elapsed_s:.2f}s")
    rate = snap.get("rate_limited", 0)
    if rate:
        lines.append(f"- ⚠️ **限流 429**: {rate} 次")
    return "\n".join(lines)
