"""精修编辑服务 —— 加载历史运行的对白、应用译文修改、重渲染单页

交互式编辑闭环：机翻打底（历史 PageSnapshot）→ 人工在表格里改译文 → 重渲染该页。
从快照重建 DialogueBlock 时气泡几何用 bbox 矩形近似（快照未保存 mask/polygon），
文字修改的渲染是准确的；如需像素级精修蒙版，请对当前内存中的运行使用调试页重试。
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
from PIL import Image

from comic_translator.gui.models.gui_models import PageSnapshot
from comic_translator.models.bubble import Bubble, DialogueBlock

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


def find_source_image(runs_root_service, run_id: str, snapshot: PageSnapshot) -> Path | None:
    """按 run 记录的 input_dir → 常见目录 的顺序定位源图"""
    candidates: list[Path] = []
    record = runs_root_service.load_run(run_id)
    if record and record.input_dir:
        candidates.append(Path(record.input_dir))
    candidates += [Path("data/comics"), Path("data/input")]
    for d in candidates:
        if not d.exists():
            continue
        for p in d.iterdir():
            if p.suffix.lower() in IMAGE_EXTS and p.stem == Path(snapshot.file_name).stem:
                return p
        # 完全同名（含扩展名）
        exact = next((d / snapshot.file_name for f in (d / snapshot.file_name,) if f.exists()), None)
        if exact:
            return exact
    return None


def load_typography_params(snapshot: PageSnapshot) -> dict[str, dict]:
    """从 render_debug.json 读取每块的已验证排版参数（字号/精确 bbox/字体）

    位置：_debug/page_NNNN/render_debug.json（RenderQualityService 写盘）。
    用于重建块时恢复原始排版，而不是从头重算。
    """
    params: dict[str, dict] = {}
    if not snapshot.rendered_path:
        return params
    debug_file = Path(snapshot.rendered_path).parent / "_debug" / f"page_{snapshot.page:04d}" / "render_debug.json"
    if not debug_file.exists():
        return params
    try:
        import json
        data = json.loads(debug_file.read_text(encoding="utf-8"))
        for b in data.get("blocks", []):
            bid = b.get("block_id")
            if bid:
                params[bid] = {
                    "font_size": int(b.get("font_size") or 0),
                    "bubble_bbox": b.get("bubble_bbox"),
                }
    except Exception as e:
        logger.debug(f"Failed to read typography params: {e}")
    return params


def snapshot_to_dialogues(
    snapshot: PageSnapshot,
    edits: dict[str, dict] | None = None,
    typo_params: dict[str, dict] | None = None,
) -> list[DialogueBlock]:
    """从 PageSnapshot 重建 DialogueBlock 列表，并应用译文/字号/位置编辑

    edits: {block_id: {"text": 新译文, "font_size": 0=跟随原值, "dx": 0, "dy": 0}}。
    typo_params: render_debug.json 的已验证排版参数（优先于快照 bbox）。
    """
    edits = edits or {}
    typo_params = typo_params or {}
    blocks: list[DialogueBlock] = []
    for d in snapshot.dialogues:
        bid = d.get("id", "")
        param = typo_params.get(bid, {})
        bbox = param.get("bubble_bbox") or d.get("bbox") or (0, 0, 10, 10)
        x1, y1, x2, y2 = [int(v) for v in bbox]
        # 位置微调（dx/dy 平移整个气泡）
        e = edits.get(bid, {})
        try:
            dx, dy = int(e.get("dx") or 0), int(e.get("dy") or 0)
        except (TypeError, ValueError):
            dx, dy = 0, 0
        x1, x2, y1, y2 = x1 + dx, x2 + dx, y1 + dy, y2 + dy
        if x2 <= x1:
            x2 = x1 + 10
        if y2 <= y1:
            y2 = y1 + 10
        if x2 <= x1:
            x2 = x1 + 10
        if y2 <= y1:
            y2 = y1 + 10
        bubble = Bubble(
            id=d.get("id", ""),
            bbox=(x1, y1, x2, y2),
            polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
            mask=[],
            class_name=d.get("class_name", "speech"),
            confidence=d.get("confidence", 0.0),
        )
        translated = d.get("translated_text", "")
        if "text" in e:
            new_text = (e.get("text") or "").strip()
            if new_text and new_text != translated:
                translated = new_text
        try:
            font_size = int(e.get("font_size") or param.get("font_size") or 0)
        except (TypeError, ValueError):
            font_size = int(param.get("font_size") or 0)
        blocks.append(DialogueBlock(
            id=bid,
            bubble=bubble,
            region_type=d.get("region_type", "speech"),
            merged_text=d.get("merged_text", ""),
            translated_text=translated,
            translation_status=d.get("translation_status", "pending"),
            render_text="",
            render_status="pending",
            font_size=font_size,
        ))
    return blocks


def rerender_with_edits(
    run_id: str,
    page: int,
    edits: dict[str, str],
    pipeline_service,
    history_service,
) -> tuple[str, str]:
    """应用编辑并重渲染指定页。

    Returns:
        (message, rendered_path_or_empty)
    """
    snapshot = history_service.load_page(run_id, page)
    if snapshot is None:
        return f"运行 {run_id} 第 {page} 页快照不存在", ""

    img_path = find_source_image(history_service, run_id, snapshot)
    if img_path is None:
        return f"找不到源图: {snapshot.file_name}（请确认输入目录仍在原位置）", ""

    applied = {k: v for k, v in (edits or {}).items() if any((v or {}).get(x) for x in ("text", "font_size", "dx", "dy"))}
    typo_params = load_typography_params(snapshot)
    dialogues = snapshot_to_dialogues(snapshot, applied, typo_params)

    pipeline_service._ensure_initialized()
    # 输出目录沿用原渲染位置（就地覆盖，保证历史页预览一致）
    out_dir = str(Path(snapshot.rendered_path).parent) if snapshot.rendered_path else "data/output/rendered"

    import time
    t0 = time.time()
    render_result = pipeline_service.render_service.render_page(
        Image.open(img_path),
        page,
        dialogues,
        output_dir=out_dir,
        source_path=str(img_path),
        render_options={"fallback_to_source": True},
    )
    logger.info(f"Re-rendered run={run_id} page={page} with {len(applied)} edit(s) "
                f"in {time.time() - t0:.2f}s -> {render_result.output_path}")
    msg = f"已应用 {len(applied)} 处修改并重渲染: {render_result.output_path}"
    if not applied:
        msg = f"未检测到修改，已按原译文重渲染: {render_result.output_path}"
    return msg, str(render_result.output_path)


def find_block_at(
    snapshot: PageSnapshot,
    typo_params: dict[str, dict],
    x: int,
    y: int,
) -> str | None:
    """返回包含点击坐标的最小气泡块 ID（画布点选用）"""
    best: tuple[int, str] | None = None
    for d in snapshot.dialogues:
        bid = d.get("id", "")
        bbox = (typo_params.get(bid, {}).get("bubble_bbox") or d.get("bbox"))
        if not bbox:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        if x1 <= x <= x2 and y1 <= y <= y2:
            area = (x2 - x1) * (y2 - y1)
            if best is None or area < best[0]:
                best = (area, bid)
    return best[1] if best else None


def draw_selection_overlay(
    img_path: str | None,
    snapshot: PageSnapshot,
    typo_params: dict[str, dict],
    selected_id: str = "",
) -> Image.Image | None:
    """在页面图上绘制气泡框：绿色=全部块，橙色=当前选中块"""
    src = img_path or snapshot.rendered_path
    if not src or not Path(src).exists():
        return None
    import cv2

    from comic_translator.gui.components.detection_overlay import bgr_to_pil

    img = cv2.imread(str(src))
    if img is None:
        return None
    for d in snapshot.dialogues:
        bid = d.get("id", "")
        bbox = (typo_params.get(bid, {}).get("bubble_bbox") or d.get("bbox"))
        if not bbox:
            continue
        x1, y1, x2, y2 = [int(v) for v in bbox]
        color = (0, 140, 255) if bid == selected_id else (80, 220, 80)
        thickness = 4 if bid == selected_id else 1
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    return bgr_to_pil(img)
