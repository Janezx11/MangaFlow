"""GUI Pipeline Service — 包装现有 Pipeline，供 Gradio 调用

只做编排和进度上报，不复制业务逻辑。
所有核心处理委托给现有 services/providers。
"""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Generator

import numpy as np
from loguru import logger
from PIL import Image

from comic_translator.models.bubble import Bubble, DialogueBlock
from comic_translator.models.detection_region import DetectionRegion
from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock
from comic_translator.models.text_region import TextRegion
from comic_translator.gui.models.gui_models import PageSnapshot, RunRecord


def _bbox_area(bbox) -> float:
    """计算 bbox 面积"""
    x1, y1, x2, y2 = bbox
    return max(0.0, (x2 - x1) * (y2 - y1))


def _bbox_iou(a, b) -> float:
    """计算两个 bbox 的 IoU"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = _bbox_area(a) + _bbox_area(b) - inter
    return inter / union if union > 0 else 0.0


def _bbox_overlap_ratio(a, b) -> float:
    """a 与 b 的重叠面积占 a 面积的比例"""
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = ix2 - ix1, iy2 - iy1
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    area = _bbox_area(a)
    return inter / area if area > 0 else 0.0


def _bubble_contain_ratio(bbox1, bbox2) -> float:
    """较小 bbox 被较大 bbox 包含的面积比例（0~1）

    用于识别"同一气泡被检测成松/紧两个框"的重复情况。
    """
    ax1, ay1, ax2, ay2 = map(int, bbox1)
    bx1, by1, bx2, by2 = map(int, bbox2)
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    small = min(a_area, b_area)
    return inter / small if small > 0 else 0.0

# ---- 全局性能计数器（供 staticmethod 翻译路径计数 429/重试） ----
_global_perf_lock = threading.Lock()
_GLOBAL_RATE_LIMITED = 0
_GLOBAL_TRANS_RETRIES = 0


def _bump_global(key: str):
    """递增模块级计数器（线程安全）"""
    global _GLOBAL_RATE_LIMITED, _GLOBAL_TRANS_RETRIES
    with _global_perf_lock:
        if key == "rate_limited":
            _GLOBAL_RATE_LIMITED += 1
        elif key == "trans_retries":
            _GLOBAL_TRANS_RETRIES += 1


def _get_global_counts() -> tuple[int, int]:
    global _GLOBAL_RATE_LIMITED, _GLOBAL_TRANS_RETRIES
    with _global_perf_lock:
        return _GLOBAL_RATE_LIMITED, _GLOBAL_TRANS_RETRIES



@dataclass
class PageResult:
    """单页处理结果，供 Debug Viewer 使用"""
    page: int
    file_name: str
    text_regions: list[TextRegion] = field(default_factory=list)
    bubbles: list[Bubble] = field(default_factory=list)
    ocr_blocks: list[NormalizedOCRBlock] = field(default_factory=list)
    dialogues: list[DialogueBlock] = field(default_factory=list)
    fused_regions: list[DetectionRegion] = field(default_factory=list)
    rendered_path: str = ""
    detection_time: float = 0.0
    ocr_time: float = 0.0
    translation_time: float = 0.0
    render_time: float = 0.0
    error: str = ""


@dataclass
class PipelineProgress:
    """进度报告"""
    current_page: int
    total_pages: int
    stage: str  # "detection" / "ocr" / "translation" / "render" / "done"
    message: str
    page_result: PageResult | None = None


class GuiPipelineService:
    """GUI 调用的 Pipeline 服务

    懒加载所有 Provider，按需初始化。
    运行 Pipeline 时逐页上报进度。
    """

    def __init__(self, history_service=None):
        import threading
        self._init_lock = threading.Lock()
        self._initialized = False
        self._init_progress: list[str] = []  # 模型加载进度消息
        self._results: dict[int, PageResult] = {}
        self._history = history_service  # RunHistoryService, optional
        self.current_run_id: str = ""
        # 性能监控（跨线程读写用锁保护）
        self._perf_lock = threading.Lock()
        self._perf: dict = {
            "run_started": 0.0,
            "pages_done": 0,
            "total_pages": 0,
            "detect_total": 0.0,
            "ocr_total": 0.0,
            "trans_total": 0.0,
            "render_total": 0.0,
            "detect_last": 0.0,
            "ocr_last": 0.0,
            "trans_last": 0.0,
            "render_last": 0.0,
            "rate_limited": 0,   # 429 / 限流次数
            "trans_retries": 0,  # 翻译重试次数
        }
        # 翻译并发 worker 数（从 config.yaml 读取，默认 4）—— 提速①
        self._translate_workers = self._load_translate_workers()
        # 是否启用翻译记忆 / 页面翻译缓存复用（从 config.yaml 读取，默认 true）—— 提速②
        self._use_translation_cache = self._load_use_translation_cache()

    def _perf_update(self, **kw):
        with self._perf_lock:
            self._perf.update(kw)

    def _perf_accumulate(self, key: str, val: float):
        with self._perf_lock:
            self._perf[key] = self._perf.get(key, 0.0) + val

    def _perf_incr(self, key: str):
        with self._perf_lock:
            self._perf[key] = self._perf.get(key, 0) + 1

    def get_perf_snapshot(self) -> dict:
        """返回性能监控快照（线程安全，供 GUI 实时显示）"""
        with self._perf_lock:
            snap = dict(self._perf)
        gl_rate, gl_retries = _get_global_counts()
        snap["rate_limited"] = snap.get("rate_limited", 0) + gl_rate
        snap["trans_retries"] = snap.get("trans_retries", 0) + gl_retries
        return snap

    def reset_perf(self, total_pages: int):
        with self._perf_lock:
            self._perf = {
                "run_started": time.time(),
                "pages_done": 0,
                "total_pages": total_pages,
                "detect_total": 0.0,
                "ocr_total": 0.0,
                "trans_total": 0.0,
                "render_total": 0.0,
                "detect_last": 0.0,
                "ocr_last": 0.0,
                "trans_last": 0.0,
                "render_last": 0.0,
                "rate_limited": 0,
                "trans_retries": 0,
            }

    def _log_init(self, msg: str):
        self._init_progress.append(msg)
        logger.info(f"[init] {msg}")

    def get_init_progress(self) -> list[str]:
        return list(self._init_progress)

    def _load_render_config(self) -> dict:
        """读取 config.yaml 中 render 段，未配置时返回默认"""
        try:
            import yaml
            config_path = Path("config.yaml")
            if config_path.exists():
                cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                return cfg.get("render", {})
        except Exception as e:
            logger.warning(f"Failed to load render config: {e}")
        return {}

    @staticmethod
    def _load_translate_workers() -> int:
        """读取 config.yaml 中 translation.workers，未配置时返回默认 4"""
        try:
            import yaml
            config_path = Path("config.yaml")
            if config_path.exists():
                cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                workers = cfg.get("translation", {}).get("workers", 4)
                return max(1, int(workers))
        except Exception as e:
            logger.warning(f"Failed to load translation.workers: {e}")
        return 4

    @staticmethod
    def _load_use_translation_cache() -> bool:
        """读取 config.yaml 中 translation.use_cache，未配置时返回 True"""
        try:
            import yaml
            config_path = Path("config.yaml")
            if config_path.exists():
                cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
                return bool(cfg.get("translation", {}).get("use_cache", True))
        except Exception as e:
            logger.warning(f"Failed to load translation.use_cache: {e}")
        return True

    def _ensure_initialized(self, conf: float = 0.3):
        if self._initialized:
            return
        # 加锁防止 warmup 线程和用户点击并发触发重复加载
        with self._init_lock:
            if self._initialized:
                return
            self._do_initialize(conf)

    def _do_initialize(self, conf: float = 0.3):
        self._log_init("加载检测模型 CTBD...")
        from comic_translator.providers.text_detection.ctbd_text_detector import CTBDTextDetectorProvider
        from comic_translator.providers.bubble.yolo_bubble_provider import YOLO11BubbleProvider
        self.text_detector = CTBDTextDetectorProvider(confidence=conf)

        self._log_init("加载气泡检测模型 YOLO11...")
        self.bubble_provider = YOLO11BubbleProvider(
            model_path="models/bubble/yolo11n-seg-mixed-bubble.pt",
        )

        self._log_init("加载 OCR 引擎 EasyOCR...")
        from comic_translator.providers.ocr.easyocr_provider import EasyOCRProvider
        import torch
        use_gpu = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        self.ocr_provider = EasyOCRProvider(lang="en", use_gpu=use_gpu)
        self._log_init(
            f"OCR 引擎 EasyOCR (GPU={'开' if use_gpu else '关，使用 CPU'})"
        )

        self._log_init("加载 OCR 后处理服务...")
        from comic_translator.services.ocr_postprocess_service import OCRPostprocessService
        self.ocr_postprocess = OCRPostprocessService()

        self._log_init("加载检测融合服务...")
        from comic_translator.services.detection_fusion_service import DetectionFusionService
        from comic_translator.services.text_region_service import TextRegionService
        from comic_translator.services.text_region_adapter import TextRegionAdapter
        self.fusion = DetectionFusionService()
        self.text_region_service = TextRegionService()
        self.text_region_adapter = TextRegionAdapter()

        self._log_init("加载排版渲染服务...")
        from comic_translator.services.text_normalizer import TextNormalizer
        from comic_translator.services.font_manager import FontManager
        from comic_translator.services.bubble_layout_service import BubbleLayoutService
        from comic_translator.services.shape_layout_solver import ShapeLayoutSolver
        from comic_translator.services.balanced_line_breaker import BalancedLineBreaker
        from comic_translator.services.padding_solver import PaddingSolver
        from comic_translator.services.font_size_solver import FontSizeSolver
        from comic_translator.services.optical_center_solver import OpticalCenterSolver
        from comic_translator.services.typography_service_v2 import TypographyServiceV2
        from comic_translator.services.typography_qa import TypographyQA
        from comic_translator.services.render_qa_service import RenderQAService
        from comic_translator.services.inpaint_service import InpaintService
        from comic_translator.services.background_cleaner import BackgroundCleaner
        from comic_translator.services.render_service import RenderService
        from comic_translator.services.bubble_interior_service import BubbleInteriorService
        from comic_translator.services.region_classifier import RegionClassifier

        normalizer = TextNormalizer()
        fm = FontManager()
        bl = BubbleLayoutService()
        sl = ShapeLayoutSolver(bl)
        brk = BalancedLineBreaker(fm)
        ps = PaddingSolver()
        fss = FontSizeSolver(fm, brk, sl, ps)
        oc = OpticalCenterSolver()
        typo_v2 = TypographyServiceV2(fm, bl, sl, brk, ps, fss, oc)
        typo_qa = TypographyQA()
        qa = RenderQAService()
        inpaint = InpaintService()
        cleaner = BackgroundCleaner(inpaint)

        # 根据 config.yaml 选择 RenderService 还是 RenderQualityService
        render_cfg = self._load_render_config()
        use_quality = render_cfg.get("use_quality_service", True)
        if use_quality:
            from comic_translator.services.render_quality_service import RenderQualityService
            self.render_service = RenderQualityService(
                normalizer, fm, typo_v2, typo_qa, qa, render_cfg,
            )
            self._log_init("使用 RenderQualityService (mask + refinement + inpaint providers)")
        else:
            from comic_translator.services.render_service import RenderService
            self.render_service = RenderService(normalizer, fm, typo_v2, typo_qa, qa, inpaint, cleaner)
            self._log_init("使用 RenderService 标准版")

        self.interior = BubbleInteriorService()
        self.classifier = RegionClassifier()

        self._initialized = True
        self._log_init("所有模型加载完成")

    def warmup(self, conf: float = 0.3):
        """在后台线程中预加载模型，避免首次运行时长时间无反馈"""
        import threading
        if self._initialized:
            return
        t = threading.Thread(target=self._ensure_initialized, args=(conf,), daemon=True)
        t.start()

    @staticmethod
    def collect_images(input_dir: str) -> list[Path]:
        d = Path(input_dir)
        if not d.exists():
            return []
        from comic_translator.utils.natural_sort import natural_key

        # 自然排序：按文件名中的数字排序（"2.jpg" < "10.jpg"），
        # 保证渲染页码 = 实际阅读顺序，且与翻译/重译顺序无关（稳定）。
        return sorted([
            f for f in d.iterdir()
            if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp", ".bmp")
        ], key=natural_key)

    @staticmethod
    def _merge_ocr_results(
        bubble_ocr: list[NormalizedOCRBlock],
        region_ocr: list[NormalizedOCRBlock],
        bubbles: list[Bubble] | None = None,
        iou_threshold: float = 0.5,
        cross_overlap_ratio: float = 0.05,
    ) -> list[NormalizedOCRBlock]:
        """合并逐气泡 OCR 与区域 OCR 结果

        策略：气泡 OCR 为主（优先保留，用于气泡内文字），
        区域 OCR 只补充非气泡文字（旁白/SFX/标题等）。
        过滤规则：
          1. 与气泡 OCR 重叠过高（IoU >= iou_threshold）→ 重复，丢弃
          2. 横跨 2+ 个气泡的合并块（重叠面积 > cross_overlap_ratio）→ 丢弃，
             它的内容已被逐气泡 OCR 完整覆盖，是"区域 OCR 把两个气泡文字
             合并成一个 block"的产物

        Args:
            bubble_ocr: 逐气泡 OCR 结果（bubble_id 关联到气泡）
            region_ocr: CTBD 区域 OCR 结果（覆盖所有文字）
            bubbles: YOLO 气泡列表（用于检测横跨多个气泡的合并块）
            iou_threshold: 与气泡 OCR 重叠超过此值则视为重复
            cross_overlap_ratio: 与单个气泡重叠面积占比超过此值计为"有交集"

        Returns:
            合并后的 OCR block 列表
        """
        if not bubble_ocr:
            return region_ocr
        if not region_ocr:
            return bubble_ocr

        # 保留气泡 OCR 全部
        merged = list(bubble_ocr)
        dropped_cross = 0
        dropped_dup = 0

        for rb in region_ocr:
            # 规则 1：与气泡 OCR 重叠过高 → 重复
            duplicate = any(
                _bbox_iou(rb.bbox, bb.bbox) >= iou_threshold
                for bb in bubble_ocr
            )
            if duplicate:
                dropped_dup += 1
                continue

            # 规则 2：横跨 2+ 个气泡的合并块 → 丢弃
            if bubbles:
                overlap_count = sum(
                    1 for b in bubbles
                    if _bbox_overlap_ratio(rb.bbox, b.bbox) > cross_overlap_ratio
                )
                if overlap_count >= 2:
                    dropped_cross += 1
                    continue

            merged.append(rb)

        logger.debug(
            f"_merge_ocr_results: kept {len(merged)} blocks "
            f"(dropped {dropped_dup} dup, {dropped_cross} cross-bubble)"
        )

        # 按 Y→X 排序
        merged.sort(key=lambda b: (b.bbox[1], b.bbox[0]))
        return merged

    @staticmethod
    def _polygon_overlap_ratio(bbox, polygon) -> float:
        """OCR block bbox 与气泡 polygon 的重叠面积占 block 面积的比例"""
        import cv2

        x1, y1, x2, y2 = map(int, bbox)
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0 or not polygon or len(polygon) < 3:
            return 0.0
        block_mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.rectangle(block_mask, (0, 0), (bw, bh), 1, -1)
        poly = np.array(
            [(px - x1, py - y1) for px, py in polygon], dtype=np.int32
        )
        bubble_mask = np.zeros((bh, bw), dtype=np.uint8)
        cv2.fillPoly(bubble_mask, [poly], 1)
        overlap = int((block_mask & bubble_mask).sum())
        return overlap / (bw * bh)

    @staticmethod
    def _merge_overlapping_bubbles(
        bubbles: list[Bubble],
        iou_threshold: float = 0.5,
        contain_ratio: float = 0.7,
    ) -> list[Bubble]:
        """合并重复气泡

        YOLO 可能把同一个气泡重复检测成多个：
        - bbox 高度重叠（IoU > 阈值），或
        - 小框大部分被大框包含（包含率 > 阈值，松/紧两个框的情况）
        只保留面积最大的一个，避免跨气泡手术被重复气泡误触发。
        """
        bubbles = list(bubbles)
        merged: list[Bubble] = []
        used: set[int] = set()

        for i, b in enumerate(bubbles):
            if i in used:
                continue
            group = [b]
            used.add(i)
            for j in range(i + 1, len(bubbles)):
                if j in used:
                    continue
                if (
                    _bbox_iou(b.bbox, bubbles[j].bbox) > iou_threshold
                    or _bubble_contain_ratio(b.bbox, bubbles[j].bbox) > contain_ratio
                ):
                    group.append(bubbles[j])
                    used.add(j)
            # 取面积最大的代表气泡
            best = max(group, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
            merged.append(best)

        if len(merged) != len(bubbles):
            logger.info(
                f"merged overlapping bubbles: {len(bubbles)} -> {len(merged)}"
            )
        return merged

    async def _resolve_cross_bubble_blocks(
        self,
        img: Image.Image,
        ocr_blocks: list[NormalizedOCRBlock],
        bubbles: list[Bubble],
        overlap_threshold: float = 0.15,
    ) -> list[NormalizedOCRBlock]:
        """跨气泡精准手术

        先合并 YOLO 重复检测的气泡，再检测 OCR block 是否横跨 2+ 个
        分离气泡（用 polygon 精确重叠判断）。只对真正合并的 block 做
        "抹除干扰文字"二次识别，其余 block 原样保留。
        """
        # 合并重复气泡，避免误触发
        bubbles = self._merge_overlapping_bubbles(bubbles)

        result: list[NormalizedOCRBlock] = []
        resolved = 0

        for block in ocr_blocks:
            # 找与此 block 有 polygon 重叠的气泡
            overlaps: list[tuple[Bubble, float]] = []
            for b in bubbles:
                if not b.polygon or len(b.polygon) < 3:
                    continue
                ratio = self._polygon_overlap_ratio(block.bbox, b.polygon)
                if ratio > overlap_threshold:
                    overlaps.append((b, ratio))

            if len(overlaps) < 2:
                result.append(block)  # 正常块，保留
                continue

            # 合并块：对重叠最大的两个气泡做精准手术
            overlaps.sort(key=lambda x: -x[1])
            resolved += 1
            logger.info(
                f"cross-bubble block {block.id}: '{block.text[:40]}...' "
                f"overlaps {overlaps[0][0].id}+{overlaps[1][0].id} -> re-OCR per bubble"
            )
            for b, _ in overlaps[:2]:
                sub_blocks = await self._reocr_in_bubble(
                    img, block.bbox, b, bubbles
                )
                result.extend(sub_blocks)

        if resolved:
            logger.info(
                f"cross-bubble surgery: {resolved} blocks resolved "
                f"({len(ocr_blocks)} -> {len(result)} blocks)"
            )
        return result

    async def _reocr_in_bubble(
        self,
        img: Image.Image,
        block_bbox,
        target_bubble: Bubble,
        bubbles: list[Bubble],
        dilate: int = 8,
    ) -> list[NormalizedOCRBlock]:
        """抹除干扰文字后重新 OCR

        在 OCR block 的 bbox 范围内，只保留 target_bubble 内的文字
        （polygon 膨胀保护防切断），把其余像素涂白（含其他气泡文字），
        再交给 OCR。不裁剪整个气泡、不破坏目标气泡文字完整性。
        """
        import cv2
        from PIL import ImageDraw

        x1, y1, x2, y2 = map(int, block_bbox)
        w, h = x2 - x1, y2 - y1
        if w <= 0 or h <= 0:
            return []

        crop = img.crop((x1, y1, x2, y2))

        # 保留区域 = 目标气泡 polygon（膨胀 dilate 像素防切断文字）
        keep = Image.new("L", (w, h), 0)
        if target_bubble.polygon and len(target_bubble.polygon) >= 3:
            pts = [(px - x1, py - y1) for px, py in target_bubble.polygon]
            ImageDraw.Draw(keep).polygon(pts, fill=255)
        else:
            bx1, by1, bx2, by2 = map(int, target_bubble.bbox)
            ImageDraw.Draw(keep).rectangle(
                [(bx1 - x1, by1 - y1), (bx2 - x1, by2 - y1)], fill=255
            )
        if dilate > 0:
            keep_np = np.array(keep)
            kernel = np.ones((dilate, dilate), np.uint8)
            keep_np = cv2.dilate(keep_np, kernel)
            keep = Image.fromarray(keep_np)

        # 涂白 keep 之外（保留目标气泡文字，抹掉其他气泡/背景文字）
        white = Image.new("RGB", (w, h), (255, 255, 255))
        masked = Image.composite(crop, white, keep)

        raw_blocks = await self.ocr_provider.detect_text(masked)
        out: list[NormalizedOCRBlock] = []
        for rb in raw_blocks:
            rx1, ry1, rx2, ry2 = map(int, rb.bbox)
            out.append(NormalizedOCRBlock(
                id=f"{target_bubble.id}_reocr_{rb.id}",
                page=rb.page,
                bbox=(rx1 + x1, ry1 + y1, rx2 + x1, ry2 + y1),
                text=rb.text,
                confidence=rb.confidence,
                language=rb.language,
                metadata={**rb.metadata, "bubble_id": target_bubble.id},
            ))
        return out

    def run_pipeline(
        self,
        input_dir: str,
        use_ctbd: bool = True,
        use_yolo: bool = True,
        use_fusion: bool = True,
        translate: bool = True,
        conf: float = 0.3,
        target_lang: str = "zh",
        render_options: dict | None = None,
        output_dir: str = "data/output/rendered",
        bubble_aware: bool = True,
        split_strategy: str = "auto",
        cross_threshold: float = 0.15,
    ) -> Generator[PipelineProgress, None, None]:
        """同步生成器，逐页处理并上报进度"""
        return self._run_sync(input_dir, use_ctbd, use_yolo, use_fusion, translate, conf,
                              target_lang, render_options, output_dir,
                              bubble_aware, split_strategy, cross_threshold)

    def _run_sync(self, input_dir, use_ctbd, use_yolo, use_fusion, translate, conf,
                  target_lang, render_options, output_dir="data/output/rendered",
                  bubble_aware=True, split_strategy="auto", cross_threshold=0.15):
        loop = asyncio.new_event_loop()
        try:
            gen = self._run_async(input_dir, use_ctbd, use_yolo, use_fusion, translate, conf,
                                  target_lang, render_options, output_dir,
                                  bubble_aware, split_strategy, cross_threshold)
            while True:
                try:
                    progress = loop.run_until_complete(gen.__anext__())
                    yield progress
                except StopAsyncIteration:
                    break
        finally:
            loop.close()

    async def _run_async(
        self, input_dir, use_ctbd, use_yolo, use_fusion, translate, conf,
        target_lang, render_options, output_dir="data/output/rendered",
        bubble_aware=True, split_strategy="auto", cross_threshold=0.15,
    ) -> Generator[PipelineProgress, None, None]:
        # 用输入目录名作为项目 ID（隔离不同漫画的术语表/角色数据）
        project_id = Path(input_dir).name
        # 如果模型还没初始化，先在线程中加载并流式回报进度
        if not self._initialized:
            yield PipelineProgress(0, 0, "init", "首次运行需加载模型，请稍候...")
            init_task = asyncio.create_task(asyncio.to_thread(self._ensure_initialized, conf))
            last_count = 0
            while not init_task.done():
                await asyncio.sleep(0.5)
                msgs = self.get_init_progress()
                for m in msgs[last_count:]:
                    yield PipelineProgress(0, 0, "init", m)
                last_count = len(msgs)
            await init_task  # 抛出异常（如果有）
            # 把剩余的初始化消息发完
            msgs = self.get_init_progress()
            for m in msgs[last_count:]:
                yield PipelineProgress(0, 0, "init", m)

        self._results.clear()

        # 生成 run_id 并记录开始时间
        run_started = time.time()
        if self._history is not None:
            self.current_run_id = self._history.generate_run_id()
        else:
            self.current_run_id = ""

        images = self.collect_images(input_dir)
        if not images:
            yield PipelineProgress(0, 0, "done", f"No images in {input_dir}")
            return

        total = len(images)
        self.reset_perf(total_pages=total)
        yield PipelineProgress(0, total, "detection", f"Found {total} pages")

        # ============================================================
        # 并行流水线：detect+OCR → translate（多 worker 并发）→ render
        # ============================================================
        #
        # 队列：
        #   q_ready       : detect+OCR 完成，待翻译
        #   q_translated  : 翻译完成，待渲染
        #   q_events      : 阶段进度事件（供主循环上报进度）
        #
        # 原理：
        #   detect+OCR 和 translate 都是 async（内部用 asyncio.to_thread / httpx），
        #   会释放 event loop，因此 3 个阶段可以真正并行。
        #   translate 是网络 IO（最慢），多个 worker 并发翻译不同页面。
        #   render 是同步 CPU 操作，放在主循环串行执行（避免多线程访问 PIL 竞态）。

        q_ready: asyncio.Queue = asyncio.Queue(maxsize=2)
        q_translated: asyncio.Queue = asyncio.Queue(maxsize=2)
        q_events: asyncio.Queue = asyncio.Queue()

        stop_event = asyncio.Event()

        # BubbleAwareOCRService —— 决定"哪些文字属于哪个气泡"
        from comic_translator.services.bubble_aware_ocr_service import BubbleAwareOCRService

        bubble_aware_service = BubbleAwareOCRService(
            ocr_provider=self.ocr_provider,
            threshold=cross_threshold,
            strategy=split_strategy,
            enabled=bubble_aware,
        )

        def _emit(stage: str, message: str, page: int = 0):
            """向事件队列发送进度事件（非阻塞，队列无限大）"""
            q_events.put_nowait((page, stage, message))

        async def producer():
            """Stage 1: detect + OCR → q_ready"""
            try:
                for i, img_path in enumerate(images):
                    page = i + 1
                    try:
                        _emit("detection", f"Page {page}/{total}: detecting...", page)
                        t0 = time.time()

                        img = Image.open(img_path)

                        text_regions: list[TextRegion] = []
                        bubbles: list[Bubble] = []

                        if use_ctbd:
                            text_regions = await self.text_detector.detect(img)
                        if use_yolo:
                            bubbles = await self.bubble_provider.detect(img)

                        detect_elapsed = time.time() - t0
                        self._perf_accumulate("detect_total", detect_elapsed)
                        self._perf_update(detect_last=detect_elapsed)

                        # Fusion
                        fused_regions: list[DetectionRegion] = []
                        if use_fusion and text_regions and bubbles:
                            det_lists = [
                                [DetectionRegion.from_text_region(tr) for tr in text_regions],
                                [DetectionRegion.from_bubble(b) for b in bubbles],
                            ]
                            fused_regions = self.fusion.fuse(det_lists)
                            text_regions = [r.to_text_region() for r in fused_regions]
                        elif text_regions:
                            fused_regions = [DetectionRegion.from_text_region(tr) for tr in text_regions]
                        elif bubbles:
                            fused_regions = [DetectionRegion.from_bubble(b) for b in bubbles]
                            text_regions = [r.to_text_region() for r in fused_regions]

                        # OCR —— 主路径：CTBD 区域 OCR（原架构，上下文完整、不碎片化）
                        _emit("ocr", f"Page {page}/{total}: OCR...", page)
                        t0 = time.time()
                        ocr_blocks = []
                        if text_regions:
                            ocr_blocks = await self.text_region_service.extract_text(
                                img, text_regions, self.ocr_provider
                            )

                        # BubbleAwareOCRService：决定"哪些文字属于哪个气泡"
                        # 只对跨 2+ 气泡的 OCR 块做拆分（细粒度优先 / 气泡裁剪 OCR），
                        # 普通块原样保留。不修改 YOLO / Translation 逻辑。
                        if ocr_blocks and use_yolo and bubbles and bubble_aware:
                            try:
                                ba_result = await bubble_aware_service.process(
                                    img, ocr_blocks, bubbles,
                                )
                                ocr_blocks = ba_result.blocks
                                if ba_result.cross_bubble_count:
                                    logger.info(
                                        f"Page {page}: BubbleAwareOCR split "
                                        f"{ba_result.cross_bubble_count} cross-bubble blocks"
                                    )
                            except Exception as e:
                                logger.warning(f"BubbleAwareOCR failed for page {page}: {e}")

                        ocr_elapsed = time.time() - t0
                        self._perf_accumulate("ocr_total", ocr_elapsed)
                        self._perf_update(ocr_last=ocr_elapsed)

                        # OCR 后处理
                        if ocr_blocks and hasattr(self, "ocr_postprocess"):
                            try:
                                post_result = self.ocr_postprocess.process_page(ocr_blocks, page=page)
                                ocr_blocks = post_result["blocks"]
                                if post_result.get("entities"):
                                    logger.debug(f"Page {page}: extracted {len(post_result['entities'])} entities")
                            except Exception as e:
                                logger.warning(f"OCR postprocess failed for page {page}: {e}")

                        # Adapter → dialogues（恢复原匹配架构：区域归属 + 气泡增强）
                        dialogues = self.text_region_adapter.to_dialogue_blocks(
                            text_regions, ocr_blocks, bubbles=bubbles if bubbles else None,
                        )

                        # Interior + Classify
                        if bubbles:
                            safe_areas = self.interior.compute_batch(bubbles, (img.height, img.width))
                            for b in bubbles:
                                b.safe_area = safe_areas.get(b.id)
                        for d in dialogues:
                            new_type = self.classifier.classify(d.bubble, img.height, img.width)
                            if new_type != d.region_type:
                                d.region_type = new_type

                        result = PageResult(page=page, file_name=img_path.name)
                        result.text_regions = text_regions
                        result.bubbles = bubbles
                        result.fused_regions = fused_regions
                        result.ocr_blocks = ocr_blocks
                        result.dialogues = dialogues
                        result.detection_time = detect_elapsed
                        result.ocr_time = ocr_elapsed

                        # 进队列（阻塞直到队列有空位，限制内存）
                        await q_ready.put((page, img, result))
                    except Exception as e:
                        logger.error(f"Page {page} detect/OCR failed: {e}")
                        result = PageResult(page=page, file_name=img_path.name, error=str(e))
                        await q_ready.put((page, None, result))
            finally:
                # 通知 worker 生产者结束
                for _ in range(translate_workers):
                    await q_ready.put(None)
                stop_event.set()

        async def translate_worker(worker_id: int):
            """Stage 2: translate → q_translated（并发 worker）"""
            try:
                while True:
                    item = await q_ready.get()
                    if item is None:
                        await q_translated.put(None)
                        break
                    page, img, result = item
                    try:
                        _emit("translation", f"Page {page}/{total}: translating...", page)
                        t0 = time.time()
                        if translate:
                            await self._translate_dialogues(
                                result.dialogues,
                                target_lang=target_lang,
                                project_id=project_id,
                                page=result.page,
                            )
                            # 并发节奏由 LLM 客户端令牌桶（rps）控制，无需固定 sleep
                        else:
                            for d in result.dialogues:
                                d.translation_status = "skipped"
                        trans_elapsed = time.time() - t0
                        result.translation_time = trans_elapsed
                        self._perf_accumulate("trans_total", trans_elapsed)
                        self._perf_update(trans_last=trans_elapsed)
                    except Exception as e:
                        logger.error(f"Page {page} translate failed: {e}")
                        result.error = f"translate: {e}"
                    await q_translated.put((page, img, result))
            except asyncio.CancelledError:
                raise

        # 启动 Stage 1 + Stage 2（多个 translate worker 并发，数量从 config 读取）
        # 并发请求的节奏由 LLM 客户端的令牌桶限速（rps）控制，无需固定 sleep。
        translate_workers = self._translate_workers if translate else 1
        producer_task = asyncio.create_task(producer())
        worker_tasks = [
            asyncio.create_task(translate_worker(i)) for i in range(translate_workers)
        ]

        # 进度事件循环：一边处理 render，一边上报 detect/ocr/translation 进度
        # 用一个独立 task 转发事件，主循环专注渲染
        async def event_forwarder():
            while True:
                if stop_event.is_set() and q_events.empty():
                    break
                try:
                    page, stage, message = await asyncio.wait_for(q_events.get(), timeout=0.5)
                    q_events.task_done()
                    yield (page, stage, message)
                except asyncio.TimeoutError:
                    continue

        # ---- Stage 3 (main loop): render + 上报进度 ----
        # 主循环串行渲染（保证输出顺序），同时非阻塞消费进度事件上报。
        # 翻译结果可能乱序到达（多 worker 并发），用 pending dict 按页序整理。
        _events_iter = event_forwarder().__aiter__()

        async def _drain_events():
            """非阻塞消费进度事件，返回 PipelineProgress 列表"""
            events: list[PipelineProgress] = []
            while True:
                try:
                    ev_page, ev_stage, ev_message = await asyncio.wait_for(
                        _events_iter.__anext__(), timeout=0.0
                    )
                    events.append(PipelineProgress(ev_page, total, ev_stage, ev_message))
                except asyncio.TimeoutError:
                    break
                except StopAsyncIteration:
                    break
            return events

        # ---- 统一按序收集翻译结果 ----
        pending: dict[int, tuple] = {}
        next_page = 1

        # 持续从 q_translated 取，直到所有页完成
        remaining = total
        finished_workers = 0
        while remaining > 0:
            # 先消费进度事件
            for ev in await _drain_events():
                yield ev

            # 取一个翻译结果（若还有 worker 未结束）
            if finished_workers < translate_workers:
                item = await q_translated.get()
                q_translated.task_done()
                if item is None:
                    finished_workers += 1
                    # 所有 worker 都结束但 pending 还有页时，继续渲染它们
                    if finished_workers >= translate_workers and not pending:
                        break
                    continue
                page, img, result = item
                pending[page] = (img, result)
            else:
                # 所有 worker 已结束：只渲染 pending 里剩余页
                if not pending:
                    break

            # 按序渲染
            while next_page in pending:
                cur_img, cur_result = pending.pop(next_page)
                cur_page = next_page
                next_page += 1
                remaining -= 1
                try:
                    yield PipelineProgress(cur_page, total, "render", f"Page {cur_page}/{total}: rendering...")
                    t0 = time.time()
                    page_render_options = dict(render_options or {})
                    if not translate:
                        page_render_options.setdefault("fallback_to_source", True)
                    if cur_img is not None:
                        render_result = self.render_service.render_page(
                            cur_img, cur_page, cur_result.dialogues,
                            output_dir=output_dir,
                            render_options=page_render_options,
                        )
                        cur_result.rendered_path = render_result.output_path
                        cur_result.render_time = time.time() - t0
                        self._perf_accumulate("render_total", cur_result.render_time)
                        self._perf_update(render_last=cur_result.render_time)
                    self._results[cur_page] = cur_result
                    self._perf_incr("pages_done")
                    yield PipelineProgress(
                        cur_page, total, "done",
                        f"Page {cur_page}/{total}: {len(cur_result.dialogues)} dialogues"
                        + (f" [ERROR: {cur_result.error}]" if cur_result.error else ""),
                        page_result=cur_result,
                    )
                except Exception as e:
                    cur_result.error = str(e)
                    logger.error(f"Page {cur_page} render failed: {e}")
                    self._results[cur_page] = cur_result
                    yield PipelineProgress(
                        cur_page, total, "done",
                        f"Page {cur_page}/{total}: [ERROR: {cur_result.error}]",
                        page_result=cur_result,
                    )

        # 等待后台 task 收尾
        for t in worker_tasks:
            t.cancel()
        producer_task.cancel()
        for t in ([producer_task] + worker_tasks):
            try:
                await t
            except (asyncio.CancelledError, Exception):
                pass

        # 保存运行记录
        if self._history is not None and self.current_run_id:
            try:
                self._save_run_record(
                    input_dir=input_dir,
                    total=total,
                    elapsed=time.time() - run_started,
                    config={
                        "use_ctbd": use_ctbd,
                        "use_yolo": use_yolo,
                        "use_fusion": use_fusion,
                        "translate": translate,
                        "conf": conf,
                        "target_lang": target_lang,
                        "render_options": render_options or {},
                    },
                )
            except Exception as e:
                logger.error(f"Failed to save run record: {e}")

        # 自动生成质检报告
        try:
            from comic_translator.services.quality_orchestrator import QualityOrchestrator
            orch = QualityOrchestrator(output_dir="data/output")
            summary = orch.aggregate_report(project_name=Path(input_dir).name)
            orch.save_report(summary, "data/output/quality_report.json")
            if summary.needs_manual_review > 0 or summary.overall_pass_rate < 90:
                yield PipelineProgress(
                    total, total, "done",
                    f"质检: {summary.overall_pass_rate}% 合格率, "
                    f"{summary.needs_manual_review} 页需人工审核"
                )
            else:
                yield PipelineProgress(
                    total, total, "done",
                    f"质检: {summary.overall_pass_rate}% 合格率"
                )
        except Exception as e:
            logger.debug(f"Quality report generation skipped: {e}")

        yield PipelineProgress(total, total, "done", f"All {total} pages completed")

    def get_result(self, page: int) -> PageResult | None:
        return self._results.get(page)

    def get_all_results(self) -> dict[int, PageResult]:
        return dict(self._results)

    # ---------- 运行历史持久化 ----------

    def _save_run_record(self, input_dir: str, total: int, elapsed: float, config: dict):
        """把当前 _results 转存为 RunRecord + PageSnapshot"""
        success = sum(1 for r in self._results.values() if not r.error)
        failed = total - success
        total_dialogues = sum(len(r.dialogues) for r in self._results.values())

        if failed == 0:
            status = "success"
        elif success == 0:
            status = "failed"
        else:
            status = "partial"

        record = RunRecord(
            id=self.current_run_id,
            project=Path(input_dir).name,
            input_dir=input_dir,
            created_time=datetime.now().isoformat(timespec="seconds"),
            total_pages=total,
            success_pages=success,
            failed_pages=failed,
            total_dialogues=total_dialogues,
            status=status,
            config=config,
            elapsed_s=round(elapsed, 2),
        )

        snapshots = [self._to_snapshot(r) for r in self._results.values()]
        self._history.save_run(record, snapshots)

    @staticmethod
    def _to_snapshot(result: PageResult) -> PageSnapshot:
        """PageResult → PageSnapshot（把 Pydantic/dataclass 模型转成 dict）"""
        detection_dicts = [
            r.model_dump() if hasattr(r, "model_dump") else asdict(r)
            for r in result.fused_regions
        ]
        ocr_dicts = [
            b.model_dump() if hasattr(b, "model_dump") else asdict(b)
            for b in result.ocr_blocks
        ]
        dialogue_dicts = []
        for d in result.dialogues:
            bubble = d.bubble
            dialogue_dicts.append({
                "id": d.id,
                "region_type": d.region_type,
                "merged_text": d.merged_text,
                "translated_text": d.translated_text,
                "render_text": d.render_text,
                "translation_status": d.translation_status,
                "render_status": d.render_status,
                "bbox": list(bubble.bbox) if bubble else None,
                "class_name": bubble.class_name if bubble else "",
                "confidence": bubble.confidence if bubble else 0.0,
            })

        return PageSnapshot(
            page=result.page,
            file_name=result.file_name,
            detection=detection_dicts,
            ocr_blocks=ocr_dicts,
            dialogues=dialogue_dicts,
            rendered_path=result.rendered_path,
            detection_time=round(result.detection_time, 3),
            ocr_time=round(result.ocr_time, 3),
            translation_time=round(result.translation_time, 3),
            render_time=round(result.render_time, 3),
            error=result.error,
        )

    @staticmethod
    def _build_translation_context(project_id: str = "") -> str:
        """从 DI 容器获取术语表 + 角色身份，构建上下文字符串

        如果传了 project_id，只加载该项目的术语表和角色身份。
        返回格式:
          Glossary:
            Avengers → 复仇者联盟
            Stark Industries → 斯塔克工业

          Characters:
            Tony Stark (speaking style: sarcastic)
            Steve Rogers (speaking style: formal)
        """
        parts = []
        try:
            from comic_translator.utils.di import get_container
            container = get_container()

            # 术语表（按项目隔离）
            try:
                glossary_service = container.get("glossary_service")
                project_arg = project_id if project_id else None
                entries = glossary_service.get_all_entries(project_id=project_arg)
                if entries:
                    glossary_lines = []
                    for e in entries:
                        glossary_lines.append(f"  {e.source} → {e.target}")
                    if glossary_lines:
                        parts.append("Glossary (use these translations consistently):")
                        parts.extend(glossary_lines)
                        parts.append("")
            except Exception as e:
                logger.debug(f"Failed to load glossary: {e}")

            # 角色身份（按项目隔离）
            try:
                char_service = container.get("character_identity_service")
                project_arg = project_id if project_id else None
                identities = char_service.get_all_identities(project_id=project_arg)
                if identities:
                    char_lines = []
                    for c in identities:
                        names = [c.canonical_name]
                        if c.aliases:
                            names.extend(c.aliases[:3])
                        name_str = " / ".join(names)
                        if c.speaking_style:
                            char_lines.append(f"  {name_str} (speaking style: {c.speaking_style})")
                        else:
                            char_lines.append(f"  {name_str}")
                    if char_lines:
                        parts.append("Characters (keep these names consistent):")
                        parts.extend(char_lines)
                        parts.append("")
            except Exception as e:
                logger.debug(f"Failed to load character identities: {e}")

        except Exception as e:
            logger.debug(f"Failed to load translation context from DI container: {e}")

        return "\n".join(parts)

    @staticmethod
    def _save_auto_glossary(
        texts: list[str],
        translations: list[str],
        project_id: str = "",
    ) -> None:
        """从源文本和译文中启发式提取专有名词并保存到术语表

        规则：
        - 提取源文本中以大写开头的英文词（可能是人名/地名/专有名词）
        - 跳过常用词（I, You, This, The 等）
        - 在译文中查找该词对应的翻译
        - 已存在的词条跳过（幂等）
        """
        if not project_id or not texts or not translations:
            return

        import re
        try:
            from comic_translator.utils.di import get_container
            from comic_translator.models.glossary import GlossaryEntry
            container = get_container()
            glossary_service = container.get("glossary_service")

            # 常用大写词跳过列表
            skip_words = {
                "I", "You", "He", "She", "We", "They", "It", "This", "That",
                "The", "A", "An", "And", "Or", "But", "So", "Yes", "No", "OK",
                "Oh", "What", "Who", "Where", "When", "Why", "How", "Let",
                "Do", "Don", "Don't", "Can", "Can't", "Will", "Would", "Should",
                "Could", "May", "Might", "Must", "Look", "Wait", "Stop", "Go",
                "Come", "Get", "Make", "Take", "There", "Here", "Now", "Then",
                "All", "Some", "Any", "Most", "Very", "Really", "Just", "Okay",
            }

            # 已存在的词条（避免重复保存）
            existing = set()
            for e in glossary_service.get_all_entries(project_id=project_id):
                existing.add(e.source.lower())

            added = 0
            for src, tgt in zip(texts, translations):
                if not src or not tgt:
                    continue
                # 提取大写开头的英文单词
                for word in re.findall(r"\b[A-Z][a-zA-Z']{1,25}\b", src):
                    if word in skip_words or word.lower() in existing:
                        continue
                    # 尝试在译文中找到对应位置（简单启发式：如果译文里包含这个词的拼音/音译，跳过）
                    # 保守策略：只保存明显的人名/地名（含常见名字后缀或多次出现）
                    if not re.search(r"\b[A-Z][a-z]+\b", word):
                        continue
                    entry = GlossaryEntry(source=word, target=word, category="general", priority=1)
                    try:
                        glossary_service.add_term(entry, project_id=project_id)
                        existing.add(word.lower())
                        added += 1
                        logger.debug(f"Auto-glossary [{project_id}]: {word} → {word}")
                    except Exception as e:
                        logger.debug(f"Failed to save auto-glossary {word}: {e}")

            if added:
                logger.info(f"Auto-glossary [{project_id}]: saved {added} new entries")

        except Exception as e:
            logger.debug(f"Auto-glossary extraction failed: {e}")

    @staticmethod
    async def _translate_dialogues(
        dialogues: list[DialogueBlock],
        target_lang: str = "zh",
        project_id: str = "",
        page: int | None = None,
    ):
        """调用 LLM 翻译，并记录每个 DialogueBlock 的翻译状态。

        优先使用 DI 容器的 LLMClient，失败时回退到直接 HTTP 调用。
        自动注入该项目下的术语表 + 角色身份上下文。

        提速②：翻译记忆 + 页面翻译缓存复用
        - 翻译前先查语义翻译记忆（精确匹配）和页面翻译缓存，命中直接复用、跳过 LLM；
        - 翻译后把结果回写翻译记忆和页面缓存，供下次复用（重跑几乎免费）。

        project_id: 项目 ID，用于隔离不同漫画的术语表和角色数据
        page: 页码（1 基），用于页面级翻译缓存键
        状态:
          - success: 成功翻译且与原文不同
          - unchanged: 翻译结果与原文相同
          - failed: 调用失败或返回为空
          - skipped: 没有需要翻译的文本
        """
        # 目标语言提示
        lang_names = {
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "en": "English",
        }
        target_name = lang_names.get(target_lang, target_lang)

        # 收集需要翻译的文本
        texts = []
        index_map: list[int] = []
        for i, d in enumerate(dialogues):
            txt = d.merged_text.strip()
            d.translation_status = "pending"
            if txt:
                texts.append(txt)
                index_map.append(i)

        if not texts:
            for d in dialogues:
                d.translation_status = "skipped"
            return

        use_reuse = GuiPipelineService._use_reuse_enabled()

        # 加载缓存 / 翻译记忆服务（best-effort，失败则退化为不启用）
        cache_service = None
        semantic_tm = None
        if use_reuse:
            try:
                from comic_translator.utils.di import get_container
                container = get_container()
                cache_service = container.get("cache_service")
                semantic_tm = container.get("semantic_tm_service")
            except Exception as e:
                logger.debug(f"Cache/TM services unavailable, translation reuse disabled: {e}")

        # 复用：翻译记忆精确匹配 + 页面翻译缓存
        reused: dict[int, str] = {}  # texts 下标 -> 译文
        if semantic_tm is not None:
            try:
                for j, txt in enumerate(texts):
                    matches = semantic_tm.find_similar(txt, limit=1)
                    if (
                        matches
                        and matches[0].similarity >= 1.0
                        and matches[0].source_text == txt
                        and matches[0].translated_text
                    ):
                        reused[j] = matches[0].translated_text
                if reused:
                    logger.info(f"TM exact match: {len(reused)}/{len(texts)} blocks reused (project={project_id})")
            except Exception as e:
                logger.debug(f"Translation memory lookup failed: {e}")

        if cache_service is not None and project_id and page is not None:
            try:
                cached_blocks = await cache_service.get_translation_cache(project_id, page)
                if cached_blocks:
                    cache_map = {
                        b.source_text: b.translated_text
                        for b in cached_blocks if b.translated_text
                    }
                    added = 0
                    for j, txt in enumerate(texts):
                        if j in reused:
                            continue
                        t = cache_map.get(txt)
                        if t:
                            reused[j] = t
                            added += 1
                    if added:
                        logger.info(
                            f"Page cache hit: {added}/{len(texts)} blocks reused (project={project_id}, page={page})"
                        )
            except Exception as e:
                logger.debug(f"Translation cache read failed: {e}")

        # 应用复用结果，剩余文本走 LLM
        remaining_texts: list[str] = []
        remaining_index_map: list[int] = []
        for j, txt in enumerate(texts):
            if j in reused:
                d = dialogues[index_map[j]]
                d.translated_text = reused[j]
                d.translation_status = "unchanged" if reused[j] == txt else "success"
            else:
                remaining_texts.append(txt)
                remaining_index_map.append(index_map[j])

        if remaining_texts:
            # 构建翻译上下文（术语表 + 角色身份，按项目隔离）
            context_str = GuiPipelineService._build_translation_context(project_id=project_id)

            # 尝试从 DI 容器获取 LLMClient，失败时 fallback
            llm_client = None
            try:
                from comic_translator.utils.di import get_container
                container = get_container()
                llm_client = container.get_llm_client()
                logger.info("Using LLMClient from DI container")
            except Exception as e:
                logger.warning(f"DI container LLM client unavailable ({e}), using direct HTTP fallback")

            if llm_client is not None:
                await GuiPipelineService._translate_via_llm_client(
                    dialogues, remaining_texts, remaining_index_map,
                    target_name, llm_client, context_str, project_id,
                )
            else:
                await GuiPipelineService._translate_via_http(
                    dialogues, remaining_texts, remaining_index_map,
                    target_name, context_str, project_id,
                )

        # 回写：翻译记忆 + 页面翻译缓存（供下次复用）
        if semantic_tm is not None:
            try:
                stored = 0
                for d in dialogues:
                    src = d.merged_text.strip()
                    tgt = d.translated_text
                    if src and tgt and d.translation_status in ("success", "unchanged"):
                        semantic_tm.store(
                            source_text=src,
                            translated_text=tgt,
                            provider="gui",
                            model=target_lang,
                        )
                        stored += 1
                if stored:
                    logger.debug(f"Stored {stored} entries to translation memory (project={project_id})")
            except Exception as e:
                logger.debug(f"Translation memory store failed: {e}")

        if cache_service is not None and project_id and page is not None:
            try:
                from comic_translator.models.translation_block import TranslationBlock
                blocks = [
                    TranslationBlock(
                        id=d.id,
                        ocr_block_id=d.id,
                        source_text=d.merged_text.strip(),
                        translated_text=d.translated_text,
                        provider="gui",
                        model=target_lang,
                    )
                    for d in dialogues
                    if d.merged_text.strip()
                    and d.translated_text
                    and d.translation_status in ("success", "unchanged")
                ]
                await cache_service.set_translation_cache(project_id, page, blocks)
            except Exception as e:
                logger.debug(f"Translation cache write failed: {e}")

    @staticmethod
    def _use_reuse_enabled() -> bool:
        """读取 config.yaml 中 translation.use_cache（进程内缓存，仅首次读文件）"""
        if not hasattr(GuiPipelineService, "_reuse_enabled_cache"):
            GuiPipelineService._reuse_enabled_cache = (
                GuiPipelineService._load_use_translation_cache()
            )
        return GuiPipelineService._reuse_enabled_cache

    @staticmethod
    async def _translate_via_llm_client(
        dialogues: list[DialogueBlock],
        texts: list[str],
        index_map: list[int],
        target_name: str,
        llm_client,
        context_str: str = "",
        project_id: str = "",
    ):
        """通过 LLMClient 翻译（推荐路径）"""
        # 构建带上下文的 system prompt
        context_parts = []
        if context_str.strip():
            context_parts.append(context_str)
        context_parts.append(
            "You are a professional comic translator. "
            f"Translate the following comic text to {target_name}. "
            "Keep the tone natural for comics. "
            "Correct obvious OCR errors in the source text. "
            "Return EXACTLY that many lines, one translation per line, in the same order. "
            "Return ONLY the translations, no numbering, no explanation."
        )
        system_prompt = "\n".join(context_parts)

        numbered_texts = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
        user_prompt = (
            f"Translate the following {len(texts)} comic text blocks to {target_name}.\n"
            f"Return exactly {len(texts)} lines, one per block, in the same order.\n\n"
            f"{numbered_texts}"
        )

        translated_lines: list[str] = []
        last_error: Exception | None = None

        for attempt in range(3):
            try:
                response = await llm_client.generate(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    temperature=0.0,
                    max_tokens=8192,
                )
                translated_lines = [l.strip() for l in response.strip().split("\n") if l.strip()]
                break
            except Exception as e:
                last_error = e
                _err_lower = str(e).lower()
                if "429" in _err_lower or "tpm exhausted" in _err_lower or "rate limit" in _err_lower:
                    _bump_global("rate_limited")
                    logger.error("⚠️ LLM 429 限流 (inference tpm exhausted)")
                else:
                    _bump_global("trans_retries")
                logger.warning(f"Translation attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(1.0)
                continue

        if not translated_lines and last_error is not None:
            status = "failed"
        else:
            status = "success"

        GuiPipelineService._apply_translation_results(
            dialogues, texts, index_map, translated_lines, status,
        )

        # 翻译成功后自动提取术语表
        if status == "success" and project_id:
            GuiPipelineService._save_auto_glossary(texts, translated_lines, project_id)

        # 对翻译失败的块逐个重试（不管整体状态，只要有失败的块就重试）
        failed_dialogues = [
            d for d in dialogues
            if d.merged_text.strip() and d.translation_status == "failed"
        ]
        if failed_dialogues:
            logger.info(f"Retrying {len(failed_dialogues)} individually failed blocks...")
            for d in failed_dialogues:
                source = d.merged_text.strip()
                single = await GuiPipelineService._translate_single(source, target_name, llm_client)
                if single:
                    d.translated_text = single
                    d.translation_status = "unchanged" if single == source else "success"
                else:
                    d.translated_text = source
                    d.translation_status = "failed"
                await asyncio.sleep(0.2)

    @staticmethod
    async def _translate_via_http(
        dialogues: list[DialogueBlock],
        texts: list[str],
        index_map: list[int],
        target_name: str,
        context_str: str = "",
        project_id: str = "",
    ):
        """通过直接 HTTP 调用翻译（DI 容器不可用时的降级路径）"""
        import yaml
        config_path = Path("config.yaml")
        if not config_path.exists():
            for d in dialogues:
                d.translation_status = "failed" if d.merged_text.strip() else "skipped"
            return

        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        llm_cfg = config.get("llm", {})
        llm_mode = llm_cfg.get("mode", "ollama")

        if llm_mode != "api":
            for d in dialogues:
                d.translation_status = "failed" if d.merged_text.strip() else "skipped"
            logger.warning(f"LLM mode is '{llm_mode}', not 'api' — translation skipped")
            return

        api_cfg = llm_cfg.get("api", {})
        api_key = api_cfg.get("api_key", "")
        if not api_key or api_key.startswith("${"):
            # 尝试环境变量
            import os
            import re
            env_key = api_key.strip("${}") if api_key.startswith("${") else ""
            api_key = os.environ.get(env_key, "")
        if not api_key:
            for d in dialogues:
                d.translation_status = "failed" if d.merged_text.strip() else "skipped"
            logger.warning("LLM API key not configured, translation skipped")
            return

        import httpx
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

        # 构建带上下文的 system message
        context_parts = []
        if context_str.strip():
            context_parts.append(context_str)
        context_parts.append(
            "You are a professional comic translator. "
            f"Translate the following comic text to {target_name}. "
            "Keep the tone natural for comics. "
            "Correct obvious OCR errors in the source text. "
            "Return EXACTLY that many lines, one translation per line, in the same order. "
            "Return ONLY the translations, no numbering, no explanation."
        )
        system_msg = "\n".join(context_parts)
        user_msg = "\n---\n".join(texts)

        translated_lines: list[str] = []
        last_error: Exception | None = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=120) as client:
                    resp = await client.post(
                        f"{api_cfg['base_url'].rstrip('/')}/chat/completions",
                        headers=headers,
                        json={
                            "model": api_cfg.get("model", "deepseek-chat"),
                            "messages": [
                                {"role": "system", "content": system_msg},
                                {"role": "user", "content": user_msg},
                            ],
                        },
                    )
                    # 429 = 限流：醒目记录 + 计数（用户要求必须提示）
                    if resp.status_code == 429:
                        _bump_global("rate_limited")
                        logger.error(
                            "⚠️ 429 限流 (inference tpm exhausted) — 第 "
                            f"{attempt + 1}/3 次尝试，等待后重试"
                        )
                    resp.raise_for_status()
                    content = resp.json()["choices"][0]["message"]["content"]
                    translated_lines = [l.strip() for l in content.strip().split("\n") if l.strip()]
                    break
            except Exception as e:
                last_error = e
                if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
                    _bump_global("rate_limited")
                    logger.error("⚠️ HTTP 429 限流 (inference tpm exhausted)")
                else:
                    _bump_global("trans_retries")
                logger.warning(f"HTTP translation attempt {attempt + 1} failed: {e}")
                if attempt < 2:
                    await asyncio.sleep(1.0)
                continue

        status = "failed" if (not translated_lines and last_error is not None) else "success"
        GuiPipelineService._apply_translation_results(
            dialogues, texts, index_map, translated_lines, status,
        )

        # 翻译成功后自动提取术语表
        if status == "success" and project_id:
            GuiPipelineService._save_auto_glossary(texts, translated_lines, project_id)

        # 对翻译失败的块逐个重试（不管整体状态，只要有失败的块就重试）
        failed_dialogues = [
            d for d in dialogues
            if d.merged_text.strip() and d.translation_status == "failed"
        ]
        if failed_dialogues:
            logger.info(f"Retrying {len(failed_dialogues)} individually failed blocks (HTTP)...")
            for d in failed_dialogues:
                source = d.merged_text.strip()
                single = await GuiPipelineService._translate_single_http(source, target_name, api_cfg, headers)
                if single:
                    d.translated_text = single
                    d.translation_status = "unchanged" if single == source else "success"
                else:
                    d.translated_text = source
                    d.translation_status = "failed"
                await asyncio.sleep(0.2)

    @staticmethod
    def _apply_translation_results(
        dialogues: list[DialogueBlock],
        texts: list[str],
        index_map: list[int],
        translated_lines: list[str],
        status: str,
    ):
        """将翻译结果应用到 DialogueBlock"""
        if status == "failed":
            for d in dialogues:
                if d.merged_text.strip():
                    d.translation_status = "failed"
                    d.translated_text = ""
                else:
                    d.translation_status = "skipped"
            return

        if len(translated_lines) != len(texts):
            logger.warning(
                f"LLM returned {len(translated_lines)} lines for {len(texts)} texts. "
                f"Expected {len(texts)}, got {len(translated_lines)}."
            )

        for idx, dialogue_index in enumerate(index_map):
            d = dialogues[dialogue_index]
            source = d.merged_text.strip()
            if idx < len(translated_lines):
                translated = translated_lines[idx].lstrip("0123456789.- ")
                if not translated:
                    d.translated_text = ""
                    d.translation_status = "failed"
                elif translated == source:
                    d.translated_text = translated
                    d.translation_status = "unchanged"
                else:
                    d.translated_text = translated
                    d.translation_status = "success"
            else:
                d.translated_text = ""
                d.translation_status = "failed"

        for d in dialogues:
            if not d.merged_text.strip() and (not d.translation_status or d.translation_status == "pending"):
                d.translation_status = "skipped"

    @staticmethod
    async def _translate_single(text: str, target_name: str, llm_client) -> str:
        """单个文本翻译（LLMClient 路径）"""
        system_prompt = (
            "You are a professional comic translator. "
            f"Translate the following comic line to {target_name}. "
            "The input may contain OCR recognition errors; infer the intended meaning. "
            "Return ONLY the translation, without explanation."
        )
        for attempt in range(2):
            try:
                response = await llm_client.generate(
                    prompt=text,
                    system_prompt=system_prompt,
                    temperature=0.0,
                    max_tokens=1024,
                )
                return response.strip()
            except Exception as e:
                if attempt == 0:
                    await asyncio.sleep(1.0)
                    continue
                logger.error(f"Single translation failed: {e}")
                return ""
        return ""

    @staticmethod
    async def retry_page_failed_bubbles(
        dialogues: list[DialogueBlock],
        target_lang: str = "zh",
        project_id: str = "",
    ) -> int:
        """重试指定页面中所有翻译失败的 DialogueBlock

        返回成功重试的数量。
        可以在 GUI 中调用，用于手动重试失败的气泡。
        """
        failed = [d for d in dialogues if d.merged_text.strip() and d.translation_status == "failed"]
        if not failed:
            return 0

        lang_names = {"zh": "Chinese", "ja": "Japanese", "ko": "Korean", "en": "English"}
        target_name = lang_names.get(target_lang, target_lang)

        # 尝试获取 LLM 客户端
        llm_client = None
        try:
            from comic_translator.utils.di import get_container
            container = get_container()
            llm_client = container.get_llm_client()
        except Exception:
            llm_client = None

        retried = 0
        for d in failed:
            source = d.merged_text.strip()
            try:
                if llm_client:
                    single = await GuiPipelineService._translate_single(source, target_name, llm_client)
                else:
                    single = ""
                if single:
                    d.translated_text = single
                    d.translation_status = "unchanged" if single == source else "success"
                    retried += 1
                else:
                    # 最后保底：用原文
                    d.translated_text = source
                    d.translation_status = "failed"
            except Exception as e:
                logger.error(f"Retry failed for block '{source[:30]}...': {e}")
                d.translated_text = source
                d.translation_status = "failed"
            await asyncio.sleep(0.2)

        logger.info(f"Retried {retried}/{len(failed)} failed blocks (project={project_id})")
        return retried

    @staticmethod
    async def _translate_single_http(text: str, target_name: str, api_cfg: dict, headers: dict) -> str:
        """单个文本翻译（HTTP 降级路径）"""
        import httpx
        system_msg = (
            "You are a professional comic translator. "
            f"Translate the following comic line to {target_name}. "
            "The input may contain OCR recognition errors; infer the intended meaning. "
            "Return ONLY the translation, without explanation."
        )
        for attempt in range(2):
            try:
                async with httpx.AsyncClient(timeout=60) as client:
                    resp = await client.post(
                        f"{api_cfg['base_url'].rstrip('/')}/chat/completions",
                        headers=headers,
                        json={
                            "model": api_cfg.get("model", "deepseek-chat"),
                            "messages": [
                                {"role": "system", "content": system_msg},
                                {"role": "user", "content": text},
                            ],
                        },
                    )
                    resp.raise_for_status()
                    return resp.json()["choices"][0]["message"]["content"].strip()
            except Exception as e:
                if attempt == 0:
                    await asyncio.sleep(1.0)
                    continue
                logger.error(f"Single HTTP translation failed: {e}")
                return ""
        return ""
