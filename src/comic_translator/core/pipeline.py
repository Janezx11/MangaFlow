"""核心翻译管线 —— CLI 与 GUI 共用的完整流程编排

检测（CTBD+YOLO）→ 气泡归属 → OCR → 翻译 → 擦除 → 渲染，逐页上报进度。
只做编排和进度上报，不复制业务逻辑；所有核心处理委托给现有 services/providers。

CLI（comic-translator run/resume）与 Gradio GUI 消费同一个
`ComicPipelineService.run_pipeline()` 同步生成器；`GuiPipelineService` 是
向后兼容别名。
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Generator
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from loguru import logger
from PIL import Image

from comic_translator.core.translation import (
    PipelineTranslationMixin,
    _get_global_counts,
)
from comic_translator.gui.models.gui_models import PageSnapshot, RunRecord
from comic_translator.models.bubble import Bubble, DialogueBlock
from comic_translator.models.detection_region import DetectionRegion
from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock
from comic_translator.models.text_region import TextRegion


def _read_bubble_cfg() -> dict:
    """从 config.yaml 读取气泡检测配置（模型路径/置信度/IoU），缺失时回退默认。

    保证 GUI 管线的气泡模型路径来自配置（模型库「启用」同步写回 config.yaml
    后即被这里读取），不再硬编码。
    """
    from pathlib import Path

    import yaml

    default = {
        "model": "models/bubble/yolo11n-seg-mixed-bubble.pt",
        "conf_threshold": 0.35,
        "iou_threshold": 0.45,
    }
    try:
        cfg_path = Path("config.yaml")
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            bubble = cfg.get("bubble", {}) or {}
            merged = dict(default)
            merged.update({k: v for k, v in bubble.items() if k in default})
            return merged
    except Exception as e:
        logger.debug(f"Failed to read bubble config: {e}")
    return default


def _read_config_section(section: str) -> dict:
    """读取 config.yaml 的指定段，缺失/失败时返回空 dict"""
    import yaml
    try:
        cfg_path = Path("config.yaml")
        if cfg_path.exists():
            cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            value = cfg.get(section, {})
            return value if isinstance(value, dict) else {}
    except Exception as e:
        logger.debug(f"Failed to read config section '{section}': {e}")
    return {}


def _detect_gpu() -> bool:
    try:
        import torch
        return bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
    except Exception:
        return False


def _build_ocr_provider():
    """按 config.yaml 的 ocr 段构建 OCR 引擎。

    约定（与 README 一致）：provider 缺省 = easyocr（默认英文 OCR），
    paddle_ocr / manga_ocr 为可选扩展。配置的 provider 导入失败时回退
    easyocr 并告警，而不是让整个管线在第一页崩溃。
    返回 (provider, 描述文本)。
    """
    cfg = _read_config_section("ocr")
    name = str(cfg.get("provider", "easyocr")).lower()
    lang = str(cfg.get("lang", "en"))
    use_gpu = cfg.get("use_gpu")
    use_gpu = _detect_gpu() if use_gpu is None else bool(use_gpu)

    def _easyocr():
        from comic_translator.providers.ocr.easyocr_provider import EasyOCRProvider
        return EasyOCRProvider(lang=lang, use_gpu=use_gpu)

    factories = {
        "easyocr": (_easyocr, f"EasyOCR (lang={lang}, GPU={'开' if use_gpu else '关'})"),
        "paddle_ocr": (
            lambda: _make_paddle(lang, use_gpu),
            f"PaddleOCR (lang={lang})",
        ),
        "paddle": (
            lambda: _make_paddle(lang, use_gpu),
            f"PaddleOCR (lang={lang})",
        ),
        "manga_ocr": (
            lambda: _make_manga(),
            "MangaOCR (ja, 竖排)",
        ),
    }

    factory, desc = factories.get(name, (None, ""))
    if factory is not None:
        try:
            return factory(), desc
        except Exception as e:
            logger.warning(f"OCR provider '{name}' unavailable ({e}); falling back to EasyOCR")

    # 缺省/未知/导入失败 → EasyOCR（easyocr 自身缺失则直接抛出，无更多退路）
    try:
        return _easyocr(), f"EasyOCR (lang={lang}, GPU={'开' if use_gpu else '关'})"
    except Exception as e:
        logger.warning(f"EasyOCR fallback failed: {e}; using NullOCR (no OCR output)")
        return _NullOCRProvider(), "NullOCR (不可用，OCR 输出为空)"


def _make_paddle(lang: str, use_gpu: bool):
    from comic_translator.providers.ocr.paddle_ocr_provider import PaddleOCRProvider
    return PaddleOCRProvider(lang=lang, use_gpu=use_gpu)


def _make_manga():
    from comic_translator.providers.ocr.manga_ocr_provider import MangaOCRProvider
    return MangaOCRProvider()


class _NullOCRProvider:
    """OCR 引擎完全不可用时的最后兜底：返回空结果，管线其余阶段照常运行"""

    async def detect_text(self, image):
        return []

    def get_name(self) -> str:
        return "null"



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


class ComicPipelineService(PipelineTranslationMixin):
    """核心翻译管线服务（CLI / GUI 共用）

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
        # 取消标志（每次 run 重建）；GUI 取消按钮调用 cancel_run() 置位
        self._cancel_event: threading.Event | None = None
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
        from comic_translator.providers.bubble.yolo_bubble_provider import YOLO11BubbleProvider
        from comic_translator.providers.text_detection.ctbd_text_detector import (
            CTBDTextDetectorProvider,
        )
        self.text_detector = CTBDTextDetectorProvider(confidence=conf)

        self._log_init("加载气泡检测模型 YOLO11...")
        bubble_cfg = _read_bubble_cfg()
        self.bubble_provider = YOLO11BubbleProvider(
            model_path=bubble_cfg["model"],
            conf_threshold=bubble_cfg["conf_threshold"],
            iou_threshold=bubble_cfg["iou_threshold"],
        )

        self.ocr_provider, ocr_desc = _build_ocr_provider()
        self._log_init(f"加载 OCR 引擎... {ocr_desc}")

        self._log_init("加载 OCR 后处理服务...")
        from comic_translator.services.ocr_postprocess_service import OCRPostprocessService
        self.ocr_postprocess = OCRPostprocessService()

        self._log_init("加载检测融合服务...")
        from comic_translator.services.detection_fusion_service import DetectionFusionService
        from comic_translator.services.text_region_adapter import TextRegionAdapter
        from comic_translator.services.text_region_service import TextRegionService
        self.fusion = DetectionFusionService()
        self.text_region_service = TextRegionService()
        self.text_region_adapter = TextRegionAdapter()

        self._log_init("加载排版渲染服务...")
        from comic_translator.services.background_cleaner import BackgroundCleaner
        from comic_translator.services.balanced_line_breaker import BalancedLineBreaker
        from comic_translator.services.bubble_interior_service import BubbleInteriorService
        from comic_translator.services.bubble_layout_service import BubbleLayoutService
        from comic_translator.services.font_manager import FontManager
        from comic_translator.services.font_size_solver import FontSizeSolver
        from comic_translator.services.inpaint_service import InpaintService
        from comic_translator.services.optical_center_solver import OpticalCenterSolver
        from comic_translator.services.padding_solver import PaddingSolver
        from comic_translator.services.region_classifier import RegionClassifier
        from comic_translator.services.render_qa_service import RenderQAService
        from comic_translator.services.render_service import RenderService
        from comic_translator.services.shape_layout_solver import ShapeLayoutSolver
        from comic_translator.services.text_normalizer import TextNormalizer
        from comic_translator.services.typography_qa import TypographyQA
        from comic_translator.services.typography_service_v2 import TypographyServiceV2

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

    def refresh_bubble_provider(self) -> None:
        """按 config.yaml 重建气泡检测 provider（模型库「启用」后立即生效）。

        YOLO 模型为懒加载：只重建 provider 对象（开销小），
        下次 detect 时才会真正载入新权重。
        """
        if not getattr(self, "_initialized", False):
            return
        from comic_translator.providers.bubble.yolo_bubble_provider import YOLO11BubbleProvider

        bubble_cfg = _read_bubble_cfg()
        self.bubble_provider = YOLO11BubbleProvider(
            model_path=bubble_cfg["model"],
            conf_threshold=bubble_cfg["conf_threshold"],
            iou_threshold=bubble_cfg["iou_threshold"],
        )
        logger.info(f"Bubble provider refreshed: {bubble_cfg['model']}")

    def cancel_run(self) -> None:
        """请求取消当前运行（GUI 取消按钮调用）。

        优雅停止：producer 不再处理剩余页，已在途的翻译/渲染自然收尾，
        避免后台线程在生成器被关闭后继续空跑（"Event loop is closed"）。
        """
        if self._cancel_event is not None:
            self._cancel_event.set()
            logger.info("Pipeline cancel requested")

    def warmup(self, conf: float = 0.3):
        """在后台线程中预加载模型，避免首次运行时长时间无反馈"""
        import threading
        if self._initialized:
            return
        t = threading.Thread(target=self._ensure_initialized, args=(conf,), daemon=True)
        t.start()

    @staticmethod
    def select_page_range(
        all_images: list[Path], start_page: int = 1, end_page: int | None = None,
    ) -> list[tuple[int, Path]]:
        """按 1 基页码范围筛选图片，返回 [(页码, 路径), ...]。

        页码保留在完整列表中的编号，使部分翻译（resume）与全量运行的
        page_NNNN 输出名一致。start_page <= 0 视为 1。
        """
        start = max(1, int(start_page))
        return [
            (i + 1, p)
            for i, p in enumerate(all_images)
            if i + 1 >= start and (end_page is None or i + 1 <= end_page)
        ]

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
        start_page: int = 1,
        end_page: int | None = None,
    ) -> Generator[PipelineProgress, None, None]:
        """同步生成器，逐页处理并上报进度

        start_page/end_page: 页码范围（1 基、含端点），页码按完整图片列表
        编号——部分翻译（resume）产出的 page_NNNN 与全量运行保持一致。
        """
        return self._run_sync(input_dir, use_ctbd, use_yolo, use_fusion, translate, conf,
                              target_lang, render_options, output_dir,
                              bubble_aware, split_strategy, cross_threshold,
                              start_page, end_page)

    def _run_sync(self, input_dir, use_ctbd, use_yolo, use_fusion, translate, conf,
                  target_lang, render_options, output_dir="data/output/rendered",
                  bubble_aware=True, split_strategy="auto", cross_threshold=0.15,
                  start_page=1, end_page=None):
        import threading
        self._cancel_event = threading.Event()
        loop = asyncio.new_event_loop()
        gen = self._run_async(input_dir, use_ctbd, use_yolo, use_fusion, translate, conf,
                              target_lang, render_options, output_dir,
                              bubble_aware, split_strategy, cross_threshold,
                              cancel_event=self._cancel_event,
                              start_page=start_page, end_page=end_page)
        try:
            while True:
                try:
                    progress = loop.run_until_complete(gen.__anext__())
                    yield progress
                except StopAsyncIteration:
                    break
        finally:
            # 消费方提前关闭生成器（GUI 取消按钮 cancels=[start_event]）时，
            # 必须先把 GeneratorExit 注入内层 async generator，让后台 task
            # 走到清理逻辑，再关闭事件循环——否则检测/OCR/翻译线程继续空跑，
            # 且 loop.close() 后任务报 "Event loop is closed"
            self._cancel_event.set()
            try:
                loop.run_until_complete(gen.aclose())
            except Exception:
                pass
            loop.close()

    async def _run_async(
        self, input_dir, use_ctbd, use_yolo, use_fusion, translate, conf,
        target_lang, render_options, output_dir="data/output/rendered",
        bubble_aware=True, split_strategy="auto", cross_threshold=0.15,
        cancel_event=None, start_page=1, end_page=None,
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

        all_images = self.collect_images(input_dir)
        images = self.select_page_range(all_images, start_page, end_page)
        if not images:
            yield PipelineProgress(
                0, 0, "done",
                f"No pages in range [{start_page}, {end_page or 'end'}] of {input_dir} "
                f"({len(all_images)} images found)",
            )
            return

        total = len(images)
        first_page = images[0][0]
        self.reset_perf(total_pages=total)
        rng = f"page {first_page}" if total == 1 else f"pages {first_page}-{images[-1][0]}"
        yield PipelineProgress(0, total, "detection", f"Found {total} pages ({rng})")

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

        # 记录 页码 → 源文件路径，供渲染时继承原文件名
        page_sources: dict[int, str] = {}

        async def producer():
            """Stage 1: detect + OCR → q_ready"""
            try:
                for page, img_path in images:
                    if cancel_event is not None and cancel_event.is_set():
                        logger.info("Producer stopped by cancel request")
                        break
                    page_sources[page] = str(img_path)
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
                # 通知 worker 生产者结束。取消路径下 worker 已被取消，
                # 不能阻塞在满队列的 put 上（否则 aclose 永久挂起）。
                if not (cancel_event is not None and cancel_event.is_set()):
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
        try:

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
            next_page = first_page

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
                                source_path=page_sources.get(cur_page, ""),
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

        finally:
            # 后台 task 必须在任何退出路径（正常结束/异常/取消触发的
            # GeneratorExit）下都被取消并 join，否则会在 loop.close() 后
            # 继续 pending，报 "Task was destroyed but it is pending"
            for t in worker_tasks:
                t.cancel()
            producer_task.cancel()
            for t in ([producer_task, *worker_tasks]):
                try:
                    await t
                except (asyncio.CancelledError, Exception):
                    pass
            try:
                await _events_iter.aclose()
            except Exception:
                pass


        # 自动质检 + 不合格页重试闭环（保守策略见 _quality_retry_loop）
        try:
            async for ev in self._quality_retry_loop(
                input_dir=input_dir,
                output_dir=output_dir,
                target_lang=target_lang,
                render_options=render_options,
                page_sources=page_sources,
                cancel_event=cancel_event,
            ):
                yield ev
        except Exception as e:
            logger.debug(f"Quality loop skipped: {e}")

        # 保存运行记录（放在质检闭环后：重试产生的最终译文/渲染路径一并入档）
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

        yield PipelineProgress(total, total, "done", f"All {total} pages completed")

    # ============================================================
    # 质检闭环：聚合报告 → 自动重试可修复页 → 重新聚合
    # ============================================================

    def _select_retry_candidates(self, summary, max_retries: int) -> tuple[list, int]:
        """从质检报告中筛选可自动重试的页。

        保守策略：只重试「needs_retry 且内存中存在 failed 翻译气泡」的页
        ——失败翻译（LLM/网络原因）重试有真实增益；纯渲染质量问题
        （模糊/残留/排版）重渲染是确定性的，重试无增益，留给人工审核。

        Returns:
            (可重试的 PageQualityReport 列表, 仅渲染问题需人工处理的页数)
        """
        candidates: list = []
        render_only = 0
        for rep in summary.pages:
            if not (rep.needs_retry and rep.retry_count < max_retries):
                continue
            result = self._results.get(rep.page)
            if result is None:
                continue
            has_failed = any(
                d.merged_text.strip() and d.translation_status == "failed"
                for d in result.dialogues
            )
            if has_failed:
                candidates.append(rep)
            else:
                render_only += 1
        return candidates, render_only

    async def _quality_retry_loop(
        self,
        input_dir: str,
        output_dir: str,
        target_lang: str,
        render_options: dict | None,
        page_sources: dict[int, str],
        cancel_event=None,
    ) -> Generator[PipelineProgress, None, None]:
        """质检闭环（异步生成器，逐条上报进度）：

        1. 聚合 render_debug.json → 质检报告并保存
        2. 对「翻译失败导致不合格」的页：LLM 逐气泡重试 → 重渲染
           （render_debug.json 随新渲染更新）
        3. 重新聚合并保存报告，产出重试前后的合格率对比
        """
        from comic_translator.services.quality_orchestrator import QualityOrchestrator

        project_name = Path(input_dir).name
        # 质检聚合跟随本次运行的渲染输出目录（默认 data/output/rendered
        # 时与历史行为一致；自定义 output_dir 不再聚合到错误的数据）
        orch = QualityOrchestrator(output_dir=str(Path(output_dir).parent))
        summary = orch.aggregate_report(
            project_name=project_name, rendered_dir=Path(output_dir).name,
        )
        orch.save_report(summary, "data/output/quality_report.json")

        if summary.total_pages == 0:
            yield PipelineProgress(0, 0, "done", "质检: 无渲染调试数据，跳过")
            return

        max_retries = orch._thresholds["max_retries"]
        candidates, render_only = self._select_retry_candidates(summary, max_retries)
        before_rate = summary.overall_pass_rate

        if not candidates:
            msg = f"质检: {before_rate}% 合格率"
            if render_only:
                msg += f"，另有 {render_only} 页触发重试阈值（渲染质量问题，建议人工复核）"
            if summary.needs_manual_review:
                msg += f"，{summary.needs_manual_review} 页需人工审核"
            yield PipelineProgress(summary.total_pages, summary.total_pages, "done", msg)
            return

        yield PipelineProgress(
            0, summary.total_pages, "quality",
            f"质检不合格 {len(candidates)} 页，自动重试中..."
            + (f"（另有 {render_only} 页渲染问题需人工处理）" if render_only else ""),
        )

        for rep in candidates:
            if cancel_event is not None and cancel_event.is_set():
                logger.info("Quality retry stopped by cancel request")
                break
            page = rep.page
            result = self._results[page]
            yield PipelineProgress(
                page, summary.total_pages, "quality",
                f"重试第 {page} 页: {rep.retry_reason}",
            )

            # 1) LLM 逐气泡重试失败翻译
            await self.retry_page_failed_bubbles(
                result.dialogues, target_lang=target_lang, project_id=project_name,
            )

            # 2) 重渲染（更新 rendered 图与 _debug/render_debug.json）
            img_path = page_sources.get(page, "")
            if img_path and Path(img_path).exists():
                t0 = time.time()
                page_render_options = dict(render_options or {})
                page_render_options.setdefault("fallback_to_source", True)
                with Image.open(img_path) as img:
                    render_result = self.render_service.render_page(
                        img, page, result.dialogues,
                        output_dir=output_dir,
                        source_path=img_path,
                        render_options=page_render_options,
                    )
                result.rendered_path = render_result.output_path
                result.render_time = time.time() - t0
                self._perf_accumulate("render_total", result.render_time)
            else:
                logger.warning(
                    f"Quality retry: source image missing for page {page} ({img_path!r}), "
                    "translations retried but page not re-rendered"
                )

        # 3) 重新聚合（重渲染已更新 _debug 数据）
        new_summary = orch.aggregate_report(project_name=project_name)
        orch.save_report(new_summary, "data/output/quality_report.json")
        yield PipelineProgress(
            new_summary.total_pages, new_summary.total_pages, "done",
            f"质检: 自动重试 {len(candidates)} 页，合格率 {before_rate}% → {new_summary.overall_pass_rate}%"
            + (f"，另有 {render_only} 页渲染质量问题建议人工复核" if render_only else ""),
        )

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


# 向后兼容别名：历史代码（GUI 页面、测试）以 GuiPipelineService 引用本服务
GuiPipelineService = ComicPipelineService
