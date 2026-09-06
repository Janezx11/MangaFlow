"""Quality Orchestrator — 翻译后质检 + 单页重跑闭环

功能:
1. aggregate_report()  — 遍历渲染输出目录，生成全局质检报告
2. needs_retry()       — 根据阈值判断页面是否需要重跑
3. retry_page()        — 单页重跑（OCR + 翻译 + 渲染），带失败上下文
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar

from loguru import logger

from comic_translator.models.quality_report import (
    PageQualityReport,
    QualityReportSummary,
    RetryRecord,
)

if TYPE_CHECKING:
    from comic_translator.clients.base import LLMClient
    from comic_translator.services.batch_cache import BatchCacheService
    from comic_translator.services.batch_ocr_service import BatchOCRService
    from comic_translator.services.batch_prompt_builder import BatchPromptBuilder
    from comic_translator.services.glossary_service import GlossaryService
    from comic_translator.services.render_quality_service import RenderQualityService


class QualityOrchestrator:
    """质检编排器"""

    # 默认阈值
    DEFAULT_THRESHOLDS: ClassVar[dict[str, Any]] = {
        "render_quality_min": 30.0,
        "translation_status_max": "FAIL",  # FAIL 必须重跑
        "length_ratio_min": 0.3,
        "block_count_diff_max": 3,
        "overall_score_min": 50.0,
        "max_retries": 3,
    }

    def __init__(
        self,
        output_dir: str = "data/output",
        thresholds: dict | None = None,
    ):
        self._output_dir = Path(output_dir)
        self._thresholds = {**self.DEFAULT_THRESHOLDS, **(thresholds or {})}

    # ============================================================
    # 报告生成
    # ============================================================

    def aggregate_report(
        self,
        project_name: str = "",
        rendered_dir: str = "rendered",
    ) -> QualityReportSummary:
        """遍历渲染输出目录，生成全局质检报告"""
        render_path = self._output_dir / rendered_dir
        if not render_path.exists():
            logger.error(f"Render directory not found: {render_path}")
            return QualityReportSummary(project_name=project_name)

        pages: list[PageQualityReport] = []
        debug_files = sorted(render_path.glob("page_*/render_debug.json"))

        if not debug_files:
            # 也尝试顶层 render_debug.json
            debug_files = sorted(render_path.glob("page_*_render_debug.json"))

        if not debug_files:
            # RenderQualityService 实际写盘位置: rendered/_debug/page_NNNN/render_debug.json
            debug_files = sorted((render_path / "_debug").glob("page_*/render_debug.json"))

        logger.info(f"Found {len(debug_files)} render debug files in {render_path}")

        for df in debug_files:
            try:
                data = json.loads(df.read_text(encoding="utf-8"))
                report = self._parse_debug_to_report(data, df)
                pages.append(report)
            except Exception as e:
                logger.warning(f"Failed to parse {df}: {e}")

        pages.sort(key=lambda p: p.page)

        # 计算汇总
        # "合格"口径统一为 not needs_retry（与 auto_retry_pages 的重跑判定一致），
        # 避免出现"总分达标但被标记需重跑"的页被计入合格率的矛盾。
        total = len(pages)
        passed = sum(1 for p in pages if not p.needs_retry)
        retry_passed = sum(1 for p in pages if p.retry_count > 0 and not p.needs_retry)
        manual = [p for p in pages if p.retry_count >= self._thresholds["max_retries"] and p.needs_retry]
        avg_render = sum(p.render_quality for p in pages) / total if total else 0.0

        # 统计翻译状态
        fail_count = sum(1 for p in pages if p.translation_status == "FAIL")
        avg_translation = "FAIL" if fail_count > total / 2 else "WARNING" if fail_count > 0 else "PASS"

        summary = QualityReportSummary(
            project_name=project_name,
            total_pages=total,
            passed=passed,
            retry_passed=retry_passed,
            needs_manual_review=len(manual),
            overall_pass_rate=round(passed / total * 100, 1) if total else 0.0,
            avg_render_quality=round(avg_render, 1),
            avg_translation_status=avg_translation,
            pages=pages,
            manual_review_pages=manual,
        )

        return summary

    def _parse_debug_to_report(
        self, data: dict, debug_path: Path
    ) -> PageQualityReport:
        """从 render_debug.json 解析为 PageQualityReport"""
        page = data.get("page", 0)
        quality = data.get("quality", {})
        ti = data.get("translation_integrity", {})
        blocks = data.get("blocks", [])
        typo_scores = data.get("typography_scores", [])

        # 翻译质量
        ti_status = ti.get("status", "PASS")
        ti_issues = ti.get("issues", [])
        ti_checks = ti.get("checks", {})
        english_residue = bool(ti_checks.get("english_residue", []))

        # 翻译块数差异
        translated_count = sum(
            1 for b in blocks if b.get("translated_text", "").strip()
        )
        # 统计 block 数差异（从 blocks 数据推断）
        source_count = sum(
            1 for b in blocks if b.get("source_text", "").strip()
        )
        block_diff = max(0, source_count - translated_count)

        # 长度比（中文字数 / 英文字数）
        source_len = sum(len(b.get("source_text", "")) for b in blocks)
        trans_len = sum(len(b.get("translated_text", "")) for b in blocks)
        length_ratio = trans_len / source_len if source_len > 0 else 0.0

        # 渲染质量
        render_q = quality.get("overall", 0.0)
        render_blur = quality.get("blur", 0.0)
        render_residual = quality.get("text_residual", 0.0)

        # 排版质量
        typo_avg = 0.0
        if typo_scores:
            typo_avg = sum(
                s.get("typography_score", 0) for s in typo_scores
            ) / len(typo_scores)

        # 综合分
        overall = self._compute_overall_score(
            render_q, ti_status, length_ratio, block_diff, typo_avg
        )

        # 是否需要重跑
        needs_retry, retry_reason = self._check_needs_retry(
            render_q, ti_status, length_ratio, block_diff, overall
        )

        # 图片路径
        img_path = ""
        rendered_png = debug_path.parent / "final_render.png"
        if not rendered_png.exists():
            # 尝试顶层文件
            page_str = f"page_{page:04d}"
            rendered_png = debug_path.parent / f"{page_str}_rendered.png"
        if rendered_png.exists():
            img_path = str(rendered_png)

        return PageQualityReport(
            page=page,
            image_path=img_path,
            ocr_block_count=source_count,
            ocr_avg_confidence=0.0,
            translation_status=ti_status,
            translation_issues=ti_issues,
            translated_block_count=translated_count,
            block_count_diff=block_diff,
            length_ratio=round(length_ratio, 2),
            english_residue=english_residue,
            render_quality=round(render_q, 1),
            render_blur=round(render_blur, 1),
            render_text_residual=round(render_residual, 1),
            typography_score=round(typo_avg, 1),
            overall_score=round(overall, 1),
            needs_retry=needs_retry,
            retry_reason=retry_reason,
        )

    def _compute_overall_score(
        self,
        render_q: float,
        ti_status: str,
        length_ratio: float,
        block_diff: int,
        typo_avg: float,
    ) -> float:
        """加权综合分 0-100"""
        # 渲染质量权重 0.4
        render_score = render_q

        # 翻译完整性权重 0.35
        ti_score = {
            "PASS": 100.0,
            "WARNING": 60.0,
            "FAIL": 0.0,
        }.get(ti_status, 50.0)

        # 长度比权重 0.1
        length_score = min(length_ratio / 0.5, 1.0) * 100

        # 块完整性权重 0.1
        block_score = max(0, 100 - block_diff * 20)

        # 排版权重 0.05
        typo_score = typo_avg

        return (
            render_score * 0.40
            + ti_score * 0.35
            + length_score * 0.10
            + block_score * 0.10
            + typo_score * 0.05
        )

    def _check_needs_retry(
        self,
        render_q: float,
        ti_status: str,
        length_ratio: float,
        block_diff: int,
        overall: float,
    ) -> tuple[bool, str]:
        """判断是否需要重跑，返回 (needs_retry, reason)"""
        reasons = []

        if render_q < self._thresholds["render_quality_min"]:
            reasons.append(f"render_quality={render_q}")

        if ti_status == "FAIL":
            # 检查是否真的需要重跑（有些 FAIL 是因为没有翻译文本，比如空页）
            reasons.append(f"translation={ti_status}")

        if length_ratio < self._thresholds["length_ratio_min"]:
            reasons.append(f"length_ratio={length_ratio}")

        if block_diff > self._thresholds["block_count_diff_max"]:
            reasons.append(f"block_diff={block_diff}")

        if overall < self._thresholds["overall_score_min"]:
            reasons.append(f"overall={overall}")

        if reasons:
            return True, "; ".join(reasons)
        return False, ""

    # ============================================================
    # 报告输出
    # ============================================================

    def save_report(
        self,
        summary: QualityReportSummary,
        output_path: str | Path = "data/output/quality_report.json",
    ) -> Path:
        """保存质检报告到 JSON 文件"""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(summary.model_dump(mode="json"), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        logger.info(f"Quality report saved: {path}")
        return path

    def print_summary(self, summary: QualityReportSummary) -> str:
        """打印人类可读的质检报告摘要"""
        lines = [
            f"{'=' * 50}",
            f"  Quality Report: {summary.project_name or '(unnamed)'}",
            f"{'=' * 50}",
            f"  Total: {summary.total_pages} pages",
            f"  Passed: {summary.passed} ({summary.overall_pass_rate}%)",
            f"    - First pass: {summary.passed - summary.retry_passed}",
            f"    - After retry: {summary.retry_passed}",
            f"  Manual review needed: {summary.needs_manual_review}",
            f"  Avg render quality: {summary.avg_render_quality}",
            f"  Avg translation: {summary.avg_translation_status}",
            f"{'=' * 50}",
        ]

        if summary.manual_review_pages:
            lines.append("")
            lines.append("  Pages needing manual review:")
            for p in summary.manual_review_pages:
                lines.append(f"    #{p.page}  quality={p.render_quality}  "
                           f"translation={p.translation_status}  "
                           f"reason={p.retry_reason}")

        if summary.pages:
            lines.append("")
            lines.append("  All pages:")
            lines.append(f"    {'Page':<6} {'Quality':<10} {'Trans':<10} {'Score':<8} {'Retry':<6} Notes")
            lines.append(f"    {'-'*6} {'-'*10} {'-'*10} {'-'*8} {'-'*6} {'-'*20}")
            for p in summary.pages:
                notes = p.retry_reason[:20] if p.needs_retry else ""
                lines.append(
                    f"    #{p.page:<4} {p.render_quality:<9} {p.translation_status:<10} "
                    f"{p.overall_score:<7} {p.retry_count:<5} {notes}"
                )

        result = "\n".join(lines)
        logger.info(f"\n{result}")
        return result

    # ============================================================
    # 单页重跑
    # ============================================================

    async def retry_page(
        self,
        page: int,
        report: PageQualityReport,
        project_name: str,
        comic_dir: str,
        ocr_service: BatchOCRService,
        llm_client: LLMClient,
        prompt_builder: BatchPromptBuilder,
        glossary_service: GlossaryService,
        cache: BatchCacheService,
        render_service: RenderQualityService | None = None,
        source_lang: str = "en",
        target_lang: str = "zh",
        comic_style: str = "western_comic",
        render_output_dir: str = "data/output/rendered",
    ) -> PageQualityReport | None:
        """单页重跑：OCR → 翻译（带失败上下文）→ 渲染 → 复检

        Args:
            page: 页码（1-based）
            report: 该页当前的质检报告
            project_name: 项目名称
            comic_dir: 漫画图片目录
            ocr_service: OCR 服务
            llm_client: LLM 客户端
            prompt_builder: Prompt 构建器
            glossary_service: 术语表服务
            cache: 缓存服务
            render_service: 渲染服务（可选，不提供则只做翻译）
            source_lang: 源语言
            target_lang: 目标语言
            comic_style: 漫画风格
            render_output_dir: 渲染输出目录
        """
        from pathlib import Path

        from comic_translator.models.batch_result import BatchInput, BatchPage

        # 1. 获取该页的图片
        images = ocr_service.collect_images(comic_dir)
        page_images = [(p, img) for p, img in images if p == page]
        if not page_images:
            logger.error(f"Page {page} image not found in {comic_dir}")
            return None

        _, img_path = page_images[0]

        # 2. 单页 OCR
        logger.info(f"[retry] OCR page {page}...")
        from PIL import Image

        image = Image.open(img_path)
        raw_blocks = await ocr_service._provider.detect_text(image)
        normalized = ocr_service._normalize.process(raw_blocks, page=page, language=source_lang)

        if ocr_service._postprocess:
            result = ocr_service._postprocess.process_page(normalized, page=page)
            processed_blocks = result["blocks"]
        else:
            processed_blocks = normalized

        batch_page = BatchPage(page=page, blocks=processed_blocks)

        # 3. 构建带失败上下文的重跑 prompt
        glossary_entries = glossary_service.get_all_entries()
        retry_reason = report.retry_reason or "quality below threshold"

        batch_input = BatchInput(
            batch_index=0,
            pages=[batch_page],
            glossary=glossary_entries,
            project_name=project_name,
            comic_style=comic_style,
            source_language=source_lang,
            target_language=target_lang,
            entity_context=f"RETRY: previous translation had issues: {retry_reason}",
        )

        system_prompt = prompt_builder.SYSTEM_PROMPT
        user_prompt = prompt_builder.build_retry_prompt(
            batch_input, retry_reason=retry_reason
        )

        # 4. 调用 LLM
        logger.info(f"[retry] Translating page {page} (reason: {retry_reason})...")
        response = await llm_client.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.0,
            max_tokens=8192,
        )
        logger.info(f"[retry] API response received ({len(response)} chars)")

        # 5. 解析结果
        batch_output = prompt_builder.parse_response(response, batch_input)

        # 6. 保存新 glossary
        if batch_output.glossary:
            for source, target in batch_output.glossary.items():
                existing = glossary_service.search_terms(source)
                if not existing:
                    from comic_translator.models.glossary import GlossaryEntry

                    glossary_service.add_term(
                        GlossaryEntry(
                            source=source, target=target, category="general"
                        )
                    )

        # 7. 如果提供了渲染服务，执行渲染
        pt = batch_output.pages[0] if batch_output.pages else None
        if render_service and pt is not None:
            # 构建 DialogueBlock 用于渲染
            dialogue_blocks = self._build_dialogue_blocks(pt, batch_page)
            if dialogue_blocks:
                render_service.render_page(
                    image=image,
                    page=page,
                    dialogue_blocks=dialogue_blocks,
                    output_dir=render_output_dir,
                    source_path=str(img_path),
                    run_qa=True,
                )

                # 重新生成质检报告
                # 兼容两种写盘布局：
                #   RenderService:      rendered/page_NNNN/render_debug.json
                #   RenderQualityService: rendered/_debug/page_NNNN/render_debug.json
                page_num = f"page_{page:04d}"
                candidates = [
                    Path(render_output_dir) / page_num / "render_debug.json",
                    Path(render_output_dir) / "_debug" / page_num / "render_debug.json",
                    Path(render_output_dir) / f"{page_num}_render_debug.json",
                ]
                render_debug_path = next((p for p in candidates if p.exists()), None)
                if render_debug_path is not None:
                    new_data = json.loads(render_debug_path.read_text(encoding="utf-8"))
                    new_report = self._parse_debug_to_report(new_data, render_debug_path)
                    new_report.retry_count = report.retry_count + 1
                    new_report.retry_history = [*report.retry_history, RetryRecord(attempt=report.retry_count + 1, reason=retry_reason, translation_status=new_report.translation_status, render_quality=new_report.render_quality, prompt_snippet=retry_reason[:100])]
                    return new_report

        # 8. 无渲染服务，只返回翻译级别的质检
        source_len = sum(len(b.source_text) for b in pt.blocks) if batch_output.pages else 0
        trans_len = sum(len(b.translated_text) for b in pt.blocks) if batch_output.pages else 0
        length_ratio = trans_len / source_len if source_len > 0 else 0.0

        # 检查翻译状态
        ti_status = "PASS"
        ti_issues = []
        for b in pt.blocks if batch_output.pages else []:
            if not b.translated_text.strip():
                ti_issues.append(f"Empty translation: {b.id}")
                ti_status = "FAIL"
            elif b.translated_text == b.source_text:
                ti_issues.append(f"Unchanged: {b.id}")
                if ti_status != "FAIL":
                    ti_status = "WARNING"

        new_report = PageQualityReport(
            page=page,
            ocr_block_count=len(processed_blocks),
            translation_status=ti_status,
            translation_issues=ti_issues,
            translated_block_count=len(pt.blocks) if batch_output.pages else 0,
            block_count_diff=max(0, len(processed_blocks) - len(pt.blocks)) if batch_output.pages else len(processed_blocks),
            length_ratio=round(length_ratio, 2),
            overall_score=round(self._compute_overall_score(0, ti_status, length_ratio, 0, 0), 1),
            retry_count=report.retry_count + 1,
            retry_history=[*report.retry_history, RetryRecord(attempt=report.retry_count + 1, reason=retry_reason, translation_status=ti_status, render_quality=0.0)],
        )
        new_report.needs_retry, new_report.retry_reason = self._check_needs_retry(
            new_report.render_quality,
            new_report.translation_status,
            new_report.length_ratio,
            new_report.block_count_diff,
            new_report.overall_score,
        )
        return new_report

    async def auto_retry_pages(
        self,
        summary: QualityReportSummary,
        project_name: str,
        comic_dir: str,
        ocr_service: BatchOCRService,
        llm_client: LLMClient,
        prompt_builder: BatchPromptBuilder,
        glossary_service: GlossaryService,
        cache: BatchCacheService,
        render_service: RenderQualityService | None = None,
        source_lang: str = "en",
        target_lang: str = "zh",
        comic_style: str = "western_comic",
        render_output_dir: str = "data/output/rendered",
        max_retries: int | None = None,
    ) -> QualityReportSummary:
        """自动重跑所有不合格页面

        Returns:
            更新后的质检报告
        """
        max_retries = max_retries or self._thresholds["max_retries"]
        retry_pages = [p for p in summary.pages if p.needs_retry and p.retry_count < max_retries]

        if not retry_pages:
            logger.info("No pages need retry")
            return summary

        logger.info(f"Auto-retrying {len(retry_pages)} pages...")

        updated = {p.page: p for p in summary.pages}

        for report in retry_pages:
            logger.info(f"\n--- Retry page {report.page} (attempt {report.retry_count + 1}/{max_retries}) ---")
            new_report = await self.retry_page(
                page=report.page,
                report=report,
                project_name=project_name,
                comic_dir=comic_dir,
                ocr_service=ocr_service,
                llm_client=llm_client,
                prompt_builder=prompt_builder,
                glossary_service=glossary_service,
                cache=cache,
                render_service=render_service,
                source_lang=source_lang,
                target_lang=target_lang,
                comic_style=comic_style,
                render_output_dir=render_output_dir,
            )

            if new_report:
                updated[report.page] = new_report
                if not new_report.needs_retry:
                    logger.info(f"  Page {report.page} passed after retry!")
                else:
                    logger.warning(f"  Page {report.page} still failing: {new_report.retry_reason}")

        # 重新生成汇总
        pages = sorted(updated.values(), key=lambda p: p.page)
        total = len(pages)
        passed = sum(1 for p in pages if not p.needs_retry)
        manual = [p for p in pages if p.retry_count >= max_retries and p.needs_retry]
        avg_render = sum(p.render_quality for p in pages) / total if total else 0.0

        return QualityReportSummary(
            project_name=project_name,
            total_pages=total,
            passed=passed,
            retry_passed=sum(1 for p in pages if p.retry_count > 0 and not p.needs_retry),
            needs_manual_review=len(manual),
            overall_pass_rate=round(passed / total * 100, 1) if total else 0.0,
            avg_render_quality=round(avg_render, 1),
            pages=pages,
            manual_review_pages=manual,
        )

    def _build_dialogue_blocks(self, pt, batch_page):
        """从翻译结果构建 DialogueBlock 列表（用于渲染）"""
        from comic_translator.models.bubble import DialogueBlock

        blocks = []
        for tb in pt.blocks:
            db = DialogueBlock(
                id=tb.id,
                merged_text=tb.source_text,
                translated_text=tb.translated_text,
                translation_status="success" if tb.translated_text else "failed",
                render_text=tb.translated_text,
                render_status="pending",
            )
            blocks.append(db)
        return blocks
