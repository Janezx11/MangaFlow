"""质检闭环的候选页选择逻辑测试

覆盖 ComicPipelineService._select_retry_candidates 的保守重试策略:
- needs_retry + 内存中存在 failed 翻译气泡 → 自动重试
- needs_retry 但翻译全部成功（纯渲染质量问题）→ 留给人工审核
- retry_count 已达上限 → 不再重试
- 内存中无该页结果（如历史渲染）→ 跳过
"""

from __future__ import annotations

from comic_translator.core.pipeline import ComicPipelineService, PageResult
from comic_translator.models.bubble import Bubble, DialogueBlock
from comic_translator.models.quality_report import PageQualityReport, QualityReportSummary


def _dialogue(status: str, text: str = "Hello world") -> DialogueBlock:
    bubble = Bubble(
        id="b1",
        bbox=(0, 0, 100, 60),
        polygon=[(0, 0), (100, 0), (100, 60), (0, 60)],
        mask=[],
        class_name="speech",
    )
    return DialogueBlock(id="b1", bubble=bubble, merged_text=text, translation_status=status)


def _report(page: int, needs_retry: bool, retry_count: int = 0) -> PageQualityReport:
    return PageQualityReport(
        page=page,
        translation_status="FAIL" if needs_retry else "PASS",
        needs_retry=needs_retry,
        retry_reason="translation=FAIL" if needs_retry else "",
        retry_count=retry_count,
    )


def _service_with_results(results: dict[int, PageResult]) -> ComicPipelineService:
    svc = ComicPipelineService(history_service=None)
    svc._results = results
    return svc


class TestSelectRetryCandidates:
    def test_failed_translation_page_is_selected(self):
        svc = _service_with_results({
            1: PageResult(page=1, file_name="1.jpg", dialogues=[_dialogue("failed")]),
        })
        summary = QualityReportSummary(pages=[_report(1, needs_retry=True)])

        candidates, render_only = svc._select_retry_candidates(summary, max_retries=3)

        assert [r.page for r in candidates] == [1]
        assert render_only == 0

    def test_render_only_page_left_for_manual_review(self):
        svc = _service_with_results({
            2: PageResult(page=2, file_name="2.jpg", dialogues=[_dialogue("success")]),
        })
        summary = QualityReportSummary(pages=[_report(2, needs_retry=True)])

        candidates, render_only = svc._select_retry_candidates(summary, max_retries=3)

        assert candidates == []
        assert render_only == 1

    def test_retry_count_limit_excluded(self):
        svc = _service_with_results({
            3: PageResult(page=3, file_name="3.jpg", dialogues=[_dialogue("failed")]),
        })
        summary = QualityReportSummary(pages=[_report(3, needs_retry=True, retry_count=3)])

        candidates, render_only = svc._select_retry_candidates(summary, max_retries=3)

        assert candidates == []
        assert render_only == 0  # 超过重试上限的页不计入"渲染问题"

    def test_page_without_inmemory_result_skipped(self):
        svc = _service_with_results({})  # 无内存结果（如重跑时的历史页）
        summary = QualityReportSummary(pages=[_report(4, needs_retry=True)])

        candidates, render_only = svc._select_retry_candidates(summary, max_retries=3)

        assert candidates == []
        assert render_only == 0

    def test_healthy_page_ignored(self):
        svc = _service_with_results({
            5: PageResult(page=5, file_name="5.jpg", dialogues=[_dialogue("success")]),
        })
        summary = QualityReportSummary(pages=[_report(5, needs_retry=False)])

        candidates, render_only = svc._select_retry_candidates(summary, max_retries=3)

        assert candidates == []
        assert render_only == 0

    def test_empty_dialogues_counts_as_render_only(self):
        # 页面有 merged_text 为空的对话块（无内容可译）→ 翻译重试无意义
        svc = _service_with_results({
            6: PageResult(page=6, file_name="6.jpg", dialogues=[_dialogue("pending", text="")]),
        })
        summary = QualityReportSummary(pages=[_report(6, needs_retry=True)])

        candidates, render_only = svc._select_retry_candidates(summary, max_retries=3)

        assert candidates == []
        assert render_only == 1
