"""回归测试：GUI 翻译复用逻辑（提速②）

覆盖 GuiPipelineService._translate_dialogues 的：
1. 语义翻译记忆精确命中 → 复用译文、跳过 LLM
2. 页面翻译缓存命中 → 复用译文、跳过 LLM
3. 未命中的文本按原样走 LLM（一次批量调用，行数与剩余数一致）
4. 翻译成功后回写语义 TM + 页面翻译缓存

用假服务隔离，不触真实 LLM / DB / DI 容器。
"""

import pytest

from comic_translator.gui.services.gui_pipeline_service import GuiPipelineService
from comic_translator.models.bubble import Bubble, DialogueBlock
from comic_translator.models.glossary import TranslationMemory
from comic_translator.models.translation_block import TranslationBlock


class _FakeTM:
    """记录 store 调用；find_similar 返回预设的精确命中"""

    def __init__(self, hits: dict[str, str]):
        self.hits = hits
        self.stored: list[tuple[str, str]] = []

    def find_similar(self, text: str, limit: int = 5) -> list[TranslationMemory]:
        t = self.hits.get(text)
        if t:
            return [
                TranslationMemory(
                    source_text=text, translated_text=t, similarity=1.0,
                    provider="test", model="", context_hash="", usage_count=1,
                )
            ]
        return []

    def store(self, source_text, translated_text, provider="", model="", context_hash=""):
        self.stored.append((source_text, translated_text))


class _FakeCache:
    def __init__(self):
        self.cache: dict[tuple[str, int], list[TranslationBlock]] = {}
        self.writes: list[tuple[str, int, int]] = []

    async def get_translation_cache(self, project_id: str, page: int):
        return self.cache.get((project_id, page))

    async def set_translation_cache(self, project_id: str, page: int, blocks):
        self.writes.append((project_id, page, len(blocks)))
        self.cache[(project_id, page)] = blocks


class _FakeLLM:
    """假 LLM：解析提示词要求的行数，返回相同数量译文"""

    def __init__(self):
        self.calls: list[str] = []

    async def generate(self, prompt, system_prompt=None, temperature=0.0, max_tokens=4096):
        self.calls.append(prompt)
        import re
        m = re.search(r"Translate the following (\d+) comic text blocks", prompt)
        n = int(m.group(1)) if m else 1
        return "\n".join(f"TR-{i + 1}" for i in range(n))


class _FakeContainer:
    def __init__(self, tm, cache, llm):
        self._tm = tm
        self._cache = cache
        self._llm = llm

    def get(self, key):
        if key == "cache_service":
            return self._cache
        if key == "semantic_tm_service":
            return self._tm
        raise KeyError(key)

    def get_llm_client(self):
        return self._llm


def _make_dialogue(did: str, text: str) -> DialogueBlock:
    b = Bubble(
        id=did, bbox=(0, 0, 10, 10),
        polygon=[(0, 0), (10, 0), (10, 10), (0, 10)],
        mask=[], class_name="speech",
    )
    return DialogueBlock(id=did, bubble=b, merged_text=text)


@pytest.mark.asyncio
async def test_translate_dialogues_reuses_tm_and_cache(monkeypatch):
    import comic_translator.utils.di as di

    tm = _FakeTM(hits={"Hello world": "你好世界"})
    cache = _FakeCache()
    llm = _FakeLLM()
    container = _FakeContainer(tm, cache, llm)
    monkeypatch.setattr(di, "get_container", lambda db_path="": container)

    # 预置页面缓存命中（项目 demo、第 1 页）
    cache.cache[("demo", 1)] = [
        TranslationBlock(id="d2", ocr_block_id="d2", source_text="Second line", translated_text="第二行")
    ]

    dialogues = [
        _make_dialogue("d1", "Hello world"),   # TM 精确命中
        _make_dialogue("d2", "Second line"),   # 页面缓存命中
        _make_dialogue("d3", "Third line"),    # 需走 LLM
    ]

    await GuiPipelineService._translate_dialogues(
        dialogues, target_lang="zh", project_id="demo", page=1,
    )

    # 命中复用
    assert dialogues[0].translated_text == "你好世界"
    assert dialogues[0].translation_status == "success"
    assert dialogues[1].translated_text == "第二行"
    assert dialogues[1].translation_status == "success"
    # 未命中走 LLM
    assert dialogues[2].translated_text.startswith("TR-")
    assert dialogues[2].translation_status == "success"

    # LLM 只调用一次，且只收到未命中的文本
    assert len(llm.calls) == 1
    assert "Third line" in llm.calls[0]
    assert "Hello world" not in llm.calls[0]
    assert "Second line" not in llm.calls[0]

    # 回写：TM 存了 3 条（含复用项与新译文），页面缓存回写一次
    tm_srcs = {s for s, _ in tm.stored}
    assert {"Hello world", "Second line", "Third line"} <= tm_srcs
    assert cache.writes and cache.writes[0][:2] == ("demo", 1)


@pytest.mark.asyncio
async def test_translate_dialogues_all_reused_no_llm_call(monkeypatch):
    """整页全部命中 TM → 完全不调用 LLM"""
    import comic_translator.utils.di as di

    tm = _FakeTM(hits={"A": "甲", "B": "乙"})
    cache = _FakeCache()
    llm = _FakeLLM()
    container = _FakeContainer(tm, cache, llm)
    monkeypatch.setattr(di, "get_container", lambda db_path="": container)

    dialogues = [_make_dialogue("d1", "A"), _make_dialogue("d2", "B")]

    await GuiPipelineService._translate_dialogues(
        dialogues, target_lang="zh", project_id="demo", page=2,
    )

    assert dialogues[0].translated_text == "甲"
    assert dialogues[1].translated_text == "乙"
    assert llm.calls == []  # 一次 LLM 都没调用
    # 依然回写页面缓存
    assert any(w[0] == "demo" and w[1] == 2 for w in cache.writes)


@pytest.mark.asyncio
async def test_translate_dialogues_use_cache_disabled(monkeypatch):
    """translation.use_cache=false 时完全不查 TM/缓存，全部走 LLM"""
    import comic_translator.utils.di as di

    tm = _FakeTM(hits={"A": "甲"})
    cache = _FakeCache()
    llm = _FakeLLM()
    container = _FakeContainer(tm, cache, llm)
    monkeypatch.setattr(di, "get_container", lambda db_path="": container)
    # 模拟 use_cache=false
    monkeypatch.setattr(GuiPipelineService, "_reuse_enabled_cache", False, raising=False)

    dialogues = [_make_dialogue("d1", "A"), _make_dialogue("d2", "B")]

    await GuiPipelineService._translate_dialogues(
        dialogues, target_lang="zh", project_id="demo", page=1,
    )

    # 全部走了 LLM（即使 TM 本可命中）
    assert len(llm.calls) == 1
    assert dialogues[0].translated_text.startswith("TR-")
    # 未启用时不回写
    assert tm.stored == []
    assert cache.writes == []
