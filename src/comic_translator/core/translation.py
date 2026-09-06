"""核心管线翻译模块 —— 翻译阶段的全部逻辑（CLI 与 GUI 共用）

原为 `gui_pipeline_service.py` 中的一批 `@staticmethod`/`@classmethod`，为拆分
单体文件而抽取到 `PipelineTranslationMixin`。方法不依赖实例状态，仅通过
DI 容器获取服务，因此全部为静态/类方法，可被 `ComicPipelineService` 继承复用，
也保持 `_translate_dialogues` 等公共 API 不变（测试直接调用）。

包含：
- 翻译上下文构建（术语表 + 角色身份，按项目隔离）
- 自动术语表提取
- 翻译记忆 + 页面翻译缓存复用
- LLM 批量翻译（统一走 DI 容器的 LLMClient，无直接 HTTP 降级路径）
- 失败气泡逐个重试
- 全局限流/重试计数器（供性能面板读取）
"""

from __future__ import annotations

import asyncio
import re
import threading
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from typing import Any

    from comic_translator.clients.base import LLMClient
    from comic_translator.models.bubble import DialogueBlock


# ---- 全局性能计数器（供翻译路径计数 429/重试，跨线程读写） ----
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


# 匹配行首序号前缀（"1. " / "12、" / "3)" 等），仅剥除一次。
# 点号后不能紧跟数字（负向前瞻），避免把 "3.14是…" 这类以小数开头的
# 正常译文误剥成 "14是…"
_NUMBERING_PREFIX_RE = re.compile(r"^\s*\d{1,3}\s*[.、)）]\s*(?!\d)")


def _parse_glossary_json(raw: str) -> list[dict]:
    """容错解析 LLM 返回的术语 JSON（支持 ```json 代码块包裹）"""
    import json
    import re
    if not raw:
        return []
    text = raw.strip()
    m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
    if m:
        text = m.group(1).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    try:
        data = json.loads(text)
    except Exception:
        return []
    if not isinstance(data, dict):
        return []
    entries = data.get("entries") or []
    return [e for e in entries if isinstance(e, dict)]


class PipelineTranslationMixin:
    """翻译阶段逻辑 —— 以 Mixin 挂在 ComicPipelineService 上。"""

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

    @classmethod
    async def _save_auto_glossary(
        cls,
        texts: list[str],
        translations: list[str],
        project_id: str = "",
    ) -> None:
        """翻译成功后，用 LLM 从源文本+译文对中抽取专有名词及真实译名，存入术语表

        质量规则（杜绝英文→英文垃圾词条）：
        - 只保存「源词 → 真实译文」且译文 ≠ 源词的词条
        - 只提取真正的专有名词（角色名/地名/特殊术语）
        - 已存在的源词跳过（幂等，跨页去重）
        - 无 LLM / 容器缺服务 / 解析失败时静默跳过，绝不降级为自指词条
        """
        if not project_id or not texts or not translations:
            return

        try:
            from comic_translator.models.glossary import GlossaryEntry
            from comic_translator.utils.di import get_container
            container = get_container()
            llm_client = container.get_llm_client()
            glossary_service = container.get("glossary_service")
        except Exception as e:
            logger.debug(f"Auto-glossary init failed (skip): {e}")
            return

        if llm_client is None or glossary_service is None:
            return

        # 幂等：已有源词集合
        existing = {
            e.source.lower()
            for e in glossary_service.get_all_entries(project_id=project_id)
        }

        # 组装平行文本（每页通常几十句，取前 40 对避免超长）
        pairs = []
        for i, (s, t) in enumerate(zip(texts, translations, strict=False)):
            if s.strip() and t.strip():
                pairs.append(f"{i + 1}. SOURCE: {s.strip()}\n   TARGET: {t.strip()}")
        if not pairs:
            return

        prompt = (
            "下面是漫画对白（SOURCE）及其翻译（TARGET）的平行文本。\n"
            + "\n".join(pairs[:40])
            + "\n\n请提取其中出现的专有名词（角色名、地名、特殊术语），"
            "并给出它们在 TARGET 中的译名。\n"
            "要求：\n"
            "1. 只提取真正的专有名词；不要提取普通单词（如 Three、Chapter、Yes、Wait）。\n"
            "2. source 用英文原名；target 用译文里实际出现的译名。\n"
            "3. 若某专名在 TARGET 中没有对应译名或不确定，不要输出该词条。\n"
            "4. 输出 JSON：{\"entries\": [{\"source\": \"...\", \"target\": \"...\"}]}\n"
            "只输出 JSON，不要任何其它文字或解释。"
        )
        try:
            raw = await llm_client.generate(
                prompt,
                system_prompt="你是漫画翻译的术语提取助手，只输出 JSON。",
                temperature=0.0,
                max_tokens=2000,
            )
        except Exception as e:
            logger.debug(f"Auto-glossary LLM call failed: {e}")
            return

        entries = _parse_glossary_json(raw)
        added = 0
        for e in entries:
            source = (e.get("source") or "").strip()
            target = (e.get("target") or "").strip()
            if not source or not target:
                continue
            if source.lower() == target.lower():
                continue  # 拒绝英文→英文
            if source.lower() in existing:
                continue
            try:
                glossary_service.add_term(
                    GlossaryEntry(source=source, target=target, category="general", priority=1),
                    project_id=project_id,
                )
                existing.add(source.lower())
                added += 1
            except Exception:
                continue

        if added:
            logger.info(f"Auto-glossary [LLM {project_id}]: saved {added} entries")

    @classmethod
    async def _translate_dialogues(
        cls,
        dialogues: list[DialogueBlock],
        target_lang: str = "zh",
        project_id: str = "",
        page: int | None = None,
    ):
        """调用 LLM 翻译，并记录每个 DialogueBlock 的翻译状态。

        统一走 DI 容器的 LLMClient（不再有直接 HTTP 降级路径）。
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

        use_reuse = cls._use_reuse_enabled()

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
            # ---- 提速：待翻译文本去重（相同原文只翻一次，映射回所有同文气泡） ----
            # 同页/跨页常见的重复文本（"HAHA"、"!!!"、重复台词）只发一次 LLM，
            # 既省请求数（rps 受限时收益大），又保证同文翻译一致。
            unique_texts: list[str] = []
            unique_to_dups: dict[str, list[int]] = {}
            for pos, txt in enumerate(remaining_texts):
                if txt not in unique_to_dups:
                    unique_to_dups[txt] = [pos]
                    unique_texts.append(txt)
                else:
                    unique_to_dups[txt].append(pos)
            deduped = len(remaining_texts) - len(unique_texts)
            unique_index_map = [remaining_index_map[unique_to_dups[t][0]] for t in unique_texts]
            if deduped:
                logger.info(
                    f"Translation dedup: {deduped} duplicate text(s) skipped "
                    f"(unique {len(unique_texts)}/{len(remaining_texts)})"
                )

            # 构建翻译上下文（术语表 + 角色身份，按项目隔离）
            context_str = cls._build_translation_context(project_id=project_id)

            # 统一从 DI 容器获取 LLMClient（不再有直接 HTTP 降级路径）
            llm_client = None
            try:
                from comic_translator.utils.di import get_container
                container = get_container()
                llm_client = container.get_llm_client()
            except Exception as e:
                logger.error(f"DI container LLM client unavailable, translation skipped: {e}")

            if llm_client is not None:
                await cls._translate_via_llm_client(
                    dialogues, unique_texts, unique_index_map,
                    target_name, llm_client, context_str, project_id,
                )
            else:
                for d in dialogues:
                    if d.merged_text.strip():
                        d.translated_text = ""
                        d.translation_status = "failed"
                    else:
                        d.translation_status = "skipped"

            # 把去重后的译文/状态复制回所有同文气泡
            if deduped:
                for u, positions in enumerate(unique_to_dups.values()):
                    if len(positions) < 2:
                        continue
                    first_d = dialogues[unique_index_map[u]]
                    for pos in positions[1:]:
                        d = dialogues[remaining_index_map[pos]]
                        d.translated_text = first_d.translated_text
                        d.translation_status = first_d.translation_status

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

    # ---- 角色一致性：说话人归属 → 自动填充角色记忆 ----

    @classmethod
    def _speaker_attribution_enabled(cls) -> bool:
        """读取 config.yaml 中 translation.speaker_attribution（进程内缓存，仅首次读文件）"""
        if not hasattr(cls, "_speaker_attr_cache"):
            try:
                from pathlib import Path

                import yaml
                cfg = {}
                cfg_path = Path("config.yaml")
                if cfg_path.exists():
                    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
                enabled = bool(cfg.get("translation", {}).get("speaker_attribution", True))
            except Exception:
                enabled = True
            cls._speaker_attr_cache = enabled
        return cls._speaker_attr_cache

    @staticmethod
    def _parse_attribution_json(raw: str) -> dict:
        """容错解析说话人归属 JSON（支持 ```json 包裹）"""
        import json
        if not raw:
            return {}
        text = raw.strip()
        m = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL)
        if m:
            text = m.group(1).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            text = text[start:end + 1]
        try:
            data = json.loads(text)
        except Exception:
            return {}
        return data if isinstance(data, dict) else {}

    @classmethod
    async def _save_speaker_attribution(
        cls,
        dialogues: list[DialogueBlock],
        llm_client: LLMClient,
        char_service: Any,
        project_id: str = "",
        page: int | None = None,
    ) -> None:
        """LLM 归属每句话的说话人，并把新角色写入角色记忆（供后续页注入上下文）。

        best-effort：任何失败静默跳过，绝不影响翻译主流程。
        """
        known = char_service.get_all_identities(project_id=project_id or None)
        known_names = {c.canonical_name.lower() for c in known}
        for c in known:
            known_names.update(a.lower() for a in (c.aliases or []))

        lines = []
        for d in dialogues:
            txt = d.merged_text.strip()
            if txt:
                lines.append(txt)

        prompt = (
            "下面是漫画某页按阅读顺序排列的对白（每行前有序号）。\n"
            + (f"已知角色：{', '.join(sorted(known_names))}\n" if known_names else "（尚无已知角色）\n")
            + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(lines))
            + "\n\n请判断每句话的说话人，并识别新出现的具名角色。规则：\n"
            "1. 说话人只能是已知角色，或从对白/称呼中能确定的新具名角色；不确定就填 \"unknown\"。\n"
            "2. 新角色必须是有名字的角色（对白中自称或被称呼），不要把普通词当角色。\n"
            "3. 只输出 JSON：{\"speakers\": [\"角色A\", \"unknown\", ...],"
            "\"new_characters\": [{\"name\": \"...\", \"aliases\": [...],"
            "\"speaking_style\": \"...\", \"description\": \"...\"}]}\n"
            "speakers 数组必须与对白行数一致。只输出 JSON。"
        )
        try:
            raw = await llm_client.generate(
                prompt=prompt,
                system_prompt="你是漫画说话人归属助手，只输出 JSON。",
                temperature=0.0,
                max_tokens=1500,
            )
        except Exception as e:
            logger.debug(f"Speaker attribution LLM call failed: {e}")
            return

        data = cls._parse_attribution_json(raw)
        speakers = data.get("speakers")
        new_chars = data.get("new_characters") or []

        # 写入新角色（去重：与已知角色及本批新角色不重名）
        import re as _re
        added = 0
        for c in new_chars:
            if not isinstance(c, dict):
                continue
            name = (c.get("name") or "").strip()
            if not name or len(name) > 40 or name.lower() in known_names:
                continue
            aliases = [a.strip() for a in (c.get("aliases") or []) if isinstance(a, str) and a.strip()]
            character_id = "auto-" + _re.sub(r"[^a-z0-9_-]", "_", name.lower())[:40]
            try:
                from comic_translator.models.character_identity import CharacterIdentity
                char_service.create_identity(
                    CharacterIdentity(
                        character_id=character_id,
                        canonical_name=name,
                        aliases=aliases,
                        speaking_style=(c.get("speaking_style") or "").strip(),
                        description=(c.get("description") or "").strip(),
                        first_seen_page=page or 0,
                        last_seen_page=page or 0,
                    ),
                    project_id=project_id or None,
                )
                known_names.add(name.lower())
                added += 1
            except Exception as e:
                logger.debug(f"Failed to save character identity '{name}': {e}")

        if speakers and len(speakers) == len(lines):
            logger.info(
                f"Speaker attribution [page {page}]: {len(lines)} lines, "
                f"{sum(1 for s in speakers if s and s != 'unknown')} attributed, "
                f"{added} new character(s) saved"
            )
        elif added:
            logger.info(f"Speaker attribution: {added} new character(s) saved (project={project_id})")

    @classmethod
    async def _maybe_attribute_speakers(
        cls, dialogues: list[DialogueBlock], project_id: str, page: int | None,
    ) -> None:
        """说话人归属入口：配置开关 + 服务/客户端获取（best-effort）"""
        if not project_id or not cls._speaker_attribution_enabled():
            return
        try:
            from comic_translator.utils.di import get_container
            container = get_container()
            char_service = container.get("character_identity_service")
            llm_client = container.get_llm_client()
        except Exception as e:
            logger.debug(f"Speaker attribution unavailable: {e}")
            return
        if llm_client is None or char_service is None:
            return
        await cls._save_speaker_attribution(
            dialogues, llm_client, char_service, project_id=project_id, page=page,
        )

    @classmethod
    def _use_reuse_enabled(cls) -> bool:
        """读取 config.yaml 中 translation.use_cache（进程内缓存，仅首次读文件）"""
        if not hasattr(cls, "_reuse_enabled_cache"):
            cls._reuse_enabled_cache = cls._load_use_translation_cache()
        return cls._reuse_enabled_cache

    @classmethod
    async def _translate_via_llm_client(
        cls,
        dialogues: list[DialogueBlock],
        texts: list[str],
        index_map: list[int],
        target_name: str,
        llm_client: LLMClient,
        context_str: str = "",
        project_id: str = "",
    ):
        """通过 LLMClient 翻译（唯一路径）"""
        # 构建带上下文的 system prompt
        context_parts = []
        if context_str.strip():
            context_parts.append(context_str)
        context_parts.append(
            "You are a professional comic translator. "
            f"Translate the following comic text blocks to {target_name}. "
            "Keep the tone natural for comics. "
            "Correct obvious OCR errors in the source text. "
            "Return one translation per line, in the same order as the input. "
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
                translated_lines = [line.strip() for line in response.strip().split("\n") if line.strip()]
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

        cls._apply_translation_results(
            dialogues, texts, index_map, translated_lines, status,
        )

        # 翻译成功后自动提取术语表（LLM 抽取真实译名）
        if status == "success" and project_id:
            await cls._save_auto_glossary(texts, translated_lines, project_id)
            # 角色一致性：说话人归属 → 新角色写入角色记忆（后续页自动注入上下文）
            await cls._maybe_attribute_speakers(dialogues, project_id, page=None)

        # 对翻译失败的块逐个重试（不管整体状态，只要有失败的块就重试）
        failed_dialogues = [
            d for d in dialogues
            if d.merged_text.strip() and d.translation_status == "failed"
        ]
        if failed_dialogues:
            logger.info(f"Retrying {len(failed_dialogues)} individually failed blocks...")
            for d in failed_dialogues:
                source = d.merged_text.strip()
                single = await cls._translate_single(source, target_name, llm_client)
                if single:
                    d.translated_text = single
                    d.translation_status = "unchanged" if single == source else "success"
                else:
                    # 失败：不要塞原文当译文，让渲染器按 fallback_to_source 决定
                    # （显示 TRANSLATION FAILED 或回退原文），避免"看似已翻译实为英文"
                    d.translated_text = ""
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
                # 只剥掉行首的"序号前缀"（如 "1. "、"12、 "），
                # 不能用 lstrip("0123456789.- ") —— 那会把以数字/点/横杠
                # 开头的正常译文（"3.14是…"、"——好极了"）也剥坏
                translated = _NUMBERING_PREFIX_RE.sub("", translated_lines[idx], count=1).strip()
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
    async def _translate_single(text: str, target_name: str, llm_client: LLMClient) -> str:
        """单个文本翻译"""
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

    @classmethod
    async def retry_page_failed_bubbles(
        cls,
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
                    single = await cls._translate_single(source, target_name, llm_client)
                else:
                    single = ""
                if single:
                    d.translated_text = single
                    d.translation_status = "unchanged" if single == source else "success"
                    retried += 1
                else:
                    # 最后保底：不塞原文当译文，交由渲染器按 fallback_to_source
                    # 显示 TRANSLATION FAILED 或回退原文
                    d.translated_text = ""
                    d.translation_status = "failed"
            except Exception as e:
                logger.error(f"Retry failed for block '{source[:30]}...': {e}")
                d.translated_text = ""
                d.translation_status = "failed"
            await asyncio.sleep(0.2)

        logger.info(f"Retried {retried}/{len(failed)} failed blocks (project={project_id})")
        return retried
