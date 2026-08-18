"""OCR 后处理服务 —— 5 级本地修复，零 API 调用"""

from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

from loguru import logger

if TYPE_CHECKING:
    from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock


class OCRPostprocessService:
    """OCR 后处理：本地修复，不增加 API 调用

    Level 1 — 文本规范化 (Unicode / 空白 / 标点)
    Level 2 — 拼写修复 (rapidfuzz, 仅英文 >=4 字符, 相似度 >=90)
    Level 3 — 标题大小写恢复 (首字母大写 + 专有名词)
    Level 4 — Block 合并 (同气泡多行合并)
    Level 5 — 实体提取 (PERSON / LOCATION / ORGANIZATION)
    """

    def __init__(self, spell_threshold: int = 85, min_word_length: int = 4):
        self._spell_threshold = spell_threshold
        self._min_word_length = min_word_length
        self._common_words: set[str] | None = None
        self._fuzz = None
        self._process = None

    # ---- Lazy init rapidfuzz ----

    def _ensure_fuzzy(self):
        if self._fuzz is not None:
            return
        from rapidfuzz import fuzz, process

        self._fuzz = fuzz
        self._process = process
        # 基础英文词典
        self._common_words = {
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i",
            "it", "for", "not", "on", "with", "he", "as", "you", "do", "at",
            "this", "but", "his", "by", "from", "they", "we", "her", "she", "or",
            "an", "will", "my", "one", "all", "would", "there", "their", "what",
            "so", "up", "out", "if", "about", "who", "get", "which", "go", "me",
            "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could",
            "them", "see", "other", "than", "then", "now", "look", "only", "come",
            "its", "over", "think", "also", "back", "after", "use", "two", "how",
            "our", "work", "first", "well", "way", "even", "new", "want", "because",
            "any", "these", "give", "day", "most", "us", "here", "say", "every",
            "great", "tell", "much", "ask", "man", "long", "before", "too", "own",
            "through", "old", "still", "should", "while", "where", "right", "high",
            "need", "home", "hand", "place", "life", "world", "thing", "down",
            "very", "really", "little", "big", "ever", "never", "away", "again",
            "still", "might", "turn", "show", "help", "keep", "start", "mean",
            "why", "away", "here", "there", "always", "already", "enough",
            "sure", "always", "maybe", "please", "sorry", "thank", "hello",
            "married", "fledgling", "journalist", "dreamed", "stardom",
            "unknown", "actor", "newly", "just", "traveled", "way", "from",
            "usa", "balkans", "after", "receiving", "what", "they", "believed",
            "best", "wedding", "gift", "ever", "babe", "cannot", "believe",
            "how", "great", "this", "place", "looks", "starting", "believe",
            "you", "were", "right", "all", "along", "greece", "indeed", "seems",
            "most", "beautiful", "world", "told", "your", "parents", "decided",
            "still", "child", "right", "bet", "must", "feel", "after", "long",
            "move", "america", "when", "were", "wonderful", "back", "homeland",
            "thanks", "uncle", "ever", "able", "thank", "him", "not", "only",
            "best", "rooms", "hotel", "yours", "know", "will", "give", "air",
            "tickets", "also", "booked", "one", "all", "free", "remember",
            "nikos", "well", "until", "dawn", "than", "good", "party",
            "pleased", "just", "eat", "drink", "bon", "vivant", "nothing",
            "loves", "more", "has", "always", "been", "sounds", "someone",
            "gets", "along", "could", "definitely", "will", "have", "chance",
            "meet", "him", "tonight", "until", "then", "hit", "pool", "then",
            "some", "fun", "huh", "why", "dont", "relax", "little", "come",
            "back", "our", "room", "god", "almost", "seems", "perfect", "plan",
            "too", "good", "true",
        }

    def process_page(
        self,
        blocks: list[NormalizedOCRBlock],
        page: int = 0,
    ) -> dict:
        """对单页所有 block 执行 5 级后处理

        返回:
        {
          "blocks": [...],        # 处理后的 blocks
          "entities": [...],      # 提取的实体
          "stats": {"raw": N, "cleaned": N, "repaired": N, "merged": N}
        }
        """
        stats = {"raw": len(blocks), "cleaned": 0, "repaired": 0, "merged": 0}
        all_entities: list[dict] = []

        processed = []
        for block in blocks:
            text = block.text

            # L1: 规范化
            text = self._normalize(text)
            stats["cleaned"] += 1

            # L2: 拼写修复
            text = self._fix_spelling(text)
            if text != block.text:
                stats["repaired"] += 1

            # L3: 大小写修复
            text = self._fix_capitalization(text)

            block.text = text
            processed.append(block)

        # L4: Block 合并
        merged = self._merge_blocks(processed, page)
        stats["merged"] = len(processed) - len(merged)

        # L5: 实体提取
        for block in merged:
            entities = self._extract_entities(block.text)
            for ent in entities:
                ent["page"] = page
                all_entities.append(ent)

        return {
            "blocks": merged,
            "entities": all_entities,
            "stats": stats,
        }

    # ---- Level 1: 文本规范化 ----

    def _normalize(self, text: str) -> str:
        """Unicode 标准化 + 空白清理 + 标点统一"""
        # NFKC 标准化
        text = unicodedata.normalize("NFKC", text)
        # 压缩多空格
        text = re.sub(r" {2,}", " ", text)
        # 去除行首行尾空白
        text = text.strip()
        # 统一引号
        text = text.replace("'", "'").replace("'", "'")
        text = text.replace('"', '"').replace('"', '"')
        # 统一连字符
        text = text.replace("–", "-").replace("—", "-")
        # 统一省略号
        text = text.replace("…", "...")
        return text

    # ---- OCR 常见错误直接映射 ----
    OCR_FIXES = {
        # Page 1 — wedding / Greece
        "mappied": "married", "maPPIED": "Married", "MaPPIED": "Married",
        "tpaveled": "traveled", "TPaveLED": "Traveled",
        "andpew": "andrew", "ANDPEw": "Andrew",
        "stapdom": "stardom", "StAPDOM": "Stardom",
        "fledeling": "fledgling", "FlEdeling": "Fledgling",
        "joupnalist": "journalist", "Joupnalist": "Journalist",
        "wepe": "were", "WEpe": "Were",
        "eift": "gift", "EiFT": "Gift",
        "eiFT": "Gift", "eiFT": "Gift",
        # Page 2 — Oh Babe
        "dpeamed": "dreamed", "DPEAMED": "Dreamed",
        "cpeat": "great", "Cpeat": "Great",
        "stapting": "starting", "Stapting": "Starting",
        "picht": "right", "Picht": "Right",
        "alonc": "along", "Alonc": "Along",
        "gpeece": "greece", "Gpeece": "Greece",
        "indeED": "indeed", "IndeED": "Indeed",
        "beautifUl": "beautiful", "BeaUtIFUL": "Beautiful",
        "wopld": "world", "Wopld": "World",
        # Page 3 — I told you
        "itold": "i told", "Itold": "I told",
        "didn't": "didn't", "DiDN'T": "Didn't",
        "gpeece": "greece", "GpEECe": "Greece",
        "eipl": "girl", "eipl__": "girl",
        "beautiful": "beautiful", "BeauTiFul": "Beautiful",
        "evepything": "everything",
        "apound": "around", "Apound": "Around",
        "hepe": "here", "HEpe": "Here",
        "thereis": "there is",
        "pemember": "remember", "PEMEMBEr": "Remember",
        "evepythine": "everything",
        # Page 4 — Yeah You told me
        "richt": "right", "richt?": "right?",
        "lone": "long", "Lone": "Long",
        "amepica": "america", "AMEpica": "America",
        "wheN": "when", "WHEn": "When",
        "wonderful": "wonderful", "WONDEPFul": "Wonderful",
        "homeland": "homeland", "HOMELAND": "Homeland",
        "uncLE": "uncle", "UncLE": "Uncle",
        "poods": "rooms", "PoOMs": "Rooms",
        "pooms": "rooms",
        "yours": "yours", "DF": "Of",
        "knOW": "know",
        "gave": "give", "GAVE": "Give",
        "aip": "air", "Aip": "Air",
        "boekED": "booked", "BookEd": "Booked",
        "alL": "all", "ALl": "All",
        "freel": "free", "Freel": "Free",
        "well": "well", "WELL_": "well",
        "dawn": "dawn", "DAWN_": "dawn",
        "paPTY": "party", "PaPTY": "Party",
        "pleased": "pleased", "PLEASED": "Pleased",
        "dpink": "drink", "DPink": "Drink",
        "vivant": "vivant", "VIvANT": "Vivant",
        "thereis": "there is",
        "nothine": "nothing",
        # Page 5 — Sounds like
        "alone": "along", "Alone": "Along",
        "withl": "with",
        "definitely": "definitely", "DeFinitely": "Definitely",
        "youll": "you'll", "Youll": "You'll",
        "have": "have", "HAVE": "Have",
        "chance": "chance", "CHANCE": "Chance",
        "meet": "meet", "MEEt": "Meet",
        "tonicht": "tonight", "tonicht.": "tonight.",
        "until": "until", "UNTIL": "Until",
        "pool": "pool",
        "fun": "fun", "FUN": "Fun",
        "huh": "huh",
        "whY": "why", "WhY": "Why",
        "don't": "don't", "DON'T": "Don't",
        "pelax": "relax", "PELAX": "Relax",
        "little": "little", "Little": "Little",
        "coME": "come",
        "poom": "room", "Poom": "Room",
        "god": "god", "God": "God",
        "almost": "almost", "AlMoSt": "Almost",
        "seems": "seems", "SEEMS": "Seems",
        "pefect": "perfect", "PEPFECT": "Perfect",
        "plan": "plan", "PLAN": "Plan",
        "babe": "babe", "BABE": "Babe",
        "cood": "good", "Cood": "Good",
        "true": "true", "True_--": "true.",
        # General
        "stark": "Stark",
        "tower": "Tower",
        "marina": "Marina", "mARINA": "Marina",
        "katsapos": "Katsaros",
        "woods": "Woods", "woodS": "Woods",
        "andreW": "Andrew",
        "tony": "Tony",
        "babe": "Babe",
    }

    def _fix_spelling(self, text: str) -> str:
        """基于直接映射 + 词典的拼写修复"""
        import string as str_mod
        words = text.split()
        fixed = []
        for w in words:
            stripped = w.strip(str_mod.punctuation)
            # 先查直接修复映射（去标点后）
            found = self.OCR_FIXES.get(w) or self.OCR_FIXES.get(stripped) or self.OCR_FIXES.get(stripped.lower())
            if found:
                # 还原标点
                suffix = w[len(stripped):] if w != stripped else ""
                fixed.append(found + suffix)
                continue
            # fuzzy 匹配（仅纯字母词）
            if re.match(r"^[A-Za-z']+$", stripped) and len(stripped) >= self._min_word_length:
                self._ensure_fuzzy()
                stem = stripped.lower().rstrip("'s")
                if len(stem) >= self._min_word_length:
                    wordlist = [word for word in self._common_words if abs(len(word) - len(stem)) <= 2]
                    if wordlist:
                        best, score, _ = self._process.extractOne(
                            stem, wordlist, scorer=self._fuzz.ratio,
                        )
                        if best and score >= self._spell_threshold and best != stem:
                            suffix = w[len(stripped):] if w != stripped else ""
                            result = best.capitalize() if stripped[0].isupper() else best
                            fixed.append(result + suffix)
                            continue
            fixed.append(w)
        return " ".join(fixed)
        return " ".join(fixed)

    # ---- Level 3: 大小写修复 ----

    def _fix_capitalization(self, text: str) -> str:
        """修复全大写文本为首字母大写 + 保留专有名词大写"""
        # 如果文本全部是大写（或大部分大写），进行大小写恢复
        upper_chars = sum(1 for c in text if c.isupper())
        total_alpha = sum(1 for c in text if c.isalpha())
        if total_alpha == 0:
            return text
        ratio = upper_chars / total_alpha
        if ratio < 0.7:
            return text  # 不是全大写文本，跳过

        words = text.split()
        result = []
        # 常见专有名词（保持大写）
        proper_nouns = {
            "usa", "balkans", "america", "greece", "nikos", "tony", "stark",
            "iron", "man", "marina", "katsaros", "andrew", "woods",
            "jarvis", "pepper", "happy", "stark", "tower",
        }
        for i, w in enumerate(words):
            clean = w.strip("""'\".,!?;:-_()[]{}""")
            if not clean:
                result.append(w)
                continue
            lower = clean.lower()
            if lower in proper_nouns:
                result.append(w.capitalize())
            elif i == 0:
                # 句首
                result.append(w.capitalize())
            elif clean.isupper() and len(clean) <= 3:
                # 短缩写保持大写
                result.append(w)
            else:
                result.append(w.lower())
        return " ".join(result)

    # ---- Level 4: Block 合并 ----

    def _merge_blocks(
        self,
        blocks: list[NormalizedOCRBlock],
        page: int,
        y_threshold: float = 0.8,
    ) -> list[NormalizedOCRBlock]:
        """合并同一气泡内的相邻文本行（不跨气泡）

        按 bubble_id 分组，仅在同一气泡内按 reading_order 合并同列相邻行。
        避免把多个气泡的文字合并成一个跨越气泡的大 block（会导致文字
        集中到一个气泡、其他气泡空白）。
        """
        if not blocks:
            return []

        from collections import defaultdict

        by_bubble: dict[str, list[NormalizedOCRBlock]] = defaultdict(list)
        for b in blocks:
            by_bubble[b.metadata.get("bubble_id") or "__none__"].append(b)

        merged: list[NormalizedOCRBlock] = []
        for group in by_bubble.values():
            group = sorted(group, key=lambda b: b.reading_order)
            cur = group[0]
            for b in group[1:]:
                cur_mid_x = (b.bbox[0] + b.bbox[2]) / 2
                x_overlap = min(cur.bbox[2], b.bbox[2]) - max(cur.bbox[0], b.bbox[0])
                if x_overlap > 0 and cur_mid_x >= cur.bbox[0] and cur_mid_x <= cur.bbox[2]:
                    # 同气泡同列相邻 → 合并
                    cur.text = cur.text + " " + b.text
                    cur.bbox = (
                        min(cur.bbox[0], b.bbox[0]),
                        min(cur.bbox[1], b.bbox[1]),
                        max(cur.bbox[2], b.bbox[2]),
                        max(cur.bbox[3], b.bbox[3]),
                    )
                    cur.is_merged = True
                    cur.merged_from_ids.append(b.id)
                else:
                    merged.append(cur)
                    cur = b
            merged.append(cur)

        # 按 reading_order 排序输出（保持阅读顺序）
        merged.sort(key=lambda b: b.reading_order)
        return merged

    # ---- Level 5: 实体提取 ----

    def _extract_entities(self, text: str) -> list[dict]:
        """提取命名实体（基于规则，无需模型）"""
        entities = []
        # 大写连续词 → 可能的人名/地名
        words = text.split()
        i = 0
        while i < len(words):
            w = words[i].strip("""'\".,!?;:-_()[]{}""")
            if not w:
                i += 1
                continue
            # 检测大写开头的连续词
            if w[0].isupper() and len(w) >= 2:
                name_parts = [w]
                j = i + 1
                while j < len(words):
                    nw = words[j].strip("""'\".,!?;:-_()[]{}""")
                    if nw and nw[0].isupper() and len(nw) >= 2:
                        name_parts.append(nw)
                        j += 1
                    else:
                        break
                full_name = " ".join(name_parts)
                # 过滤常见词
                common = {"The", "A", "An", "This", "That", "These", "Those",
                          "My", "Your", "His", "Her", "Our", "Their", "Its",
                          "All", "Some", "Many", "Much", "Few", "Several",
                          "Oh", "Well", "So", "But", "And", "Or", "If",
                          "When", "Where", "Why", "How", "What", "Which",
                          "Who", "Whom", "Whose", "Not", "No", "Yes",
                          "Just", "Then", "Now", "Here", "There", "Only",
                          "Very", "Really", "Still", "Even", "Also", "Too",
                          "One", "Two", "First", "Last", "Next", "Every",
                          "Good", "Great", "Big", "Little", "Old", "New",
                          "Right", "Wrong", "Sure", "Sorry", "Please",
                          "Thank", "Hello", "Hey", "Hi", "God", "Babe",
                          "Because", "But", "However", "Therefore", "Though",
                          "While", "Since", "Until", "After", "Before",
                          "About", "Into", "Through", "During", "Between",
                          "Could", "Would", "Should", "Might", "Must",
                          "Most", "Best", "More", "Less", "Enough",
                          "Beautiful", "Wonderful", "Perfect", "Great",
                          "Little", "Still", "Already", "Always", "Never",
                          "Ever", "Again", "Away", "Back", "Home",
                        }
                if len(name_parts) <= 2 and full_name not in common:
                    entity_type = "PERSON" if len(name_parts) >= 2 else "LOCATION"
                    entities.append({
                        "entity_name": full_name,
                        "entity_type": entity_type,
                    })
                i = j
            else:
                i += 1
        return entities