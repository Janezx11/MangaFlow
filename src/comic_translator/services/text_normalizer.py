"""Chinese Text Normalizer —— 统一中文排版格式"""

import re


class TextNormalizer:
    """中文文本规范化

    职责:
    - 中英文标点统一
    - 连续空格处理
    - 换行规范化
    - 中英文间距
    """

    def normalize(self, text: str) -> str:
        """完整规范化流程"""
        text = self._unify_punctuation(text)
        text = self._fix_spacing(text)
        text = self._normalize_newlines(text)
        return text.strip()

    def _unify_punctuation(self, text: str) -> str:
        """统一中英文标点"""
        replacements = {
            # 英文引号 → 中文引号
            '"': '“', '"': '”',
            "'": '‘', "'": '’',
            # 省略号
            "...": "…",
            "..": "…",
            # 破折号
            "--": "—",
            # 句号
            ".": "。",
            # 逗号
            ",": "，",
            # 问号
            "?": "？",
            # 感叹号
            "!": "！",
            # 冒号
            ":": "：",
            # 分号
            ";": "；",
        }
        # 按长度降序替换（避免短模式干扰长模式）
        for old, new in sorted(replacements.items(), key=lambda x: -len(x[0])):
            text = text.replace(old, new)
        return text

    def _fix_spacing(self, text: str) -> str:
        """处理中英文间距"""
        # 中英文之间加空格
        text = re.sub(r"([一-鿿])([a-zA-Z])", r"\1 \2", text)
        text = re.sub(r"([a-zA-Z])([一-鿿])", r"\1 \2", text)
        # 压缩多空格
        text = re.sub(r" {2,}", " ", text)
        return text

    def _normalize_newlines(self, text: str) -> str:
        """统一换行为单空格（中文文本通常不需要硬换行）"""
        text = text.replace("\n", " ")
        text = text.replace("\r", "")
        return text