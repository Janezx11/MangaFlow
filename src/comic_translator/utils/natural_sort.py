"""自然排序工具 — 按文件名中的数字块排序

解决普通字典序（str 比较）导致的问题：
  "10.jpg" < "2.jpg"、"page_10" < "page_2"
使得带编号的漫画文件被错误排序，渲染/翻译页码与实际阅读顺序不符。

用法: sorted(files, key=natural_key) 或 files.sort(key=natural_key)
"""

from __future__ import annotations

import re
from pathlib import Path

_NUM_RE = re.compile(r"(\d+)")


def natural_key(value) -> tuple:
    """生成自然排序键。

    - 文件名中的数字块按数值比较（"2.jpg" < "10.jpg"）
    - 非数字块按小写文本比较
    - 无数字的文件退化为按文本比较（保持字典序行为）

    返回由 ((0, int) | (1, str), ...) 组成的元组，可直接用于 sorted/sort 的 key。
    """
    s = Path(value).stem if isinstance(value, (str, Path)) else str(value)
    parts: list[tuple] = []
    for chunk in _NUM_RE.split(s):
        if not chunk:
            continue
        if chunk.isdigit():
            parts.append((0, int(chunk)))
        else:
            parts.append((1, chunk.lower()))
    return tuple(parts)
