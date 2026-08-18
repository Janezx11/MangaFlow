"""目录扫描工具 — 查找包含图片的目录"""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

SCAN_ROOTS = ["data", "data/comics", "data/input", "dataset"]


def scan_image_dirs(roots: list[str] | None = None) -> list[str]:
    """扫描根目录下所有包含图片的子目录"""
    roots = roots or SCAN_ROOTS
    found: list[str] = []

    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        # 检查根目录本身
        if _has_images(root_path):
            found.append(root_path.as_posix())
        # 检查一级子目录
        for child in sorted(root_path.iterdir()):
            if child.is_dir() and _has_images(child):
                found.append(child.as_posix())

    return sorted(set(found))


def _has_images(d: Path) -> bool:
    try:
        return any(f.suffix.lower() in IMAGE_EXTS for f in d.iterdir() if f.is_file())
    except PermissionError:
        return False
