"""DEPRECATED（已被 typography_service_v2.py 取代）——仅 demo/examples 脚本引用。

字体配置、文本绘制（Debug V3，原模块 docstring：Typography Engine）。
实际渲染管线使用 TypographyServiceV2。
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from PIL import ImageDraw, ImageFont

from comic_translator.models.render_models import FontProfile, RegionType

if TYPE_CHECKING:
    from PIL import Image


# ---- 字体路径检测 ----
def _find_font(name: str) -> str:
    """查找字体文件：优先自定义路径（>=1KB 才认为是有效字体），回退系统字体"""
    custom = Path(f"data/fonts/{name}")
    if custom.exists() and custom.stat().st_size >= 1024:
        return str(custom)
    system_paths = {
        "NotoSansSC-Regular.ttf": [
            "C:/Windows/Fonts/msyh.ttc",
            "/System/Library/Fonts/PingFang.ttc",
        ],
        "NotoSansSC-Bold.ttf": [
            "C:/Windows/Fonts/msyhbd.ttc",
            "/System/Library/Fonts/PingFang.ttc",
        ],
    }
    candidates = system_paths.get(name, [])
    for cp in candidates:
        if Path(cp).exists():
            logger.info(f"Using system font: {cp}")
            return cp
    raise FileNotFoundError(
        f"Font not found: {name}\n"
        f"Place in data/fonts/ or install a system Chinese font.\n"
        f"Download: https://github.com/notofonts/noto-cjk/releases"
    )


# ---- 默认字体配置（自动检测路径） ----
DIALOGUE_FONT = FontProfile(
    name="dialogue",
    font_path=_find_font("NotoSansSC-Regular.ttf"),
    min_size=12, max_size=64, default_size=24,
    line_spacing=1.3, stroke_width=1,
)

NARRATION_FONT = FontProfile(
    name="narration",
    font_path=_find_font("NotoSansSC-Regular.ttf"),
    min_size=10, max_size=48, default_size=18,
    line_spacing=1.4, stroke_width=1,
)

TITLE_FONT = FontProfile(
    name="title",
    font_path=_find_font("NotoSansSC-Bold.ttf"),
    min_size=20, max_size=80, default_size=36,
    line_spacing=1.2, stroke_width=2,
)

SFX_FONT = FontProfile(
    name="sfx",
    font_path=_find_font("NotoSansSC-Bold.ttf"),
    min_size=16, max_size=72, default_size=32,
    line_spacing=1.1, stroke_width=2,
)

REGION_FONT_MAP: dict[RegionType, FontProfile] = {
    RegionType.DIALOGUE: DIALOGUE_FONT,
    RegionType.NARRATION: NARRATION_FONT,
    RegionType.TITLE: TITLE_FONT,
    RegionType.SFX: SFX_FONT,
    RegionType.UNKNOWN: DIALOGUE_FONT,
}


class TypographyService:
    """排版引擎 —— 字体管理、文本绘制、样式控制"""

    def get_font_profile(self, region_type: RegionType) -> FontProfile:
        return REGION_FONT_MAP.get(region_type, DIALOGUE_FONT)

    def get_font_profile_by_name(self, region_name: str) -> FontProfile:
        """通过字符串名称获取字体配置（用于 YOLO 的 class_name）"""
        from comic_translator.models.render_models import RegionType
        mapping = {
            "speech": RegionType.DIALOGUE,
            "narration": RegionType.NARRATION,
            "title": RegionType.TITLE,
            "sfx": RegionType.SFX,
        }
        rt = mapping.get(region_name, RegionType.DIALOGUE)
        return REGION_FONT_MAP.get(rt, DIALOGUE_FONT)

    @lru_cache(maxsize=256)
    def load_font(self, font_path: str, size: int) -> ImageFont.FreeTypeFont:
        """加载字体（带 LRU 缓存）"""
        path = Path(font_path)
        if not path.exists():
            raise FileNotFoundError(font_path)
        font = ImageFont.truetype(str(path), size)
        logger.debug(f"Loaded font: {font_path} size={size}")
        return font

    def get_text_size(self, font: ImageFont.FreeTypeFont, text: str) -> tuple[int, int]:
        bbox = font.getbbox(text)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])

    def draw_text(
        self,
        draw: ImageDraw.ImageDraw,
        x: int,
        y: int,
        text: str,
        font: ImageFont.FreeTypeFont,
        profile: FontProfile,
    ) -> None:
        """绘制带轮廓的文本"""
        if profile.stroke_width > 0:
            draw.text(
                (x, y), text, font=font,
                fill=profile.stroke_color,
                stroke_width=profile.stroke_width,
                stroke_fill=profile.stroke_color,
            )
        draw.text(
            (x, y), text, font=font,
            fill=profile.color,
        )