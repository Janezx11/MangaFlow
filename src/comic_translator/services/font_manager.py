"""FontManager V2 — Comic Font System with Fallback Chain

每个 RegionType 可指定字体，支持 fallback chain。
内置漫画字体：Wild Words（对白）、Anime Ace（SFX）、Komika（标题）。

字体查找优先级:
  1. data/fonts/{name} (用户自定义)
  2. 系统字体 (Windows / macOS)
  3. Noto Sans SC (通用中文字体)
  4. 系统回退字体 (msyh.ttc / PingFang.ttc)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from loguru import logger
from PIL import ImageFont

from comic_translator.models.render_models import FontProfile, RegionType

if TYPE_CHECKING:
    pass

# ---- 系统字体路径 ----
SYSTEM_FONTS: list[str] = [
    # Windows
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/msyhbd.ttc",
    "C:/Windows/Fonts/simsun.ttc",
    "C:/Windows/Fonts/yahei.ttf",
    # macOS
    "/System/Library/Fonts/PingFang.ttc",
    "/System/Library/Fonts/STHeiti Light.ttc",
    "/System/Library/Fonts/Helvetica.ttc",
]


def _find_font(names: list[str], min_size_kb: int = 1) -> str:
    """在 data/fonts/ 和系统目录中查找字体"""
    # 1. 检查 data/fonts/
    for name in names:
        custom = Path(f"data/fonts/{name}")
        if custom.exists() and custom.stat().st_size >= min_size_kb * 1024:
            logger.info(f"Found custom font: {custom}")
            return str(custom)

    # 2. 检查系统字体（按名字匹配）
    for name in names:
        for sys_path in SYSTEM_FONTS:
            p = Path(sys_path)
            if p.exists() and name.lower() in p.name.lower():
                logger.info(f"Using system font: {p}")
                return str(p)

    # 3. 回退到任何存在的系统字体
    for sys_path in SYSTEM_FONTS:
        if Path(sys_path).exists():
            logger.warning(f"Font {names[0]} not found, fallback to {sys_path}")
            return str(sys_path)

    raise FileNotFoundError(
        f"Font not found: {names}\n"
        f"Place in data/fonts/ or install a system Chinese font.\n"
        f"Download Noto Sans SC: https://github.com/notofonts/noto-cjk/releases"
    )


# ---- 字体配置文件 ----
FONT_PROFILES: dict[str, FontProfile] = {}

# RegionType → 字体名称列表（按优先级排列）
REGION_FONT_NAMES: dict[RegionType, list[str]] = {
    RegionType.DIALOGUE: [
        "WildWords.ttf",
        "CCWildWords.ttf",
        "NotoSansSC-Regular.ttf",
        "msyh.ttc",
    ],
    RegionType.NARRATION: [
        "NotoSansSC-Regular.ttf",
        "msyh.ttc",
    ],
    RegionType.TITLE: [
        "Komika-Bold.ttf",
        "NotoSansSC-Bold.ttf",
        "msyhbd.ttc",
    ],
    RegionType.SFX: [
        "AnimeAce.ttf",
        "AnimeAce-Bold.ttf",
        "NotoSansSC-Bold.ttf",
        "msyhbd.ttc",
    ],
    RegionType.UNKNOWN: [
        "NotoSansSC-Regular.ttf",
        "msyh.ttc",
    ],
}

# 默认字体属性
FONT_CONFIG: dict[RegionType, dict] = {
    RegionType.DIALOGUE: dict(min_size=12, max_size=64, default_size=24, line_spacing=1.3, stroke_width=1),
    RegionType.NARRATION: dict(min_size=10, max_size=48, default_size=18, line_spacing=1.4, stroke_width=1),
    RegionType.TITLE: dict(min_size=20, max_size=80, default_size=36, line_spacing=1.2, stroke_width=2),
    RegionType.SFX: dict(min_size=16, max_size=72, default_size=32, line_spacing=1.1, stroke_width=2),
    RegionType.UNKNOWN: dict(min_size=12, max_size=64, default_size=24, line_spacing=1.3, stroke_width=1),
}


def _ensure_font_profiles():
    """延迟初始化字体配置（在 _find_font 可能抛出异常时）"""
    if FONT_PROFILES:
        return
    for rt in RegionType:
        names = REGION_FONT_NAMES.get(rt, REGION_FONT_NAMES[RegionType.UNKNOWN])
        config = FONT_CONFIG.get(rt, FONT_CONFIG[RegionType.UNKNOWN])
        try:
            path = _find_font(names)
        except FileNotFoundError:
            # 最后回退：只要任何一个系统字体存在
            fallback = [p for p in SYSTEM_FONTS if Path(p).exists()]
            if fallback:
                path = fallback[0]
            else:
                raise
        FONT_PROFILES[rt.value] = FontProfile(
            name=rt.value,
            font_path=path,
            min_size=config["min_size"],
            max_size=config["max_size"],
            default_size=config["default_size"],
            line_spacing=config["line_spacing"],
            stroke_width=config["stroke_width"],
        )
        logger.debug(f"FontProfile [{rt.value}]: {path}")


class FontManager:
    """漫画字体管理器

    功能:
      - 按 RegionType 获取 FontProfile（带 fallback）
      - 加载字体缓存 (LRU)
      - 测量文本尺寸
    """

    def __init__(self) -> None:
        _ensure_font_profiles()
        self._override_profiles: dict[str, FontProfile] | None = None

    def set_font_override(self, font_name: str | None) -> None:
        """切换字体覆盖：'comic' 使用默认漫画字体，'system'/'fallback' 使用系统字体。"""
        if font_name in (None, "", "comic"):
            self._override_profiles = None
            return

        override: dict[str, FontProfile] = {}
        for rt in RegionType:
            profile = FONT_PROFILES.get(rt.value, FONT_PROFILES[RegionType.UNKNOWN.value])
            if font_name in ("system", "fallback"):
                fallback = self.get_fallback_profile(rt)
                override[rt.value] = FontProfile(
                    name=fallback.name,
                    font_path=fallback.font_path,
                    min_size=profile.min_size,
                    max_size=profile.max_size,
                    default_size=profile.default_size,
                    line_spacing=profile.line_spacing,
                    stroke_width=profile.stroke_width,
                    color=profile.color,
                )
            else:
                # 如果给出具体字体文件名，尝试在所有 region 上复用
                from pathlib import Path
                custom = Path(f"data/fonts/{font_name}")
                if not custom.exists():
                    custom = Path(font_name)
                if not custom.exists():
                    raise FileNotFoundError(f"Font override not found: {font_name}")
                override[rt.value] = FontProfile(
                    name=custom.name,
                    font_path=str(custom),
                    min_size=profile.min_size,
                    max_size=profile.max_size,
                    default_size=profile.default_size,
                    line_spacing=profile.line_spacing,
                    stroke_width=profile.stroke_width,
                    color=profile.color,
                )
        self._override_profiles = override

    def get_profile(self, region_type: RegionType | str) -> FontProfile:
        """获取 RegionType 对应的 FontProfile"""
        if isinstance(region_type, str):
            rt = RegionType(region_type) if region_type in RegionType._value2member_map_ else RegionType.UNKNOWN
        else:
            rt = region_type
        key = rt.value
        if self._override_profiles is not None and key in self._override_profiles:
            return self._override_profiles[key]
        return FONT_PROFILES.get(key, FONT_PROFILES[RegionType.UNKNOWN.value])

    def get_profile_by_name(self, region_name: str) -> FontProfile:
        mapping = {
            "speech": RegionType.DIALOGUE,
            "narration": RegionType.NARRATION,
            "title": RegionType.TITLE,
            "sfx": RegionType.SFX,
        }
        rt = mapping.get(region_name, RegionType.UNKNOWN)
        return self.get_profile(rt)

    def get_fallback_profile(self, region_type: RegionType | str) -> FontProfile:
        """获取回退字体（仅 Noto Sans SC / 系统）"""
        if isinstance(region_type, str):
            rt = RegionType(region_type) if region_type in RegionType._value2member_map_ else RegionType.UNKNOWN
        else:
            rt = region_type
        # 跳过漫画字体，直接用系统字体
        names = REGION_FONT_NAMES.get(rt, REGION_FONT_NAMES[RegionType.UNKNOWN])
        # 只保留系统字体
        system_names = [n for n in names if "Noto" in n or "msyh" in n or "PingFang" in n]
        path = _find_font(system_names or ["msyh.ttc", "PingFang.ttc"])
        config = FONT_CONFIG.get(rt, FONT_CONFIG[RegionType.UNKNOWN])
        return FontProfile(
            name=f"{rt.value}_fallback",
            font_path=path,
            min_size=config["min_size"],
            max_size=config["max_size"],
            default_size=config["default_size"],
            line_spacing=config["line_spacing"],
            stroke_width=config["stroke_width"],
        )

    @lru_cache(maxsize=128)
    def load_font(self, font_path: str, size: int) -> ImageFont.FreeTypeFont:
        if not Path(font_path).exists():
            raise FileNotFoundError(font_path)
        return ImageFont.truetype(font_path, size)

    def measure_text(self, text: str, font_path: str, size: int) -> tuple[int, int]:
        font = self.load_font(font_path, size)
        bbox = font.getbbox(text)
        return (bbox[2] - bbox[0], bbox[3] - bbox[1])

    def draw_text(
        self,
        draw,
        x: int,
        y: int,
        text: str,
        font: ImageFont.FreeTypeFont,
        profile: FontProfile,
    ) -> None:
        # Pillow 的 draw.text 支持 fill（文字填充色）+ stroke_fill（描边色）
        # 一次调用即可，不要画两次否则第二次覆盖描边
        draw.text(
            (x, y), text, font=font,
            fill=profile.color,
            stroke_width=profile.stroke_width,
            stroke_fill=profile.stroke_color,
        )
