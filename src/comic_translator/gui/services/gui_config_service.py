"""GUI Config Service — 读取/修改 config.yaml"""

from __future__ import annotations

import os
import re
from pathlib import Path

import yaml
from loguru import logger


def mask_api_key(key: str) -> str:
    """掩码 API key，仅用于界面展示：sk-****abcd"""
    if not key:
        return ""
    if len(key) <= 8:
        return "*" * len(key)
    return f"{key[:3]}****{key[-4:]}"


class GuiConfigService:
    """读取项目配置，供 GUI Settings 页面使用"""

    def __init__(self, config_path: str = "config.yaml"):
        self._path = Path(config_path)
        self._config: dict = {}
        self.reload()

    def reload(self) -> dict:
        if self._path.exists():
            self._config = yaml.safe_load(self._path.read_text(encoding="utf-8")) or {}
        else:
            self._config = {}
            logger.warning(f"Config not found: {self._path}")
        return self._config

    @property
    def config(self) -> dict:
        return self._config

    def get(self, key: str, default=None):
        keys = key.split(".")
        val = self._config
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
            if val is None:
                return default
        return val

    def set(self, key: str, value):
        keys = key.split(".")
        d = self._config
        for k in keys[:-1]:
            d = d.setdefault(k, {})
        d[keys[-1]] = value

    def get_api_key_state(self) -> dict:
        """返回 API key 状态，供设置页展示。

        支持三种情况：
          - 未配置
          - 配置为 "${VAR}" → 尝试从环境变量解析
          - 明文 key（config.yaml，已 gitignore）
        返回: {"configured": bool, "source": str, "preview": str(掩码)}
        """
        raw = self.get("llm.api.api_key", "")
        if not raw:
            return {"configured": False, "source": "none", "preview": ""}
        raw = str(raw).strip()
        m = re.match(r"^\$\{(\w+)\}$", raw)
        if m:
            var = m.group(1)
            val = os.environ.get(var, "")
            if val:
                return {"configured": True, "source": f"环境变量 {var}", "preview": mask_api_key(val)}
            return {"configured": False, "source": f"环境变量 {var}（未设置）", "preview": ""}
        return {"configured": True, "source": "config.yaml", "preview": mask_api_key(raw)}

    def set_api_key(self, value: str) -> None:
        """设置 API key（仅修改内存；调用方负责 save() 落盘）。

        留空/纯空白时不改动已有配置。
        """
        value = (value or "").strip()
        if value:
            self.set("llm.api.api_key", value)

    def save(self):
        self._path.write_text(
            yaml.dump(self._config, default_flow_style=False, allow_unicode=True),
            encoding="utf-8",
        )
        logger.info(f"Config saved: {self._path}")

    def summary(self) -> dict:
        """返回 GUI 显示用的配置摘要"""
        return {
            "llm_mode": self.get("llm.mode", "unknown"),
            "llm_model": self.get("llm.api.model", "unknown"),
            "llm_base_url": self.get("llm.api.base_url", ""),
            "ocr_provider": self.get("ocr.provider", "easyocr"),
            "bubble_model": self.get("bubble.model", ""),
            "bubble_conf": self.get("bubble.conf_threshold", 0.25),
            "target_lang": self.get("translation.target_lang", "zh"),
            "style": self.get("translation.style", "western_comic"),
            "output_dir": self.get("project.output_dir", "data/output"),
        }
