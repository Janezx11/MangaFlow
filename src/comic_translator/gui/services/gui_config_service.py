"""GUI Config Service — 读取/修改 config.yaml"""

from __future__ import annotations

from pathlib import Path

import yaml
from loguru import logger


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
