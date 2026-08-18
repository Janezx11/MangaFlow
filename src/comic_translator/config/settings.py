from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Any


class Settings(BaseSettings):
    """全局配置 —— 支持环境变量覆盖 (CT_ 前缀)

    注意: config.yaml 由 di.py 的 _load_config() 直接读取，
    本类仅提供环境变量覆盖和默认值。
    """

    model_config = SettingsConfigDict(
        env_prefix="CT_",
        env_nested_delimiter="__",
        extra="allow",
    )

    # Top-level provider selection
    ocr_provider: str = "paddle_ocr"
    translation_provider: str = "qwen"
    qa_provider: str = "gemma"
    render_provider: str = "opencv"
    cache_enabled: bool = True
    comic_style: str = "western_comic"
    log_level: str = "INFO"
    log_file: str = "data/logs/translator.log"

    # Nested provider configs (可从环境变量覆盖)
    ocr: dict[str, Any] = {}
    translation: dict[str, Any] = {}
    qa: dict[str, Any] = {}
    render: dict[str, Any] = {}
    cache: dict[str, Any] = {}
    comic_style_configs: dict[str, Any] = {}
    project: dict[str, Any] = {}
    logging: dict[str, Any] = {}