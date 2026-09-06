from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """环境变量覆盖（CT_ 前缀）。

    config.yaml 是主要配置源（由 di.py 的 _load_config() 读取）；
    本类只暴露支持环境变量覆盖的少量运行时开关。
    """

    model_config = SettingsConfigDict(
        env_prefix="CT_",
        extra="ignore",
    )

    cache_enabled: bool = True
    log_level: str = "INFO"
    log_file: str = "data/logs/translator.log"
