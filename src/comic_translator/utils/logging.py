import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: str = "data/logs/translator.log",
    rotation: str = "100 MB",
    retention: str = "30 days",
) -> None:
    """配置 loguru 日志"""
    logger.remove()

    # stdout (human readable)
    logger.add(
        sys.stderr,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | <cyan>{name}</cyan> | {message}",
    )

    # file (structured)
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        str(log_path),
        level=level,
        rotation=rotation,
        retention=retention,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <7} | {name}:{function}:{line} | {message}",
    )
