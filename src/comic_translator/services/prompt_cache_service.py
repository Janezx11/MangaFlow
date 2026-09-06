from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from loguru import logger


class PromptCacheService:
    """Prompt 缓存服务 —— 记录每次翻译的完整输入输出

    保存内容：
    - system_prompt
    - user_prompt
    - glossary 上下文
    - character memory 上下文
    - 模型原始响应

    用于调试翻译质量和 Prompt 优化。
    文件位置: data/cache/{project_id}/prompts/page_{page:06d}_prompt.json
    """

    def __init__(self, cache_dir: str = "data/cache") -> None:
        self._cache_dir = Path(cache_dir)

    def save_prompt(
        self,
        project_id: str,
        page: int,
        system_prompt: str,
        user_prompt: str,
        glossary_context: list[dict[str, Any]] | None = None,
        character_context: list[dict[str, Any]] | None = None,
        raw_model_response: str | None = None,
    ) -> str:
        """保存完整 Prompt 记录，返回文件路径"""
        path = self._prompt_path(project_id, page)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "project_id": project_id,
            "page": page,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "glossary_context": glossary_context or [],
            "character_context": character_context or [],
            "raw_model_response": raw_model_response or "",
        }

        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        logger.debug(f"Prompt cache saved to {path}")
        return str(path)

    def get_prompt(self, project_id: str, page: int) -> dict[str, Any] | None:
        """读取 Prompt 记录"""
        path = self._prompt_path(project_id, page)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def prompt_exists(self, project_id: str, page: int) -> bool:
        return self._prompt_path(project_id, page).exists()

    def clear(self, project_id: str) -> None:
        path = self._cache_dir / project_id / "prompts"
        if path.exists():
            import shutil
            shutil.rmtree(path)
            logger.info(f"Cleared prompt cache for project {project_id}")

    def _prompt_path(self, project_id: str, page: int) -> Path:
        return self._cache_dir / project_id / "prompts" / f"page_{page:06d}_prompt.json"
