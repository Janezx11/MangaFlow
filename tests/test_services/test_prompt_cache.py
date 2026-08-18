"""Tests for PromptCacheService"""

import json

from comic_translator.services.prompt_cache_service import PromptCacheService


class TestPromptCacheService:
    def test_save_and_get_prompt(self, tmp_path):
        cache = PromptCacheService(cache_dir=str(tmp_path))
        path = cache.save_prompt(
            project_id="test_proj",
            page=1,
            system_prompt="You are a translator",
            user_prompt="Translate: Hello",
            glossary_context=[{"source": "Stark", "target": "斯塔克"}],
            character_context=[{"name": "Tony Stark", "speaking_style": "sarcastic"}],
            raw_model_response='[{"id": "b1", "translated_text": "你好"}]',
        )
        assert path.endswith("_prompt.json")

        data = cache.get_prompt("test_proj", 1)
        assert data is not None
        assert data["system_prompt"] == "You are a translator"
        assert data["user_prompt"] == "Translate: Hello"
        assert len(data["glossary_context"]) == 1
        assert data["raw_model_response"] is not None

    def test_prompt_not_found(self, tmp_path):
        cache = PromptCacheService(cache_dir=str(tmp_path))
        data = cache.get_prompt("nonexistent", 99)
        assert data is None

    def test_prompt_exists(self, tmp_path):
        cache = PromptCacheService(cache_dir=str(tmp_path))
        assert cache.prompt_exists("p", 1) is False
        cache.save_prompt("p", 1, "sys", "user")
        assert cache.prompt_exists("p", 1) is True

    def test_clear(self, tmp_path):
        cache = PromptCacheService(cache_dir=str(tmp_path))
        cache.save_prompt("p", 1, "sys", "user")
        cache.save_prompt("p", 2, "sys", "user")
        assert cache.prompt_exists("p", 1) is True
        cache.clear("p")
        assert cache.prompt_exists("p", 1) is False