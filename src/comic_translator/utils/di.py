from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from comic_translator.cache.file_cache_provider import FileCacheProvider
from comic_translator.cache.null_cache_provider import NullCacheProvider
from comic_translator.clients.base import LLMClient
from comic_translator.clients.ollama_client import OllamaClient
from comic_translator.clients.openai_compatible_client import OpenAICompatibleClient
from comic_translator.config.settings import Settings
from comic_translator.prompts.manga_prompt_builder import MangaPromptBuilder
from comic_translator.prompts.webtoon_prompt_builder import WebtoonPromptBuilder
from comic_translator.prompts.western_comic_prompt_builder import (
    WesternComicPromptBuilder,
)
from comic_translator.providers.ocr.base import OCRProvider
from comic_translator.providers.qa.gemma_qa_provider import GemmaQAProvider
from comic_translator.providers.render.opencv_render_provider import OpenCVRenderProvider
from comic_translator.providers.translation.deepseek_translator import DeepSeekTranslator
from comic_translator.providers.translation.qwen_translator import QwenTranslator
from comic_translator.services.cache_service import CacheService
from comic_translator.services.character_identity_service import CharacterIdentityService
from comic_translator.services.character_memory_service import CharacterMemoryService
from comic_translator.services.context_assembler import ContextAssembler
from comic_translator.services.glossary_service import GlossaryService
from comic_translator.services.merge_confidence_service import MergeConfidenceService
from comic_translator.services.ocr_normalize_service import OCRNormalizeService
from comic_translator.services.ocr_service import OCRService
from comic_translator.services.page_summary_service import PageSummaryService
from comic_translator.services.project_service import ProjectService
from comic_translator.services.prompt_cache_service import PromptCacheService
from comic_translator.services.qa_service import QAService
from comic_translator.services.render_service import RenderService
from comic_translator.services.semantic_tm_service import SemanticTranslationMemoryService
from comic_translator.services.translation_memory_service import TranslationMemoryService
from comic_translator.services.translation_service import TranslationService
from comic_translator.services.text_normalizer import TextNormalizer
from comic_translator.services.font_manager import FontManager
from comic_translator.services.bubble_layout_service import BubbleLayoutService
from comic_translator.services.shape_layout_solver import ShapeLayoutSolver
from comic_translator.services.balanced_line_breaker import BalancedLineBreaker
from comic_translator.services.padding_solver import PaddingSolver
from comic_translator.services.font_size_solver import FontSizeSolver
from comic_translator.services.optical_center_solver import OpticalCenterSolver
from comic_translator.services.typography_service_v2 import TypographyServiceV2
from comic_translator.services.typography_qa import TypographyQA
from comic_translator.services.render_qa_service import RenderQAService
from comic_translator.services.inpaint_service import InpaintService
from comic_translator.services.background_cleaner import BackgroundCleaner
from comic_translator.storage.database import init_db
from comic_translator.storage.repositories import ProjectRepository


def _load_config() -> dict:
    """从 config.yaml 加载配置，不存在时返回空 dict"""
    config_path = Path("config.yaml")
    if config_path.exists():
        return yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return {}


def _get_config_value(config: dict, key: str, default: Any = None) -> Any:
    """从嵌套 dict 中读取配置值，如 'llm.api.model'"""
    keys = key.split(".")
    val = config
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
        if val is None:
            return default
    return val


def _resolve_env_var(value: str) -> str:
    """解析配置中的环境变量引用

    支持两种格式:
      - "${VAR_NAME}" → os.environ.get("VAR_NAME", "")
      - "plain text" → 原样返回
    """
    import os
    import re
    if not isinstance(value, str):
        return value
    match = re.match(r'^\$\{(\w+)\}$', value.strip())
    if match:
        return os.environ.get(match.group(1), "")
    return value


def create_llm_client_from_config(config: dict) -> LLMClient:
    """根据 config.yaml 的 llm 段创建 LLMClient

    支持两种 mode:
      - ollama: 使用 OllamaClient
      - api: 使用 OpenAICompatibleClient
    """
    llm_mode = _get_config_value(config, "llm.mode", "ollama")

    if llm_mode == "api":
        api_cfg = _get_config_value(config, "llm.api", {})
        raw_key = api_cfg.get("api_key", "")
        api_key = _resolve_env_var(raw_key) if isinstance(raw_key, str) else raw_key
        return OpenAICompatibleClient(
            model=api_cfg.get("model", "deepseek-chat"),
            base_url=api_cfg.get("base_url", "https://api.deepseek.com"),
            api_key=api_key or raw_key,  # 回退到原始值
            timeout=api_cfg.get("timeout", 120),
            max_retries=api_cfg.get("max_retries", 5),
            rps=api_cfg.get("rps", 3.0),
        )
    else:
        ollama_cfg = _get_config_value(config, "llm.ollama", {})
        return OllamaClient(
            model=ollama_cfg.get("model", "qwen3.5:9b"),
            base_url=ollama_cfg.get("base_url", "http://localhost:11434"),
            timeout=ollama_cfg.get("timeout", 300),
            max_retries=ollama_cfg.get("max_retries", 3),
        )


def create_ocr_provider_from_config(config: dict) -> OCRProvider:
    """根据 config.yaml 创建 OCR Provider

    支持:
      - paddle: PaddleOCRProvider
      - easyocr: EasyOCRProvider
      - manga_ocr: MangaOCRProvider
    """
    ocr_cfg = _get_config_value(config, "ocr", {})
    provider_name = ocr_cfg.get("provider", "paddle_ocr")
    lang = ocr_cfg.get("lang", "en")

    if provider_name == "easyocr":
        from comic_translator.providers.ocr.easyocr_provider import EasyOCRProvider
        use_gpu = ocr_cfg.get("use_gpu", False)
        if not isinstance(use_gpu, bool):
            try:
                import torch
                use_gpu = bool(torch.cuda.is_available())
            except Exception:
                use_gpu = False
        return EasyOCRProvider(lang=lang, use_gpu=use_gpu)
    elif provider_name == "manga_ocr":
        from comic_translator.providers.ocr.manga_ocr_provider import MangaOCRProvider
        return MangaOCRProvider()
    else:
        from comic_translator.providers.ocr.paddle_ocr_provider import PaddleOCRProvider
        return PaddleOCRProvider()


def create_translation_provider_from_config(config: dict, llm_client: LLMClient) -> Any:
    """根据 config.yaml 创建翻译 Provider

    自动根据 llm.mode 和配置选择合适的 Provider:
      - api mode → DeepSeekTranslator (使用 OpenAI compatible API)
      - ollama mode → QwenTranslator (使用 Ollama)
    """
    llm_mode = _get_config_value(config, "llm.mode", "ollama")

    if llm_mode == "api":
        api_cfg = _get_config_value(config, "llm.api", {})
        return DeepSeekTranslator(
            client=llm_client,
            model=api_cfg.get("model", "deepseek-chat"),
        )
    else:
        ollama_cfg = _get_config_value(config, "llm.ollama", {})
        return QwenTranslator(
            client=llm_client,
            model=ollama_cfg.get("model", "qwen3.5:9b"),
        )


class Container:
    """手动 DI 容器 —— 从 config.yaml 动态读取配置"""

    def __init__(self, settings: Settings | None = None, db_path: str = "data/comic_translator.db") -> None:
        self._settings = settings or Settings()
        self._db_path = db_path
        self._config = _load_config()
        self._instances: dict[str, Any] = {}
        self._init_storage()
        self._init_services()

    def _init_storage(self) -> None:
        init_db(self._db_path)

    def _init_services(self) -> None:
        # -- Cache Layer --
        if self._settings.cache_enabled:
            cache_dir = _get_config_value(
                self._config, "cache.providers.file.dir", "data/cache"
            )
            cache_provider = FileCacheProvider(cache_dir=str(cache_dir))
        else:
            cache_provider = NullCacheProvider()
        cache_service = CacheService(provider=cache_provider)
        self._instances["cache_service"] = cache_service

        # -- LLM Client (从 config.yaml 动态创建) --
        llm_client = create_llm_client_from_config(self._config)
        self._instances["llm_client"] = llm_client

        # -- Prompt Builders --
        self._instances["prompt_builders"] = {
            "western_comic": WesternComicPromptBuilder(),
            "manga": MangaPromptBuilder(),
            "webtoon": WebtoonPromptBuilder(),
        }

        # Prompt builder for current style
        prompt_builder = self._resolve_prompt_builder()
        self._instances["prompt_builder"] = prompt_builder

        # -- Database Services (SQLite-backed) --
        glossary_service = GlossaryService(db_path=self._db_path)
        character_memory_service = CharacterMemoryService(db_path=self._db_path)
        character_identity_service = CharacterIdentityService(db_path=self._db_path)
        translation_memory_service = TranslationMemoryService(db_path=self._db_path)
        semantic_tm_service = SemanticTranslationMemoryService(db_path=self._db_path)
        prompt_cache_service = PromptCacheService(
            cache_dir=_get_config_value(
                self._config, "cache.providers.file.dir", "data/cache"
            )
        )
        page_summary_service = PageSummaryService(
            cache_dir=_get_config_value(
                self._config, "cache.providers.file.dir", "data/cache"
            ),
            prompt_builder=prompt_builder,
        )
        merge_confidence_service = MergeConfidenceService()

        self._instances.update(
            glossary_service=glossary_service,
            character_memory_service=character_memory_service,
            character_identity_service=character_identity_service,
            translation_memory_service=translation_memory_service,
            semantic_tm_service=semantic_tm_service,
            prompt_cache_service=prompt_cache_service,
            page_summary_service=page_summary_service,
            merge_confidence_service=merge_confidence_service,
        )

        # -- Providers (from config) --
        ocr_provider = create_ocr_provider_from_config(self._config)
        translation_provider = create_translation_provider_from_config(
            self._config, llm_client
        )
        qa_provider = GemmaQAProvider(client=llm_client, model="gemma4:e4b")
        render_provider = OpenCVRenderProvider()

        self._instances["ocr_provider"] = ocr_provider
        self._instances["translation_provider"] = translation_provider

        # -- ContextAssembler --
        context_assembler = ContextAssembler(
            glossary_service=glossary_service,
            character_identity_service=character_identity_service,
            semantic_tm_service=semantic_tm_service,
            page_summary_service=page_summary_service,
            prompt_builder=prompt_builder,
        )
        self._instances["context_assembler"] = context_assembler

        # -- Services --
        ocr_normalize_service = OCRNormalizeService()
        ocr_service = OCRService(provider=ocr_provider, cache=cache_service)
        translation_service = TranslationService(
            provider=translation_provider,
            prompt_builder=prompt_builder,
            cache=cache_service,
            context_assembler=context_assembler,
            semantic_tm_service=semantic_tm_service,
            prompt_cache_service=prompt_cache_service,
            page_summary_service=page_summary_service,
        )
        qa_service = QAService(provider=qa_provider, prompt_builder=prompt_builder)

        # -- RenderService 依赖（Typography Engine） --
        text_normalizer = TextNormalizer()
        font_manager = FontManager()
        bubble_layout = BubbleLayoutService()
        shape_layout_solver = ShapeLayoutSolver(bubble_layout)
        balanced_line_breaker = BalancedLineBreaker(font_manager)
        padding_solver = PaddingSolver()
        font_size_solver = FontSizeSolver(font_manager, balanced_line_breaker, shape_layout_solver, padding_solver)
        optical_center_solver = OpticalCenterSolver()
        typography_v2 = TypographyServiceV2(
            font_manager, bubble_layout, shape_layout_solver,
            balanced_line_breaker, padding_solver, font_size_solver, optical_center_solver,
        )
        typography_qa = TypographyQA()
        render_qa_service = RenderQAService()
        inpaint_service = InpaintService()
        background_cleaner = BackgroundCleaner(inpaint_service)

        render_service = RenderService(
            text_normalizer=text_normalizer,
            font_manager=font_manager,
            typography_v2=typography_v2,
            typography_qa=typography_qa,
            qa_service=render_qa_service,
            inpaint_service=inpaint_service,
            background_cleaner=background_cleaner,
        )
        project_service = ProjectService(repository=ProjectRepository())

        self._instances.update(
            ocr_service=ocr_service,
            ocr_normalize_service=ocr_normalize_service,
            translation_service=translation_service,
            qa_service=qa_service,
            render_service=render_service,
            project_service=project_service,
        )

    def _resolve_prompt_builder(self):
        style = _get_config_value(self._config, "translation.style", "western_comic")
        builders = {
            "western_comic": WesternComicPromptBuilder,
            "manga": MangaPromptBuilder,
            "webtoon": WebtoonPromptBuilder,
        }
        cls = builders.get(style, WesternComicPromptBuilder)
        return cls()

    def get(self, key: str) -> Any:
        return self._instances[key]

    def get_llm_client(self) -> LLMClient:
        """获取当前 LLM 客户端"""
        return self._instances["llm_client"]

    def refresh_llm_client(self) -> LLMClient:
        """从 config.yaml 重新读取配置并重建 LLM 客户端。

        供 GUI 设置页保存 API key 后热更新，无需重启应用。
        会一并重建依赖该客户端的 translation_provider。
        """
        self._config = _load_config()
        llm_client = create_llm_client_from_config(self._config)
        self._instances["llm_client"] = llm_client
        self._instances["translation_provider"] = create_translation_provider_from_config(
            self._config, llm_client
        )
        logger.info("DI container: LLM client refreshed from config.yaml")
        return llm_client


_container: Container | None = None


def get_container(db_path: str = "data/comic_translator.db") -> Container:
    global _container
    if _container is None:
        _container = Container(db_path=db_path)
        logger.info("DI container initialized")
    return _container


def refresh_llm_client() -> LLMClient | None:
    """热刷新全局容器的 LLM 客户端；容器未初始化时返回 None。"""
    global _container
    if _container is None:
        return None
    return _container.refresh_llm_client()
