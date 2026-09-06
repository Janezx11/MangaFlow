"""Tests for ABC interfaces"""

from comic_translator.cache.base import CacheProvider
from comic_translator.clients.base import LLMClient
from comic_translator.prompts.base import PromptBuilder
from comic_translator.providers.ocr.base import OCRProvider
from comic_translator.providers.qa.base import QAProvider
from comic_translator.providers.render.base import RenderProvider
from comic_translator.providers.translation.base import TranslationProvider


class TestABCInterfaces:
    """验证所有 ABC 可以被继承"""

    def test_ocr_provider_abstract(self):
        """OCRProvider 有抽象方法，不能直接实例化"""
        try:
            OCRProvider()  # type: ignore
            raise AssertionError("Should not be instantiable")
        except TypeError:
            pass

    def test_translation_provider_abstract(self):
        try:
            TranslationProvider()  # type: ignore
            raise AssertionError()
        except TypeError:
            pass

    def test_qa_provider_abstract(self):
        try:
            QAProvider()  # type: ignore
            raise AssertionError()
        except TypeError:
            pass

    def test_render_provider_abstract(self):
        try:
            RenderProvider()  # type: ignore
            raise AssertionError()
        except TypeError:
            pass

    def test_llm_client_abstract(self):
        try:
            LLMClient()  # type: ignore
            raise AssertionError()
        except TypeError:
            pass

    def test_cache_provider_abstract(self):
        try:
            CacheProvider()  # type: ignore
            raise AssertionError()
        except TypeError:
            pass

    def test_prompt_builder_abstract(self):
        try:
            PromptBuilder()  # type: ignore
            raise AssertionError()
        except TypeError:
            pass
