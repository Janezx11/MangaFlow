from pydantic import BaseModel, Field
from comic_translator.models.normalized_ocr_block import NormalizedOCRBlock
from comic_translator.models.translation_block import TranslationBlock
from comic_translator.models.glossary import GlossaryEntry


class BatchPage(BaseModel):
    """Batch 中单页的 OCR 结果"""

    page: int
    blocks: list[NormalizedOCRBlock] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class PageTranslation(BaseModel):
    """翻译后的单页结果"""

    page: int
    blocks: list[TranslationBlock] = Field(default_factory=list)


class BatchInput(BaseModel):
    """Batch 翻译的输入"""

    batch_index: int
    pages: list[BatchPage] = Field(default_factory=list)
    glossary: list[GlossaryEntry] = Field(default_factory=list)
    project_name: str = ""
    comic_style: str = "western_comic"
    source_language: str = "en"
    target_language: str = "zh"


class BatchOutput(BaseModel):
    """Batch 翻译的输出"""

    batch_index: int
    pages: list[PageTranslation] = Field(default_factory=list)
    glossary: dict[str, str] = Field(default_factory=dict)
    raw_response: str = ""


class PipelineProject(BaseModel):
    """整个翻译项目"""

    project_name: str
    total_pages: int
    batch_size: int = 20
    pages: list[BatchPage] = Field(default_factory=list)
    batches: list[BatchOutput] = Field(default_factory=list)
    source_language: str = "en"
    target_language: str = "zh"
    comic_style: str = "western_comic"