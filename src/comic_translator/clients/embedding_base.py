from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Embedding 抽象层 —— 预留未来扩展

    目标实现:
    - BGEEmbeddingProvider (本地, 推荐)
    - SentenceTransformerProvider (本地, 简单)
    - OpenAIEmbeddingProvider (远程)
    - JinaEmbeddingProvider (远程)
    """

    @abstractmethod
    async def embed(self, text: str) -> list[float]:
        """生成单条文本的 embedding 向量"""
        ...

    @abstractmethod
    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """批量生成 embedding 向量"""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回 embedding 向量维度"""
        ...