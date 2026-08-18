from abc import ABC, abstractmethod
from typing import AsyncIterator


class LLMClient(ABC):
    """LLM 调用抽象层 —— 所有模型调用的唯一入口

    Provider 不得直接调用模型 SDK，只能依赖 LLMClient。
    切换模型/后端时，只需更换 LLMClient 实现，Provider 代码不变。
    """

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> str:
        ...

    @abstractmethod
    async def generate_stream(
        self,
        prompt: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> AsyncIterator[str]:
        ...

    @abstractmethod
    async def count_tokens(self, text: str) -> int:
        ...

    @property
    @abstractmethod
    def model_name(self) -> str:
        ...