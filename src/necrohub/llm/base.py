"""The chat-model interface shared by the generator, the LLM de-identifier and RAG."""

from collections.abc import Sequence
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    role: Literal["system", "user", "assistant"]
    content: str


class LlmError(Exception):
    """A backend call failed (connection, HTTP status, quota). Never carries a secret."""


class ChatLLM(Protocol):
    name: str

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        seed: int | None = None,
    ) -> str: ...


def get_llm(backend: Literal["ollama", "gemini"], model: str) -> ChatLLM:
    """The configured backend; host and key come from the environment through Settings."""
    from necrohub.settings import Settings

    settings = Settings()
    if backend == "ollama":
        from necrohub.llm.ollama import OllamaLLM

        return OllamaLLM(settings.ollama_host, model)
    from necrohub.llm.gemini import GeminiLLM

    if settings.gemini_api_key is None:
        raise LlmError("GEMINI_API_KEY is not set")
    return GeminiLLM(settings.gemini_api_key, model)
