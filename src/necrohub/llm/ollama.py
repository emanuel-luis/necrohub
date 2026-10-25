"""Local models through Ollama (`ollama` extra)."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from necrohub.llm.base import ChatMessage, LlmError

if TYPE_CHECKING:
    import ollama


class OllamaLLM:
    def __init__(self, host: str, model: str, timeout_s: float = 120.0) -> None:
        self.host, self.model, self.timeout_s = host, model, timeout_s
        self.name = f"ollama/{model}"
        self._client: ollama.Client | None = None

    def _get_client(self) -> "ollama.Client":
        if self._client is None:
            import ollama

            self._client = ollama.Client(host=self.host, timeout=self.timeout_s)
        return self._client

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        seed: int | None = None,
    ) -> str:
        import httpx
        import ollama

        options: dict[str, Any] = {"temperature": temperature}
        if seed is not None:
            options["seed"] = seed
        try:
            response = self._get_client().chat(
                model=self.model,
                messages=[m.model_dump() for m in messages],
                format=json_schema or None,
                options=options,
            )
        except (ollama.ResponseError, ConnectionError, httpx.HTTPError) as exc:
            raise LlmError(f"{self.name}: {type(exc).__name__}: {exc}") from exc
        return response.message.content or ""
