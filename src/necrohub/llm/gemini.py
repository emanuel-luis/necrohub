"""Gemini through the AI Studio API (`genai` extra). Optional, paid, never used in CI."""

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from pydantic import SecretStr

from necrohub.llm.base import ChatMessage, LlmError

if TYPE_CHECKING:
    from google import genai

_ATTEMPTS = 3


def _transient(exc: BaseException) -> bool:
    from google.genai import errors

    return isinstance(exc, errors.ServerError) or (
        isinstance(exc, errors.ClientError) and exc.code == 429
    )


class GeminiLLM:
    def __init__(self, api_key: SecretStr, model: str) -> None:
        self._api_key, self.model = api_key, model
        self.name = f"gemini/{model}"
        self._client: genai.Client | None = None

    def __repr__(self) -> str:  # never show the key
        return f"GeminiLLM(model={self.model!r})"

    def _get_client(self) -> "genai.Client":
        if self._client is None:
            from google import genai

            self._client = genai.Client(api_key=self._api_key.get_secret_value())
        return self._client

    def complete(
        self,
        messages: Sequence[ChatMessage],
        *,
        json_schema: dict[str, Any] | None = None,
        temperature: float = 0.2,
        seed: int | None = None,
    ) -> str:
        from google.genai import errors, types
        from tenacity import Retrying, retry_if_exception, stop_after_attempt, wait_exponential

        system = "\n\n".join(m.content for m in messages if m.role == "system") or None
        contents = [
            types.Content(
                role="model" if m.role == "assistant" else "user",
                parts=[types.Part(text=m.content)],
            )
            for m in messages
            if m.role != "system"
        ]
        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            seed=seed,
            response_mime_type="application/json" if json_schema else None,
            response_json_schema=json_schema,
        )
        client = self._get_client()
        try:
            # retry only the API call, a bounded number of times
            for attempt in Retrying(
                stop=stop_after_attempt(_ATTEMPTS),
                wait=wait_exponential(multiplier=1, max=20),
                retry=retry_if_exception(_transient),
                reraise=True,
            ):
                with attempt:
                    response = client.models.generate_content(
                        model=self.model, contents=contents, config=config
                    )
        except errors.APIError as exc:
            raise LlmError(f"{self.name}: {type(exc).__name__} {exc.code}") from exc
        return response.text or ""
