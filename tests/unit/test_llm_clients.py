"""Request shapes of the LLM clients with their SDK clients replaced: no network, no keys."""

from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import SecretStr

from necrohub.llm.base import ChatMessage, LlmError

MESSAGES = [
    ChatMessage(role="system", content="sys"),
    ChatMessage(role="user", content="oi"),
    ChatMessage(role="assistant", content="olá"),
    ChatMessage(role="user", content="de novo"),
]


def test_ollama_request_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    ollama = pytest.importorskip("ollama")
    from necrohub.llm.ollama import OllamaLLM

    calls: list[dict[str, Any]] = []

    class Client:
        def __init__(self, host: str, timeout: float) -> None:
            calls.append({"host": host, "timeout": timeout})

        def chat(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            return SimpleNamespace(message=SimpleNamespace(content="resposta"))

    monkeypatch.setattr(ollama, "Client", Client)
    llm = OllamaLLM("http://ollama.example.org:11434", "m:1b", timeout_s=30)
    schema = {"type": "object"}
    assert llm.complete(MESSAGES, json_schema=schema, temperature=0.8, seed=3) == "resposta"
    assert llm.complete(MESSAGES) == "resposta"
    assert calls[0] == {"host": "http://ollama.example.org:11434", "timeout": 30}
    assert calls[1] == {
        "model": "m:1b",
        "messages": [m.model_dump() for m in MESSAGES],
        "format": schema,
        "options": {"temperature": 0.8, "seed": 3},
    }
    assert calls[2]["format"] is None and calls[2]["options"] == {"temperature": 0.2}
    assert len(calls) == 3  # one client, built lazily


def test_ollama_connection_error_becomes_llm_error(monkeypatch: pytest.MonkeyPatch) -> None:
    ollama = pytest.importorskip("ollama")
    from necrohub.llm.ollama import OllamaLLM

    class Client:
        def __init__(self, host: str, timeout: float) -> None:
            pass

        def chat(self, **kwargs: Any) -> Any:
            raise ConnectionError("refused")

    monkeypatch.setattr(ollama, "Client", Client)
    with pytest.raises(LlmError, match="ollama/m"):
        OllamaLLM("http://localhost:1", "m").complete(MESSAGES)


def test_gemini_request_shape_and_secret(monkeypatch: pytest.MonkeyPatch) -> None:
    genai = pytest.importorskip("google.genai")
    from google.genai import errors

    from necrohub.llm.gemini import GeminiLLM

    calls: list[dict[str, Any]] = []
    key = "not-a-real-key-" + "x" * 8

    class Models:
        def generate_content(self, **kwargs: Any) -> Any:
            calls.append(kwargs)
            if len(calls) > 1:
                raise errors.ClientError(400, {"error": {"message": "bad request"}})
            return SimpleNamespace(text="resposta")

    class Client:
        def __init__(self, api_key: str, http_options: Any) -> None:
            assert api_key == key and http_options.timeout == 30_000
            self.models = Models()

    monkeypatch.setattr(genai, "Client", Client)
    llm = GeminiLLM(SecretStr(key), "gemini-x", timeout_s=30)
    assert key not in repr(llm)
    assert llm.complete(MESSAGES, temperature=0.8, seed=3) == "resposta"
    config = calls[0]["config"]
    assert calls[0]["model"] == "gemini-x"
    assert config.system_instruction == "sys" and config.seed == 3 and config.temperature == 0.8
    assert [c.role for c in calls[0]["contents"]] == ["user", "model", "user"]
    with pytest.raises(LlmError) as info:  # a 400 is not retried
        llm.complete(MESSAGES)
    assert len(calls) == 2 and key not in str(info.value)


@pytest.mark.parametrize("error", ["httpx", "connection", "timeout"])
def test_gemini_transport_errors_become_llm_error(
    monkeypatch: pytest.MonkeyPatch, error: str
) -> None:
    genai = pytest.importorskip("google.genai")
    import httpx

    from necrohub.llm.gemini import GeminiLLM

    detail = "detail-that-must-not-leak"
    exc = {
        "httpx": httpx.ConnectError(detail),
        "connection": ConnectionError(detail),
        "timeout": TimeoutError(detail),
    }[error]

    class Models:
        def generate_content(self, **kwargs: Any) -> Any:
            raise exc

    class Client:
        def __init__(self, api_key: str, http_options: Any) -> None:
            self.models = Models()

    monkeypatch.setattr(genai, "Client", Client)
    with pytest.raises(LlmError, match=type(exc).__name__) as info:
        GeminiLLM(SecretStr("k"), "gemini-x").complete(MESSAGES)
    assert detail not in str(info.value)


def test_get_llm_needs_a_gemini_key(monkeypatch: pytest.MonkeyPatch) -> None:
    from necrohub.llm.base import get_llm

    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LlmError, match="GEMINI_API_KEY"):
        get_llm("gemini", "gemini-x")
