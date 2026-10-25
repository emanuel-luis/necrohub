"""Real LLM backends: a running Ollama server or a Gemini key. Never run in CI."""

import os

import pytest
from pydantic import SecretStr

from necrohub.llm.base import ChatMessage, LlmError

pytestmark = pytest.mark.external

MESSAGES = [
    ChatMessage(role="system", content="Responda em uma palavra."),
    ChatMessage(role="user", content="Qual é a cor do céu num dia limpo?"),
]


def test_ollama_completes() -> None:
    pytest.importorskip("ollama")
    from necrohub.llm.ollama import OllamaLLM
    from necrohub.params import LlmParams
    from necrohub.settings import Settings

    llm = OllamaLLM(Settings().ollama_host, LlmParams().model, timeout_s=120)
    try:
        out = llm.complete(MESSAGES, temperature=0.0, seed=7)
    except LlmError as exc:
        pytest.skip(f"Ollama unreachable or model missing: {exc}")
    assert out.strip()


def test_gemini_completes() -> None:
    pytest.importorskip("google.genai")
    key = os.environ.get("GEMINI_API_KEY")
    if not key:
        pytest.skip("GEMINI_API_KEY is not set")
    from necrohub.llm.gemini import GeminiLLM

    out = GeminiLLM(SecretStr(key), "gemini-2.5-flash").complete(MESSAGES, temperature=0.0, seed=7)
    assert out.strip()
